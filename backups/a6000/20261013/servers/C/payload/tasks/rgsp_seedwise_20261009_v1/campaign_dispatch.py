#!/usr/bin/env python3
"""Finite host dispatch: explicit dependencies -> seed0 -> global barrier -> seed1.

No SSH, HTTP service, installation, key access, automatic retry, signal or job
discovery. --detach starts only this finite dispatcher in a new Linux session.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from manifest import (absolute, atomic_json, digest, frozen, load_plan, read_json,
                      sha, trial_by_id, utc, verify_barrier, verify_code, within)

SCHEMA = "rgsp-campaign-dispatch-v1"
HOST_KEYS = ("a", "b", "c", "f0", "f1")
MAX_WAIT = 7 * 24 * 60 * 60


def process_identity(pid, expected_start=None):
    """Read argv only for the expected process, not an unrelated reused PID."""
    root = Path("/proc") / str(int(pid))
    try:
        text = (root / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        result = {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                  "session": int(fields[3]), "start_ticks": int(fields[19])}
        if expected_start is None or result["start_ticks"] == expected_start:
            result["argv"] = [v.decode(errors="replace") for v in (root / "cmdline").read_bytes().split(b"\0") if v]
        return result
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(expected):
    if not expected or "pid" not in expected or "start_ticks" not in expected:
        return False
    current = process_identity(expected["pid"], expected["start_ticks"])
    if current is None or current["start_ticks"] != expected["start_ticks"] or current["state"] == "Z":
        return False
    if current.get("argv") != expected.get("argv"):
        raise RuntimeError("An expected live process changed argv; inspect before another dispatch")
    return True


def selected_lanes(plan, key):
    entry = plan["dispatch_map"][key]
    if socket.gethostname() != entry["hostname"]:
        raise RuntimeError(f"Wrong host for dispatch {key}: {socket.gethostname()}")
    lane_map = {lane["id"]: lane for lane in plan["lanes"]}
    selected = {}
    for seed in (0, 1):
        lane = lane_map[entry[f"seed{seed}_lane"]]
        if lane["hostname"] != entry["hostname"] or lane["gpu_uuid"] != entry["gpu_uuid"]:
            raise RuntimeError("Dispatch and lane host/GPU identity differ")
        if any(trial_by_id(plan, tid)["seed"] != seed for tid in lane["trial_ids"]):
            raise RuntimeError(f"Seed{seed} lane contains a different seed; no execution")
        selected[seed] = lane
    if selected[0]["id"] == selected[1]["id"]:
        raise RuntimeError("The seed phases require distinct finite lane IDs")
    for requirement in entry.get("requires", []):
        within(requirement["path"], plan["campaign_root"])
        if not isinstance(requirement.get("equals"), dict) or not requirement["equals"]:
            raise RuntimeError("Each dependency needs explicit exact expected fields")
    if key.startswith("f"):
        if not entry.get("requires") or not entry.get("finalizer_launch"):
            raise RuntimeError("F dispatch requires completion dependencies and an explicit finalizer_launch path")
        within(entry["finalizer_launch"], plan["campaign_root"])
    return entry, selected


def acquire_lock(path, inherited_fd=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if inherited_fd is not None:
        actual, expected = os.fstat(inherited_fd), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("Inherited dispatch lock is not the expected lock file")
        handle = os.fdopen(inherited_fd, "a+")
    else:
        handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise RuntimeError(f"Another dispatcher owns {path}")
    return handle


def gpu_lock_available(plan, entry):
    path = absolute(plan["campaign_root"]) / "locks" / (entry["gpu_uuid"] + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


def dependencies(plan, entry):
    missing, observed = [], []
    finalizer = None
    finalizer_alive = False
    if entry.get("finalizer_launch"):
        path = absolute(entry["finalizer_launch"])
        if not path.is_file():
            raise RuntimeError(f"Reviewed finalizer launch evidence is missing: {path}")
        launch = read_json(path)
        # Observe finalizer liveness before completion paths: an exited finalizer
        # cannot publish a marker between a missing-marker check and this check.
        finalizer = {key: launch[key] for key in ("pid", "start_ticks", "argv")}
        finalizer_alive = identity_alive(finalizer)
        observed.append({"path": str(path), "sha256": sha(path),
                         "finalizer": finalizer, "finalizer_exited": not finalizer_alive})
    for requirement in entry.get("requires", []):
        path = absolute(requirement["path"])
        if not path.exists():
            missing.append(str(path))
            continue
        item = read_json(path)
        for key, expected in requirement["equals"].items():
            value = item.get(key)
            if type(value) is not type(expected) or value != expected:
                raise RuntimeError(f"Dependency has an unexpected result: {path}, {key}={value!r}")
        observed.append({"path": str(path), "sha256": sha(path), "equals": requirement["equals"]})
    if missing:
        if finalizer is not None and not finalizer_alive:
            raise RuntimeError(f"Reviewed finalizer exited without required completion evidence: {missing}")
        return False, {"missing": missing, "verified": observed}
    if finalizer_alive:
        return False, {"waiting_finalizer_exit": finalizer, "verified": observed}
    if not gpu_lock_available(plan, entry):
        return False, {"waiting_campaign_gpu_lock": entry["gpu_uuid"], "verified": observed}
    return True, {"verified": observed}


def assert_no_old_live_owner(root, plan):
    for path, fields in ((root / "launch.json", ("owner",)), (root / "status.json", ("owner", "child"))):
        if not path.exists():
            continue
        old = read_json(path)
        if old.get("plan_sha256") != digest(plan):
            raise RuntimeError("This dispatch directory belongs to a different immutable plan")
        for field in fields:
            if identity_alive(old.get(field)):
                raise RuntimeError(f"Existing {field} is still alive; do not dispatch twice: {path}")


def run_dispatch(plan_path, plan, key, entry, lanes, root, attempt, owner):
    status = {"schema": SCHEMA, "plan_sha256": digest(plan), "host_key": key,
              "hostname": socket.gethostname(), "gpu_uuid": entry["gpu_uuid"],
              "owner": owner, "attempt_dir": str(attempt), "started_utc": utc(),
              "completed_seeds": [], "child": None}

    def update(**values):
        status.update(values, updated_utc=utc())
        atomic_json(attempt / "status.json", status)
        atomic_json(root / "status.json", status)

    def wait_dependencies():
        deadline = time.monotonic() + MAX_WAIT
        while True:
            ready, evidence = dependencies(plan, entry)
            update(state="dependencies_ready" if ready else "waiting_dependencies", dependency_evidence=evidence)
            if ready:
                frozen(attempt / "dependencies_verified.json", evidence)
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Seven-day finite dependency wait expired; no task skipped")
            time.sleep(min(30, max(0.1, deadline - time.monotonic())))

    def execute_seed(seed):
        verify_code(plan)
        if seed == 1:
            not_ready = [tid for tid in plan["seed0_trial_ids"] if trial_by_id(plan, tid)["kind"] == "not_ready"]
            if not_ready:
                raise RuntimeError(f"Seed1 remains forbidden while intended seed0 trials are not ready: {not_ready}")
            verify_barrier(plan)
        lane = lanes[seed]
        command = [plan["python"], "-B", "-u", str(Path(__file__).with_name("run_lane.py")),
                   "--plan", str(plan_path), "--lane", lane["id"]]
        log_path = attempt / f"seed{seed}.log"
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, cwd=plan["campaign_root"], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            identity = process_identity(child.pid) or {"pid": child.pid, "already_exited": True}
            identity["argv"] = command
            record = {"seed": seed, "lane": lane["id"], "child": identity, "command": command,
                      "log": str(log_path), "started_utc": utc()}
            frozen(attempt / f"seed{seed}.child_launch.json", record)
            update(state=f"running_seed{seed}", seed=seed, child=identity, child_log=str(log_path))
            result = child.wait()
        finished = {**record, "returncode": result, "finished_utc": utc()}
        frozen(attempt / f"seed{seed}.child_exit.json", finished)
        status["child"] = {**identity, "returncode": result}
        if result:
            raise RuntimeError(f"Seed{seed} lane failed with exit={result}; no retry/kill/further phase: {log_path}")
        complete_path = absolute(lane["output_dir"]) / "lane_complete.json"
        complete = read_json(complete_path)
        if (complete.get("plan_sha256") != digest(plan) or complete.get("lane") != lane["id"]
                or complete.get("trial_ids") != lane["trial_ids"] or complete.get("completed") != lane["trial_ids"]):
            raise RuntimeError("Lane exited without exact full-manifest completion evidence")
        frozen(attempt / f"seed{seed}.lane_verified.json", {"path": str(complete_path), "sha256": sha(complete_path),
               "plan_sha256": digest(plan), "trial_ids": lane["trial_ids"]})
        status["completed_seeds"].append(seed)
        update(state=f"seed{seed}_lane_complete")

    try:
        frozen(attempt / "owner_start.json", {"plan_sha256": digest(plan), "owner": owner, "started_utc": utc()})
        update(state="checking_dependencies")
        wait_dependencies()
        execute_seed(0)
        update(state="waiting_global_seed0", seed=0)
        deadline = time.monotonic() + MAX_WAIT
        while not absolute(plan["barrier_path"]).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("Seven-day finite global seed0 wait expired; seed1 was not started")
            time.sleep(min(30, max(0.1, deadline - time.monotonic())))
            update(state="waiting_global_seed0")
        payload = verify_barrier(plan)  # malformed, old-plan or partial barriers fail closed
        frozen(attempt / "global_seed0_barrier_verified.json", {"path": plan["barrier_path"],
               "sha256": sha(plan["barrier_path"]), "plan_sha256": digest(plan),
               "seed0_trial_ids": payload["seed0_trial_ids"]})
        execute_seed(1)
        update(state="complete", finished_utc=utc())
        frozen(attempt / "dispatch_complete.json", status)
    except BaseException as error:
        update(state="failed", error=repr(error), failed_utc=utc(),
               failure_policy="Stop, preserve evidence, no automatic retry or process signal")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--host-key", required=True, choices=HOST_KEYS)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--inherited-lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Server Linux execution only; no local fallback")
    plan_path = absolute(args.plan)
    plan = load_plan(plan_path)
    entry, lanes = selected_lanes(plan, args.host_key)
    verify_code(plan)
    if args.verify_only:
        print(json.dumps({"status": "verified", "plan_sha256": digest(plan), "host_key": args.host_key,
                          "seed0_lane": lanes[0]["id"], "seed1_lane": lanes[1]["id"]}))
        return
    root = absolute(plan["campaign_root"]) / "dispatch" / args.host_key
    handle = acquire_lock(root / "dispatch.lock", args.inherited_lock_fd)
    try:
        if args.inherited_lock_fd is None:
            assert_no_old_live_owner(root, plan)
            attempt = root / ("attempt_" + utc().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8])
            attempt.mkdir()
        else:
            if args.detach or args.attempt_dir is None:
                raise RuntimeError("Invalid inherited dispatch invocation")
            attempt = within(args.attempt_dir, root)
            if not attempt.is_dir():
                raise RuntimeError("Inherited attempt directory is missing")
        if args.detach:
            command = [plan["python"], "-B", "-u", str(Path(__file__).resolve()), "--plan", str(plan_path),
                       "--host-key", args.host_key, "--inherited-lock-fd", str(handle.fileno()), "--attempt-dir", str(attempt)]
            with (attempt / "dispatcher.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=plan["campaign_root"], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(handle.fileno(),), close_fds=True)
            time.sleep(1)
            owner = process_identity(child.pid) or {"pid": child.pid, "already_exited": True}
            record = {"schema": SCHEMA, "plan_sha256": digest(plan), "host_key": args.host_key,
                      "owner": owner, "command": command, "attempt_dir": str(attempt),
                      "log": str(attempt / "dispatcher.log"), "launched_utc": utc(), "detached": True,
                      "initial_returncode": child.poll(), "script_sha256": sha(__file__)}
            frozen(attempt / "launch.json", record)
            atomic_json(root / "launch.json", record)
            print(json.dumps(record, indent=2, sort_keys=True))
            if record["initial_returncode"] is not None:
                raise RuntimeError("Detached dispatcher exited during launch; inspect its independent log")
            return
        owner = process_identity(os.getpid())
        if args.inherited_lock_fd is None:
            record = {"schema": SCHEMA, "plan_sha256": digest(plan), "host_key": args.host_key,
                      "owner": owner, "attempt_dir": str(attempt), "launched_utc": utc(),
                      "detached": False, "script_sha256": sha(__file__)}
            frozen(attempt / "launch.json", record)
            atomic_json(root / "launch.json", record)
        run_dispatch(plan_path, plan, args.host_key, entry, lanes, root, attempt, owner)
    finally:
        handle.close()


if __name__ == "__main__":
    main()
