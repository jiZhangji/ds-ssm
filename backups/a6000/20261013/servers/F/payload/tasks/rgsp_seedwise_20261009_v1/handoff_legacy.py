#!/usr/bin/env python3
"""Explicit parent-only handoff of two already-running legacy RGSP lanes.

inspect is read-only. freeze/finalize require separate confirmation flags.
Only identity-pinned scheduler parents receive signals, never children or groups.
The script does not create/resume training and never writes old result files.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from manifest import absolute, atomic_json, digest, frozen, read_json, sha, utc, within

SCHEMA = "rgsp-legacy-handoff-v1"
SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def process(pid):
    """Read only process identity/signal metadata; never read its environment."""
    root = Path("/proc") / str(int(pid))
    try:
        stat = (root / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        argv = [x.decode(errors="replace") for x in (root / "cmdline").read_bytes().split(b"\0") if x]
        status = {}
        for line in (root / "status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Uid", "SigIgn", "SigCgt", "SigBlk", "SigPnd", "ShdPnd", "Threads"}:
                status[key] = value.strip()
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "process_group": int(fields[2]), "session": int(fields[3]),
                "start_ticks": int(fields[19]), "exit_code": int(fields[49]),
                "argv": argv, "signal_metadata": status}
    except (FileNotFoundError, ProcessLookupError):
        return None


def require_identity(observed, expected_pid, expected_start, expected_argv=None, allowed_states=None):
    if observed is None or observed["pid"] != expected_pid or observed["start_ticks"] != expected_start:
        raise RuntimeError(f"Missing/reused PID; refusing to act: {expected_pid}")
    if expected_argv is not None and observed["argv"] != expected_argv:
        raise RuntimeError(f"PID command changed; refusing to act: {expected_pid}")
    if allowed_states is not None and observed["state"] not in allowed_states:
        raise RuntimeError(f"Unexpected process state for {expected_pid}: {observed['state']}")
    if int(observed["signal_metadata"]["Uid"].split()[0]) != os.geteuid():
        raise RuntimeError("Scheduler/training process does not belong to the current user")
    return observed


def scheduler_argv(spec, lane):
    return [spec["python"], str(absolute(spec["recovery_root"]) / "recovery_queue.py"), "queue", "--key", lane["id"]]


def child_argv(spec, lane):
    return [spec["python"], str(absolute(spec["recovery_root"]) / "train_redirect.py"),
            "--code", spec["frozen_code_root"], "--output", lane["recovery_lane"],
            "--variant", lane["variant"], "--seed", str(lane["seed"]), "--epochs", "150", "--patience", "30"]


def gpu(lane):
    out = subprocess.check_output(["nvidia-smi", "-i", str(lane["gpu_index"]),
           "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=20)
    rows = list(csv.reader(out.strip().splitlines()))
    if len(rows) != 1 or rows[0][0].strip() != lane["gpu_uuid"]:
        raise RuntimeError("GPU index/UUID identity changed")
    row = [v.strip() for v in rows[0]]
    pids = subprocess.check_output(["nvidia-smi", "-i", lane["gpu_uuid"], "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    return {"uuid": row[0], "name": row[1], "memory_mib": int(row[2]),
            "utilization_percent": int(row[3]), "pids": [int(p) for p in pids.splitlines()] if pids else []}


def load_spec(path):
    spec = read_json(absolute(path))
    if spec.get("schema") != SCHEMA or socket.gethostname() != spec["hostname"]:
        raise RuntimeError("Wrong handoff schema/host")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != spec["boot_id"]:
        raise RuntimeError("Server rebooted; old PID identities cannot be used")
    for key in ("python", "recovery_root", "frozen_code_root", "campaign_root", "evidence_dir"):
        absolute(spec[key])
    within(spec["evidence_dir"], spec["campaign_root"])
    if len(spec["lanes"]) != 2 or {r["id"] for r in spec["lanes"]} != {"f0", "f1"}:
        raise RuntimeError("This handoff is restricted to the two explicitly inspected F lanes")
    for lane in spec["lanes"]:
        if lane["seed"] != (0 if lane["id"] == "f0" else 1) or lane["variant"] != "local_energy":
            raise RuntimeError("Only the inspected, already-started local_energy trials may be adopted")
        within(lane["recovery_lane"], spec["recovery_root"])
        within(lane["adopted_dir"], spec["campaign_root"])
        absolute(lane["source_lane"])
        if lane["parent_pid"] == lane["child_pid"]:
            raise RuntimeError("Parent and training child must be distinct PIDs")
    return spec


def inspect_lane(spec, lane, frozen_parent=False):
    parent = require_identity(process(lane["parent_pid"]), lane["parent_pid"], lane["parent_start_ticks"],
                              scheduler_argv(spec, lane), {"T"} if frozen_parent else {"S", "R", "D"})
    child = require_identity(process(lane["child_pid"]), lane["child_pid"], lane["child_start_ticks"],
                             child_argv(spec, lane), {"R", "S", "D"})
    if child["ppid"] != parent["pid"]:
        raise RuntimeError("Training process is no longer the direct child of the inspected scheduler")
    info = gpu(lane)
    if info["pids"] != [child["pid"]]:
        raise RuntimeError(f"GPU does not contain exactly the inspected training child: {info}")
    root = absolute(lane["recovery_lane"])
    state = read_json(root / "queue_state.json")
    recovery = read_json(root / "recovery_manifest.json")
    expected = {"pid": parent["pid"], "host": spec["hostname"], "gpu_uuid": lane["gpu_uuid"],
                "current": lane["variant"], "state": "running", "command": child_argv(spec, lane)}
    if any(state.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Legacy queue state no longer describes the inspected active child")
    expected_recovery = {"key": lane["id"], "host": spec["hostname"], "seed": lane["seed"],
                         "gpu_uuid": lane["gpu_uuid"], "output": lane["recovery_lane"],
                         "code": spec["frozen_code_root"], "source_lane": lane["source_lane"]}
    if any(recovery.get(k) != v for k, v in expected_recovery.items()):
        raise RuntimeError("Recovery manifest does not match explicit handoff paths and identity")
    return {"parent": parent, "child": child, "gpu": info, "queue_state": state,
            "recovery_manifest_sha256": sha(root / "recovery_manifest.json"), "observed_utc": utc()}


def audit_code(spec, lane):
    root, code = absolute(lane["recovery_lane"]), absolute(spec["frozen_code_root"])
    recovery = read_json(root / "recovery_manifest.json")
    path = absolute(lane["source_lane"]) / "manifest/protocol.json"
    if sha(path) != recovery["protocol_sha256"]:
        raise RuntimeError("Original protocol hash changed")
    protocol = read_json(path)
    if protocol["epochs"] != 150 or protocol["seed"] != lane["seed"] or protocol["batch"] != 16:
        raise RuntimeError("Unexpected original SSDD protocol")
    for name, expected in protocol["code"].items():
        if sha(within(code / name, code)) != expected:
            raise RuntimeError(f"Frozen legacy code hash changed: {name}")
    if sha(root / "manifest/configs" / (lane["variant"] + ".yaml")) != protocol["config_sha256"][lane["variant"]]:
        raise RuntimeError("Legacy training config changed")
    if sha(root / "manifest/data_content.json") != protocol["data_manifest_sha256"]:
        raise RuntimeError("Legacy data manifest changed")
    return {"protocol_path": str(path), "protocol_sha256": sha(path), "code_files_verified": len(protocol["code"]),
            "data_manifest_sha256": protocol["data_manifest_sha256"]}


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def journal(spec, lane_id, event):
    root = absolute(spec["evidence_dir"])
    root.mkdir(parents=True, exist_ok=True)
    row = {"utc": utc(), "spec_sha256": digest(spec), "operator_pid": os.getpid(), **event}
    with (root / (lane_id + ".actions.jsonl")).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pinned_parent(spec, lane, state):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("pidfd support is required; no PID-racy signal fallback")
    fd = os.pidfd_open(lane["parent_pid"], 0)
    try:
        require_identity(process(lane["parent_pid"]), lane["parent_pid"], lane["parent_start_ticks"],
                         scheduler_argv(spec, lane), state)
    except BaseException:
        os.close(fd)
        raise
    return fd


def send_parent(spec, lane, fd, sig):
    # pidfd targets exactly one previously verified process. Never killpg/-PID.
    journal(spec, lane["id"], {"action": "signal_requested", "parent_pid": lane["parent_pid"],
                              "signal": signal.Signals(sig).name, "training_child_not_signaled": True})
    signal.pidfd_send_signal(fd, sig, None, 0)
    journal(spec, lane["id"], {"action": "signal_sent", "parent_pid": lane["parent_pid"],
                              "signal": signal.Signals(sig).name})


def freeze(spec):
    evidence = absolute(spec["evidence_dir"])
    handle = lock(evidence / "freeze.lock")
    if (evidence / "freeze_receipt.json").exists():
        raise RuntimeError("Freeze receipt already exists; inspect it instead of issuing duplicate signals")
    observed = {lane["id"]: inspect_lane(spec, lane) for lane in spec["lanes"]}
    for lane in spec["lanes"]:
        audit_code(spec, lane)
        root = absolute(lane["recovery_lane"])
        dest = evidence / lane["id"] / "pre_freeze"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("queue_state.json", "recovery_manifest.json", "launch_receipt.json"):
            source = root / name
            if source.exists():
                data = source.read_bytes()
                target = dest / name
                with target.open("xb") as stream:
                    stream.write(data)
                frozen(dest / (name + ".sha256.json"), {"source": str(source), "sha256": hashlib.sha256(data).hexdigest()})
    frozen(evidence / "handoff_spec.json", spec)
    frozen(evidence / "pre_freeze_inspection.json", observed)
    stopped = []
    try:
        for lane in spec["lanes"]:
            inspect_lane(spec, lane)  # queue/child may have advanced since the initial read
            fd = pinned_parent(spec, lane, {"R", "S", "D"})
            try:
                send_parent(spec, lane, fd, signal.SIGSTOP)
                deadline = time.monotonic() + 5
                while True:
                    parent = require_identity(process(lane["parent_pid"]), lane["parent_pid"], lane["parent_start_ticks"], scheduler_argv(spec, lane))
                    if parent["state"] == "T":
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Scheduler did not reach SIGSTOP state")
                    time.sleep(0.05)
                stopped.append(lane["id"])
                child = process(lane["child_pid"])
                require_identity(child, lane["child_pid"], lane["child_start_ticks"],
                                 None if child and child["state"] == "Z" else child_argv(spec, lane), {"R", "S", "D", "Z"})
                journal(spec, lane["id"], {"action": "parent_frozen_child_preserved", "parent": parent, "child": child})
            finally:
                os.close(fd)
        frozen(evidence / "freeze_receipt.json", {"schema": SCHEMA, "spec_sha256": digest(spec),
               "status": "both_scheduler_parents_frozen", "parent_lanes": stopped, "finished_utc": utc(),
               "script_sha256": sha(__file__), "children_were_not_signaled": True})
    except BaseException as error:
        atomic_json(evidence / "freeze_failure.json", {"spec_sha256": digest(spec), "stopped_lanes": stopped,
                    "error": repr(error), "policy": "No automatic SIGCONT rollback; already-frozen parents remain stopped for inspection"})
        raise
    finally:
        handle.close()
    return {"status": "frozen", "lanes": stopped, "children_not_signaled": True}


def parent_still_frozen(spec, lane):
    return require_identity(process(lane["parent_pid"]), lane["parent_pid"], lane["parent_start_ticks"],
                            scheduler_argv(spec, lane), {"T"})


def completed_training(spec, lane):
    parent_still_frozen(spec, lane)
    child = process(lane["child_pid"])
    require_identity(child, lane["child_pid"], lane["child_start_ticks"],
                     None if child and child["state"] == "Z" else child_argv(spec, lane), {"R", "S", "D", "Z"})
    if child["ppid"] != lane["parent_pid"]:
        raise RuntimeError("Training child parent changed while scheduler was frozen")
    if child["state"] != "Z":
        return None
    if child["exit_code"] != 0:
        raise RuntimeError(f"Training child exited unsuccessfully: {child['exit_code']}; scheduler remains frozen")
    run = absolute(lane["recovery_lane"]) / "runs" / lane["variant"]
    train = read_json(run / "train_complete.json")
    if train.get("smoke_only") or not 0 < int(train["last_epoch_observed"]) <= 150:
        raise RuntimeError("No valid formal training-complete marker")
    if sha(run / "weights/best.pt") != train["best_sha256"]:
        raise RuntimeError("Completed training marker does not match best checkpoint")
    return {"child": child, "source_run": str(run), "train_complete": train,
            "train_complete_sha256": sha(run / "train_complete.json"), "best_sha256": train["best_sha256"]}


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    expected = sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise RuntimeError(f"Independent adoption destination already differs: {target}")
    else:
        # Exclusive destination creation; never overwrite an existing artifact.
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if sha(target) != expected or sha(source) != expected:
        raise RuntimeError("Source/destination changed while copying adoption evidence")
    return {"source": str(source), "path": str(target), "sha256": expected}


def stage_adoption(spec, lane, done):
    adopted = within(lane["adopted_dir"], spec["campaign_root"])
    source_run, recovery = Path(done["source_run"]), absolute(lane["recovery_lane"])
    adopted.mkdir(parents=True, exist_ok=True)
    copies = [copy_verified(source_run / "train_complete.json", adopted / "train_complete.json"),
              copy_verified(source_run / "weights/best.pt", adopted / "weights/best.pt"),
              copy_verified(absolute(lane["source_lane"]) / "manifest/protocol.json", adopted / "protocol.json"),
              copy_verified(recovery / "manifest/data_content.json", adopted / "data_manifest.json")]
    for name in ("runtime_audit.json", "runtime_audit_resume.json", "controls_by_epoch.jsonl", "results.csv", "args.yaml"):
        if (source_run / name).exists():
            copies.append(copy_verified(source_run / name, adopted / "source_training" / name))
    inventory = read_json(adopted / "data_manifest.json")
    counts = {}
    for row in inventory:
        split, name = row["split"], row.get("name", row.get("image"))
        if split not in {"train", "val", "test"} or not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("Invalid legacy dataset inventory")
        counts[split] = counts.get(split, 0) + 1
        for kind, filename, key in (("images", name, "image_sha256"), ("labels", Path(name).stem + ".txt", "label_sha256")):
            source = recovery / "data" / kind / split / filename
            if sha(source) != row[key]:
                raise RuntimeError("Legacy dataset content no longer matches frozen inventory")
            copy_verified(source, adopted / "data" / kind / split / filename)
    if counts != {"train": 730, "val": 186, "test": 232}:
        raise RuntimeError(f"Unexpected SSDD split: {counts}")
    # JSON is valid YAML; use stdlib only. All evaluator-created caches stay in the new copy.
    frozen(adopted / "data.yaml", {"path": str(adopted / "data"), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {"0": "ship"}})
    frozen(adopted / "adoption_provenance.json", {"schema": SCHEMA, "spec_sha256": digest(spec),
           "lane": lane["id"], "seed": lane["seed"], "variant": lane["variant"], "copies": copies,
           "source_training_not_signaled": True, "child_exit_code": done["child"]["exit_code"],
           "source_train_complete_sha256": done["train_complete_sha256"], "best_sha256": done["best_sha256"],
           "grandfathered_started_seed1": lane["seed"] == 1,
           "limitations": "Already-running legacy frozen-code trial; same seed is not cross-architecture tensor alignment. Seed1 began before the global seedwise policy and does not authorize any additional seed1."})
    return adopted


def retire_parent(spec, lane):
    parent = parent_still_frozen(spec, lane)
    mask = 1 << (signal.SIGTERM - 1)
    if any(int(parent["signal_metadata"][key], 16) & mask for key in ("SigIgn", "SigCgt", "SigBlk")):
        raise RuntimeError("Unexpected SIGTERM handler/ignore/block mask; not safe to resume scheduler for retirement")
    fd = pinned_parent(spec, lane, {"T"})
    try:
        send_parent(spec, lane, fd, signal.SIGTERM)
        # Default SIGTERM is pending on the stopped parent. SIGCONT allows default
        # termination; no Python scheduler loop can handle/ignore SIGTERM here.
        send_parent(spec, lane, fd, signal.SIGCONT)
        deadline = time.monotonic() + 10
        while True:
            after = process(lane["parent_pid"])
            if after is None or (after["start_ticks"] == lane["parent_start_ticks"] and after["state"] == "Z"):
                break
            require_identity(after, lane["parent_pid"], lane["parent_start_ticks"], scheduler_argv(spec, lane))
            if time.monotonic() >= deadline:
                raise RuntimeError("Scheduler did not retire after parent-only TERM+CONT; no stronger signal issued")
            time.sleep(0.1)
        journal(spec, lane["id"], {"action": "scheduler_retired", "parent_after": after})
        return after
    finally:
        os.close(fd)


def finalize(spec, lane, wait_seconds):
    evidence = absolute(spec["evidence_dir"])
    receipt = read_json(evidence / "freeze_receipt.json")
    if receipt.get("spec_sha256") != digest(spec) or receipt.get("status") != "both_scheduler_parents_frozen":
        raise RuntimeError("No matching successful freeze receipt")
    lane_lock = lock(evidence / (lane["id"] + ".finalize.lock"))
    gpu_lock = lock(absolute(spec["campaign_root"]) / "locks" / (lane["gpu_uuid"] + ".lock"))
    deadline = time.monotonic() + max(0, wait_seconds)
    try:
        while True:
            done = completed_training(spec, lane)
            atomic_json(evidence / (lane["id"] + ".status.json"), {"spec_sha256": digest(spec),
                        "state": "training_complete" if done else "waiting_existing_child", "checked_utc": utc(),
                        "parent_pid": lane["parent_pid"], "child_pid": lane["child_pid"]})
            if done:
                break
            if time.monotonic() >= deadline:
                return {"status": "waiting_existing_child", "lane": lane["id"], "scheduler_remains_frozen": True}
            time.sleep(min(30, max(0.1, deadline - time.monotonic())))
        audit = audit_code(spec, lane)
        adopted = stage_adoption(spec, lane, done)
        frozen(adopted / "legacy_code_audit.json", audit)
        snapshot = gpu(lane)
        if snapshot["pids"] or snapshot["memory_mib"] > 2500 or snapshot["utilization_percent"] > 10:
            raise RuntimeError(f"Assigned GPU is not idle after child exit; no takeover: {snapshot}")
        parent_still_frozen(spec, lane)
        eval_path = adopted / "eval_test.json"
        completion = adopted / "evaluation_complete.json"
        if not completion.exists():
            if eval_path.exists():
                raise RuntimeError("An uncompleted adopted evaluation exists; inspect before retrying")
            command = [spec["python"], str(absolute(spec["frozen_code_root"]) / "tools/evaluate_sarmix.py"),
                       "--weights", str(adopted / "weights/best.pt"), "--data", str(adopted / "data.yaml"),
                       "--experiment", "rgsp_adopted_" + lane["id"] + "_local_energy", "--placement", "p3",
                       "--variant", "local_energy", "--state-dim", "16", "--seed", str(lane["seed"]),
                       "--subset", "SSDD", "--split", "test", "--output-json", str(eval_path),
                       "--imgsz", "640", "--batch", "16", "--device", str(lane["gpu_index"]), "--workers", "0",
                       "--conf", "0.001", "--iou", "0.6", "--project", str(adopted / "evaluation"), "--name", "test"]
            env = os.environ.copy()
            # This old evaluator expects a physical ordinal. Its current mapping
            # was checked against the fixed UUID immediately above; frozen code is unchanged.
            env.update(CUDA_VISIBLE_DEVICES=str(lane["gpu_index"]), PYTHONPATH=spec["frozen_code_root"],
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       YOLO_AUTOINSTALL="false", WANDB_MODE="disabled", YOLO_CONFIG_DIR=str(adopted / "runtime-config"),
                       MPLCONFIGDIR=str(adopted / "matplotlib"))
            journal(spec, lane["id"], {"action": "independent_test_started", "argv": command, "gpu": snapshot})
            with (adopted / "evaluation.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=spec["frozen_code_root"], env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Independent evaluation failed with exit {result.returncode}; parent remains frozen")
            evaluation = read_json(eval_path)
            if (evaluation.get("weights_sha256") != done["best_sha256"] or evaluation.get("seed") != lane["seed"]
                    or evaluation.get("split") != "test" or sha(adopted / "weights/best.pt") != done["best_sha256"]):
                raise RuntimeError("Independent evaluation is not bound to the completed best/seed/test")
            audit_code(spec, lane)
            frozen(completion, {"status": "complete", "best_sha256": done["best_sha256"], "eval_test_sha256": sha(eval_path),
                   "exit_code": 0, "smoke_only": False, "finished_utc": utc(), "independent_frozen_code_test": True})
        check = read_json(completion)
        if check["best_sha256"] != done["best_sha256"] or check["eval_test_sha256"] != sha(eval_path):
            raise RuntimeError("Existing adopted completion artifact changed")
        if sha(Path(done["source_run"]) / "weights/best.pt") != done["best_sha256"]:
            raise RuntimeError("Original completed best changed during adoption")
        retired = retire_parent(spec, lane)
        final = {"schema": SCHEMA, "spec_sha256": digest(spec), "status": "adopted_and_scheduler_retired",
                 "lane": lane["id"], "seed": lane["seed"], "adopted_dir": str(adopted), "best_sha256": done["best_sha256"],
                 "eval_test_sha256": sha(eval_path), "retired_parent": retired, "finished_utc": utc(),
                 "child_not_signaled": True, "grandfathered_started_seed1": lane["seed"] == 1}
        frozen(evidence / (lane["id"] + ".complete.json"), final)
        atomic_json(evidence / (lane["id"] + ".status.json"), final)
        return final
    except BaseException as error:
        atomic_json(evidence / (lane["id"] + ".status.json"), {"status": "failed", "spec_sha256": digest(spec),
                    "error": repr(error), "updated_utc": utc(), "policy": "No child/group signal, no automatic retry or unfreeze"})
        raise
    finally:
        gpu_lock.close()
        lane_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "freeze", "finalize"))
    parser.add_argument("--spec", required=True)
    parser.add_argument("--lane", choices=("f0", "f1"))
    parser.add_argument("--confirm-parent-only-freeze", action="store_true")
    parser.add_argument("--confirm-evaluate-and-retire", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Finite wait for the existing child only; operator may detach this foreground process")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.mode == "inspect":
        output = {"schema": SCHEMA, "mode": "read_only", "spec_sha256": digest(spec),
                  "lanes": {lane["id"]: inspect_lane(spec, lane) for lane in spec["lanes"]}}
    elif args.mode == "freeze":
        if not args.confirm_parent_only_freeze:
            parser.error("freeze requires --confirm-parent-only-freeze after operator review")
        output = freeze(spec)
    else:
        if not args.confirm_evaluate_and_retire or args.lane is None:
            parser.error("finalize requires --lane and --confirm-evaluate-and-retire after operator review")
        if args.wait_seconds < 0 or args.wait_seconds > 172800:
            parser.error("wait-seconds must be finite and between 0 and 172800")
        output = finalize(spec, next(lane for lane in spec["lanes"] if lane["id"] == args.lane), args.wait_seconds)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
