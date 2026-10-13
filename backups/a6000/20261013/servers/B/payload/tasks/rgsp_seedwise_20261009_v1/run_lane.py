#!/usr/bin/env python3
"""Finite preassigned lane. No task discovery, process killing, or seed unlocks."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from manifest import (SCHEMA, absolute, atomic_json, digest, frozen, load_plan,
                      read_json, receipt_path, sha, trial_by_id, utc,
                      validate_receipt, verify_barrier, verify_code, within,
                      write_receipt, check_file)


def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise RuntimeError(f"Another process holds exclusive lock: {path}")
    return handle  # caller retains handle for the whole lane lifetime


def gpu_snapshot(uuid):
    def query(fields, kind):
        command = ["nvidia-smi", "-i", uuid, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(command, text=True, timeout=20).strip()
        return list(csv.reader(output.splitlines())) if output else []
    rows = query("uuid,name,memory.used,memory.total,utilization.gpu", "gpu")
    if len(rows) != 1 or rows[0][0].strip() != uuid:
        raise RuntimeError(f"GPU identity mismatch for assigned {uuid}: {rows}")
    row = [cell.strip() for cell in rows[0]]
    processes = query("gpu_uuid,pid,process_name", "compute-apps")
    # A compute process is never whitelisted or terminated, regardless of owner.
    return {"uuid": row[0], "name": row[1], "memory_used_mib": int(row[2]),
            "memory_total_mib": int(row[3]), "utilization_percent": int(row[4]),
            "compute_processes": processes, "checked_utc": utc()}


def wait_idle(uuid, lane, status_path, status):
    hold = float(lane.get("idle_seconds", 15))
    deadline = time.monotonic() + float(lane.get("idle_timeout_seconds", 3600))
    quiet_since = None
    while True:
        snapshot = gpu_snapshot(uuid)
        quiet = (not snapshot["compute_processes"]
                 and snapshot["memory_used_mib"] <= int(lane.get("max_idle_memory_mib", 1024))
                 and snapshot["utilization_percent"] <= int(lane.get("max_idle_utilization_percent", 5)))
        status.update(state="waiting_idle", gpu=snapshot, updated_utc=utc())
        atomic_json(status_path, status)
        if quiet:
            quiet_since = quiet_since if quiet_since is not None else time.monotonic()
            if time.monotonic() - quiet_since >= hold:
                return snapshot
        else:
            quiet_since = None
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Assigned GPU did not become safely idle: {uuid}; no process was interrupted")
        time.sleep(min(5, max(0.1, hold)))


def execute(command, cwd, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"started_utc": utc(), "argv": command, "cwd": str(cwd)}) + "\n")
        handle.flush()
        # Inherit the parent's process group: nohup from the launcher is enough.
        # No timeout-kill and no automatic retry after an external failure.
        result = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(json.dumps({"finished_utc": utc(), "returncode": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError(f"Trial failed with exit {result.returncode}; stopped lane; see {log}")


def verify_reuse(plan, trial):
    """Require exact audited historical evidence; never relabel old runs as new."""
    evidence = trial["reuse"]
    required = ("train_complete", "eval_test", "evaluation_complete", "best", "protocol", "audit")
    for name in required:
        check_file(evidence[name])
    train = read_json(evidence["train_complete"]["path"])
    result = read_json(evidence["eval_test"]["path"])
    completion = read_json(evidence["evaluation_complete"]["path"])
    best_hash = evidence["best"]["sha256"]
    if train.get("smoke_only") or result.get("split") != "test":
        raise RuntimeError("Smoke/non-test results cannot complete a formal trial")
    if train.get("best_sha256") != best_hash or result.get("weights_sha256") != best_hash:
        raise RuntimeError("Historical best checkpoint is not bound to its training/test evidence")
    if result.get("seed") != trial["seed"]:
        raise RuntimeError("Historical seed mismatch")
    if not any(completion.get(k) == best_hash for k in ("best_sha256", "weights_sha256")):
        raise RuntimeError("Historical evaluation completion is not bound to best checkpoint")
    audit = read_json(evidence["audit"]["path"])
    if audit.get("status") != "verified" or audit.get("trial_id") != trial["id"]:
        raise RuntimeError("Reuse requires a specific verified compatibility audit")
    if not audit.get("limitations") or not audit.get("protocol_compatible"):
        raise RuntimeError("Reuse audit must explicitly retain limitations and affirm protocol compatibility")
    return write_receipt(plan, trial, [evidence[name]["path"] for name in required],
                         {"reused": True, "origin": evidence.get("origin"), "limitations": audit["limitations"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Validate frozen plan/code, without acquiring GPU or running trials")
    parser.add_argument("--resume-rich", nargs=2, metavar=("TRIAL_ID", "CHECKPOINT"),
                        help="Explicitly restore ONE compatible raw-FP32 checkpoint; never legacy last.pt")
    args = parser.parse_args()
    plan_path = absolute(args.plan)
    plan = load_plan(plan_path)
    lane = next(item for item in plan["lanes"] if item["id"] == args.lane)
    if socket.gethostname() != lane["hostname"]:
        raise RuntimeError(f"Wrong server: {socket.gethostname()} != {lane['hostname']}")
    verify_code(plan)
    if args.verify_only:
        print(json.dumps({"status": "verified", "plan_sha256": digest(plan), "lane": args.lane}))
        return
    if args.resume_rich and args.resume_rich[0] not in lane["trial_ids"]:
        raise ValueError("Explicit resume trial is not in this lane")
    out = within(lane["output_dir"], plan["campaign_root"])
    out.mkdir(parents=True, exist_ok=True)
    locks = [lock_file(out / "lane.lock"),
             lock_file(absolute(plan["campaign_root"]) / "locks" / (lane["gpu_uuid"] + ".lock"))]
    status_path = out / "status.json"
    status = {"schema": SCHEMA, "plan_sha256": digest(plan), "lane": args.lane, "host": socket.gethostname(),
              "pid": os.getpid(), "gpu_uuid": lane["gpu_uuid"], "started_utc": utc(), "completed": []}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=lane["gpu_uuid"], PYTHONUNBUFFERED="1", PYTHONPATH=plan["source_root"],
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", YOLO_AUTOINSTALL="false",
               WANDB_MODE="disabled", YOLO_CONFIG_DIR=str(out / "runtime-config"), MPLCONFIGDIR=str(out / "matplotlib"))
    try:
        for tid in lane["trial_ids"]:
            trial = trial_by_id(plan, tid)
            status.update(current=tid, state="checking", updated_utc=utc())
            atomic_json(status_path, status)
            if trial["seed"] == 1:
                # Not based on this lane's progress. No marker => stop, never self-unlock.
                verify_barrier(plan)
            if trial["kind"] == "not_ready":
                raise RuntimeError(f"Planned trial is not ready: {tid}")
            receipt = receipt_path(plan, trial)
            if receipt.exists():
                validate_receipt(plan, trial, read_json(receipt), verify_files=True)
                status["completed"].append(tid)
                continue
            verify_code(plan)
            if trial["kind"] == "reuse":
                verify_reuse(plan, trial)
            else:
                snapshot = wait_idle(lane["gpu_uuid"], lane, status_path, status)
                status.update(state="running", gpu=snapshot, updated_utc=utc())
                atomic_json(status_path, status)
                if trial["kind"] == "detection":
                    command = [plan["python"], "-u", str(Path(__file__).with_name("train_trial.py")),
                               "--plan", str(plan_path), "--trial", tid, "--gpu-uuid", lane["gpu_uuid"]]
                    if args.resume_rich and args.resume_rich[0] == tid:
                        command.extend(["--resume-rich", str(absolute(args.resume_rich[1]))])
                    cwd = absolute(plan["source_root"])
                else:
                    command, cwd = trial["command"], absolute(trial["cwd"])
                    trial_dir = within(trial["run_dir"], plan["campaign_root"])
                    if trial_dir.exists() and any(trial_dir.iterdir()):
                        raise RuntimeError(f"Refusing to rerun command over existing output: {trial_dir}")
                    trial_dir.mkdir(parents=True, exist_ok=True)
                execute(command, cwd, out / "logs" / (tid + ".log"), env)
                if trial["kind"] == "command":
                    for artifact in trial["success_artifacts"]:
                        if not absolute(artifact).is_file() or absolute(artifact).stat().st_size == 0:
                            raise RuntimeError(f"Command exited without required completion evidence: {artifact}")
                    write_receipt(plan, trial, trial["success_artifacts"], {"kind": "command", "argv": command})
                validate_receipt(plan, trial, read_json(receipt_path(plan, trial)), verify_files=True)
            status["completed"].append(tid)
        status.update(state="complete", finished_utc=utc())
        frozen(out / "lane_complete.json", {"schema": SCHEMA, "plan_sha256": digest(plan), "lane": lane["id"],
               "trial_ids": lane["trial_ids"], "completed": status["completed"]})
    except BaseException as error:
        status.update(state="failed", error=repr(error), failed_utc=utc())
        raise
    finally:
        status["updated_utc"] = utc()
        atomic_json(status_path, status)
        for handle in reversed(locks):
            handle.close()


if __name__ == "__main__":
    main()
