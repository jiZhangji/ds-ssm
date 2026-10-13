#!/usr/bin/env python3
"""Dispatch queued experiments after a GPU has been idle for a stable window.

The scheduler is intentionally conservative: it never kills processes, never
overwrites completed outputs, and only launches a job when its data and runtime
exist on the selected host. A single controller can manage the A6000 LAN over
key-based SSH.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if isinstance(jobs, list):
        for job in jobs:
            expanded = list(job.get("completion", []))
            for completion_path in job.pop("completion_paths", []):
                expanded.extend({"host": host, "path": completion_path} for host in job.get("hosts", []))
            job["completion"] = expanded
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class HostRunner:
    def __init__(self, spec: dict):
        self.spec = spec

    def run(self, script: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        if self.spec.get("local"):
            command = ["bash", "-s"]
        else:
            command = [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
            ]
            if self.spec.get("identity_file"):
                command.extend(["-o", "IdentitiesOnly=yes", "-i", self.spec["identity_file"]])
            command.extend([self.spec["ssh"], "bash", "-s"])
        return subprocess.run(
            command,
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )


def job_pid_path(root: str, job_id: str) -> str:
    return f"{root}/scheduler/pids/{job_id}.pid"


def job_log_path(root: str, job_id: str) -> str:
    return f"{root}/scheduler/logs/{job_id}.log"


def paths_for_host(config: dict, state: dict, host_name: str) -> set[str]:
    root = config["hosts"][host_name]["root"]
    paths: set[str] = set()
    for job in config["jobs"]:
        for evidence in job.get("completion", []):
            if evidence["host"] == host_name:
                paths.add(evidence["path"])
        if host_name in job.get("hosts", []):
            paths.update(job.get("requires", []))
            paths.add(job_pid_path(root, job["id"]))
    return paths


def snapshot_host(runner: HostRunner, paths: set[str]) -> dict:
    lines = [
        "set +e",
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits | sed 's/^/GPU|/'",
    ]
    for path in sorted(paths):
        quoted = shlex.quote(path)
        lines.append(
            f"test -e {quoted} && printf 'FILE|1|%s\\n' {quoted} || printf 'FILE|0|%s\\n' {quoted}"
        )
        if path.endswith(".pid"):
            lines.append(
                f"if test -s {quoted}; then p=$(cat {quoted}); kill -0 \"$p\" 2>/dev/null && "
                f"printf 'PID|1|%s|%s\\n' {quoted} \"$p\" || "
                f"printf 'PID|0|%s|%s\\n' {quoted} \"$p\"; fi"
            )
    result = runner.run("\n".join(lines) + "\n")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"snapshot failed with {result.returncode}")

    snapshot = {"gpus": [], "files": set(), "pids": {}}
    for raw in result.stdout.splitlines():
        if raw.startswith("GPU|"):
            fields = [item.strip() for item in raw[4:].split(",")]
            if len(fields) == 5:
                snapshot["gpus"].append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "memory_total": int(fields[2]),
                        "memory_used": int(fields[3]),
                        "utilization": int(fields[4]),
                    }
                )
        elif raw.startswith("FILE|1|"):
            snapshot["files"].add(raw.split("|", 2)[2])
        elif raw.startswith("PID|"):
            _, alive, path, pid = raw.split("|", 3)
            snapshot["pids"][path] = {"alive": alive == "1", "pid": int(pid)}
    return snapshot


def is_complete(job: dict, snapshots: dict[str, dict]) -> bool:
    evidence = job.get("completion", [])
    return any(item["path"] in snapshots.get(item["host"], {}).get("files", set()) for item in evidence)


def requirements_met(job: dict, host_name: str, snapshot: dict) -> bool:
    return all(path in snapshot["files"] for path in job.get("requires", []))


def gpu_is_busy(gpu: dict, config: dict) -> bool:
    return (
        gpu["memory_used"] > config.get("idle_memory_mib", 2048)
        or gpu["utilization"] > config.get("idle_utilization_percent", 10)
    )


def launch_job(runner: HostRunner, host: dict, job: dict, gpu: int) -> tuple[bool, str]:
    root = host["root"]
    pid_path = job_pid_path(root, job["id"])
    log_path = job_log_path(root, job["id"])
    command = job["command"].format(root=root, gpu=gpu)
    wrapped = f"export CUDA_VISIBLE_DEVICES={gpu}; {command}"
    script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(root + '/scheduler/pids')} {shlex.quote(root + '/scheduler/logs')}",
            f"if test -s {shlex.quote(pid_path)}; then old=$(cat {shlex.quote(pid_path)}); "
            "kill -0 \"$old\" 2>/dev/null && { echo ALREADY:$old; exit 0; } || true; fi",
            f"nohup bash -lc {shlex.quote(wrapped)} >> {shlex.quote(log_path)} 2>&1 </dev/null &",
            "pid=$!",
            f"printf '%s\n' \"$pid\" > {shlex.quote(pid_path + '.tmp')}",
            f"mv {shlex.quote(pid_path + '.tmp')} {shlex.quote(pid_path)}",
            "echo STARTED:$pid",
        ]
    ) + "\n"
    result = runner.run(script)
    message = (result.stdout + result.stderr).strip()
    return result.returncode == 0 and ("STARTED:" in message or "ALREADY:" in message), message


def reconcile_jobs(config: dict, state: dict, snapshots: dict[str, dict]) -> None:
    for job in config["jobs"]:
        item = state["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
        if is_complete(job, snapshots):
            item.update({"status": "complete", "completed_at": item.get("completed_at", utc_now())})
            continue
        assigned = item.get("host")
        if assigned and assigned in snapshots:
            root = config["hosts"][assigned]["root"]
            pid = snapshots[assigned]["pids"].get(job_pid_path(root, job["id"]))
            if pid and pid["alive"]:
                item.update({"status": "running", "pid": pid["pid"]})
                continue
            if item.get("status") == "running":
                item.update({"status": "pending", "last_exit_observed_at": utc_now()})
        if item.get("attempts", 0) >= job.get("max_attempts", config.get("max_attempts", 3)):
            item["status"] = "failed"
        elif item.get("status") not in {"running", "complete"}:
            item["status"] = "pending"


def dispatch_once(config: dict, state: dict, snapshots: dict[str, dict], runners: dict[str, HostRunner]) -> None:
    now = time.time()
    idle_seconds = int(config.get("idle_seconds", 900))
    for host_name, host in config["hosts"].items():
        if not host.get("enabled", True) or host_name not in snapshots:
            continue
        snapshot = snapshots[host_name]
        for gpu in snapshot["gpus"]:
            gpu_key = f"{host_name}:{gpu['index']}"
            resource = state["gpus"].setdefault(gpu_key, {})
            resource.update({"last_snapshot": utc_now(), **gpu})
            if gpu_is_busy(gpu, config):
                resource["idle_since"] = None
                continue
            if not resource.get("idle_since"):
                resource["idle_since"] = now
                continue
            if now - float(resource["idle_since"]) < idle_seconds:
                continue

            candidate = None
            for job in sorted(config["jobs"], key=lambda value: (value.get("priority", 100), value["id"])):
                item = state["jobs"][job["id"]]
                if item["status"] != "pending" or host_name not in job.get("hosts", []):
                    continue
                if not requirements_met(job, host_name, snapshot):
                    continue
                candidate = job
                break
            if candidate is None:
                continue

            item = state["jobs"][candidate["id"]]
            ok, message = launch_job(runners[host_name], host, candidate, gpu["index"])
            item["last_launch_message"] = message
            item["last_launch_at"] = utc_now()
            if ok:
                item.update(
                    {
                        "status": "running",
                        "host": host_name,
                        "gpu": gpu["index"],
                        "attempts": item.get("attempts", 0) + 1,
                    }
                )
                resource["idle_since"] = None
            else:
                item["status"] = "pending"


def write_status(config: dict, state: dict, status_path: Path, snapshots: dict[str, dict]) -> None:
    jobs_by_status: dict[str, list[str]] = {}
    for job_id, item in state["jobs"].items():
        jobs_by_status.setdefault(item["status"], []).append(job_id)
    payload = {
        "updated_at": utc_now(),
        "idle_seconds": config.get("idle_seconds", 900),
        "jobs": {key: sorted(value) for key, value in sorted(jobs_by_status.items())},
        "hosts": {
            name: {
                "reachable": name in snapshots,
                "gpus": snapshots.get(name, {}).get("gpus", []),
                "error": state["hosts"].get(name, {}).get("error"),
            }
            for name in config["hosts"]
        },
    }
    save_json(status_path, payload)


def cycle(config: dict, state: dict) -> None:
    runners = {name: HostRunner(spec) for name, spec in config["hosts"].items()}
    snapshots: dict[str, dict] = {}
    for name, runner in runners.items():
        if not config["hosts"][name].get("enabled", True):
            continue
        try:
            snapshots[name] = snapshot_host(runner, paths_for_host(config, state, name))
            state["hosts"].setdefault(name, {}).update({"last_seen": utc_now(), "error": None})
        except Exception as error:
            state["hosts"].setdefault(name, {}).update({"error": str(error), "last_error": utc_now()})
    reconcile_jobs(config, state, snapshots)
    dispatch_once(config, state, snapshots, runners)
    state["updated_at"] = utc_now()
    state["_snapshots"] = snapshots


def print_jobs(config: dict) -> None:
    print("ID\tTASK\tDATASET\tVARIANT\tHOSTS")
    for job in sorted(config["jobs"], key=lambda value: (value.get("priority", 100), value["id"])):
        print(f"{job['id']}\t{job['task']}\t{job['dataset']}\t{job['variant']}\t{','.join(job['hosts'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--loop-seconds", type=int, default=60)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    if args.list:
        print_jobs(config)
        return
    base = args.config.resolve().parent
    state_path = (args.state or base / "gpu_scheduler_state.json").resolve()
    status_path = (args.status or base / "gpu_scheduler_status.json").resolve()
    lock_path = (args.lock or base / "gpu_scheduler.lock").resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another scheduler instance already owns the lock")
        lock_handle.write(str(os.getpid()))
        lock_handle.flush()

        state = load_json(state_path) if state_path.exists() else {"jobs": {}, "hosts": {}, "gpus": {}}
        while True:
            try:
                cycle(config, state)
                snapshots = state.pop("_snapshots", {})
                save_json(state_path, state)
                write_status(config, state, status_path, snapshots)
            except Exception as error:
                print(f"{utc_now()} scheduler cycle failed: {error}", file=sys.stderr, flush=True)
            if args.once:
                break
            time.sleep(max(args.loop_seconds, 15))


if __name__ == "__main__":
    main()
