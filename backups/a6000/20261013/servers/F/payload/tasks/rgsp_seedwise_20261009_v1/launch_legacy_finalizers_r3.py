#!/usr/bin/env python3
"""Start only the two explicitly reviewed finite legacy finalizers, detached."""
import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from manifest import absolute, digest, frozen, read_json, sha, utc, within


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--launch-tag", default="initial")
    parser.add_argument("--confirm-replace-owned-waiters", action="store_true")
    args = parser.parse_args()
    spec_path, script = absolute(args.spec), absolute(args.script)
    spec = read_json(spec_path)
    if socket.gethostname() != spec["hostname"] or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != spec["boot_id"]:
        raise RuntimeError("Wrong host or boot identity")
    evidence = within(spec["evidence_dir"], spec["campaign_root"])
    receipt = read_json(evidence / "freeze_receipt.json")
    if receipt["spec_sha256"] != digest(spec) or receipt["status"] != "both_scheduler_parents_frozen":
        raise RuntimeError("Matching parent freeze receipt is required")
    if sha(script) != receipt["script_sha256"] and not args.confirm_replace_owned_waiters:
        raise RuntimeError("Finalizer script must be the exact reviewed script that froze the parents")
    if args.confirm_replace_owned_waiters:
        if args.launch_tag != "r3":
            raise RuntimeError("Only the specifically approved r2-to-r3 waiter replacement is supported")
        module_spec = importlib.util.spec_from_file_location("reviewed_handoff", script)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.load_spec(str(spec_path))
        old_records = {}
        for lane_spec in spec["lanes"]:
            lane = lane_spec["id"]
            module.inspect_lane(spec, lane_spec, frozen_parent=True)
            record = read_json(evidence / (lane + ".finalizer_launch.json"))
            if record["script_sha256"] != receipt["script_sha256"] or record["spec_sha256"] != digest(spec):
                raise RuntimeError("Old waiter launch identity is not the reviewed r2 process")
            current = module.require_identity(module.process(record["pid"]), record["pid"], record["start_ticks"], record["argv"], {"S", "R"})
            descendants = Path("/proc") / str(record["pid"]) / "task" / str(record["pid"]) / "children"
            if descendants.read_text().strip():
                raise RuntimeError("Old waiter has an active subprocess; replacement is not allowed")
            state = read_json(evidence / (lane + ".status.json"))
            if state.get("state") != "waiting_existing_child":
                raise RuntimeError("Old waiter has left the CPU-only waiting phase")
            if any(record["pid"] in module.gpu(item)["pids"] for item in spec["lanes"]):
                raise RuntimeError("Old waiter is using a GPU; replacement is not allowed")
            old_records[lane] = {"launch": record, "observed": current}
        frozen(evidence / "approved_waiter_replacement_r3.json", {
            "spec_sha256": digest(spec), "old_script_sha256": receipt["script_sha256"],
            "new_script_sha256": sha(script), "old_waiters": old_records,
            "scope": "Root explicitly approved replacing only these owned CPU waiters; original scheduler parents stay stopped, training children untouched",
            "approved_changes": ["Explicit PCI_BUS_ID before legacy evaluation", "15-second continuous post-training idle gate"]})
        for lane_spec in spec["lanes"]:
            lane = lane_spec["id"]
            old = old_records[lane]["launch"]
            module.inspect_lane(spec, lane_spec, frozen_parent=True)
            fd = (os.pidfd_open(old["pid"], 0) if hasattr(os, "pidfd_open") else
                  module.pidfd_syscall(434, module.ctypes.c_int(old["pid"]), module.ctypes.c_uint(0)))
            try:
                module.require_identity(module.process(old["pid"]), old["pid"], old["start_ticks"], old["argv"], {"S", "R"})
                children = Path("/proc") / str(old["pid"]) / "task" / str(old["pid"]) / "children"
                if children.read_text().strip():
                    raise RuntimeError("Old waiter started a child during verification; no signal sent")
                module.journal(spec, lane, {"action": "approved_owned_cpu_waiter_SIGTERM", "pid": old["pid"], "start_ticks": old["start_ticks"], "original_parent_and_child_not_signaled": True})
                module.pidfd_signal(fd, signal.SIGTERM)
            finally:
                os.close(fd)
            deadline = time.monotonic() + 10
            while True:
                after = module.process(old["pid"])
                if after is None or (after["start_ticks"] == old["start_ticks"] and after["state"] == "Z"):
                    break
                module.require_identity(after, old["pid"], old["start_ticks"], old["argv"])
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned CPU waiter did not exit; no stronger signal issued")
                time.sleep(0.1)
            module.inspect_lane(spec, lane_spec, frozen_parent=True)
            module.journal(spec, lane, {"action": "owned_cpu_waiter_retired_for_r3", "pid": old["pid"], "after": after})
    launched = []
    for lane in ("f0", "f1"):
        suffix = "" if args.launch_tag == "initial" else "." + args.launch_tag
        path = evidence / (lane + suffix + ".finalizer_launch.json")
        if path.exists():
            raise RuntimeError(f"An existing finalizer launch needs inspection: {path}")
        command = [spec["python"], "-B", "-u", str(script), "finalize", "--spec", str(spec_path),
                   "--lane", lane, "--confirm-evaluate-and-retire", "--wait-seconds", "172800"]
        log_path = evidence / (lane + suffix + ".finalizer.log")
        with log_path.open("xb") as log:
            child = subprocess.Popen(command, cwd=spec["campaign_root"], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        time.sleep(1)
        code = child.poll()
        proc = Path("/proc") / str(child.pid)
        record = {"spec_sha256": digest(spec), "script_sha256": sha(script), "lane": lane,
                  "pid": child.pid, "launched_utc": utc(), "argv": command, "log": str(log_path),
                  "poll_returncode": code, "detached_session": True,
                  "scope": "existing child completion -> copied independent test -> parent-only retirement; no new training"}
        if proc.exists():
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            record.update(start_ticks=int(fields[19]), state=fields[0], session=int(fields[3]),
                          actual_argv=[s.decode() for s in (proc / "cmdline").read_bytes().split(b"\0") if s])
        frozen(path, record)
        launched.append(record)
        if code is not None:
            raise RuntimeError(f"Finalizer failed to stay running: {lane}, exit={code}; see {log_path}")
    print(json.dumps({"status": "detached_finalizers_started", "launched": launched}, indent=2))


if __name__ == "__main__":
    main()
