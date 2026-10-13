#!/usr/bin/env python3
"""Start only the two explicitly reviewed finite legacy finalizers, detached."""
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from manifest import absolute, digest, frozen, read_json, sha, utc, within


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--script", required=True)
    args = parser.parse_args()
    spec_path, script = absolute(args.spec), absolute(args.script)
    spec = read_json(spec_path)
    if socket.gethostname() != spec["hostname"] or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != spec["boot_id"]:
        raise RuntimeError("Wrong host or boot identity")
    evidence = within(spec["evidence_dir"], spec["campaign_root"])
    receipt = read_json(evidence / "freeze_receipt.json")
    if receipt["spec_sha256"] != digest(spec) or receipt["status"] != "both_scheduler_parents_frozen":
        raise RuntimeError("Matching parent freeze receipt is required")
    if sha(script) != receipt["script_sha256"]:
        raise RuntimeError("Finalizer script must be the exact reviewed script that froze the parents")
    launched = []
    for lane in ("f0", "f1"):
        path = evidence / (lane + ".finalizer_launch.json")
        if path.exists():
            raise RuntimeError(f"An existing finalizer launch needs inspection: {path}")
        command = [spec["python"], "-B", "-u", str(script), "finalize", "--spec", str(spec_path),
                   "--lane", lane, "--confirm-evaluate-and-retire", "--wait-seconds", "172800"]
        log_path = evidence / (lane + ".finalizer.log")
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
