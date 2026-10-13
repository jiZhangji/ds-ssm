#!/usr/bin/env python3
"""Start only the verified finite waiter; no system service configuration."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import recovery_queue as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", choices=r.HOSTS, required=True)
    a = p.parse_args()
    cfg = r.layout(a.key)
    output = Path(cfg["output"])
    assert r.read(output / "verification.json")["status"] == "PASS"
    assert r.read(output / "queue_state.json")["state"] == "prepared"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    command = [str(r.PYTHON), str(r.ROOT / "recovery_queue.py"), "queue", "--key", a.key]
    with (output / "queue.log").open("a") as log:
        child = subprocess.Popen(command, cwd=r.ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(40):
        if child.poll() is not None:
            raise RuntimeError(f"Queue exited: {child.returncode}; inspect queue.log")
        state = r.read(output / "queue_state.json")
        if state.get("pid") == child.pid and state["state"] == "waiting_predecessor":
            receipt = {"key": a.key, "pid": child.pid, "created_unix": time.time(),
                       "state": state["state"], "predecessor_completed": state["predecessor_completed"],
                       "predecessor_total": state["predecessor_total"], "output": str(output),
                       "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                       "proc_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]}
            r.write(output / "launch_receipt.json", receipt)
            print(json.dumps(receipt))
            return
        time.sleep(0.25)
    raise RuntimeError("No queue acknowledgement within 10 seconds; inspect before retrying")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
