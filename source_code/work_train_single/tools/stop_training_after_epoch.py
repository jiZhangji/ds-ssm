#!/usr/bin/env python3
"""Stop one training process after a completed epoch is safely checkpointed."""

import argparse
import os
import signal
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, help="Training process ID")
    parser.add_argument("--pid-file", type=Path, help="File containing the training process ID")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to weights/last.pt")
    parser.add_argument("--epoch", type=int, default=61, help="Stop after this many epochs are complete")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def checkpoint_epoch(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return int(checkpoint.get("epoch", -1)) + 1
    except (EOFError, OSError, RuntimeError, ValueError):
        return 0


def main():
    args = parse_args()
    if (args.pid is None) == (args.pid_file is None):
        raise SystemExit("Provide exactly one of --pid or --pid-file")
    if args.epoch < 1:
        raise SystemExit("--epoch must be at least 1")

    pid = args.pid if args.pid is not None else int(args.pid_file.read_text().strip())
    print(f"Monitoring PID {pid}; stop after epoch {args.epoch}: {args.checkpoint}", flush=True)

    while process_exists(pid):
        completed = checkpoint_epoch(args.checkpoint) if args.checkpoint.exists() else 0
        print(f"completed_epoch={completed}", flush=True)
        if completed >= args.epoch:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {pid} after checkpointed epoch {completed}", flush=True)
            return
        time.sleep(args.poll_seconds)

    raise SystemExit(f"Training PID {pid} exited before epoch {args.epoch}")


if __name__ == "__main__":
    main()
