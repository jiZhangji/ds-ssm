#!/usr/bin/env python3
"""Process-local PyTorch 1.13 UUID device-count compatibility; no install edits.

Keep CUDA_VISIBLE_DEVICES as the exact assigned GPU UUID. Only the known case
of PyTorch 1.13 reporting zero devices while CUDA reports one is repaired, by
using the CUDA runtime count. Hardware identity is independently checked through
nvidia-smi. Call ensure_uuid_cuda(expected_uuid) immediately after importing
torch, before constructing models or checking torch.cuda.device_count().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_uuid_cuda(expected_uuid):
    import torch

    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", expected_uuid):
        raise RuntimeError("An exact, full GPU UUID is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal the exact assigned GPU UUID")
    result = subprocess.run(
        ["nvidia-smi", "--id=" + expected_uuid, "--query-gpu=uuid,index,name", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=15,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("Assigned GPU UUID did not resolve to exactly one physical device")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3 or parts[0] != expected_uuid:
        raise RuntimeError("nvidia-smi returned a different GPU identity")
    native_count = int(torch._C._cuda_getDeviceCount())
    reported_count = int(torch.cuda.device_count())
    if native_count != 1:
        raise RuntimeError(f"CUDA runtime must expose one assigned GPU, got {native_count}")
    applied = False
    if reported_count != native_count:
        if not torch.__version__.startswith("1.13.") or reported_count != 0:
            raise RuntimeError(f"Unexpected CUDA device-count disagreement: {reported_count} vs {native_count}")

        def runtime_device_count():
            return int(torch._C._cuda_getDeviceCount())

        torch.cuda.device_count = runtime_device_count
        applied = True
    if int(torch.cuda.device_count()) != 1 or not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA GPU remains unavailable after compatibility check")
    torch.cuda.set_device(0)
    return {
        "hostname": socket.gethostname(), "torch": torch.__version__,
        "cuda_visible_devices": expected_uuid, "physical_gpu_uuid": parts[0],
        "physical_gpu_index": int(parts[1]), "physical_gpu_name": parts[2],
        "reported_count_before": reported_count, "cuda_runtime_count": native_count,
        "reported_count_after": int(torch.cuda.device_count()), "compatibility_applied": applied,
        "method": "process-local CUDA runtime device_count for known PyTorch 1.13 UUID parser bug" if applied else "none",
        "torch_installation_modified": False, "gpu_mask_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace compatibility evidence: {args.output}")
    if not args.script.is_file():
        raise FileNotFoundError(args.script)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "script": str(args.script.resolve()),
        "script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        report["cuda_identity"] = ensure_uuid_cuda(args.expected_uuid)
        forwarded = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
        sys.argv = [str(args.script), *forwarded]
        sys.path.insert(0, str(args.script.resolve().parent))
        runpy.run_path(str(args.script), run_name="__main__")
        report["status"] = "PASS"
    except BaseException as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"compatibility_launcher": report.get("status", "FAIL"), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
