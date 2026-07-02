#!/usr/bin/env python3
"""Compute params, FLOPs, and inference latency for Mamba-YOLO style models.

Run from the Mamba-YOLO project root:
    python complexity_benchmark/model_complexity_benchmark.py \
        --configs ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml \
                  ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml \
        --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="Model yaml files or .pt weights.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--out", default="complexity_benchmark/results/model_complexity.csv")
    return parser.parse_args()


def import_yolo():
    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    return YOLO, get_flops


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency(model: torch.nn.Module, device: torch.device, imgsz: int, batch: int, warmup: int, runs: int, half: bool):
    model.eval().to(device)
    if half and device.type == "cuda":
        model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_images = batch * runs
    return {
        "latency_ms_per_batch": elapsed * 1000.0 / runs,
        "latency_ms_per_image": elapsed * 1000.0 / total_images,
        "fps": total_images / elapsed if elapsed > 0 else 0.0,
    }


def profile_one(config: str, args: argparse.Namespace, YOLO, get_flops) -> dict:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    yolo = YOLO(config)
    model = yolo.model
    try:
        info = model.info(detailed=False, verbose=False, imgsz=args.imgsz)
    except Exception:
        info = None
    if isinstance(info, (tuple, list)) and len(info) >= 4:
        layers, info_params, gradients, info_flops = info[:4]
    else:
        layers = len(list(model.modules()))
        info_params = gradients = info_flops = 0
    total_params, trainable_params = count_params(model)

    try:
        flops = float(get_flops(model, imgsz=args.imgsz))
    except Exception as exc:
        flops = 0.0
        flops_error = repr(exc)
    else:
        flops_error = ""

    try:
        latency = measure_latency(model, device, args.imgsz, args.batch, args.warmup, args.runs, args.half)
    except Exception as exc:
        latency = {"latency_ms_per_batch": 0.0, "latency_ms_per_image": 0.0, "fps": 0.0}
        latency_error = repr(exc)
    else:
        latency_error = ""

    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    return {
        "model": Path(config).name,
        "config_or_weight": config,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": "fp16" if args.half and device.type == "cuda" else "fp32",
        "device": str(device),
        "device_name": cuda_name,
        "layers": layers,
        "params": total_params,
        "params_M": round(total_params / 1e6, 4),
        "trainable_params": trainable_params,
        "GFLOPs": round(flops or info_flops or 0.0, 4),
        "latency_ms_per_batch": round(latency["latency_ms_per_batch"], 4),
        "latency_ms_per_image": round(latency["latency_ms_per_image"], 4),
        "fps": round(latency["fps"], 4),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "flops_error": flops_error,
        "latency_error": latency_error,
    }


def main() -> None:
    args = parse_args()
    YOLO, get_flops = import_yolo()

    rows = []
    for config in args.configs:
        try:
            rows.append(profile_one(config, args, YOLO, get_flops))
        except Exception as exc:
            rows.append(
                {
                    "model": Path(config).name,
                    "config_or_weight": config,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "precision": "fp16" if args.half else "fp32",
                    "device": args.device,
                    "device_name": "",
                    "layers": 0,
                    "params": 0,
                    "params_M": 0.0,
                    "trainable_params": 0,
                    "GFLOPs": 0.0,
                    "latency_ms_per_batch": 0.0,
                    "latency_ms_per_image": 0.0,
                    "fps": 0.0,
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "flops_error": "",
                    "latency_error": f"profile_failed: {exc!r}",
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved CSV: {out}")
    print(f"Saved JSON: {json_out}")
    for row in rows:
        print(
            f"{row['model']}: params={row['params_M']}M, GFLOPs={row['GFLOPs']}, "
            f"latency={row['latency_ms_per_image']} ms/img, fps={row['fps']}"
        )


if __name__ == "__main__":
    main()
