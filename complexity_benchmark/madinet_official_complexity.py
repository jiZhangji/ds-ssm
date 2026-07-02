#!/usr/bin/env python3
"""Profile the official MaDiNet MambaSAR module.

This script is intentionally separated from our Mamba-YOLO code. It loads the
official MaDiNet `MambaSAR.py` file and profiles the MambaSAR res4/res5 module
shown in the official source comments. If full Detectron2 configs are available,
use MaDiNet's own `tools/analyze_model.py` for end-to-end detector FLOPs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--madinet-root", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="complexity_benchmark/results/madinet_official_complexity.csv")
    return parser.parse_args()


def install_detectron2_stub_if_needed():
    """MambaSAR imports detectron2.layers.CNNBlockBase but does not use it here."""
    if "detectron2" in sys.modules:
        return
    detectron2 = types.ModuleType("detectron2")
    layers = types.ModuleType("detectron2.layers")
    detectron2.__spec__ = importlib.machinery.ModuleSpec("detectron2", loader=None)
    layers.__spec__ = importlib.machinery.ModuleSpec("detectron2.layers", loader=None)

    class CNNBlockBase(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    layers.CNNBlockBase = CNNBlockBase
    detectron2.layers = layers
    sys.modules["detectron2"] = detectron2
    sys.modules["detectron2.layers"] = layers


def load_mambasar(madinet_root: Path):
    install_detectron2_stub_if_needed()
    try:
        import timm.models._builder as timm_builder

        if not hasattr(timm_builder, "_update_default_kwargs"):
            def _update_default_kwargs(*args, **kwargs):
                return None

            timm_builder._update_default_kwargs = _update_default_kwargs
    except Exception:
        pass
    module_path = madinet_root / "MambaSAR.py"
    spec = importlib.util.spec_from_file_location("official_madinet_mambasar", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OfficialMambaSARRes4Res5(nn.Module):
    """Standalone official MambaSAR res4/res5 block from source comments."""

    def __init__(self, official):
        super().__init__()
        drop_path_res4 = [0.067, 0.083, 0.100, 0.117, 0.133, 0.150]
        drop_path_res5 = [0.167, 0.183, 0.200]
        self.official = official
        self.res4_blocks = nn.ModuleList(
            [
                official.Block(
                    dim=1024,
                    counter=i,
                    transformer_blocks=[3, 4, 5],
                    num_heads=8,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0,
                    attn_drop=0,
                    drop_path=drop_path_res4[i],
                    layer_scale=None,
                )
                for i in range(6)
            ]
        )
        self.down_sample = official.Downsample(dim=1024)
        self.res5_blocks = nn.ModuleList(
            [
                official.Block(
                    dim=2048,
                    counter=i,
                    transformer_blocks=[2],
                    num_heads=16,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0,
                    attn_drop=0,
                    drop_path=drop_path_res5[i],
                    layer_scale=None,
                )
                for i in range(3)
            ]
        )

    def forward(self, x):
        # x is res4-like feature map, e.g. Bx1024x40x40 for 640 input at stride 16.
        b, c, h, w = x.shape
        x = self.official.window_partition(x, h)
        for block in self.res4_blocks:
            x = block(x)
        x = self.official.window_reverse(x, h, h, w)
        x = self.down_sample(x)
        b, c, h, w = x.shape
        x = self.official.window_partition(x, h)
        for block in self.res5_blocks:
            x = block(x)
        x = self.official.window_reverse(x, h, h, w)
        return x


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency(model, x, warmup, runs, device):
    model.eval()
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
    return elapsed * 1000.0 / runs


def try_thop(model, x):
    try:
        import thop

        flops = thop.profile(model, inputs=(x,), verbose=False)[0] / 1e9 * 2
        return flops, ""
    except Exception as exc:
        return 0.0, repr(exc)


def main():
    args = parse_args()
    root = Path(args.madinet_root)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    official = load_mambasar(root)
    model = OfficialMambaSARRes4Res5(official).to(device)
    x = torch.randn(1, 1024, args.imgsz // 16, args.imgsz // 16, device=device)
    params, trainable = count_params(model)
    gflops, flops_error = try_thop(model, x)
    latency = measure_latency(model, x, args.warmup, args.runs, device)
    row = {
        "method": "MaDiNet official MambaSAR res4+res5",
        "source": str(root),
        "scope": "official MambaSAR module only; not full detector",
        "input_feature": f"1x1024x{args.imgsz // 16}x{args.imgsz // 16}",
        "imgsz_assumption": args.imgsz,
        "params": params,
        "params_M": round(params / 1e6, 4),
        "trainable_params": trainable,
        "GFLOPs_thop": round(gflops, 4),
        "latency_ms": round(latency, 4),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "flops_note": "THOP may undercount selective_scan/custom CUDA ops; use latency as supporting evidence.",
        "flops_error": flops_error,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out.with_suffix(".json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
