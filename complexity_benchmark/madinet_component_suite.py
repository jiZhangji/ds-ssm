#!/usr/bin/env python3
"""Profile additional modules from the official MaDiNet repository."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
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
    parser.add_argument("--device", default="0")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="complexity_benchmark/results/madinet_component_suite.csv")
    return parser.parse_args()


def install_stubs():
    if "detectron2" not in sys.modules:
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
    try:
        import timm.models._builder as timm_builder

        if not hasattr(timm_builder, "_update_default_kwargs"):
            timm_builder._update_default_kwargs = lambda *args, **kwargs: None
    except Exception:
        pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def try_flops(model, x):
    try:
        import thop

        return thop.profile(model, inputs=(x,), verbose=False)[0] / 1e9 * 2, ""
    except Exception as exc:
        return 0.0, repr(exc)


@torch.no_grad()
def latency(model, x, warmup, runs, device):
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
    return (time.perf_counter() - start) * 1000.0 / runs


def profile(name, model, x, args, device, note):
    model = model.to(device)
    x = x.to(device)
    params = count_params(model)
    flops, flops_error = try_flops(model, x)
    try:
        lat = latency(model, x, args.warmup, args.runs, device)
        latency_error = ""
    except Exception as exc:
        lat = 0.0
        latency_error = repr(exc)
    return {
        "component": name,
        "input_shape": "x".join(map(str, x.shape)),
        "params": params,
        "params_M": round(params / 1e6, 4),
        "GFLOPs_thop": round(flops, 4),
        "latency_ms": round(lat, 4),
        "device": str(device),
        "note": note,
        "flops_error": flops_error,
        "latency_error": latency_error,
    }


def main():
    args = parse_args()
    install_stubs()
    root = Path(args.madinet_root)
    m = load_module(root / "MambaSAR.py", "official_madinet_mambasar_suite")
    pdc = load_module(root / "pixel difference convPDC.py", "official_madinet_pdc_suite")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    rows = []

    seq = torch.randn(1, 1600, 1024)
    rows.append(profile("MambaVisionMixer-d1024-L1600", m.MambaVisionMixer(d_model=1024, d_state=8, d_conv=3, expand=1), seq, args, device, "official Mamba mixer at res4 token scale"))
    rows.append(profile("SelfAttention-d1024-L1600", m.Attention(dim=1024, num_heads=8, qkv_bias=True), seq, args, device, "attention branch used in official Block"))
    rows.append(profile("AgentAttention-d1024-L1600", m.AgentAttention(dim=1024, num_patches=1600, num_heads=8, qkv_bias=True, sr_ratio=1, agent_num=49, downstream_agent_shape=(7, 7)), seq, args, device, "agent attention candidate in official source comments"))

    rows.append(profile("Block-Mamba-d1024-L1600", m.Block(dim=1024, num_heads=8, counter=0, transformer_blocks=[3, 4, 5], mlp_ratio=4, qkv_bias=True), seq, args, device, "official Block using MambaVisionMixer"))
    rows.append(profile("Block-Attention-d1024-L1600", m.Block(dim=1024, num_heads=8, counter=3, transformer_blocks=[3, 4, 5], mlp_ratio=4, qkv_bias=True), seq, args, device, "official Block using self-attention"))

    feat = torch.randn(1, 256, 80, 80)
    for fname, fn in [("CPDC", pdc.CPDC), ("APDC", pdc.APDC), ("RPDC", pdc.RPDC)]:
        rows.append(profile(f"PDC-{fname}-256", pdc.PDC_Conv2d(fn, 256, 256, 3, stride=1, padding=1, groups=1, bias=False), feat, args, device, "official pixel difference convolution candidate"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
