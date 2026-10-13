#!/usr/bin/env python3
"""Record the exact parsed architecture used by one controlled OGSOD run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import parse_model, yaml_model_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_count(data: dict) -> int:
    names = data.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    if "nc" in data:
        return int(data["nc"])
    raise ValueError("Dataset YAML has neither names nor nc")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    expected_nc = class_count(data)
    model_dict = yaml_model_load(str(config_path))
    config_nc = int(model_dict.get("nc", expected_nc))
    if config_nc != expected_nc:
        raise ValueError(
            f"Class-count mismatch: model config nc={config_nc}, dataset classes={expected_nc}"
        )

    model, _ = parse_model(model_dict, ch=3, verbose=False)
    payload = {
        "run_name": args.run_name,
        "variant": args.variant,
        "imgsz": args.imgsz,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "data": str(data_path),
        "data_yaml_sha256": sha256(data_path),
        "nc": expected_nc,
        "scale": model_dict.get("scale"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameter_tensors": sum(1 for _ in model.parameters()),
        "module_instances": sum(1 for _ in model.modules()),
        "flops": None,
        "flops_note": (
            "Not computed by parse-only snapshot. Profile the frozen server model separately; "
            "custom selective-scan operations are not fully handled by generic profilers."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
