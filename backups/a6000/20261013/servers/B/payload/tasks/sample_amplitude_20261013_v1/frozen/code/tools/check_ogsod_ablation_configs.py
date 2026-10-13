#!/usr/bin/env python3
"""Parse OGSOD ablation YAMLs and verify parameter-matched controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import parse_model, yaml_model_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named model YAML. Repeat for each variant to parse.",
    )
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def model_counts(path: Path) -> tuple[int, int]:
    model_dict = yaml_model_load(str(path))
    model, _ = parse_model(model_dict, ch=3, verbose=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return parameters, trainable


def main() -> None:
    args = parse_args()
    counts = {}
    for item in args.config:
        name, path = parse_named_path(item)
        if name in counts:
            raise ValueError(f"Duplicate variant name: {name}")
        counts[name] = model_counts(path)
        parameters, trainable = counts[name]
        print(f"{name}: parameters={parameters}, trainable_parameters={trainable}, config={path}")

    matched = [name for name in ("full", "global", "no_2dstd") if name in counts]
    if len(matched) >= 2:
        reference_name = matched[0]
        reference_counts = counts[reference_name]
        mismatched = {
            name: counts[name]
            for name in matched[1:]
            if counts[name] != reference_counts
        }
        if mismatched:
            raise SystemExit(
                f"Parameter-matched controls disagree with {reference_name}={reference_counts}: {mismatched}"
            )
        print(f"PASS: parameter-matched controls agree: {', '.join(matched)}")
    else:
        print("INFO: fewer than two parameter-matched controls were requested")


if __name__ == "__main__":
    main()
