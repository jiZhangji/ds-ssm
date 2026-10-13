#!/usr/bin/env python3
"""Evaluate one validation-selected OGSOD checkpoint and save reproducible metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    weights_path = Path(args.weights).resolve()
    data_path = Path(args.data).resolve()

    model = YOLO(weights_path)
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        half=False,
        plots=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    box = metrics.box
    class_ids = [int(x) for x in as_list(box.ap_class_index)]
    precision = [float(x) for x in as_list(box.p)]
    recall = [float(x) for x in as_list(box.r)]
    ap50 = [float(x) for x in as_list(box.ap50)]
    ap50_95 = [float(x) for x in as_list(box.ap)]
    all_ap = as_list(box.all_ap)
    ap75 = [float(row[5]) for row in all_ap]

    names = metrics.names or model.names
    if isinstance(names, list):
        names = dict(enumerate(names))

    per_class = []
    for row_index, class_id in enumerate(class_ids):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "precision": precision[row_index],
                "recall": recall[row_index],
                "ap50": ap50[row_index],
                "ap75": ap75[row_index],
                "ap50_95": ap50_95[row_index],
            }
        )

    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "ap50": float(box.map50),
        "ap75": sum(ap75) / len(ap75),
        "ap50_95": float(box.map),
    }
    payload = {
        "run_name": args.run_name,
        "variant": args.variant,
        "seed": args.seed,
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "data": str(data_path),
        "data_yaml_sha256": sha256(data_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "precision": "fp32",
        "result_dir": str(metrics.save_dir),
        "overall": overall,
        "per_class": per_class,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = output_json.with_name("test_per_class_metrics.csv")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)

    print("\n===== OGSOD BEST-CHECKPOINT TEST =====")
    print(f"run:       {args.run_name}")
    print(f"variant:   {args.variant}")
    print(f"seed:      {args.seed}")
    print(f"P:         {overall['precision']:.6f}")
    print(f"R:         {overall['recall']:.6f}")
    print(f"AP50:      {overall['ap50']:.6f}")
    print(f"AP75:      {overall['ap75']:.6f}")
    print(f"AP50-95:   {overall['ap50_95']:.6f}")
    print(f"JSON:      {output_json}")
    print(f"per-class: {output_csv}")


if __name__ == "__main__":
    main()
