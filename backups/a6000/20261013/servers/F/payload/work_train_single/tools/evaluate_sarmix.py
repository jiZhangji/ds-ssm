#!/usr/bin/env python3
"""Evaluate one SAR checkpoint on a validation or test split."""

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
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset", default="overall")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
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
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()

    model = YOLO(weights)
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        half=False,
        plots=False,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    box = metrics.box
    class_ids = [int(value) for value in as_list(box.ap_class_index)]
    precision = [float(value) for value in as_list(box.p)]
    recall = [float(value) for value in as_list(box.r)]
    ap50 = [float(value) for value in as_list(box.ap50)]
    ap50_95 = [float(value) for value in as_list(box.ap)]
    all_ap = as_list(box.all_ap)
    ap75 = [float(row[5]) for row in all_ap]

    names = metrics.names or model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    per_class = [
        {
            "class_id": class_id,
            "class_name": str(names[class_id]),
            "precision": precision[index],
            "recall": recall[index],
            "ap50": ap50[index],
            "ap75": ap75[index],
            "ap50_95": ap50_95[index],
        }
        for index, class_id in enumerate(class_ids)
    ]
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "ap50": float(box.map50),
        "ap75": float(box.map75),
        "ap50_95": float(box.map),
    }
    payload = {
        "experiment": args.experiment,
        "placement": args.placement,
        "variant": args.variant,
        "state_dim": args.state_dim,
        "seed": args.seed,
        "subset": args.subset,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(data),
        "data_yaml_sha256": sha256(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "precision_mode": "fp32",
        "result_dir": str(metrics.save_dir),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "overall": overall,
        "per_class": per_class,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".per_class.csv")
    fieldnames = ("class_id", "class_name", "precision", "recall", "ap50", "ap75", "ap50_95")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)

    print(f"experiment: {args.experiment}")
    print(f"subset:     {args.subset}")
    print(f"P/R:        {overall['precision']:.6f}/{overall['recall']:.6f}")
    print(f"AP50:       {overall['ap50']:.6f}")
    print(f"AP75:       {overall['ap75']:.6f}")
    print(f"AP50-95:    {overall['ap50_95']:.6f}")
    print(f"JSON:       {output}")


if __name__ == "__main__":
    main()
