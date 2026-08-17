#!/usr/bin/env python3
"""Evaluate one SSDD sweep checkpoint on validation or test data."""

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
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--state-dim", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--mode", choices=("standard", "tta"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
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
    weights = Path(args.weights).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    info = model.model.info(verbose=True, imgsz=args.imgsz)
    layers, parameters, gradients, gflops = info

    use_tta = args.mode == "tta"
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        half=False,
        augment=use_tta,
        plots=False,
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

    per_class = []
    for row_index, class_id in enumerate(class_ids):
        p = precision[row_index]
        r = recall[row_index]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "precision": p,
                "recall": r,
                "f1": 2.0 * p * r / (p + r + 1e-12),
                "ap50": ap50[row_index],
                "ap75": ap75[row_index],
                "ap50_95": ap50_95[row_index],
            }
        )

    mp = float(box.mp)
    mr = float(box.mr)
    overall = {
        "precision": mp,
        "recall": mr,
        "f1": 2.0 * mp * mr / (mp + mr + 1e-12),
        "ap50": float(box.map50),
        "ap75": sum(ap75) / len(ap75),
        "ap50_95": float(box.map),
    }

    payload = {
        "run_name": args.run_name,
        "placement": args.placement,
        "recipe": args.recipe,
        "state_dim": args.state_dim,
        "seed": args.seed,
        "mode": args.mode,
        "augment": use_tta,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(data),
        "data_yaml_sha256": sha256(data),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "split": args.split,
        "precision_mode": "fp32",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "result_dir": str(metrics.save_dir),
        "model": {
            "layers": int(layers),
            "parameters": int(parameters),
            "gradients": int(gradients),
            "gflops": float(gflops),
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in (metrics.speed or {}).items()
        },
        "overall": overall,
        "per_class": per_class,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_csv = output_json.with_name(f"{args.split}_{args.mode}_per_class.csv")
    if per_class:
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
            writer.writeheader()
            writer.writerows(per_class)

    print(f"\n===== SSDD BEST {args.split.upper()} ({args.mode.upper()}) =====")
    print(f"run:       {args.run_name}")
    print(f"placement: {args.placement}")
    print(f"recipe:    {args.recipe}")
    print(f"state_dim: {args.state_dim}")
    print(f"seed:      {args.seed}")
    print(f"P:         {overall['precision']:.6f}")
    print(f"R:         {overall['recall']:.6f}")
    print(f"F1:        {overall['f1']:.6f}")
    print(f"AP50:      {overall['ap50']:.6f}")
    print(f"AP75:      {overall['ap75']:.6f}")
    print(f"AP50-95:   {overall['ap50_95']:.6f}")
    print(f"Params:    {parameters}")
    print(f"GFLOPs:    {gflops:.4f}")
    print(f"JSON:      {output_json}")


if __name__ == "__main__":
    main()
