#!/usr/bin/env python3
"""Evaluate fixed OGSOD checkpoints across density, scale, and spacing strata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from snapshot_yolo_dataset import collect_images, label_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="repeat as LABEL=/absolute/path/to/best.pt",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--criteria", nargs="+", default=["density", "area", "nearest"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_models(items: list[str]) -> dict[str, Path]:
    models = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=WEIGHTS, received: {item}")
        label, value = item.split("=", 1)
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if label in models:
            raise ValueError(f"Duplicate model label: {label}")
        models[label] = path
    return models


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        boxes.append((int(float(values[0])), *(float(value) for value in values[1:5])))
    return boxes


def nearest_stat(boxes: list[tuple[int, float, float, float, float]]) -> float | None:
    if len(boxes) < 2:
        return None
    centers = [(box[1], box[2]) for box in boxes]
    distances = []
    for index, (x1, y1) in enumerate(centers):
        distances.append(
            min(
                math.hypot(x1 - x2, y1 - y2)
                for other, (x2, y2) in enumerate(centers)
                if other != index
            )
        )
    return median(distances)


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def as_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def metric_payload(metrics, model: YOLO) -> tuple[dict, list[dict]]:
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
    for row, class_id in enumerate(class_ids):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "precision": precision[row],
                "recall": recall[row],
                "ap50": ap50[row],
                "ap75": ap75[row],
                "ap50_95": ap50_95[row],
            }
        )
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "ap50": float(box.map50),
        "ap75": sum(ap75) / len(ap75),
        "ap50_95": float(box.map),
    }
    return overall, per_class


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    yaml_dir = data_path.parent
    root = Path(data.get("path", yaml_dir)).expanduser()
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    images = collect_images(data[args.split], root, yaml_dir)

    records = []
    for image in images:
        boxes = read_boxes(label_path(image))
        records.append(
            {
                "image": image,
                "boxes": boxes,
                "density": len(boxes),
                "area": median([box[3] * box[4] for box in boxes]) if boxes else None,
                "nearest": nearest_stat(boxes),
            }
        )

    strata = {}
    if "density" in args.criteria:
        strata.update(
            {
                "density_01_04": [row for row in records if 1 <= row["density"] <= 4],
                "density_05_12": [row for row in records if 5 <= row["density"] <= 12],
                "density_13_plus": [row for row in records if row["density"] >= 13],
            }
        )

    thresholds = {}
    if "area" in args.criteria:
        values = [row["area"] for row in records if row["area"] is not None]
        q1, q2 = quantile(values, 1 / 3), quantile(values, 2 / 3)
        thresholds["area"] = {"q33": q1, "q67": q2}
        strata.update(
            {
                "area_small": [row for row in records if row["area"] is not None and row["area"] <= q1],
                "area_medium": [row for row in records if row["area"] is not None and q1 < row["area"] <= q2],
                "area_large": [row for row in records if row["area"] is not None and row["area"] > q2],
            }
        )

    if "nearest" in args.criteria:
        values = [row["nearest"] for row in records if row["nearest"] is not None]
        q1, q2 = quantile(values, 1 / 3), quantile(values, 2 / 3)
        thresholds["nearest"] = {"q33": q1, "q67": q2}
        strata.update(
            {
                "nearest_crowded": [row for row in records if row["nearest"] is not None and row["nearest"] <= q1],
                "nearest_medium": [row for row in records if row["nearest"] is not None and q1 < row["nearest"] <= q2],
                "nearest_separated": [row for row in records if row["nearest"] is not None and row["nearest"] > q2],
                "nearest_single": [row for row in records if row["nearest"] is None and row["density"] == 1],
            }
        )

    manifest = {
        "data": str(data_path),
        "split": args.split,
        "images": len(images),
        "thresholds": thresholds,
        "strata": {},
    }
    results = []
    per_class_results = []
    for stratum, rows in strata.items():
        if not rows:
            continue
        subset_dir = output / "subsets" / stratum
        subset_dir.mkdir(parents=True, exist_ok=True)
        image_list = subset_dir / "images.txt"
        image_text = "\n".join(str(row["image"]) for row in rows) + "\n"
        image_list.write_text(image_text, encoding="utf-8")
        subset_yaml = subset_dir / "data.yaml"
        subset_data = dict(data)
        subset_data["path"] = str(root)
        subset_data["train"] = str(image_list)
        subset_data["val"] = str(image_list)
        subset_data["test"] = str(image_list)
        subset_yaml.write_text(yaml.safe_dump(subset_data, sort_keys=False), encoding="utf-8")

        class_counts = {}
        for row in rows:
            for box in row["boxes"]:
                class_counts[str(box[0])] = class_counts.get(str(box[0]), 0) + 1
        manifest["strata"][stratum] = {
            "images": len(rows),
            "instances": sum(row["density"] for row in rows),
            "class_counts": class_counts,
            "image_list_sha256": hashlib.sha256(image_text.encode("utf-8")).hexdigest(),
            "image_list": str(image_list),
        }

        for label, weights in models.items():
            model = YOLO(str(weights))
            metrics = model.val(
                data=str(subset_yaml),
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=False,
                plots=False,
                project=str(output / "eval"),
                name=f"{label}_{stratum}",
                exist_ok=True,
            )
            overall, per_class = metric_payload(metrics, model)
            result = {
                "model": label,
                "weights": str(weights),
                "stratum": stratum,
                "images": len(rows),
                "instances": sum(row["density"] for row in rows),
                **overall,
            }
            results.append(result)
            for class_result in per_class:
                per_class_results.append({**result, **class_result})
            print(
                f"{label:16s} {stratum:20s} images={len(rows):4d} "
                f"AP50={overall['ap50']:.6f} AP75={overall['ap75']:.6f} "
                f"AP50-95={overall['ap50_95']:.6f}"
            )

    manifest_path = output / "strata_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results_json = output / "strata_metrics.json"
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with (output / "strata_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with (output / "strata_per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_results[0].keys())
        writer.writeheader()
        writer.writerows(per_class_results)

    print(f"manifest:  {manifest_path}")
    print(f"metrics:   {results_json}")


if __name__ == "__main__":
    main()
