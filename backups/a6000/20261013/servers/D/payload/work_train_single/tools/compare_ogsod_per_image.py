#!/usr/bin/env python3
"""Create a fixed-threshold per-image baseline/full comparison for OGSOD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from snapshot_yolo_dataset import collect_images, label_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_ground_truth(path: Path, image_height: int, image_width: int):
    boxes, classes = [], []
    labels = label_path(path)
    if not labels.is_file():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    for line in labels.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id = int(float(values[0]))
        x, y, width, height = (float(value) for value in values[1:5])
        boxes.append(
            [
                (x - width / 2) * image_width,
                (y - height / 2) * image_height,
                (x + width / 2) * image_width,
                (y + height / 2) * image_height,
            ]
        )
        classes.append(class_id)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    first_area = np.clip(first[:, 2:] - first[:, :2], 0, None).prod(axis=1)
    second_area = np.clip(second[:, 2:] - second[:, :2], 0, None).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)


def match_counts(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    class_ids: list[int],
    threshold: float,
) -> dict:
    matched = set()
    tp = {class_id: 0 for class_id in class_ids}
    fp = {class_id: 0 for class_id in class_ids}
    ious = box_iou(pred_boxes, gt_boxes)
    for pred_index in np.argsort(-pred_conf):
        class_id = int(pred_classes[pred_index])
        candidates = [
            gt_index
            for gt_index, gt_class in enumerate(gt_classes)
            if int(gt_class) == class_id and gt_index not in matched
        ]
        if candidates:
            best = max(candidates, key=lambda gt_index: float(ious[pred_index, gt_index]))
            if ious[pred_index, best] >= threshold:
                matched.add(best)
                tp[class_id] = tp.get(class_id, 0) + 1
                continue
        fp[class_id] = fp.get(class_id, 0) + 1
    fn = {
        class_id: int(np.sum(gt_classes == class_id)) - tp.get(class_id, 0)
        for class_id in class_ids
    }
    return {"tp": tp, "fp": fp, "fn": fn}


def evaluate_model(model_path: Path, images: list[Path], args, class_ids, class_keys):
    model = YOLO(str(model_path))
    predictions = model.predict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.nms_iou,
        half=False,
        verbose=False,
    )
    records = {}
    for result in predictions:
        image_path = Path(result.path).resolve()
        image_height, image_width = result.orig_shape
        gt_boxes, gt_classes = read_ground_truth(image_path, image_height, image_width)
        if result.boxes is None:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            pred_classes = np.empty((0,), dtype=np.int64)
            pred_conf = np.empty((0,), dtype=np.float32)
        else:
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
            pred_classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            pred_conf = result.boxes.conf.detach().cpu().numpy()
        counts50 = match_counts(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, class_ids, 0.50
        )
        counts75 = match_counts(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, class_ids, 0.75
        )
        row = {
            "gt": len(gt_boxes),
            "pred": len(pred_boxes),
            "tp50": sum(counts50["tp"].values()),
            "fp50": sum(counts50["fp"].values()),
            "fn50": sum(counts50["fn"].values()),
            "tp75": sum(counts75["tp"].values()),
            "fp75": sum(counts75["fp"].values()),
            "fn75": sum(counts75["fn"].values()),
        }
        for class_id, key in zip(class_ids, class_keys):
            row[f"{key}_gt"] = int(np.sum(gt_classes == class_id))
            row[f"{key}_tp50"] = counts50["tp"].get(class_id, 0)
            row[f"{key}_tp75"] = counts75["tp"].get(class_id, 0)
            row[f"{key}_fn75"] = counts75["fn"].get(class_id, 0)
        records[str(image_path)] = row
    return records


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    yaml_dir = data_path.parent
    root = Path(data.get("path", yaml_dir)).expanduser()
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()
    images = collect_images(data[args.split], root, yaml_dir)
    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in names.items()}
    class_ids = sorted(names)
    class_keys = [safe_name(names[class_id]) for class_id in class_ids]

    baseline_weights = Path(args.baseline).expanduser().resolve()
    full_weights = Path(args.full).expanduser().resolve()
    baseline = evaluate_model(
        baseline_weights, images, args, class_ids, class_keys
    )
    full = evaluate_model(
        full_weights, images, args, class_ids, class_keys
    )
    rows = []
    for image in images:
        key = str(image.resolve())
        row = {"image": key}
        row.update({f"baseline_{name}": value for name, value in baseline[key].items()})
        row.update({f"full_{name}": value for name, value in full[key].items()})
        for metric in ("tp50", "tp75", "fp75", "fn75"):
            row[f"delta_{metric}"] = full[key][metric] - baseline[key][metric]
        for class_key in class_keys:
            row[f"delta_{class_key}_tp75"] = (
                full[key][f"{class_key}_tp75"] - baseline[key][f"{class_key}_tp75"]
            )
        rows.append(row)

    csv_path = output / "per_image_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OGSOD Per-Image Candidate Audit",
        "",
        f"Fixed confidence: {args.conf}; NMS IoU: {args.nms_iou}; match IoU: 0.75.",
        "",
        "These rankings select qualitative candidates only; they are not AP estimates.",
    ]
    for class_key, class_name in zip(class_keys, (names[class_id] for class_id in class_ids)):
        lines.extend(["", f"## {class_name}", "", "### Full Improves", ""])
        improved = sorted(
            (row for row in rows if row[f"delta_{class_key}_tp75"] > 0),
            key=lambda row: (row[f"delta_{class_key}_tp75"], row["delta_tp75"]),
            reverse=True,
        )[:10]
        lines.extend(
            f"- `{row['image']}`: class dTP75={row[f'delta_{class_key}_tp75']:+d}, all dTP75={row['delta_tp75']:+d}"
            for row in improved
        )
        if not improved:
            lines.append("- None")
        lines.extend(["", "### Full Regresses", ""])
        regressed = sorted(
            (row for row in rows if row[f"delta_{class_key}_tp75"] < 0),
            key=lambda row: (row[f"delta_{class_key}_tp75"], row["delta_tp75"]),
        )[:10]
        lines.extend(
            f"- `{row['image']}`: class dTP75={row[f'delta_{class_key}_tp75']:+d}, all dTP75={row['delta_tp75']:+d}"
            for row in regressed
        )
        if not regressed:
            lines.append("- None")

    report_path = output / "candidate_examples.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    image_list_digest = hashlib.sha256(
        "\n".join(str(path.resolve()) for path in images).encode("utf-8")
    ).hexdigest()
    manifest = {
        "tool": "compare_ogsod_per_image.py",
        "tool_sha256": sha256(Path(__file__).resolve()),
        "data": str(data_path),
        "data_yaml_sha256": sha256(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "workers": args.workers,
        "baseline": str(baseline_weights),
        "baseline_sha256": sha256(baseline_weights),
        "full": str(full_weights),
        "full_sha256": sha256(full_weights),
        "image_count": len(images),
        "image_list_sha256": image_list_digest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"per-image CSV: {csv_path}")
    print(f"candidate report: {report_path}")
    print(f"manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
