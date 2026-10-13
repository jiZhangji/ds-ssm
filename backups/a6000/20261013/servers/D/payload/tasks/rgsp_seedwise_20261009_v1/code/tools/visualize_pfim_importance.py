#!/usr/bin/env python3
"""Render PFIM importance maps for fixed-checkpoint qualitative analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.mamba_yolo import DS_VSSBlock

from snapshot_yolo_dataset import label_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--block-index", type=int, default=0)
    return parser.parse_args()


def find_blocks(model) -> list[DS_VSSBlock]:
    return [module for module in model.modules() if isinstance(module, DS_VSSBlock)]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (230, 90, 40),
        (50, 180, 70),
        (40, 130, 230),
        (190, 90, 210),
        (30, 190, 190),
        (210, 150, 30),
        (140, 140, 140),
    )
    return palette[class_id % len(palette)]


def draw_ground_truth(image: np.ndarray, path: Path, names: dict) -> np.ndarray:
    output = image.copy()
    labels = label_path(path)
    if not labels.is_file():
        return output
    height, width = output.shape[:2]
    for line in labels.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id = int(float(values[0]))
        x, y, box_width, box_height = (float(value) for value in values[1:5])
        x1 = int((x - box_width / 2) * width)
        y1 = int((y - box_height / 2) * height)
        x2 = int((x + box_width / 2) * width)
        y2 = int((y + box_height / 2) * height)
        color = color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"GT {names[class_id]}",
            (max(0, x1), max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def draw_predictions(image: np.ndarray, result, names: dict) -> np.ndarray:
    output = image.copy()
    if result.boxes is None:
        return output
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    for box, class_id, confidence in zip(xyxy, classes, confidences):
        x1, y1, x2, y2 = (int(value) for value in box)
        color = color_for_class(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{names[class_id]} {confidence:.2f}",
            (max(0, x1), max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def title_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = np.full((image.shape[0] + 34, image.shape[1], 3), 255, dtype=np.uint8)
    panel[34:] = image
    cv2.putText(panel, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    blocks = find_blocks(model.model)
    if not blocks:
        raise RuntimeError("No DS_VSSBlock was found; PFIM importance is unavailable for this checkpoint")
    if not 0 <= args.block_index < len(blocks):
        raise IndexError(f"block-index {args.block_index} is outside 0..{len(blocks) - 1}")

    ds_ssm = blocks[args.block_index].ds_ssm
    if ds_ssm.pfim is None:
        raise RuntimeError("Selected block does not contain PFIM")
    ds_ssm.capture_importance = True
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))

    summaries = []
    for value in args.images:
        image_path = Path(value).expanduser().resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            half=False,
            verbose=False,
        )[0]
        importance = getattr(ds_ssm, "last_importance", None)
        hw_shape = getattr(ds_ssm, "last_hw_shape", None)
        if importance is None or hw_shape is None:
            raise RuntimeError("PFIM importance capture did not run")
        importance = importance[0, :, 0].float().cpu().numpy()
        height_tokens, width_tokens = (int(value) for value in hw_shape)
        if importance.size != height_tokens * width_tokens:
            raise ValueError(
                f"importance size {importance.size} does not match feature shape {hw_shape}"
            )
        importance = importance.reshape(height_tokens, width_tokens)
        minimum, maximum = float(importance.min()), float(importance.max())
        normalized = (importance - minimum) / max(maximum - minimum, 1e-8)
        resized = cv2.resize(normalized, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.58, heatmap, 0.42, 0)

        panels = [
            title_panel(draw_ground_truth(image, image_path, names), "Ground truth"),
            title_panel(draw_predictions(image, result, names), "PFIM predictions"),
            title_panel(heatmap, "PFIM importance"),
            title_panel(overlay, "Importance overlay"),
        ]
        target_height = max(panel.shape[0] for panel in panels)
        panels = [
            cv2.copyMakeBorder(panel, 0, target_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for panel in panels
        ]
        figure = cv2.hconcat(panels)
        output_path = output_dir / f"{image_path.stem}_pfim_importance.png"
        cv2.imwrite(str(output_path), figure)
        np.save(output_dir / f"{image_path.stem}_pfim_importance.npy", importance)
        summaries.append(
            {
                "image": str(image_path),
                "output": str(output_path),
                "block_index": args.block_index,
                "feature_height": height_tokens,
                "feature_width": width_tokens,
                "importance_min": minimum,
                "importance_max": maximum,
                "importance_mean": float(importance.mean()),
                "importance_std": float(importance.std()),
                "predictions": len(result.boxes) if result.boxes is not None else 0,
            }
        )
        print(output_path)

    with (output_dir / "importance_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    weights_path = Path(args.weights).expanduser().resolve()
    image_paths = [Path(value).expanduser().resolve() for value in args.images]
    manifest = {
        "tool": "visualize_pfim_importance.py",
        "tool_sha256": sha256(Path(__file__).resolve()),
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "block_index": args.block_index,
        "precision": "fp32",
        "images": [
            {"path": str(path), "sha256": sha256(path)}
            for path in image_paths
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
