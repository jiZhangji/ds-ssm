#!/usr/bin/env python3
"""Audit prepared paper-aligned HBB datasets and render representative overlays."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def audit_dataset(yaml_path: Path, output: Path, seed: int) -> dict:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    names = payload["names"]
    class_names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
    report = {"yaml": str(yaml_path), "root": str(root), "classes": class_names, "splits": {}}
    colors = [(255, 40, 40), (40, 220, 40), (40, 120, 255), (255, 200, 40), (220, 40, 220), (40, 220, 220)]
    rng = random.Random(seed)

    for split in ("train", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        class_counts = Counter()
        box_areas, box_widths, box_heights = [], [], []
        empty = invalid = 0
        parsed = {}
        for image_path in images:
            rows = []
            label_path = label_dir / f"{image_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
                fields = line.split()
                if len(fields) != 5:
                    invalid += 1
                    continue
                class_id = int(fields[0])
                cx, cy, width, height = map(float, fields[1:])
                if not (0 <= class_id < len(class_names) and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    invalid += 1
                    continue
                class_counts[class_names[class_id]] += 1
                box_widths.append(width)
                box_heights.append(height)
                box_areas.append(width * height)
                rows.append((class_id, cx, cy, width, height))
            if not rows:
                empty += 1
            parsed[image_path] = rows

        sample_paths = [path for path in images if parsed[path]]
        rng.shuffle(sample_paths)
        overlay_dir = output / yaml_path.stem / split
        overlay_dir.mkdir(parents=True, exist_ok=True)
        pixel_stats = []
        for image_path in sample_paths[:6]:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            array = np.asarray(image)
            pixel_stats.append({"min": int(array.min()), "max": int(array.max()), "mean": float(array.mean()), "std": float(array.std())})
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for class_id, cx, cy, bw, bh in parsed[image_path]:
                x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                color = colors[class_id % len(colors)]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                draw.text((x1 + 2, y1 + 2), class_names[class_id], fill=color)
            image.save(overlay_dir / f"{image_path.stem}_overlay.jpg", quality=92)

        report["splits"][split] = {
            "images": len(images),
            "empty_images": empty,
            "invalid_rows": invalid,
            "boxes": sum(class_counts.values()),
            "class_counts": dict(class_counts),
            "box_width_p01_p50_p99": [percentile(box_widths, q) for q in (1, 50, 99)],
            "box_height_p01_p50_p99": [percentile(box_heights, q) for q in (1, 50, 99)],
            "box_area_p01_p50_p99": [percentile(box_areas, q) for q in (1, 50, 99)],
            "sample_pixel_stats": pixel_stats,
            "overlay_dir": str(overlay_dir),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    reports = [audit_dataset(Path(path).resolve(), output, args.seed) for path in args.yaml]
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "audit.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
