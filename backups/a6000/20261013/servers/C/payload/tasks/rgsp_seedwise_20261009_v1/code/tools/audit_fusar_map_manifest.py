#!/usr/bin/env python3
"""Audit FUSAR-Map labels and split protocol without changing training data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def exact_palette_counts(mask: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, int, list[list[int]]]:
    pixels = mask.reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    lookup = {tuple(color.tolist()): index for index, color in enumerate(palette)}
    totals = np.zeros(len(palette), dtype=np.int64)
    unknown = 0
    unknown_colors: list[list[int]] = []
    for color, count in zip(unique, counts):
        index = lookup.get(tuple(color.tolist()))
        if index is None:
            unknown += int(count)
            if len(unknown_colors) < 20:
                unknown_colors.append(color.tolist())
        else:
            totals[index] += int(count)
    return totals, unknown, unknown_colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-masks", type=int, default=0, help="0 audits every mask")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "segmentation":
        raise SystemExit(f"Expected segmentation manifest, got {manifest.get('task')!r}")

    palette = np.asarray(manifest["palette"], dtype=np.uint8)
    class_names = manifest.get("class_names") or [f"class_{index}" for index in range(len(palette))]
    split_counts = Counter(row["split"] for row in manifest["records"])
    city_counts: dict[str, Counter] = defaultdict(Counter)
    pairs = set()
    duplicate_pairs = 0
    records = manifest["records"][: args.max_masks or None]
    class_pixels = np.zeros(len(palette), dtype=np.int64)
    unknown_pixels = 0
    unknown_colors: list[list[int]] = []
    shape_counts = Counter()

    for row in records:
        pair = (row["image"], row["mask"])
        duplicate_pairs += int(pair in pairs)
        pairs.add(pair)
        city = row.get("city") or row.get("scene") or row.get("group") or "unknown"
        city_counts[row["split"]][city] += 1
        mask_path = Path(row["mask"])
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        shape_counts[f"{mask.shape[1]}x{mask.shape[0]}"] += 1
        counts, unknown, colors = exact_palette_counts(mask, palette)
        class_pixels += counts
        unknown_pixels += unknown
        for color in colors:
            if color not in unknown_colors and len(unknown_colors) < 20:
                unknown_colors.append(color)

    known_pixels = int(class_pixels.sum())
    total_pixels = known_pixels + unknown_pixels
    classes = []
    for index, count in enumerate(class_pixels):
        classes.append(
            {
                "index": index,
                "name": class_names[index] if index < len(class_names) else f"class_{index}",
                "rgb": palette[index].tolist(),
                "pixels": int(count),
                "fraction_of_known": float(count / max(known_pixels, 1)),
            }
        )

    report = {
        "manifest": str(manifest_path),
        "protocol": manifest.get("protocol"),
        "declared_classes": manifest.get("classes"),
        "class_names": class_names,
        "palette": palette.tolist(),
        "records_total": len(manifest["records"]),
        "records_audited": len(records),
        "split_counts": dict(split_counts),
        "city_counts_by_split": {split: dict(counts) for split, counts in city_counts.items()},
        "mask_shapes": dict(shape_counts),
        "duplicate_pairs": duplicate_pairs,
        "known_pixels": known_pixels,
        "unknown_pixels": unknown_pixels,
        "unknown_fraction": float(unknown_pixels / max(total_pixels, 1)),
        "unknown_color_examples": unknown_colors,
        "class_pixel_distribution": classes,
        "metric_warning": (
            "The current trainer reports both all-class mIoU and foreground-only mIoU. "
            "Published values are comparable only after confirming whether background is included."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
