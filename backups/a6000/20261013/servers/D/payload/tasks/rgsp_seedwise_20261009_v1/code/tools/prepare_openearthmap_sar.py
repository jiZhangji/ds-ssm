#!/usr/bin/env python3
"""Materialize the OpenEarthMap-SAR SAR-only P-label benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = (
    "bareland",
    "rangeland",
    "developed_space",
    "road",
    "tree",
    "water",
    "agriculture_land",
    "building",
)
OFFICIAL_COLORS = np.asarray(
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 255, 36),
        (148, 148, 148),
        (255, 255, 255),
        (34, 97, 38),
        (0, 69, 255),
        (75, 181, 73),
        (222, 31, 7),
    ],
    dtype=np.uint8,
)


def label_ids(image: Image.Image) -> np.ndarray:
    if image.mode in {"1", "L", "I", "I;16", "P"}:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        values = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for index, color in enumerate(OFFICIAL_COLORS):
            values[np.all(rgb == color, axis=2)] = index
        if bool((values == 255).any()):
            unknown = np.unique(rgb[values == 255].reshape(-1, 3), axis=0)
            raise ValueError(f"Unknown OpenEarthMap-SAR label colors: {unknown[:10].tolist()}")
    unique = set(int(value) for value in np.unique(values))
    if not unique.issubset(set(range(9))):
        raise ValueError(f"Expected label IDs 0..8, found {sorted(unique)}")
    return values


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def materialize_split(dataset, split: str, output: Path) -> list[dict[str, str]]:
    split_name = "val" if split == "validation" else split
    image_dir = output / split_name / "sar_images"
    mask_dir = output / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []
    label_histogram = np.zeros(9, dtype=np.int64)
    for index, row in enumerate(dataset):
        image_id = safe_name(str(row.get("image_id") or Path(row["filename"]).stem))
        image_path = image_dir / f"{image_id}.png"
        mask_path = mask_dir / f"{image_id}.png"
        if not image_path.exists():
            row["image"].convert("L").save(image_path, compress_level=1)
        if row.get("label") is None:
            raise ValueError(f"Missing released label for {split}:{image_id}")
        ids = label_ids(row["label"])
        label_histogram += np.bincount(ids.ravel(), minlength=9)[:9]
        if not mask_path.exists():
            Image.fromarray(ids, mode="L").save(mask_path, compress_level=1)
        records.append(
            {
                "split": split_name,
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "image_id": image_id,
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(dataset):
            print(f"{split_name}: {index + 1}/{len(dataset)}", flush=True)
    (output / split_name / "label_histogram.json").write_text(
        json.dumps(label_histogram.tolist(), indent=2) + "\n", encoding="utf-8"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--dataset", default="doem1997/openearthmap-sar")
    args = parser.parse_args()

    from datasets import load_dataset

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.dataset, cache_dir=args.cache_dir)
    expected = {"train": 4333, "validation": 210, "test": 490}
    actual = {split: len(data[split]) for split in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected OpenEarthMap-SAR split sizes: {actual}, expected {expected}")

    records = []
    for split in ("train", "validation", "test"):
        records.extend(materialize_split(data[split], split, output))

    gray_palette = [[value, value, value] for value in range(9)]
    manifest = {
        "dataset": "OpenEarthMap-SAR",
        "task": "segmentation",
        "protocol": "SAR-only; train on 4,333 released pseudo labels (P); select on 210 released real validation labels; evaluate once on 490 released real test labels",
        "source": args.dataset,
        "official_source": "https://zenodo.org/records/14950559",
        "classes": 8,
        "class_names": list(CLASS_NAMES),
        "palette": gray_palette,
        "label_map": [-1, 0, 1, 2, 3, 4, 5, 6, 7],
        "ignore_index": -1,
        "primary_metric": "miou_all",
        "selection_policy": "best validation mIoU; test evaluated once after selection",
        "segmentation_loss": "weighted_cross_entropy_plus_dice",
        "records": records,
        "split_sizes": {"train": 4333, "val": 210, "test": 490},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
