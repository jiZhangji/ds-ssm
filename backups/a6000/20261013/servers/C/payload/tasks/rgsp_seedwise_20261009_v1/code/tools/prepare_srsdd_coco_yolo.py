#!/usr/bin/env python3
"""Convert the COCO release of SRSDD-v1.0 to a deterministic YOLO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(source: Path, split: str) -> tuple[dict, Path, Path]:
    split_dir = source / split
    annotation = split_dir / f"{split}.json"
    image_dir = split_dir / "images"
    if not annotation.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"Missing SRSDD {split} JSON/images under {split_dir}")
    return json.loads(annotation.read_text(encoding="utf-8")), annotation, image_dir


def link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Existing image link points elsewhere: {target}")
        return
    os.symlink(source.resolve(), target)


def write_labels(
    payload: dict,
    image_dir: Path,
    output: Path,
    split: str,
    selected_ids: set[int],
    class_by_name: dict[str, int],
) -> tuple[int, int]:
    category_names = {int(row["id"]): row["name"] for row in payload["categories"]}
    annotations = defaultdict(list)
    for annotation in payload["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[int(annotation["image_id"])].append(annotation)

    image_count = instance_count = 0
    for image in sorted(payload["images"], key=lambda row: row["file_name"]):
        image_id = int(image["id"])
        if image_id not in selected_ids:
            continue
        source_image = image_dir / image["file_name"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        link_image(source_image, output / "images" / split / source_image.name)

        width, height = float(image["width"]), float(image["height"])
        lines = []
        for annotation in annotations.get(image_id, []):
            category_name = category_names[int(annotation["category_id"])]
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                f"{class_by_name[category_name]} "
                f"{((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        label = output / "labels" / split / f"{source_image.stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_count += 1
        instance_count += len(lines)
    return image_count, instance_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")

    train, train_json, train_images = load_split(source, "train")
    test, test_json, test_images = load_split(source, "test")
    ordered_names = [row["name"] for row in sorted(train["categories"], key=lambda row: int(row["id"]))]
    for name in sorted({row["name"] for row in test["categories"]}):
        if name not in ordered_names:
            ordered_names.append(name)
    class_by_name = {name: index for index, name in enumerate(ordered_names)}

    train_rows = sorted(train["images"], key=lambda row: row["file_name"])
    random.Random(args.seed).shuffle(train_rows)
    val_count = max(1, round(len(train_rows) * args.val_fraction))
    val_ids = {int(row["id"]) for row in train_rows[:val_count]}
    train_ids = {int(row["id"]) for row in train_rows[val_count:]}
    test_ids = {int(row["id"]) for row in test["images"]}

    counts = {}
    counts["train"] = write_labels(train, train_images, output, "train", train_ids, class_by_name)
    counts["val"] = write_labels(train, train_images, output, "val", val_ids, class_by_name)
    counts["test"] = write_labels(test, test_images, output, "test", test_ids, class_by_name)

    yaml_lines = [
        f"path: {output}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(ordered_names)),
        "",
    ]
    (output / "SRSDD_v1.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    snapshot = {
        "dataset": "SRSDD-v1.0",
        "source": str(source),
        "format": "COCO bbox to horizontal YOLO",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "classes": ordered_names,
        "counts": {
            split: {"images": values[0], "instances": values[1]}
            for split, values in counts.items()
        },
        "source_sha256": {"train_json": sha256(train_json), "test_json": sha256(test_json)},
    }
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
