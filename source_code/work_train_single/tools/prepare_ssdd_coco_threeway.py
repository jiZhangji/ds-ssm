#!/usr/bin/env python3
"""Prepare the COCO-format SSDD release as a three-way YOLO dataset.

The source release contains an official training set and test set but no
validation set. This tool deterministically splits the official training set
into train/val, keeps the official test set untouched, and optionally prepares
the inshore/offshore test subsets. Source images and annotations are never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="SSDD root containing annotations/ and images/")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How output images reference source images",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
        raise ValueError(f"Not a COCO annotation file: {path}")
    return data


def find_image(image_dir: Path, file_name: str) -> Path:
    direct = image_dir / file_name
    if direct.is_file():
        return direct

    stem = Path(file_name).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    matches = [path for path in image_dir.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Image not found for {file_name!r} below {image_dir}")
    raise RuntimeError(f"Ambiguous image stem {stem!r}: {matches[:5]}")


def materialize_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return
    if mode == "symlink":
        target.symlink_to(source)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def yolo_lines(image: dict, annotations: list[dict]) -> list[str]:
    width = float(image["width"])
    height = float(image["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {image}")

    lines = []
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, box_width, box_height = map(float, bbox)
        x1 = max(0.0, min(x, width))
        y1 = max(0.0, min(y, height))
        x2 = max(0.0, min(x + box_width, width))
        y2 = max(0.0, min(y + box_height, height))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        center_x = (x1 + x2) / 2.0 / width
        center_y = (y1 + y2) / 2.0 / height
        lines.append(
            f"0 {center_x:.8f} {center_y:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    return lines


def convert_records(
    coco: dict,
    image_dir: Path,
    output: Path,
    split: str,
    selected_ids: set[int] | None,
    link_mode: str,
) -> dict:
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    images = [
        image for image in coco["images"]
        if selected_ids is None or int(image["id"]) in selected_ids
    ]
    images.sort(key=lambda item: (str(item.get("file_name", "")), int(item["id"])))

    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    object_count = 0
    empty_count = 0
    manifest_images = []

    for image in images:
        source = find_image(image_dir, str(image["file_name"]))
        target_name = source.name
        if target_name in seen_names:
            raise RuntimeError(f"Duplicate output image name in {split}: {target_name}")
        seen_names.add(target_name)

        lines = yolo_lines(image, annotations_by_image.get(int(image["id"]), []))
        if not lines:
            empty_count += 1
        object_count += len(lines)

        materialize_image(source, image_output / target_name, link_mode)
        (label_output / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="ascii"
        )
        manifest_images.append(
            {
                "id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "output_name": target_name,
                "objects": len(lines),
            }
        )

    return {
        "split": split,
        "images": len(images),
        "objects": object_count,
        "images_without_objects": empty_count,
        "records": manifest_images,
    }


def write_yaml(path: Path, dataset_root: Path, test_split: str) -> None:
    path.write_text(
        f"path: {dataset_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: images/{test_split}\n\n"
        "names:\n"
        "  0: ship\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if output == root or root in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")

    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists; pass --force to replace it: {output}")
        if output == Path(output.anchor):
            raise ValueError(f"Refusing to remove filesystem root: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    annotation_dir = root / "annotations"
    train_json = annotation_dir / "train.json"
    test_json = annotation_dir / "test.json"
    train_coco = load_coco(train_json)
    test_coco = load_coco(test_json)

    train_ids = [int(image["id"]) for image in train_coco["images"]]
    random.Random(args.seed).shuffle(train_ids)
    val_count = max(1, min(len(train_ids) - 1, round(len(train_ids) * args.val_ratio)))
    val_ids = set(train_ids[:val_count])
    kept_train_ids = set(train_ids[val_count:])

    summaries = [
        convert_records(
            train_coco, root / "images" / "train", output, "train", kept_train_ids, args.link_mode
        ),
        convert_records(train_coco, root / "images" / "train", output, "val", val_ids, args.link_mode),
        convert_records(test_coco, root / "images" / "test", output, "test", None, args.link_mode),
    ]

    subset_sources = {
        "test_inshore": (annotation_dir / "test_inshore.json", root / "images" / "test_inshore"),
        "test_offshore": (annotation_dir / "test_offshore.json", root / "images" / "test_offshore"),
    }
    for split, (json_path, image_dir) in subset_sources.items():
        if json_path.is_file() and image_dir.is_dir():
            summaries.append(
                convert_records(load_coco(json_path), image_dir, output, split, None, args.link_mode)
            )

    write_yaml(output / "SSDD.yaml", output, "test")
    if (output / "images" / "test_inshore").is_dir():
        write_yaml(output / "SSDD_inshore.yaml", output, "test_inshore")
    if (output / "images" / "test_offshore").is_dir():
        write_yaml(output / "SSDD_offshore.yaml", output, "test_offshore")

    manifest = {
        "source_root": str(root),
        "output_root": str(output),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "link_mode": args.link_mode,
        "annotation_sha256": {
            "train": sha256(train_json),
            "test": sha256(test_json),
        },
        "splits": summaries,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("SSDD conversion complete")
    print(f"Output: {output}")
    for summary in summaries:
        print(
            f"{summary['split']}: images={summary['images']} "
            f"objects={summary['objects']} empty={summary['images_without_objects']}"
        )
    print(f"Dataset YAML: {output / 'SSDD.yaml'}")


if __name__ == "__main__":
    main()
