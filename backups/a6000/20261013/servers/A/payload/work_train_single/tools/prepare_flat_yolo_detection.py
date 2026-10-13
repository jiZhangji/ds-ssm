#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def paired_images(src: Path):
    images = []
    for img in sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES):
        label = img.with_suffix(".txt")
        if label.exists():
            images.append(img)
    return images


def write_split(items, out: Path, split: str):
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in items:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(img.with_suffix(".txt"), lbl_dir / f"{img.stem}.txt")


def main():
    parser = argparse.ArgumentParser(description="Convert a flat YOLO jpg/txt detection dataset into YOLO train/val/test folders.")
    parser.add_argument("--src", required=True, help="Directory containing flat image/label pairs")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--name", default="dataset", help="Dataset name used in generated YAML")
    parser.add_argument("--class-name", default="ship", help="Class name for class id 0")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise FileNotFoundError(src)

    images = paired_images(src)
    if not images:
        raise RuntimeError(f"No image/txt pairs found in {src}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, items in splits.items():
        write_split(items, out, split)
        print(f"{split}: {len(items)}")

    yaml_text = f"""path: {out.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: {args.class_name}
"""
    yaml_path = out / f"{args.name}.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()

