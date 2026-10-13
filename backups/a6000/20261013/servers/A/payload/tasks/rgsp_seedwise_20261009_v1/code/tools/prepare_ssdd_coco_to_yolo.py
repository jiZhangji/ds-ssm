#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


IMG_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_image(img_dir: Path, file_name: str):
    p = img_dir / file_name
    if p.exists():
        return p

    stem = Path(file_name).stem
    for suf in IMG_SUFFIXES:
        p = img_dir / f"{stem}{suf}"
        if p.exists():
            return p

    for suf in IMG_SUFFIXES:
        matches = list(img_dir.rglob(f"{stem}{suf}"))
        if matches:
            return matches[0]

    matches = list(img_dir.rglob(file_name))
    if matches:
        return matches[0]

    return None


def convert_split(json_path: Path, img_dir: Path, out: Path, split: str):
    out_img_dir = out / "images" / split
    out_lbl_dir = out / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    anns_by_img = defaultdict(list)
    for ann in anns:
        anns_by_img[ann["image_id"]].append(ann)

    copied = 0
    missing_img = 0
    empty_label = 0
    bad_box = 0

    print(f"\n[{split}] json: {json_path}")
    print(f"[{split}] image dir: {img_dir}")
    print(f"[{split}] images in json: {len(images)}")
    print(f"[{split}] annotations in json: {len(anns)}")

    for im in tqdm(images, desc=f"Processing {split}"):
        img_id = im["id"]
        file_name = im["file_name"]

        img_path = find_image(img_dir, file_name)
        if img_path is None:
            missing_img += 1
            continue

        # 优先用 json 里的 width/height；如果没有，再读图片
        img_w = im.get("width", None)
        img_h = im.get("height", None)
        if img_w is None or img_h is None:
            with Image.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

        yolo_lines = []

        for ann in anns_by_img.get(img_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue

            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                bad_box += 1
                continue

            # COCO bbox: [x, y, width, height]
            x, y, w, h = map(float, bbox)

            if w <= 1 or h <= 1:
                bad_box += 1
                continue

            x = max(0.0, min(x, img_w - 1))
            y = max(0.0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w <= 1 or h <= 1:
                bad_box += 1
                continue

            xc = (x + w / 2.0) / img_w
            yc = (y + h / 2.0) / img_h
            wn = w / img_w
            hn = h / img_h

            # SSDD 单类别 ship，所以 class_id = 0
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            empty_label += 1
            continue

        shutil.copy2(img_path, out_img_dir / img_path.name)

        label_path = out_lbl_dir / f"{img_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        copied += 1

    print(f"[{split}] copied images:", copied)
    print(f"[{split}] missing images:", missing_img)
    print(f"[{split}] empty labels:", empty_label)
    print(f"[{split}] bad boxes:", bad_box)


def auto_find_json(ann_dir: Path, split_keywords):
    jsons = sorted(ann_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No json files found in {ann_dir}")

    for key in split_keywords:
        for p in jsons:
            if key.lower() in p.name.lower():
                return p

    raise FileNotFoundError(
        f"Could not find json with keywords {split_keywords} in {ann_dir}. "
        f"Available jsons: {[p.name for p in jsons]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to BBox_SSDD/coco_style")
    parser.add_argument("--out", default="datasets/SSDD_YOLO")
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    ann_dir = root / "annotations"
    train_img_dir = root / "images" / "train"
    val_img_dir = root / "images" / "test"

    if args.train_json is None:
        train_json = auto_find_json(ann_dir, ["train"])
    else:
        train_json = Path(args.train_json)

    if args.val_json is None:
        val_json = auto_find_json(ann_dir, ["test", "val"])
    else:
        val_json = Path(args.val_json)

    assert train_img_dir.exists(), f"not found: {train_img_dir}"
    assert val_img_dir.exists(), f"not found: {val_img_dir}"
    assert train_json.exists(), f"not found: {train_json}"
    assert val_json.exists(), f"not found: {val_json}"

    convert_split(train_json, train_img_dir, out, "train")
    convert_split(val_json, val_img_dir, out, "val")

    yaml_text = f"""path: {out.resolve()}
train: images/train
val: images/val

names:
  0: ship
"""

    with open(out / "ssdd.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("\nDone.")
    print("Dataset yaml:", out / "ssdd.yaml")


if __name__ == "__main__":
    main()
