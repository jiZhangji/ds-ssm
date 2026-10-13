#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from tqdm import tqdm


IMG_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_image(img_dir: Path, stem: str):
    for suf in IMG_SUFFIXES:
        p = img_dir / f"{stem}{suf}"
        if p.exists():
            return p

    for suf in IMG_SUFFIXES:
        matches = list(img_dir.rglob(f"{stem}{suf}"))
        if matches:
            return matches[0]

    return None


def xml_to_yolo(xml_path: Path, img_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    labels = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        xmin = max(0.0, min(xmin, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        xmax = max(0.0, min(xmax, img_w - 1))
        ymax = max(0.0, min(ymax, img_h - 1))

        bw = xmax - xmin
        bh = ymax - ymin

        if bw <= 1 or bh <= 1:
            continue

        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        bw = bw / img_w
        bh = bh / img_h

        labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return labels


def process_split(split, ann_dir, img_dir, out):
    out_img_dir = out / "images" / split
    out_lbl_dir = out / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    xmls = sorted(Path(ann_dir).glob("*.xml"))

    print(f"{split} xml count:", len(xmls))
    print(f"{split} ann dir:", ann_dir)
    print(f"{split} img dir:", img_dir)

    bad = []
    missing_img = []
    empty_label = []

    for xml in tqdm(xmls, desc=f"Processing {split}"):
        if xml.stat().st_size == 0:
            bad.append((str(xml), "empty xml"))
            continue

        img = find_image(Path(img_dir), xml.stem)
        if img is None:
            missing_img.append(str(xml))
            continue

        try:
            labels = xml_to_yolo(xml, img)
        except Exception as e:
            bad.append((str(xml), repr(e)))
            continue

        if not labels:
            empty_label.append(str(xml))
            continue

        shutil.copy2(img, out_img_dir / img.name)
        with open(out_lbl_dir / f"{img.stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    print(f"{split} bad xml:", len(bad))
    print(f"{split} missing image:", len(missing_img))
    print(f"{split} empty labels:", len(empty_label))

    if bad:
        print(f"{split} first bad:", bad[:5])
    if missing_img:
        print(f"{split} first missing:", missing_img[:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to BBox_SSDD/voc_style")
    parser.add_argument("--out", default="datasets/SSDD_YOLO")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    train_ann = root / "Annotations_train"
    train_img = root / "JPEGImages_train"
    val_ann = root / "Annotations_test"
    val_img = root / "JPEGImages_test"

    assert train_ann.exists(), f"not found: {train_ann}"
    assert train_img.exists(), f"not found: {train_img}"
    assert val_ann.exists(), f"not found: {val_ann}"
    assert val_img.exists(), f"not found: {val_img}"

    process_split("train", train_ann, train_img, out)
    process_split("val", val_ann, val_img, out)

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