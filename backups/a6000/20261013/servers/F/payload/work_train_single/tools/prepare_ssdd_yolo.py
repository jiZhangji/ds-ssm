#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from PIL import Image
from tqdm import tqdm


IMG_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def find_dir(root: Path, names):
    names = {n.lower() for n in names}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def find_image(img_root: Path, stem: str):
    for suf in IMG_SUFFIXES:
        p = img_root / f"{stem}{suf}"
        if p.exists():
            return p

    for suf in IMG_SUFFIXES:
        matches = list(img_root.rglob(f"{stem}{suf}"))
        if matches:
            return matches[0]

    return None


def parse_float(node, name):
    text = node.findtext(name)
    if text is None:
        raise ValueError(f"missing {name}")
    return float(text)


def voc_xml_to_yolo(xml_path: Path, img_path: Path):
    if xml_path.stat().st_size == 0:
        raise ParseError("empty xml file")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    labels = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")

        # 有些 SSDD 版本可能是 robndbox 旋转框，这里优先处理普通 bndbox。
        # 如果只有 robndbox，本脚本会跳过，需要另写旋转框转水平框逻辑。
        if bndbox is None:
            continue

        try:
            xmin = parse_float(bndbox, "xmin")
            ymin = parse_float(bndbox, "ymin")
            xmax = parse_float(bndbox, "xmax")
            ymax = parse_float(bndbox, "ymax")
        except Exception:
            continue

        xmin = max(0.0, min(xmin, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        xmax = max(0.0, min(xmax, img_w - 1))
        ymax = max(0.0, min(ymax, img_h - 1))

        box_w = xmax - xmin
        box_h = ymax - ymin

        if box_w <= 1 or box_h <= 1:
            continue

        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        box_w = box_w / img_w
        box_h = box_h / img_h

        labels.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, default="datasets/SSDD_YOLO")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)

    ann_dir = find_dir(raw, ["Annotations", "annotations"])
    img_dir = find_dir(raw, ["JPEGImages", "images", "Imgs", "img"])

    if ann_dir is None:
        xmls = sorted(raw.rglob("*.xml"))
        if not xmls:
            raise RuntimeError("No XML annotation files found.")
        ann_dir = xmls[0].parent
    else:
        xmls = sorted(ann_dir.glob("*.xml"))

    if img_dir is None:
        imgs = []
        for suf in IMG_SUFFIXES:
            imgs.extend(raw.rglob(f"*{suf}"))
        if not imgs:
            raise RuntimeError("No image files found.")
        img_dir = imgs[0].parent

    print("Annotation dir:", ann_dir)
    print("Image dir:", img_dir)
    print("XML count:", len(xmls))

    random.seed(args.seed)
    random.shuffle(xmls)

    val_num = int(len(xmls) * args.val_ratio)
    val_xmls = xmls[:val_num]
    train_xmls = xmls[val_num:]

    bad_xmls = []
    missing_images = []
    empty_labels = []

    for split, split_xmls in [("train", train_xmls), ("val", val_xmls)]:
        image_out_dir = out / "images" / split
        label_out_dir = out / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for xml in tqdm(split_xmls, desc=f"Processing {split}"):
            img = find_image(img_dir, xml.stem)
            if img is None:
                missing_images.append(str(xml))
                continue

            try:
                labels = voc_xml_to_yolo(xml, img)
            except Exception as e:
                bad_xmls.append((str(xml), repr(e)))
                continue

            if len(labels) == 0:
                empty_labels.append(str(xml))
                continue

            shutil.copy2(img, image_out_dir / img.name)
            with open(label_out_dir / f"{img.stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

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

    print("\nSummary:")
    print("bad xmls:", len(bad_xmls))
    print("missing images:", len(missing_images))
    print("empty labels:", len(empty_labels))

    if bad_xmls:
        print("\nFirst 10 bad XML files:")
        for x, err in bad_xmls[:10]:
            print(x, err)

    if missing_images:
        print("\nFirst 10 missing image XML files:")
        for x in missing_images[:10]:
            print(x)

    if empty_labels:
        print("\nFirst 10 empty-label XML files:")
        for x in empty_labels[:10]:
            print(x)


if __name__ == "__main__":
    main()