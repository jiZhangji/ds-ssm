#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASETS = ("OGSOD", "SAR_AIRcraft", "SSDD", "ship_dataset_v0")


def image_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def safe_reset_dir(path: Path, force: bool):
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} exists. Use --force to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def maybe_unzip(dataset_root: Path, name: str):
    extracted = dataset_root / name
    if extracted.exists():
        return extracted
    zip_path = dataset_root / f"{name}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Neither {extracted} nor {zip_path} exists")
    print(f"[unzip] {zip_path} -> {extracted}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted)
    return extracted


def collapse_single_child(root: Path):
    cur = root
    while True:
        children = [p for p in cur.iterdir() if p.is_dir()]
        files = [p for p in cur.iterdir() if p.is_file()]
        if len(children) == 1 and not files:
            cur = children[0]
        else:
            return cur


def write_yaml(out: Path, name: str, names):
    lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, cls_name in enumerate(names):
        lines.append(f"  {idx}: {cls_name}")
    (out / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_items(items, train_ratio, val_ratio, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def copy_pair(img: Path, label: Path, out: Path, split: str):
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dir / img.name)
    shutil.copy2(label, lbl_dir / f"{img.stem}.txt")


def prepare_flat_yolo(src: Path, out: Path, name: str, names, train_ratio, val_ratio, seed):
    pairs = []
    for img in image_files(src):
        label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    if not pairs:
        return False

    splits = split_items(pairs, train_ratio, val_ratio, seed)
    for split, items in splits.items():
        for img, label in items:
            copy_pair(img, label, out, split)
        print(f"[{name}] {split}: {len(items)}")
    write_yaml(out, name, names)
    return True


def prepare_yolo_tree(src: Path, out: Path, name: str, names, train_ratio, val_ratio, seed):
    images_root = src / "images"
    labels_root = src / "labels"
    if not images_root.exists() or not labels_root.exists():
        return False

    known_splits = [s for s in ("train", "val", "valid", "test") if (images_root / s).exists()]
    if known_splits:
        for split in known_splits:
            dst_split = "val" if split == "valid" else split
            count = 0
            for img in image_files(images_root / split):
                rel = img.relative_to(images_root / split)
                label = labels_root / split / rel.with_suffix(".txt")
                if label.exists():
                    copy_pair(img, label, out, dst_split)
                    count += 1
            print(f"[{name}] {dst_split}: {count}")
    else:
        pairs = []
        for img in image_files(images_root):
            rel = img.relative_to(images_root)
            label = labels_root / rel.with_suffix(".txt")
            if label.exists():
                pairs.append((img, label))
        if not pairs:
            return False
        splits = split_items(pairs, train_ratio, val_ratio, seed)
        for split, items in splits.items():
            for img, label in items:
                copy_pair(img, label, out, split)
            print(f"[{name}] {split}: {len(items)}")
    write_yaml(out, name, names)
    return True


def find_image_by_name(root: Path, file_name: str):
    direct = root / file_name
    if direct.exists():
        return direct
    stem = Path(file_name).stem
    for img in image_files(root):
        if img.name == file_name or img.stem == stem:
            return img
    return None


def convert_coco_split(json_path: Path, image_root: Path, out: Path, split: str, cat_id_to_yolo):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) != 1:
            anns_by_img[ann["image_id"]].append(ann)

    count = 0
    for img_id, im in images.items():
        img_path = find_image_by_name(image_root, im["file_name"])
        if img_path is None:
            continue
        width = im.get("width")
        height = im.get("height")
        if width is None or height is None:
            with Image.open(img_path) as pil_img:
                width, height = pil_img.size

        label_lines = []
        for ann in anns_by_img.get(img_id, []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1:
                continue
            cls = cat_id_to_yolo.get(ann["category_id"], 0)
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w / width:.6f} {h / height:.6f}")
        if not label_lines:
            continue

        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, img_dir / img_path.name)
        (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        count += 1
    return count


def prepare_coco(src: Path, out: Path, name: str, fallback_names):
    jsons = sorted(src.rglob("*.json"))
    if not jsons:
        return False

    by_name = {p.name.lower(): p for p in jsons}
    train_json = by_name.get("train.json") or next(
        (p for p in jsons if "train" in p.name.lower() and "trainval" not in p.name.lower()), None
    )
    val_json = next((by_name.get(n) for n in ("val.json", "valid.json", "validation.json") if by_name.get(n)), None)
    if val_json is None:
        val_json = next(
            (
                p
                for p in jsons
                if any(k in p.name.lower() for k in ("val", "valid")) and "trainval" not in p.name.lower()
            ),
            None,
        )
    test_json = by_name.get("test.json") or next((p for p in jsons if "test" in p.name.lower()), None)
    split_jsons = {"train": train_json, "val": val_json, "test": test_json}
    if train_json is None:
        return False

    train_data = json.loads(train_json.read_text(encoding="utf-8"))
    categories = sorted(train_data.get("categories", []), key=lambda c: c.get("id", 0))
    if categories:
        cat_id_to_yolo = {cat["id"]: i for i, cat in enumerate(categories)}
        names = [cat.get("name", str(i)) for i, cat in enumerate(categories)]
    else:
        cat_id_to_yolo = defaultdict(int)
        names = fallback_names

    image_root = src / "images" if (src / "images").exists() else src
    for split, json_path in split_jsons.items():
        if json_path is None:
            continue
        count = convert_coco_split(json_path, image_root, out, split, cat_id_to_yolo)
        print(f"[{name}] {split}: {count}")
    write_yaml(out, name, names)
    return True


def voc_to_yolo(xml_path: Path, img_path: Path, class_to_id):
    with Image.open(img_path) as im:
        width, height = im.size
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall("object"):
        cls_name = (obj.findtext("name") or "ship").strip()
        cls = class_to_id.setdefault(cls_name, len(class_to_id))
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        if w <= 1 or h <= 1:
            continue
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def prepare_voc(src: Path, out: Path, name: str, train_ratio, val_ratio, seed):
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        return False
    class_to_id = {}
    pairs = []
    for xml in xmls:
        img = find_image_by_name(src, f"{xml.stem}.jpg")
        if img is None:
            img = find_image_by_name(src, xml.stem)
        if img is not None:
            pairs.append((img, xml))
    if not pairs:
        return False

    splits = split_items(pairs, train_ratio, val_ratio, seed)
    for split, items in splits.items():
        count = 0
        for img, xml in items:
            lines = voc_to_yolo(xml, img, class_to_id)
            if not lines:
                continue
            img_dir = out / "images" / split
            lbl_dir = out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_dir / img.name)
            (lbl_dir / f"{img.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1
        print(f"[{name}] {split}: {count}")

    names = [None] * len(class_to_id)
    for cls_name, idx in class_to_id.items():
        names[idx] = cls_name
    write_yaml(out, name, names or ["ship"])
    return True


def prepare_one(dataset_root: Path, det_root: Path, name: str, force: bool, train_ratio: float, val_ratio: float, seed: int):
    src = collapse_single_child(maybe_unzip(dataset_root, name))
    out = det_root / name
    safe_reset_dir(out, force)

    fallback_names = ["ship"]
    if name == "SAR_AIRcraft":
        fallback_names = ["A220", "A320/321", "A330", "ARJ21", "Boeing737", "Boeing787", "other"]

    handlers = (
        lambda: prepare_yolo_tree(src, out, name, fallback_names, train_ratio, val_ratio, seed),
        lambda: prepare_flat_yolo(src, out, name, fallback_names, train_ratio, val_ratio, seed),
        lambda: prepare_coco(src, out, name, fallback_names),
        lambda: prepare_voc(src, out, name, train_ratio, val_ratio, seed),
    )
    for handler in handlers:
        if handler():
            print(f"[done] {name}: {out}")
            return
    shutil.rmtree(out, ignore_errors=True)
    raise RuntimeError(
        f"Could not recognize {name} under {src}. Expected YOLO images/labels, flat jpg/txt, COCO json, or VOC xml."
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare Hugging Face shimiandeshu/sar-m ZIPs into dataset/det YOLO detection folders.")
    parser.add_argument("--dataset-root", default="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset")
    parser.add_argument("--out", default=None, help="Default: <dataset-root>/det")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset/det outputs")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    det_root = Path(args.out) if args.out else dataset_root / "det"
    det_root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        prepare_one(dataset_root, det_root, name, args.force, args.train_ratio, args.val_ratio, args.seed)

    print(f"All requested datasets are prepared under: {det_root}")


if __name__ == "__main__":
    main()
