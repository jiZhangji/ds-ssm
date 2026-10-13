#!/usr/bin/env python3
"""Prepare corrected cross-task datasets without overwriting the v1 artifacts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FUSAR_PALETTE = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def extract_once(archive: Path, destination: Path) -> Path:
    if not destination.exists() or not any(destination.iterdir()):
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    return destination


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_detection_yaml(root: Path, names: list[str]) -> None:
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    (root / f"{root.name}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def split_ids(items: list, seed: int, fractions=(0.8, 0.1, 0.1)) -> dict[str, list]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    first = round(len(shuffled) * fractions[0])
    second = first + round(len(shuffled) * fractions[1])
    return {"train": shuffled[:first], "val": shuffled[first:second], "test": shuffled[second:]}


def copy_pair(image: Path, label: str, root: Path, split: str, name: str | None = None) -> None:
    name = name or image.name
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_dir / name)
    (label_dir / f"{Path(name).stem}.txt").write_text(label, encoding="utf-8")


def prepare_sadd(raw_root: Path, output: Path, seed: int) -> dict:
    source = extract_once(raw_root / "SAR-aircraft.zip", output.parent / "_raw_sadd")
    label_by_stem = {path.stem: path for path in source.rglob("*.txt")}
    positive = []
    for image in image_files(source):
        label = label_by_stem.get(image.stem)
        if label and label.read_text(encoding="utf-8", errors="ignore").strip():
            positive.append((image, label))
    if len(positive) != 883:
        raise RuntimeError(f"Expected 883 positive SADD images, found {len(positive)}")
    random.Random(seed).shuffle(positive)
    partitions = {"train": positive[:795], "val": positive[795:839], "test": positive[839:883]}
    for split, pairs in partitions.items():
        for image, label in pairs:
            copy_pair(image, label.read_text(encoding="utf-8"), output, split)
    write_detection_yaml(output, ["aircraft"])
    return {"protocol": "official-positive-883", "split": {key: len(value) for key, value in partitions.items()}}


def airship_annotations(xml_root: Path, image_root: Path) -> list[dict]:
    records = []
    for xml in sorted(xml_root.glob("*.xml")):
        image = image_root / f"{xml.stem}.tiff"
        if not image.exists():
            continue
        with Image.open(image) as handle:
            width, height = handle.size
        boxes = []
        for obj in ET.parse(xml).getroot().findall(".//object"):
            points = []
            for point in obj.findall(".//point"):
                try:
                    x, y = (float(value.strip()) for value in point.text.split(","))
                    points.append((x, y))
                except (AttributeError, TypeError, ValueError):
                    continue
            if len(points) >= 4:
                xs, ys = zip(*points)
                boxes.append((max(0.0, min(xs)), max(0.0, min(ys)), min(width, max(xs)), min(height, max(ys))))
        records.append({"image": image, "boxes": boxes, "size": (width, height)})
    return records


def tile_detection_record(record: dict, output: Path, split: str, tile=640, stride=512, keep_empty=False) -> tuple[int, int]:
    with Image.open(record["image"]) as image:
        image = image.convert("RGB")
        width, height = image.size
        xs = sorted(set(list(range(0, max(width - tile + 1, 1), stride)) + [max(width - tile, 0)]))
        ys = sorted(set(list(range(0, max(height - tile + 1, 1), stride)) + [max(height - tile, 0)]))
        tiles = objects = 0
        for top in ys:
            for left in xs:
                clipped = []
                for x1, y1, x2, y2 in record["boxes"]:
                    ix1, iy1 = max(x1, left), max(y1, top)
                    ix2, iy2 = min(x2, left + tile), min(y2, top + tile)
                    intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
                    original = max(x2 - x1, 0) * max(y2 - y1, 0)
                    if intersection <= 0 or intersection / max(original, 1e-9) < 0.5:
                        continue
                    cx, cy = ((ix1 + ix2) / 2 - left) / tile, ((iy1 + iy2) / 2 - top) / tile
                    bw, bh = (ix2 - ix1) / tile, (iy2 - iy1) / tile
                    clipped.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if not clipped and not keep_empty:
                    continue
                name = f"{record['image'].stem}_x{left:04d}_y{top:04d}.png"
                image_dir, label_dir = output / "images" / split, output / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                image.crop((left, top, min(left + tile, width), min(top + tile, height))).save(image_dir / name)
                (label_dir / f"{Path(name).stem}.txt").write_text("\n".join(clipped) + ("\n" if clipped else ""), encoding="utf-8")
                tiles += 1
                objects += len(clipped)
    return tiles, objects


def prepare_airship(raw_root: Path, output: Path, seed: int) -> dict:
    data = extract_once(raw_root / "AIR-SARShip-2.0-data.zip", output.parent / "_raw_airship_data")
    xml = extract_once(raw_root / "AIR-SARShip-2.0-xml.zip", output.parent / "_raw_airship_xml")
    image_root = next(path for path in data.rglob("AIR-SARShip-2.0-data") if path.is_dir())
    xml_root = next(path for path in xml.rglob("AIR-SARShip-2.0-xml") if path.is_dir())
    records = airship_annotations(xml_root, image_root)
    if len(records) != 300 or sum(len(item["boxes"]) for item in records) != 2040:
        raise RuntimeError("AIR-SARShip pairing or annotation count is incorrect")
    parts = split_ids(records, seed, fractions=(0.7, 0.15, 0.15))
    report = {"source_images": {}, "tiles": {}, "objects": {}}
    for split, items in parts.items():
        report["source_images"][split] = len(items)
        report["tiles"][split] = report["objects"][split] = 0
        for item in items:
            # Keep the complete tiled image extent in every split. Dropping empty
            # validation/test tiles would hide false positives and inflate AP.
            tiles, objects = tile_detection_record(item, output, split, keep_empty=True)
            report["tiles"][split] += tiles
            report["objects"][split] += objects
    write_detection_yaml(output, ["ship"])
    return report


def parse_srsdd_label(path: Path, width: int, height: int, class_to_id: dict[str, int]) -> tuple[str, Counter]:
    lines, stats = [], Counter()
    for row in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = row.split()
        if len(tokens) < 10:
            continue
        try:
            coordinates = [float(value) for value in tokens[:8]]
        except ValueError:
            continue
        xs = [min(max(value, 0.0), float(width)) for value in coordinates[0::2]]
        ys = [min(max(value, 0.0), float(height)) for value in coordinates[1::2]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        if x2 <= x1 or y2 <= y1:
            stats["degenerate"] += 1
            continue
        class_name = tokens[8]
        lines.append(
            f"{class_to_id[class_name]} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} "
            f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
        )
        stats[class_name] += 1
        if any(value < 0 for value in coordinates) or max(coordinates[0::2]) > width or max(coordinates[1::2]) > height:
            stats["clipped"] += 1
    return "\n".join(lines) + ("\n" if lines else ""), stats


def prepare_srsdd(raw_root: Path, output: Path, seed: int) -> dict:
    root = raw_root / "SRSDD-V1.0"
    train_candidates = [path for path in root.rglob("train") if (path / "images").is_dir() and (path / "labels").is_dir()]
    test_candidates = [path for path in root.rglob("test") if (path / "images").is_dir() and (path / "labels").is_dir()]
    if not train_candidates:
        extracted = extract_once(root / "train.zip", root / "_train_extracted")
        train_candidates = [path for path in extracted.rglob("train") if (path / "images").is_dir() and (path / "labels").is_dir()]
    if not test_candidates:
        extracted = extract_once(root / "test.zip", root / "_test_extracted")
        test_candidates = [path for path in extracted.rglob("test") if (path / "images").is_dir() and (path / "labels").is_dir()]
    if not train_candidates or not test_candidates:
        raise RuntimeError(f"Invalid SRSDD archive layout under {root}")
    train_root, test_root = train_candidates[0], test_candidates[0]
    class_names = sorted({row.split()[8] for label in root.rglob("labels/*.txt") for row in label.read_text(encoding="utf-8", errors="ignore").splitlines() if len(row.split()) >= 10})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    train_images = image_files(train_root / "images")
    train_parts = split_ids(train_images, seed, fractions=(0.8, 0.2, 0.0))
    partitions = {"train": train_parts["train"], "val": train_parts["val"], "test": image_files(test_root / "images")}
    report = {"classes": class_names, "split": {}, "stats": Counter()}
    for split, images in partitions.items():
        report["split"][split] = len(images)
        for image in images:
            label_root = image.parent.parent / "labels"
            with Image.open(image) as handle:
                width, height = handle.size
            text, stats = parse_srsdd_label(label_root / f"{image.stem}.txt", width, height, class_to_id)
            report["stats"].update(stats)
            copy_pair(image, text, output, split)
    report["stats"] = dict(report["stats"])
    write_detection_yaml(output, class_names)
    return report


def prepare_fusar(raw_root: Path, output: Path, seed: int) -> dict:
    root = extract_once(raw_root / "FUSAR-Map1.0.zip", output / "_raw")
    image_root = next(path for path in root.rglob("SAR_1024") if path.is_dir())
    label_root = next(path for path in root.rglob("LAB_1024") if path.is_dir())
    by_city: dict[str, list[dict]] = {}
    for mask in image_files(label_root):
        candidates = list(image_root.glob(mask.stem.replace("_Label", "_SAR") + ".*"))
        if not candidates:
            continue
        city = mask.name.split("_")[0]
        by_city.setdefault(city, []).append({"image": str(candidates[0].resolve()), "mask": str(mask.resolve()), "group": city})
    cities = sorted(by_city)
    tie_order = list(cities)
    random.Random(seed).shuffle(tie_order)
    tie_rank = {city: index for index, city in enumerate(tie_order)}
    target = sum(map(len, by_city.values())) * 0.15
    # Pick representative-sized held-out cities. A purely shuffled choice can
    # produce a 12-image test set, which is too unstable for model selection.
    test_city = min(cities, key=lambda city: (abs(len(by_city[city]) - target), tie_rank[city]))
    val_city = min(
        (city for city in cities if city != test_city),
        key=lambda city: (abs(len(by_city[city]) - target), tie_rank[city]),
    )
    records = []
    for city, items in by_city.items():
        split = "test" if city == test_city else "val" if city == val_city else "train"
        records.extend({**item, "split": split} for item in items)
    payload = {
        "task": "segmentation",
        "classes": 5,
        "class_names": ["background", "building", "vegetation", "water", "road"],
        "palette": FUSAR_PALETTE,
        "records": records,
        "protocol": "city-disjoint",
    }
    write_json(output / "manifest.json", payload)
    return {"split": dict(Counter(row["split"] for row in records)), "test_city": test_city, "val_city": val_city}


def prepare_classification(archive: Path, output: Path, seed: int, sample_real=False) -> dict:
    root = extract_once(archive, output / "_raw")
    if sample_real:
        root = next(path for path in root.rglob("real") if path.parent.name == "decibel")
    else:
        image_roots = [path for path in root.rglob("images") if path.is_dir()]
        if image_roots:
            root = image_roots[0]
    class_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and any(child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir())
    )
    classes = [path.name for path in class_dirs]
    if not classes or len(classes) != len(set(classes)):
        raise RuntimeError(f"Invalid classification class layout in {archive}: {classes}")
    records = []
    for class_id, directory in enumerate(class_dirs):
        for split, images in split_ids(image_files(directory), seed).items():
            records.extend({"path": str(image.resolve()), "label": class_id, "class_name": directory.name, "split": split} for image in images)
    write_json(output / "manifest.json", {"task": "classification", "classes": classes, "records": records})
    return {
        "classes": classes,
        "split": dict(Counter(row["split"] for row in records)),
        "class_count": dict(Counter(row["class_name"] for row in records)),
        "protocol": "stratified-random-80-10-10-real-only" if sample_real else "stratified-random-80-10-10",
    }


def prepare_change(raw_root: Path, output: Path, archive_name: str, seed: int) -> dict:
    root = extract_once(raw_root / archive_name, output / "_raw")
    images = image_files(root)
    mask = next(path for path in images if "gt" in path.stem.lower())
    pair = sorted(path for path in images if path != mask)[:2]
    with Image.open(mask) as handle:
        width, height = handle.size
        change = np.asarray(handle.convert("L")) > 0
    patch, stride = 48, 8
    records = []
    # Split by non-overlapping vertical bands, then sample densely inside each band.
    bounds = {"train": (0, round(width * 0.6)), "val": (round(width * 0.6), round(width * 0.8)), "test": (round(width * 0.8), width)}
    for split, (start, end) in bounds.items():
        x_positions = list(range(start, max(end - patch + 1, start + 1), stride))
        if not x_positions:
            x_positions = [start]
        y_positions = list(range(0, max(height - patch + 1, 1), stride)) + [max(height - patch, 0)]
        for y in sorted(set(y_positions)):
            for x in sorted(set(x_positions)):
                crop = [x, y, min(x + patch, end), min(y + patch, height)]
                ratio = float(change[crop[1]:crop[3], crop[0]:crop[2]].mean())
                records.append({"image1": str(pair[0].resolve()), "image2": str(pair[1].resolve()), "mask": str(mask.resolve()), "crop": crop, "split": split, "change_ratio": ratio})
    write_json(output / "manifest.json", {"task": "change_detection", "records": records, "patch_size": patch, "stride": stride, "protocol": "dense-spatial-bands"})
    return {"split": dict(Counter(row["split"] for row in records)), "positive_ratio": {split: float(np.mean([row["change_ratio"] for row in records if row["split"] == split])) for split in bounds}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=["SADD", "AIR-SARShip-2.0", "SRSDD-v1.0", "SAR-ACD", "SAMPLE", "FUSAR-Map", "Ottawa", "Yellow River", "Farmland", "Sulzberger"])
    args = parser.parse_args()
    raw, output = Path(args.raw_root).resolve(), Path(args.out).resolve()
    search_roots = [raw, *(Path(path).resolve() for path in args.extra_root)]

    def find_archive(name: str) -> Path:
        return next(path for root in search_roots for path in root.rglob(name))

    functions = {
        "SADD": lambda: prepare_sadd(raw, output / "detection" / "SADD", args.seed),
        "AIR-SARShip-2.0": lambda: prepare_airship(raw, output / "detection" / "AIR_SARShip_2", args.seed),
        "SRSDD-v1.0": lambda: prepare_srsdd(raw, output / "detection" / "SRSDD_v1", args.seed),
        "SAR-ACD": lambda: prepare_classification(find_archive("SAR_ACD.zip"), output / "classification" / "SAR_ACD", args.seed),
        "SAMPLE": lambda: prepare_classification(find_archive("SAMPLE_dataset_public.zip"), output / "classification" / "SAMPLE", args.seed, True),
        "FUSAR-Map": lambda: prepare_fusar(raw, output / "segmentation" / "FUSAR_Map", args.seed),
        "Ottawa": lambda: prepare_change(raw, output / "change_detection" / "Ottawa", "Ottawa_data.zip", args.seed),
        "Yellow River": lambda: prepare_change(raw, output / "change_detection" / "Yellow_River", "Yellowriver_data.zip", args.seed),
        "Farmland": lambda: prepare_change(raw, output / "change_detection" / "Farmland", "Farmland_data.zip", args.seed),
        "Sulzberger": lambda: prepare_change(raw, output / "change_detection" / "Sulzberger", "Sulzberger_data.zip", args.seed),
    }
    report_path = output / "preparation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for dataset in args.datasets:
        report[dataset] = functions[dataset]()
        print(dataset, json.dumps(report[dataset], ensure_ascii=False), flush=True)
    write_json(report_path, report)


if __name__ == "__main__":
    main()
