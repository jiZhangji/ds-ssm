#!/usr/bin/env python3
"""Prepare the authoritative cross-task SAR datasets into reproducible manifests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
import yaml

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DETECTION = ("SSDD", "SAR-AIRcraft", "AIR-SARShip-2.0", "SRSDD-v1.0", "SADD")
CLASSIFICATION = ("SAR-ACD", "SAMPLE")
SEGMENTATION = ("FUSAR-Map",)
CHANGE = ("Ottawa", "Yellow River", "Farmland", "Sulzberger")
ALL = DETECTION + CLASSIFICATION + SEGMENTATION + CHANGE

DATASET_DIRS = {
    "SSDD": ("detection", "SSDD"),
    "SAR-AIRcraft": ("detection", "SAR_AIRcraft"),
    "AIR-SARShip-2.0": ("detection", "AIR_SARShip_2"),
    "SRSDD-v1.0": ("detection", "SRSDD_v1"),
    "SADD": ("detection", "SADD"),
    "SAR-ACD": ("classification", "SAR_ACD"),
    "SAMPLE": ("classification", "SAMPLE"),
    "FUSAR-Map": ("segmentation", "FUSAR_Map"),
    "Ottawa": ("change_detection", "Ottawa"),
    "Yellow River": ("change_detection", "Yellow_River"),
    "Farmland": ("change_detection", "Farmland"),
    "Sulzberger": ("change_detection", "Sulzberger"),
}


def files(root: Path, suffixes=IMG_SUFFIXES):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def unzip_once(path: Path, out: Path) -> Path:
    if out.exists() and any(out.iterdir()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)
    return out


def split(items, seed=0, train=0.8, val=0.1):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    a, b = int(n * train), int(n * (train + val))
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_det_yaml(out: Path, names):
    lines = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "", "names:"]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(names))
    (out / f"{out.name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_yaml_root(source_yaml: Path, payload: dict) -> Path:
    root = Path(payload.get("path", source_yaml.parent))
    if not root.is_absolute():
        root = source_yaml.parent / root
    return root.resolve()


def link_existing_detection(source_yaml: Path, out: Path) -> dict:
    """Create a stable YAML view without copying a large existing dataset."""
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    root = resolve_yaml_root(source_yaml, payload)
    names = payload.get("names", {0: "target"})
    rendered = {
        "path": str(root),
        "train": payload.get("train", "images/train"),
        "val": payload.get("val", "images/val"),
        "test": payload.get("test", "images/test"),
        "names": names,
    }
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / f"{out.name}.yaml"
    yaml_path.write_text(yaml.safe_dump(rendered, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"format": "existing YOLO dataset (linked YAML)", "yaml": str(yaml_path), "source": str(source_yaml.resolve())}


def find_file(roots: list[Path], names: tuple[str, ...]) -> Path | None:
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
        for name in names:
            matches = sorted(root.rglob(name)) if root.exists() else []
            if matches:
                return matches[0]
    return None


def prepare_existing_detection(name: str, archive: Path, target: Path, seed: int) -> dict:
    """Reuse the maintained COCO/YOLO/VOC converter for existing archives."""
    existing_yaml = target / f"{target.name}.yaml"
    if existing_yaml.exists():
        return {"format": "existing converted dataset (reused)", "yaml": str(existing_yaml)}
    converter_path = Path(__file__).resolve().with_name("prepare_hf_sar_det_datasets.py")
    spec = importlib.util.spec_from_file_location("prepare_hf_sar_det_datasets", converter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load detection converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_name = "SAR_AIRcraft" if name == "SAR-AIRcraft" else "SSDD"
    staging = target.parent / "_archives"
    staging.mkdir(parents=True, exist_ok=True)
    staged_archive = staging / f"{source_name}.zip"
    if not staged_archive.exists():
        shutil.copy2(archive, staged_archive)
    module.prepare_one(staging, target.parent, source_name, force=False, train_ratio=0.8, val_ratio=0.1, seed=seed)
    generated = target.parent / source_name
    if generated != target:
        if target.exists():
            shutil.rmtree(target)
        generated.rename(target)
    yaml_path = target / f"{source_name}.yaml"
    renamed_yaml = target / f"{target.name}.yaml"
    if yaml_path.exists() and yaml_path != renamed_yaml:
        renamed_yaml.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"format": "existing archive converted by prepare_hf_sar_det_datasets.py", "yaml": str(renamed_yaml)}


def copy_det_pair(img: Path, label_text: str, out: Path, split_name: str):
    (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, out / "images" / split_name / img.name)
    (out / "labels" / split_name / f"{img.stem}.txt").write_text(label_text, encoding="utf-8")


def prepare_flat_detection(src: Path, out: Path, names=("target",), seed=0, include_empty=False):
    if (out / f"{out.name}.yaml").exists():
        return {"format": "existing converted dataset (reused)", "yaml": str(out / f"{out.name}.yaml")}
    label_index = {label.stem: label for label in src.rglob("*.txt")}
    pairs = []
    for img in files(src):
        label = img.with_suffix(".txt")
        if not label.exists():
            label = label_index.get(img.stem)
        if (label is not None and label.exists()) or include_empty:
            pairs.append((img, label.read_text(encoding="utf-8", errors="ignore") if label is not None and label.exists() else ""))
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {src}")
    for split_name, items in split(pairs, seed).items():
        for img, text in items:
            copy_det_pair(img, text, out, split_name)
    write_det_yaml(out, names)
    return {"format": "YOLO", "images": len(pairs), "classes": list(names), "yaml": str(out / f"{out.name}.yaml")}


def prepare_airship(raw: Path, out: Path, seed=0):
    if (out / f"{out.name}.yaml").exists():
        return {"format": "existing converted dataset (reused)", "yaml": str(out / f"{out.name}.yaml")}
    data_zip = raw / "AIR-SARShip-2.0-data.zip"
    xml_zip = raw / "AIR-SARShip-2.0-xml.zip"
    data_root = next(p for p in unzip_once(data_zip, raw / "_AIR_data").rglob("*") if p.is_dir() and any(p.glob("*.tiff")))
    xml_root = next(p for p in unzip_once(xml_zip, raw / "_AIR_xml").rglob("*") if p.is_dir() and any(p.glob("*.xml")))
    pairs = []
    for xml in sorted(xml_root.glob("*.xml")):
        img = data_root / f"{xml.stem}.tiff"
        if not img.exists():
            continue
        with Image.open(img) as im:
            width, height = im.size
        root = ET.parse(xml).getroot()
        labels = []
        for point_group in root.findall(".//object"):
            points = []
            for point in point_group.findall(".//point"):
                try:
                    x, y = (float(v.strip()) for v in point.text.split(","))
                    points.append((x, y))
                except Exception:
                    pass
            if len(points) >= 4:
                xs, ys = zip(*points)
                x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
                y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
                if x2 > x1 and y2 > y1:
                    labels.append(f"0 {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        if labels:
            pairs.append((img, "\n".join(labels) + "\n"))
    for split_name, items in split(pairs, seed).items():
        for img, text in items:
            copy_det_pair(img, text, out, split_name)
    write_det_yaml(out, ["ship"])
    return {"format": "VOC/XML to HBB YOLO", "images": len(pairs), "classes": ["ship"], "yaml": str(out / f"{out.name}.yaml")}


def prepare_srsdd(raw: Path, out: Path, seed=0):
    if (out / f"{out.name}.yaml").exists():
        return {"format": "existing converted dataset (reused)", "yaml": str(out / f"{out.name}.yaml")}
    train_root = next(p for p in unzip_once(raw / "train.zip", raw / "_SRSDD_train").rglob("labels") if p.is_dir())
    test_root = next(p for p in unzip_once(raw / "test.zip", raw / "_SRSDD_test").rglob("labels") if p.is_dir())

    class_names = set()
    for label_root in (train_root, test_root):
        for label in label_root.glob("*.txt"):
            for row in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                tok = row.split()
                if len(tok) >= 10 and tok[0].replace(".", "", 1).isdigit():
                    class_names.add(tok[8])
    classes = sorted(class_names)
    class_to_id = {name: index for index, name in enumerate(classes)}
    if not classes:
        raise RuntimeError(f"No SRSDD class labels found under {raw}")

    def read_split(label_root: Path):
        result = []
        image_root = label_root.parent / "images"
        for img in sorted(image_root.glob("*")):
            if not img.is_file() or img.suffix.lower() not in IMG_SUFFIXES:
                continue
            label = label_root / f"{img.stem}.txt"
            with Image.open(img) as im:
                width, height = im.size
            lines = []
            label_text = label.read_text(encoding="utf-8", errors="ignore") if label.exists() else ""
            for row in label_text.splitlines():
                tok = row.split()
                if len(tok) < 8 or not all(v.replace(".", "", 1).isdigit() for v in tok[:8]):
                    continue
                coords = [float(v) for v in tok[:8]]
                xs, ys = coords[0::2], coords[1::2]
                x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
                y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
                class_name = tok[8] if len(tok) >= 10 else classes[0]
                cls = class_to_id[class_name]
                if x2 > x1 and y2 > y1:
                    lines.append(f"{cls} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
            # Keep images without valid objects as negative samples. Dropping them
            # changes the split distribution and inflates detector metrics.
            result.append((img, "\n".join(lines) + ("\n" if lines else "")))
        return result

    train_pairs, test_pairs = read_split(train_root), read_split(test_root)
    # SRSDD already provides an official train archive and test archive. Split
    # only the official train archive into train/val; never discard its tail.
    train_parts = split(train_pairs, seed, train=0.8, val=0.2)
    for name, items in (("train", train_parts["train"]), ("val", train_parts["val"]), ("test", test_pairs)):
        for img, text in items:
            copy_det_pair(img, text, out, name)
    write_det_yaml(out, classes)
    return {"format": "SRSDD rotated quadrilateral to HBB YOLO", "images": len(train_pairs) + len(test_pairs), "classes": classes, "yaml": str(out / f"{out.name}.yaml")}


def prepare_classification(src: Path, out: Path, seed=0):
    if (out / "manifest.json").exists():
        return {"format": "existing prepared manifest (reused)", "manifest": str(out / "manifest.json")}
    class_dirs = sorted(p for p in src.rglob("*") if p.is_dir() and any(x.suffix.lower() in IMG_SUFFIXES for x in p.iterdir()))
    samples = [(img, directory.name) for directory in class_dirs for img in files(directory)]
    if not samples:
        raise RuntimeError(f"No class-directory images found in {src}")
    classes = sorted({label for _, label in samples})
    records = []
    by_class = {label: [] for label in classes}
    for img, label in samples:
        by_class[label].append(img)
    for label, imgs in by_class.items():
        for split_name, items in split(imgs, seed).items():
            records.extend({"path": str(img.resolve()), "label": classes.index(label), "class_name": label, "split": split_name} for img in items)
    payload = {"task": "classification", "classes": classes, "records": records}
    write_json(out / "manifest.json", payload)
    return {"format": "class-directory", "images": len(records), "classes": classes, "manifest": str(out / "manifest.json")}


def find_sample_real(root: Path) -> Path:
    direct = root / "decibel" / "real"
    if direct.is_dir():
        return direct
    matches = sorted(root.rglob("decibel/real"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"SAMPLE decibel/real directory not found under {root}")


def prepare_fusar(raw_zip: Path, out: Path):
    if (out / "manifest.json").exists():
        return {"format": "existing prepared manifest (reused)", "manifest": str(out / "manifest.json")}
    root = unzip_once(raw_zip, out / "_raw")
    sar_root = next(p for p in root.rglob("SAR_1024") if p.is_dir())
    lab_root = next(p for p in root.rglob("LAB_1024") if p.is_dir())
    pairs = []
    for label in files(lab_root):
        stem = label.stem.replace("_Label", "_SAR")
        candidates = list(sar_root.glob(f"{stem}.*"))
        if candidates:
            pairs.append({"image": str(candidates[0].resolve()), "mask": str(label.resolve())})
    payload = {"task": "segmentation", "classes": 5, "records": [{**item, "split": s} for s, items in split(pairs, 0).items() for item in items]}
    write_json(out / "manifest.json", payload)
    return {"format": "SAR_1024/LAB_1024 pairs", "images": len(pairs), "manifest": str(out / "manifest.json")}


def prepare_change(raw_zip: Path, out: Path, stem: str):
    if (out / "manifest.json").exists():
        return {"format": "existing prepared manifest (reused)", "manifest": str(out / "manifest.json")}
    root = unzip_once(raw_zip, out / "_raw")
    imgs = files(root, {".bmp"})
    gt = next((p for p in imgs if "gt" in p.stem.lower()), None)
    pair = [p for p in imgs if p != gt]
    if len(pair) < 2 or gt is None:
        raise RuntimeError(f"Cannot identify two images and GT in {root}")
    with Image.open(pair[0]) as im:
        width, height = im.size
    patch_size, stride = min(128, width, height), min(128, width, height)
    patches = []
    for y in range(0, max(height - patch_size + 1, 1), stride):
        for x in range(0, max(width - patch_size + 1, 1), stride):
            patches.append({"image1": str(pair[0].resolve()), "image2": str(pair[1].resolve()), "mask": str(gt.resolve()), "crop": [x, y, min(x + patch_size, width), min(y + patch_size, height)]})
    # Keep spatial regions disjoint. Random overlapping patches would leak
    # neighboring pixels from the same image pair across train/val/test.
    records = []
    total = len(patches)
    for index, item in enumerate(patches):
        if total >= 3:
            # Reserve at least one spatially distinct patch for each split.
            split_name = "train" if index < total - 2 else "val" if index == total - 2 else "test"
        else:
            split_name = "train" if index == 0 else "val" if index == 1 else "test"
        records.append({**item, "split": split_name})
    payload = {"task": "change_detection", "records": records, "patch_size": patch_size}
    write_json(out / "manifest.json", payload)
    return {"format": "two-date BMP + binary GT", "images": 1, "manifest": str(out / "manifest.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", nargs="+", required=True, help="One or more roots containing the raw archives/datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", choices=ALL, default=list(ALL))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    raw_roots = [Path(item).expanduser().resolve() for item in args.raw_root]
    raw, out = raw_roots[0], Path(args.out).expanduser().resolve()
    results = {}
    for name in args.datasets:
        task_dir, dir_name = DATASET_DIRS[name]
        target = out / task_dir / dir_name
        try:
            if name == "AIR-SARShip-2.0":
                data_archive = find_file(raw_roots, ("AIR-SARShip-2.0-data.zip",))
                xml_archive = find_file(raw_roots, ("AIR-SARShip-2.0-xml.zip",))
                if data_archive is None or xml_archive is None:
                    raise FileNotFoundError("AIR-SARShip-2.0 data/xml archive not found")
                staging = raw / "_AIR_SARShip"
                if data_archive.parent != staging:
                    staging.mkdir(parents=True, exist_ok=True)
                    for archive in (data_archive, xml_archive):
                        target_archive = staging / archive.name
                        if not target_archive.exists():
                            shutil.copy2(archive, target_archive)
                result = prepare_airship(staging, target, args.seed)
            elif name == "SRSDD-v1.0":
                srsdd = find_file(raw_roots, ("SRSDD-V1.0",))
                if srsdd is None:
                    train_zip = find_file(raw_roots, ("train.zip",))
                    test_zip = find_file(raw_roots, ("test.zip",))
                    if train_zip is None or test_zip is None:
                        raise FileNotFoundError("SRSDD-V1.0 source not found")
                    srsdd = train_zip.parent
                result = prepare_srsdd(srsdd, target, args.seed)
            elif name == "SADD":
                sadd_root = find_file(raw_roots, ("SAR-aircraft.zip",))
                sadd_root = unzip_once(sadd_root, raw / "_SADD") if sadd_root else find_file(raw_roots, ("SAR-aircraft",))
                if sadd_root is None:
                    raise FileNotFoundError("SADD source not found (expected SAR-aircraft.zip or SAR-aircraft directory)")
                result = prepare_flat_detection(sadd_root, target, ("aircraft",), args.seed, include_empty=True)
            elif name == "SAR-ACD":
                src = find_file(raw_roots, ("SAR_ACD", "SAR_ACD.zip"))
                if src and src.suffix.lower() == ".zip":
                    src = unzip_once(src, raw / "_SAR_ACD")
                if src is None:
                    raise FileNotFoundError("SAR-ACD source not found")
                result = prepare_classification(src, target, args.seed)
            elif name == "SAMPLE":
                src = find_file(raw_roots, ("SAMPLE_dataset_public", "SAMPLE_dataset_public.zip"))
                if src and src.suffix.lower() == ".zip":
                    src = unzip_once(src, raw / "_SAMPLE")
                if src is None:
                    raise FileNotFoundError("SAMPLE source not found")
                result = prepare_classification(find_sample_real(src), target, args.seed)
            elif name == "FUSAR-Map":
                archive = find_file(raw_roots, ("FUSAR-Map1.0.zip",))
                if archive is None:
                    raise FileNotFoundError("FUSAR-Map1.0.zip not found")
                result = prepare_fusar(archive, target)
            elif name in CHANGE:
                archive = find_file(raw_roots, ({"Ottawa":"Ottawa_data.zip", "Yellow River":"Yellowriver_data.zip", "Farmland":"Farmland_data.zip", "Sulzberger":"Sulzberger_data.zip"}[name],))
                if archive is None:
                    raise FileNotFoundError(f"{name} archive not found")
                result = prepare_change(archive, target, name)
            else:
                yaml_names = ("SSDD.yaml", "SSDD_HF.yaml") if name == "SSDD" else ("SAR_AIRcraft.yaml", "SAR-AIRcraft.yaml", "SAR-AIRcraft1.0.yaml")
                source = find_file(raw_roots, yaml_names)
                if source is not None:
                    result = link_existing_detection(source, target)
                else:
                    archive_names = ("SSDD.zip",) if name == "SSDD" else ("SAR_AIRcraft.zip",)
                    archive = find_file(raw_roots, archive_names)
                    if archive is None:
                        raise FileNotFoundError(f"{name} YAML/archive not found")
                    result = prepare_existing_detection(name, archive, target, args.seed)
            results[name] = {"status": "ready", **result}
            print(f"[ready] {name}: {result}")
        except Exception as exc:
            results[name] = {"status": "error", "error": str(exc)}
            print(f"[error] {name}: {exc}")
    write_json(out / "cross_task_prepare_status.json", results)
    if any(item["status"] == "error" for item in results.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
