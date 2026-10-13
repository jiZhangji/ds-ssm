#!/usr/bin/env python3
"""Build a small, deterministic four-source SAR detection proxy dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
GLOBAL_NAMES = [
    "ship",
    "bridge",
    "harbor",
    "tank",
    "Boeing737",
    "Boeing787",
    "A330",
    "A220",
    "A320/321",
    "ARJ21",
    "other",
]
DEFAULT_COUNTS = {
    "SSDD": (200, 80),
    "OGSOD": (250, 100),
    "SAR-Ship": (200, 80),
    "SAR-Aircraft": (350, 140),
}
ALIASES = {
    "ship": "ship",
    "bridge": "bridge",
    "harbor": "harbor",
    "harbour": "harbor",
    "tank": "tank",
    "boeing737": "Boeing737",
    "b737": "Boeing737",
    "boeing787": "Boeing787",
    "b787": "Boeing787",
    "a330": "A330",
    "a220": "A220",
    "a320321": "A320/321",
    "a320": "A320/321",
    "a321": "A320/321",
    "arj21": "ARJ21",
    "other": "other",
    "others": "other",
}


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path
    source_split: str
    classes: frozenset[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssdd", required=True, help="SSDD dataset YAML")
    parser.add_argument("--ogsod", required=True, help="OGSOD dataset YAML")
    parser.add_argument("--sar-ship", required=True, help="SAR-Ship dataset YAML")
    parser.add_argument("--sar-aircraft", required=True, help="SAR-Aircraft dataset YAML")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    return parser.parse_args()


def normalized_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def names_from_yaml(data: dict) -> dict[int, str]:
    names = data["names"]
    if isinstance(names, list):
        return dict(enumerate(str(item) for item in names))
    return {int(key): str(value) for key, value in names.items()}


def resolve_entry(root: Path, entry) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for value in entries:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                item = Path(line.strip())
                if not item.is_absolute():
                    item = path.parent / item
                if item.suffix.lower() in IMAGE_SUFFIXES and item.exists():
                    images.append(item.resolve())
        elif path.is_dir():
            images.extend(
                sorted(item.resolve() for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES and path.exists():
            images.append(path)
    return images


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indices = [index for index, value in enumerate(parts) if value == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_classes(label: Path, remap: dict[int, int]) -> frozenset[int]:
    classes = set()
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            source_id = int(float(fields[0]))
            if source_id in remap:
                classes.add(remap[source_id])
    return frozenset(classes)


def load_source(yaml_path: Path, source: str) -> tuple[list[Pair], list[Pair], dict[int, int]]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data.get("path") or yaml_path.parent)
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    source_names = names_from_yaml(data)
    remap = {}
    for class_id, class_name in source_names.items():
        canonical = ALIASES.get(normalized_name(class_name))
        if canonical is None:
            raise ValueError(f"Unsupported class '{class_name}' in {yaml_path}")
        remap[class_id] = GLOBAL_NAMES.index(canonical)

    def pairs_for(split: str) -> list[Pair]:
        if not data.get(split):
            return []
        pairs = []
        for image in resolve_entry(root, data[split]):
            label = label_for_image(image)
            if not label.exists():
                continue
            classes = read_classes(label, remap)
            if classes:
                pairs.append(Pair(image, label, split, classes))
        return pairs

    train = pairs_for("train")
    val = pairs_for("val")
    if not train:
        raise RuntimeError(f"No labeled training images found through {yaml_path}")

    # Some historical SSDD YAMLs called the official test set `val`. Keep the
    # official test untouched and derive both proxy splits from official train.
    if source == "SSDD":
        val = []
    return train, val, remap


def choose(items: list[Pair], count: int, seed: int) -> list[Pair]:
    if len(items) < count:
        raise RuntimeError(f"Requested {count} samples from a pool of {len(items)}")
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    selected: list[Pair] = []
    remaining = set(range(len(order)))
    all_classes = sorted(set().union(*(item.classes for item in order)))
    for class_id in all_classes:
        match = next((index for index in sorted(remaining) if class_id in order[index].classes), None)
        if match is not None and len(selected) < count:
            selected.append(order[match])
            remaining.remove(match)
    selected.extend(order[index] for index in sorted(remaining)[: count - len(selected)])
    rng.shuffle(selected)
    return selected


def split_source(train: list[Pair], val: list[Pair], n_train: int, n_val: int, seed: int):
    if val:
        return choose(train, n_train, seed), choose(val, n_val, seed + 1)
    combined = choose(train, n_train + n_val, seed)
    return combined[:n_train], combined[n_train:]


def transfer(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        target.hardlink_to(source)
    else:
        shutil.copy2(source, target)


def remap_label(source: Path, target: Path, remap: dict[int, int]) -> tuple[int, Counter]:
    lines = []
    counts = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        source_id = int(float(fields[0]))
        if source_id not in remap:
            continue
        target_id = remap[source_id]
        fields[0] = str(target_id)
        lines.append(" ".join(fields))
        counts[target_id] += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines), counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_yamls(out: Path, sources: list[str]) -> None:
    names = {index: name for index, name in enumerate(GLOBAL_NAMES)}
    common = {"train": "images/train", "val": "images/val", "names": names}
    (out / "SAR-Mix.yaml").write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    subset_dir = out / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        payload = {"train": "../images/train", "val": f"../images/val/{source}", "names": names}
        (subset_dir / f"{source}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {out}; add --force to rebuild")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    inputs = {
        "SSDD": Path(args.ssdd).expanduser().resolve(),
        "OGSOD": Path(args.ogsod).expanduser().resolve(),
        "SAR-Ship": Path(args.sar_ship).expanduser().resolve(),
        "SAR-Aircraft": Path(args.sar_aircraft).expanduser().resolve(),
    }
    rows = []
    source_summary = {}
    global_counts = Counter()
    for source_index, (source, yaml_path) in enumerate(inputs.items()):
        train_pool, val_pool, remap = load_source(yaml_path, source)
        n_train, n_val = DEFAULT_COUNTS[source]
        train, val = split_source(train_pool, val_pool, n_train, n_val, args.seed + 1000 * source_index)
        counts = Counter()
        for split, items in (("train", train), ("val", val)):
            for index, item in enumerate(items):
                name = f"{source.lower().replace('-', '_')}_{index:05d}{item.image.suffix.lower()}"
                image_out = out / "images" / split / source / name
                label_out = out / "labels" / split / source / f"{Path(name).stem}.txt"
                transfer(item.image, image_out, args.copy_mode)
                objects, object_counts = remap_label(item.label, label_out, remap)
                counts.update(object_counts)
                global_counts.update(object_counts)
                rows.append(
                    {
                        "source": source,
                        "split": split,
                        "source_split": item.source_split,
                        "source_image": f"{source}/{item.source_split}/{item.image.name}",
                        "output_image": str(image_out.relative_to(out)),
                        "objects": objects,
                        "image_sha256": sha256(image_out),
                        "label_sha256": sha256(label_out),
                    }
                )
        source_summary[source] = {
            "train_images": len(train),
            "val_images": len(val),
            "objects_by_class": {GLOBAL_NAMES[key]: value for key, value in sorted(counts.items())},
        }

    write_yamls(out, list(inputs))
    manifest_csv = out / "sample_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest_sha = sha256(manifest_csv)
    summary = {
        "seed": args.seed,
        "classes": GLOBAL_NAMES,
        "sources": source_summary,
        "total_train_images": sum(item[0] for item in DEFAULT_COUNTS.values()),
        "total_val_images": sum(item[1] for item in DEFAULT_COUNTS.values()),
        "objects_by_class": {GLOBAL_NAMES[key]: value for key, value in sorted(global_counts.items())},
        "manifest_sha256": manifest_sha,
    }
    (out / "dataset_snapshot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {out / 'SAR-Mix.yaml'}")


if __name__ == "__main__":
    main()
