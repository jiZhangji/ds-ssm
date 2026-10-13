#!/usr/bin/env python3
"""Create a deterministic split and annotation fingerprint for a YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def identifier(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_entry(entry: str, root: Path, yaml_dir: Path) -> Path:
    path = Path(entry).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_candidate = (root / path).resolve()
    if root_candidate.exists():
        return root_candidate
    return (yaml_dir / path).resolve()


def collect_images(value, root: Path, yaml_dir: Path) -> list[Path]:
    entries = value if isinstance(value, list) else [value]
    images = []
    for entry in entries:
        source = resolve_entry(str(entry), root, yaml_dir)
        if source.is_dir():
            images.extend(
                path.resolve()
                for path in source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                images.append(resolve_entry(line, root, source.parent))
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source)
        else:
            raise FileNotFoundError(f"Cannot resolve dataset split entry: {entry}")
    return sorted(set(images), key=lambda path: str(path))


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if not image_indices:
        raise ValueError(f"Image path has no 'images' component: {image}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def fingerprint_split(images: list[Path], root: Path) -> dict:
    image_manifest = hashlib.sha256()
    image_contents = hashlib.sha256()
    label_contents = hashlib.sha256()
    image_bytes = 0
    label_bytes = 0
    missing_labels = []

    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"Missing image: {image}")
        image_size = image.stat().st_size
        image_bytes += image_size
        image_manifest.update(identifier(image, root).encode("utf-8"))
        image_manifest.update(b"\0")
        image_manifest.update(str(image_size).encode("ascii"))
        image_manifest.update(b"\n")

        image_contents.update(identifier(image, root).encode("utf-8"))
        image_contents.update(b"\0")
        with image.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                image_contents.update(chunk)
        image_contents.update(b"\n")

        label = label_path(image)
        label_id = identifier(label, root)
        label_contents.update(label_id.encode("utf-8"))
        label_contents.update(b"\0")
        if label.is_file():
            content = label.read_bytes()
            label_bytes += len(content)
            label_contents.update(content)
        else:
            missing_labels.append(label_id)
            label_contents.update(b"<MISSING>")
        label_contents.update(b"\n")

    return {
        "images": len(images),
        "image_bytes": image_bytes,
        "image_manifest_sha256": image_manifest.hexdigest(),
        "image_contents_sha256": image_contents.hexdigest(),
        "labels_present": len(images) - len(missing_labels),
        "label_bytes": label_bytes,
        "label_contents_sha256": label_contents.hexdigest(),
        "missing_labels": missing_labels,
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    yaml_dir = data_path.parent
    root_value = data.get("path", yaml_dir)
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (yaml_dir / root).resolve()

    payload = {
        "data_yaml": str(data_path),
        "data_yaml_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_root": str(root),
        "fingerprint_scope": "image relative paths, sizes, and bytes; label relative paths and contents",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        images = collect_images(data[split], root, yaml_dir)
        payload["splits"][split] = fingerprint_split(images, root)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
