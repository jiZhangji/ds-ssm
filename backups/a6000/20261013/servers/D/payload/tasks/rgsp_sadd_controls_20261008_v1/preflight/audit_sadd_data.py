#!/usr/bin/env python3
"""Audit the recovered SADD positive-only split without changing source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED = {"train": 795, "val": 44, "test": 44}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def audit(source, raw=None):
    rows, issues = [], []
    for split, expected in EXPECTED.items():
        images = sorted(p for p in (source / "images" / split).iterdir() if p.suffix.lower() in SUFFIXES)
        if len(images) != expected:
            issues.append(f"{split}: expected {expected}, got {len(images)}")
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            issues.append(f"{split}: duplicate image stems")
        labels = {p.stem for p in (source / "labels" / split).glob("*.txt")}
        if labels != set(stems):
            issues.append(f"{split}: unmatched image/label stems")
        for image in images:
            label = source / "labels" / split / (image.stem + ".txt")
            boxes = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if (len(values) != 5 or not all(math.isfinite(v) for v in values)
                        or values[0] != 0 or not all(0 <= v <= 1 for v in values[1:])
                        or min(values[3:]) <= 0):
                    issues.append(f"Invalid label: {split}/{label.name}")
                boxes.append(values)
            if not boxes:
                issues.append(f"Empty positive-only label: {split}/{label.name}")
            with Image.open(image) as im:
                im.load()
                rgb = im.convert("RGB")
                pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                shape = list(im.size)
            rows.append({"split": split, "name": image.name, "image_sha256": sha(image),
                         "pixel_sha256": pixel, "label_sha256": sha(label), "size": shape,
                         "boxes": len(boxes)})
    duplicates = {}
    for key in ("name", "image_sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(f"{row['split']}/{row['name']}")
        duplicates[key] = [v for v in groups.values() if len(v) > 1]
        for group in duplicates[key]:
            if len({p.split('/')[0] for p in group}) > 1:
                issues.append(f"Cross-split {key}: {group}")
    report = {"dataset": "SADD", "protocol": "recovered local positive-only 883, fixed 795/44/44",
              "source": str(source.resolve()), "counts": dict(Counter(r['split'] for r in rows)),
              "boxes": {s: sum(r['boxes'] for r in rows if r['split'] == s) for s in EXPECTED},
              "duplicates": duplicates, "inventory": rows, "inventory_sha256": inventory_digest(rows),
              "limitations": ["Not SAR-AIRcraft1.0 (seven-class dataset)",
                              "Local random image split, not verified official benchmark split",
                              "No near-duplicate, scene/source independence proof",
                              "Positive-only evaluation does not measure false alarms on empty scenes",
                              "44 test images give limited statistical precision"]}
    if raw:
        images = sorted(p for p in (raw / "images").iterdir() if p.suffix.lower() in SUFFIXES)
        positives, missing = [], []
        for image in images:
            label = raw / "labels" / (image.stem + ".txt")
            if not label.exists():
                missing.append(image.name)
            elif label.read_text().strip():
                positives.append(image)
        by_name = {p.name: p for p in positives}
        if len(by_name) != 883 or set(by_name) != {r['name'] for r in rows}:
            issues.append("Prepared data does not exactly cover the 883 raw positive images")
        for row in rows:
            if row['name'] not in by_name:
                continue
            image = by_name[row['name']]
            label = raw / "labels" / (image.stem + ".txt")
            if sha(image) != row['image_sha256'] or sha(label) != row['label_sha256']:
                issues.append(f"Raw/prepared content mismatch: {row['name']}")
        shuffled = list(positives)
        random.Random(0).shuffle(shuffled)
        reconstructed = {"train": shuffled[:795], "val": shuffled[795:839], "test": shuffled[839:]}
        report['raw'] = {"path": str(raw.resolve()), "images": len(images), "positive": len(positives),
                         "empty_or_unannotated": len(images) - len(positives), "missing_labels": missing,
                         "seed0_sorted_positive_split_matches": all(
                             {p.name for p in reconstructed[s]} == {r['name'] for r in rows if r['split'] == s}
                             for s in EXPECTED)}
    report["issues"] = issues
    report["status"] = "PASS" if not issues else "FAIL"
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--raw", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.source, args.raw)
    if args.output.exists():
        raise RuntimeError(f"Do not overwrite audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != 'inventory'}, indent=2))
    if report['issues']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
