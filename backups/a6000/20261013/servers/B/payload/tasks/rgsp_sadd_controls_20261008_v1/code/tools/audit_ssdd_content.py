#!/usr/bin/env python3
"""Read-only SSDD content duplicate audit, without importing a model."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
groups = defaultdict(list)
for split in ("train", "test"):
    for image in sorted((args.source / "images" / split).iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        label = args.source / "labels" / split / (image.stem + ".txt")
        groups[hashlib.sha256(image.read_bytes()).hexdigest()].append({
            "split": split, "image": image.name,
            "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
            "labels": label.read_text().splitlines(),
        })
duplicates = [{"image_sha256": key, "items": items} for key, items in groups.items() if len(items) > 1]
report = {
    "total_images": sum(len(v) for v in groups.values()), "unique_content": len(groups),
    "duplicate_groups": duplicates,
    "cross_official_train_test": [g for g in duplicates if len({r["split"] for r in g["items"]}) > 1],
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({
    "total_images": report["total_images"], "unique_content": report["unique_content"],
    "duplicate_groups": len(duplicates),
    "cross_official_train_test_groups": len(report["cross_official_train_test"]),
    "train_copies_overlapping_test": sum(x["split"] == "train" for g in report["cross_official_train_test"] for x in g["items"]),
    "output": str(args.output),
}, indent=2))
