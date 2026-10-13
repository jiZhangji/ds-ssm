#!/usr/bin/env python3
"""Audit/stage only the five authorized cross-task datasets; never alter sources.

This helper imports no model code. Image decoding is an integrity/content audit,
not a training, inference, or benchmark operation. Run on the selected server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from manifest import canonical, digest, frozen, read_json, sha

DATASETS = {
    "SAR-ACD": "classification", "SAMPLE": "classification",
    "Ottawa": "change_detection", "Farmland": "change_detection",
    "Yellow_River": "change_detection",
}
SPLITS = ("train", "val", "test")
SCHEMA = "rgsp-cross-data-v1"


def resolve_image(raw, manifest):
    """Accept an explicit in-tree path, or relocate its recorded _raw suffix."""
    root = manifest.parent.resolve()
    value = str(raw).replace("\\", "/")
    candidates = [Path(raw), root / value]
    if "/_raw/" in value:
        candidates.append(root / "_raw" / value.split("/_raw/", 1)[1])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and root in resolved.parents:
            return resolved
    raise RuntimeError(f"Missing or out-of-source-tree image: {raw}")


def image_info(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        header = canonical({"mode": "RGB", "width": rgb.width, "height": rgb.height})
        pixel_hash = hashlib.sha256(header + rgb.tobytes()).hexdigest()
        return {"sha256": sha(path), "pixel_sha256": pixel_hash,
                "width": rgb.width, "height": rgb.height, "source_mode": image.mode,
                "bytes": path.stat().st_size}


def intersections(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(width, 0) * max(height, 0)


def separation(a, b):
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def geometry_audit(records):
    """Verify native source-pixel isolation, report adjacency without hiding it."""
    overlap, adjacent, minimum_gap = [], 0, None
    within_overlap = Counter()
    duplicate_rows = set()
    group_splits = defaultdict(set)
    for index, row in enumerate(records):
        key = (row["image1"], row["image2"], row["mask"], tuple(row["crop"]))
        if key in duplicate_rows:
            raise RuntimeError(f"Duplicate CD crop record: {key}")
        duplicate_rows.add(key)
        group_splits[(row["image1"], row["image2"], str(row["group"]))].add(row["split"])
        crop, bounds = row["crop"], row["group_bounds"]
        if not (bounds[0] <= crop[0] < crop[2] <= bounds[2]
                and bounds[1] <= crop[1] < crop[3] <= bounds[3]):
            raise RuntimeError(f"Crop crosses its native macroblock: {index}")
        for other in records[:index]:
            if (row["image1"], row["image2"]) != (other["image1"], other["image2"]):
                continue
            area = intersections(crop, other["crop"])
            if row["split"] == other["split"]:
                within_overlap[row["split"]] += int(area > 0)
                continue
            if area:
                overlap.append({"left": index, "right_crop": other["crop"], "area": area})
            distance = separation(crop, other["crop"])
            minimum_gap = distance if minimum_gap is None else min(minimum_gap, distance)
            adjacent += int(distance == 0 and area == 0)
    if overlap or any(len(value) != 1 for value in group_splits.values()):
        raise RuntimeError(f"CD spatial group/pixel leakage: {overlap[:10]}")
    return {"cross_split_intersection_count": 0, "cross_split_touching_crop_pairs": adjacent,
            "minimum_cross_split_rectangle_gap_pixels": minimum_gap,
            "within_split_overlapping_crop_pairs": dict(within_overlap),
            "scene_disjoint": False, "buffer_zone_required_or_claimed": False,
            "limitation": "One-scene spatial holdout, adjacent correlated regions; not unseen-scene generalization.",
            "evaluation": "Resize predictions to native crop; mean overlapping probabilities; count each heldout source pixel once."}


def stage(source_manifest, output, dataset, annotation_policy):
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    if not source_manifest.is_file() or dataset not in DATASETS:
        raise ValueError("An existing source manifest and authorized dataset are required")
    if output == source_manifest.parent or source_manifest.parent in output.parents:
        raise ValueError("Output must be an independent directory outside the source dataset")
    old = read_json(source_manifest)
    if old.get("task") != DATASETS[dataset]:
        raise ValueError("Dataset task does not match the authorized dataset")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    files, file_cache = {}, {}
    path_keys = ("path",) if old["task"] == "classification" else ("image1", "image2", "mask")
    for index, source_row in enumerate(old["records"]):
        row = dict(source_row)
        if row.get("split") not in SPLITS:
            raise ValueError(f"Invalid split on row {index}")
        row["source_record_index"] = index
        for key in path_keys:
            source = resolve_image(row[key], source_manifest)
            if source not in file_cache:
                file_cache[source] = image_info(source)
            info = file_cache[source]
            relative = f"files/{info['sha256'][:2]}/{info['sha256']}{source.suffix.lower()}"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if sha(target) != info["sha256"]:
                raise RuntimeError(f"Staged bytes differ: {target}")
            files[relative] = {"path": relative, **info}
            row[key] = relative
            row[key + "_source"] = str(source.relative_to(source_manifest.parent))
        records.append(row)
    audit = {"schema": SCHEMA, "dataset": dataset, "task": old["task"],
             "source_manifest": str(source_manifest), "source_manifest_sha256": sha(source_manifest),
             "source_counts": dict(Counter(row["split"] for row in records)),
             "source_records": len(records), "unique_staged_files": len(files),
             "all_referenced_files_exist_and_decode": True,
             "annotation_policy": annotation_policy, "source_files_modified": False}
    exclusions, conflicts, duplicate_groups = [], [], []
    if old["task"] == "classification":
        classes = old.get("classes")
        if (not isinstance(classes, list) or len(classes) != (5 if dataset == "SAR-ACD" else 10)
                or len(set(classes)) != len(classes)):
            raise ValueError("Unexpected classification class list")
        paths, groups = set(), defaultdict(list)
        for row in records:
            source_path = row["path_source"]
            if source_path in paths:
                raise RuntimeError("A classification source path occurs more than once")
            paths.add(source_path)
            if row["label"] not in range(len(classes)) or classes[row["label"]] != row["class_name"]:
                raise ValueError("Classification label/class_name disagree")
            if dataset == "SAMPLE" and "/decibel/real/" not in source_path.replace("\\", "/"):
                raise RuntimeError("SAMPLE must use decibel/real only, not synthetic or magnitude duplicates")
            groups[files[row["path"]]["pixel_sha256"]].append(row)
        kept = []
        priority = {"train": 0, "val": 1, "test": 2}
        for pixel_hash, group in groups.items():
            labels = {row["label"] for row in group}
            ambiguous = len(labels) > 1
            highest = max(priority[row["split"]] for row in group)
            details = {"pixel_sha256": pixel_hash,
                       "rows": [{key: row[key] for key in ("source_record_index", "split", "class_name", "path_source")} for row in group]}
            if len(group) > 1:
                duplicate_groups.append(details)
            if ambiguous:
                conflicts.append(details)
            for row in group:
                reason = None
                if priority[row["split"]] < highest:
                    reason = "Identical decoded content exists in higher-priority heldout split (test > val > train)."
                elif ambiguous and row["split"] == "train":
                    reason = "Contradictory class labels for identical content; exclude all ambiguous training copies without relabeling."
                if reason:
                    exclusions.append({**row, "reason": reason})
                else:
                    kept.append(row)
        if conflicts and annotation_policy == "fail":
            frozen(output / "audit_blocked.json", {**audit, "ready": False,
                   "conflicting_content_groups": conflicts, "duplicate_content_groups": duplicate_groups,
                   "message": "Explicit preserve-heldout/exclude-ambiguous-training policy required; no training manifest emitted."})
            raise RuntimeError("Contradictory classification labels; see audit_blocked.json")
        records = sorted(kept, key=lambda row: row["source_record_index"])
        for split in SPLITS:
            if {row["label"] for row in records if row["split"] == split} != set(range(len(classes))):
                raise RuntimeError(f"Cleaning removed a class from {split}")
        serials = defaultdict(lambda: defaultdict(set))
        if dataset == "SAMPLE":
            for row in records:
                match = re.search(r"_serial_([^.]*)", Path(row["path_source"]).name)
                serials[row["class_name"]][match.group(1) if match else "unknown"].add(row["split"])
        audit.update({"duplicate_content_groups": duplicate_groups,
                      "conflicting_content_groups": conflicts, "excluded_records": exclusions,
                      "test_policy": "Preserve original test rows/labels; val has priority over train; retain ambiguous heldout annotations and disclose them.",
                      "retained_annotation_conflict_records": [r for g in conflicts for r in g["rows"] if r["source_record_index"] in {x["source_record_index"] for x in records}],
                      "sample_serial_split_membership": {c: {s: sorted(v) for s, v in by.items()} for c, by in serials.items()},
                      "instance_or_scene_disjoint": False,
                      "limitation": "Decoded-exact-content isolation only, not a near-duplicate, multi-view, instance or scene independence guarantee."})
    else:
        for row in records:
            infos = [files[row[key]] for key in path_keys]
            if len({(info["width"], info["height"]) for info in infos}) != 1:
                raise RuntimeError("CD temporal pair/mask dimensions differ")
            for name in ("crop", "group_bounds"):
                box = row.get(name)
                if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                    raise ValueError("CD crop/group bounds require four native integer coordinates")
                if not (0 <= box[0] < box[2] <= infos[0]["width"] and 0 <= box[1] < box[3] <= infos[0]["height"]):
                    raise ValueError("CD rectangle is out of image bounds")
        audit["geometry"] = geometry_audit(records)
        audit["split_policy"] = "Preserve pre-existing label-balanced native 96px macroblocks; no split reoptimization."
    if set(row["split"] for row in records) != set(SPLITS):
        raise RuntimeError("Every task requires nonempty independent train/val/test")
    audit["counts"] = dict(Counter(row["split"] for row in records))
    audit["ready"] = True
    metadata = {key: value for key, value in old.items() if key != "records"}
    metadata.update({"protocol": SCHEMA, "dataset": dataset, "source_protocol": old.get("protocol"),
                     "records": records,
                     "selection_policy": "validation-only; sealed test evaluated after checkpoint/threshold selection",
                     "normalization": "fixed RGB /255, then (x-0.5)/0.5; no estimated test statistics"})
    logical_hash = digest(metadata)
    inventory = sorted(files.values(), key=lambda row: row["path"])
    frozen(output / "files.json", inventory)
    frozen(output / "audit.json", audit)
    staged = dict(metadata)
    staged["records"] = [{**row, **{key: str(output / row[key]) for key in path_keys}} for row in records]
    staged["rgsp_data"] = {"schema": SCHEMA, "logical_manifest_sha256": logical_hash,
                           "files_sha256": sha(output / "files.json"), "audit_sha256": sha(output / "audit.json"),
                           "annotation_policy": annotation_policy}
    frozen(output / "manifest.json", staged)
    return verify_staged(output / "manifest.json")


def verify_staged(path):
    path = Path(path).resolve()
    manifest, root = read_json(path), path.parent
    meta = manifest.get("rgsp_data", {})
    if meta.get("schema") != SCHEMA or manifest.get("dataset") not in DATASETS:
        raise RuntimeError("Only audited/staged RGSP cross-data manifests are accepted")
    for name, key in (("files.json", "files_sha256"), ("audit.json", "audit_sha256")):
        if sha(root / name) != meta[key]:
            raise RuntimeError(f"Audit/inventory hash mismatch: {name}")
    inventory = read_json(root / "files.json")
    by_path = {}
    for row in inventory:
        target = (root / row["path"]).resolve()
        if root not in target.parents or not target.is_file() or sha(target) != row["sha256"]:
            raise RuntimeError(f"Missing/changed staged image: {target}")
        by_path[str(target)] = row
    records = manifest["records"]
    path_keys = ("path",) if manifest["task"] == "classification" else ("image1", "image2", "mask")
    normalized = []
    for row in records:
        item = dict(row)
        for key in path_keys:
            resolved = str(Path(row[key]).resolve())
            if resolved not in by_path:
                raise RuntimeError("Manifest references an unaudited image")
            item[key] = by_path[resolved]["path"]
        normalized.append(item)
    logical = {key: value for key, value in manifest.items() if key != "rgsp_data"}
    logical["records"] = normalized
    if digest(logical) != meta["logical_manifest_sha256"]:
        raise RuntimeError("Logical manifest hash mismatch")
    if manifest["task"] == "classification":
        memberships = defaultdict(set)
        for row in records:
            memberships[by_path[str(Path(row["path"]).resolve())]["pixel_sha256"]].add(row["split"])
        if any(len(value) != 1 for value in memberships.values()):
            raise RuntimeError("Decoded-identical content crosses staged classification splits")
    else:
        geometry_audit(records)
    audit = read_json(root / "audit.json")
    if audit.get("ready") is not True or Counter(row["split"] for row in records) != Counter(audit["counts"]):
        raise RuntimeError("Prepared dataset is not marked ready or counts changed")
    return {"manifest": str(path), "manifest_sha256": sha(path),
            "logical_manifest_sha256": meta["logical_manifest_sha256"],
            "counts": audit["counts"], "audit_sha256": meta["audit_sha256"],
            "files_sha256": meta["files_sha256"], "annotation_policy": meta["annotation_policy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest")
    parser.add_argument("--output")
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--verify")
    parser.add_argument("--annotation-policy", choices=("fail", "preserve-heldout-exclude-ambiguous-training"), default="fail")
    args = parser.parse_args()
    if args.verify:
        result = verify_staged(args.verify)
    else:
        if not all((args.source_manifest, args.output, args.dataset)):
            parser.error("--source-manifest, --output, --dataset are required to stage data")
        result = stage(args.source_manifest, args.output, args.dataset, args.annotation_policy)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
