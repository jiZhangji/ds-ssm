#!/usr/bin/env python3
"""Prepare the full, positive-and-negative SADD image-level protocol.

Run only on an explicitly selected Linux server with Pillow already available.
The source is read-only; all images and labels are retained as symlinks in a
new SADD_full directory. An existing dataset is never adopted or overwritten.
readiness.json is published last, only after a second complete source/link
audit and a read-back of the saved manifests. This is not a scene-disjoint or
official-split protocol, and it does not replace the historical 883-image run.

Public integration entry point:
    prepare_sadd(taskdata_root, source_root=DEFAULT_SOURCE_ROOT) -> readiness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

if __package__:
    from .preparation_common import (
        SCHEMA, SPLIT_SEED, canonical, create_dataset_dir, digest,
        ensure_taskdata_root, fixed_group_split, image_info, sha_file, utc,
        write_json_new, write_text_new,
    )
else:
    from preparation_common import (
        SCHEMA, SPLIT_SEED, canonical, create_dataset_dir, digest,
        ensure_taskdata_root, fixed_group_split, image_info, sha_file, utc,
        write_json_new, write_text_new,
    )


DEFAULT_SOURCE_ROOT = (
    "/home/powerleader/ig-ds-ssm/data/cross_task_prepared_v2/"
    "detection/_raw_sadd/SAR-aircraft"
)
DATASET = "SADD"
DATASET_DIRECTORY = "SADD_full"
PROTOCOL_ID = "sadd-full2966-posneg-stratified-80-10-10-rgbgroup-v1"
SPLITS = ("train", "val", "test")
EXPECTED_IMAGES = 2966
EXPECTED_STRATA = {"positive": 883, "negative": 2083}
STRATIFIED_TARGETS = {
    name: {"train": count * 8 // 10, "val": count // 10,
           "test": count - count * 8 // 10 - count // 10}
    for name, count in EXPECTED_STRATA.items()
}
EXPECTED_SPLITS = {
    split: sum(targets[split] for targets in STRATIFIED_TARGETS.values())
    for split in SPLITS
}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
CORNER_TOLERANCE = Fraction(1, 1_000_000)


def _protocol():
    return {
        "id": PROTOCOL_ID,
        "task": "detection",
        "dataset": DATASET,
        "scope": "all 2966 source images, including 2083 negative images",
        "split_seed": SPLIT_SEED,
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "stratification": "positive iff a valid source label contains at least one box",
        "stratified_targets": STRATIFIED_TARGETS,
        "integer_rule": "within each stratum: floor(0.8*N), floor(0.1*N), remainder",
        "split_unit": "complete exact decoded RGB content group within each stratum",
        "ordering": "SHA256 rank of [20261012, namespace, pixel_sha256]; no training seed",
        "quota_solver": "preparation_common.fixed_group_split exact bounded subset-sum",
        "infeasible_quota_policy": "fail; never split a group, delete, or resample images",
        "pixel_sha256": (
            "SHA256(canonical JSON of RGB mode/width/height + PIL-decoded RGB bytes)"
        ),
        "labels": {
            "format": "YOLO normalized class x_center y_center width height",
            "classes": {"0": "aircraft"},
            "validation": (
                "ASCII numeric text; exactly five finite numeric values per nonblank line; "
                "integer class 0; centers in [0,1]; widths/heights in (0,1]; "
                "valid normalized xywh edge boxes may have corners beyond [0,1]; "
                "raw extents retained, native Ultralytics augmentation clipping unchanged; "
                "duplicate semantic/float32 boxes, float32-degenerate boxes and "
                "loader-incompatible internal whitespace-only lines rejected"
            ),
            "semantic_comparison": (
                "sorted exact rational-valued boxes; numeric spelling, whitespace, "
                "line endings and row order do not matter; no rounding or clipping"
            ),
            "conflicting_rgb_group_policy": "fail, including positive/negative conflicts",
            "empty_label_policy": "retain; every image must still have a source TXT file",
            "source_mutation": "none, including no newline or coordinate normalization",
        },
        "storage": "absolute symlinks to original image/label files; no source writes",
        "official_split": False,
        "scene_disjoint": False,
        "scene_ids_available": False,
        "near_duplicate_audit_performed": False,
        "limitations": [
            "Image-level content grouping is not proof of scene/source independence.",
            "No near-duplicate or scene mapping is inferred from numeric filenames.",
            "This full-data protocol is separate from the historical positive-only 795/44/44.",
            "Readiness is a point-in-time audit; consumers must revalidate mutable sources.",
        ],
    }


def _real_source_root(raw):
    requested = Path(raw).expanduser()
    if not requested.is_absolute() or ".." in requested.parts:
        raise ValueError("--source-root must be an explicit absolute source directory")
    resolved = requested.resolve(strict=True)
    if resolved != requested or requested.is_symlink() or not requested.is_dir():
        raise ValueError(f"Source must be a real directory with no symlink ancestors: {requested}")
    return resolved


def _source_inventory(source):
    """Reject unexpected entries, case collisions, or an incomplete pairing."""
    inventories = {}
    for directory_name, suffix in (("images", ".bmp"), ("labels", ".txt")):
        directory = source / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Expected a real source directory: {directory}")
        by_stem, folded = {}, set()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if (path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
                    or path.suffix.lower() != suffix
                    or any(ord(character) < 32 for character in path.name)):
                raise ValueError(f"Unexpected source entry; no files will be silently ignored: {path}")
            if path.stem in by_stem or path.stem.casefold() in folded:
                raise ValueError(f"Duplicate/case-colliding source stem: {path}")
            by_stem[path.stem] = path
            folded.add(path.stem.casefold())
        inventories[directory_name] = by_stem
    images, labels = inventories["images"], inventories["labels"]
    if images.keys() != labels.keys():
        missing = sorted(images.keys() - labels.keys())
        orphaned = sorted(labels.keys() - images.keys())
        raise ValueError(f"SADD pairing is not one-to-one: missing labels={missing}, orphan labels={orphaned}")
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"Expected all {EXPECTED_IMAGES} SADD pairs, found {len(images)}")
    return [(images[name], labels[name]) for name in sorted(images)]


def _label_info(path):
    """Read immutable raw bytes and compare annotation semantics without rounding."""
    before = path.stat()
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"Non-ASCII YOLO label: {path}") from error
    boxes = []
    float32_boxes = []
    tolerated_corners = 0
    # Match the current YOLO loader's strip/splitlines behavior without importing it.
    for line_number, line in enumerate(text.strip().splitlines(), start=1):
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5 or any(NUMBER.fullmatch(token) is None for token in tokens):
            raise ValueError(f"Expected exactly five finite YOLO numbers: {path}:{line_number}")
        try:
            category, x, y, width, height = map(Fraction, tokens)
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            raise ValueError(f"Invalid numeric YOLO label: {path}:{line_number}") from error
        if (category != 0 or category.denominator != 1
                or not 0 <= x <= 1 or not 0 <= y <= 1
                or not 0 < width <= 1 or not 0 < height <= 1):
            raise ValueError(f"Invalid single-class normalized YOLO box: {path}:{line_number}")
        corners = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)
        # The frozen Ultralytics verifier validates normalized xywh, not all
        # derived corners. Preserve valid edge-object extents byte-for-byte;
        # native augmentations retain their existing clipping behavior.
        tolerated_corners += int(any(value < 0 or value > 1 for value in corners))
        boxes.append((category, x, y, width, height))
        packed = struct.pack("!5f", *(float(value) for value in boxes[-1]))
        if min(struct.unpack("!5f", packed)[3:]) <= 0:
            raise ValueError(f"YOLO box degenerates to zero size in float32: {path}:{line_number}")
        float32_boxes.append(packed)
    if len(boxes) != len(set(boxes)):
        raise ValueError(f"Duplicate semantic boxes would be silently deduplicated by a loader: {path}")
    if len(float32_boxes) != len(set(float32_boxes)):
        raise ValueError(f"Distinct decimal boxes collapse to loader-deduplicated float32 rows: {path}")
    semantic = [[f"{value.numerator}/{value.denominator}" for value in box]
                for box in sorted(boxes)]
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ino)
            != (after.st_size, after.st_mtime_ns, after.st_ino)
            or before.st_size != len(raw) or sha_file(path) != raw_sha):
        raise RuntimeError(f"Source label changed during audit: {path}")
    return {
        "label_sha256": raw_sha,
        "semantic_label_sha256": digest(semantic),
        "label_bytes": len(raw),
        "box_count": len(boxes),
        "positive": bool(boxes),
        "boundary_overhang_box_count": tolerated_corners,
    }


def _audit_source(source, pairs):
    records = []
    for image, label in pairs:
        info = image_info(image)
        if info["format"] != "BMP" or min(info["width"], info["height"]) < 10:
            raise ValueError(f"Expected a decodable BMP of at least 10 pixels on each side: {image}")
        records.append({
            "name": image.name,
            "image_source": str(image),
            "label_source": str(label),
            "source_relative_path": str(image.relative_to(source)),
            "label_source_relative_path": str(label.relative_to(source)),
            "image_sha256": info["sha256"],
            "pixel_sha256": info["pixel_sha256"],
            "width": info["width"],
            "height": info["height"],
            "image_bytes": info["bytes"],
            "image_format": info["format"],
            "source_mode": info["source_mode"],
            **_label_info(label),
        })
    actual = Counter("positive" if row["positive"] else "negative" for row in records)
    if dict(actual) != EXPECTED_STRATA:
        raise ValueError(f"Expected positive/negative counts {EXPECTED_STRATA}, found {dict(actual)}")
    _duplicate_audit(records, require_split=False)
    return records


def _duplicate_audit(records, require_split=True):
    report = {}
    for field in ("image_sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row)
        duplicates = []
        for group_id, rows in sorted(groups.items()):
            if len({row["semantic_label_sha256"] for row in rows}) != 1:
                raise ValueError(f"Semantic label conflict in {field} group {group_id}: "
                                 f"{[row['name'] for row in rows]}")
            if len({row["positive"] for row in rows}) != 1:
                raise ValueError(f"Positive/negative conflict in content group {group_id}")
            if field == "image_sha256" and len({row["pixel_sha256"] for row in rows}) != 1:
                raise ValueError(f"Identical source bytes decoded differently in group {group_id}")
            splits = sorted({row["split"] for row in rows}) if require_split else []
            if require_split and len(splits) != 1:
                raise ValueError(f"Content group crosses splits: {field} {group_id}")
            if len(rows) > 1:
                duplicates.append({
                    "sha256": group_id, "count": len(rows), "splits": splits,
                    "names": sorted(row["name"] for row in rows),
                    "semantic_label_sha256": rows[0]["semantic_label_sha256"],
                })
        report[field] = {"unique_groups": len(groups), "duplicate_groups": duplicates,
                         "duplicate_group_count": len(duplicates),
                         "duplicate_member_count": sum(item["count"] for item in duplicates)}
    return report


def _assign_splits(source_records, dataset_root):
    records = []
    for stratum in ("positive", "negative"):
        rows = [row for row in source_records if row["positive"] == (stratum == "positive")]
        partitions = fixed_group_split(rows, STRATIFIED_TARGETS[stratum], f"SADD_full/{stratum}")
        for split in SPLITS:
            for row in partitions[split]:
                image = dataset_root / "images" / split / row["name"]
                label = dataset_root / "labels" / split / (Path(row["name"]).stem + ".txt")
                records.append({**row, "split": split, "path": str(image),
                                "image": str(image), "label_path": str(label), "label": str(label)})
    records.sort(key=lambda row: (SPLITS.index(row["split"]), row["name"]))
    _verify_partition(records, source_records)
    return records


def _verify_partition(records, source_records):
    if len(records) != EXPECTED_IMAGES or len({row["name"] for row in records}) != EXPECTED_IMAGES:
        raise ValueError("Output must contain each of the 2966 source image names exactly once")
    for field in ("image_source", "label_source"):
        if Counter(row[field] for row in records) != Counter(row[field] for row in source_records):
            raise ValueError(f"Split construction lost or duplicated a source member: {field}")
    if dict(Counter(row["split"] for row in records)) != EXPECTED_SPLITS:
        raise ValueError("SADD split totals differ from the frozen quotas")
    for stratum, targets in STRATIFIED_TARGETS.items():
        observed = Counter(row["split"] for row in records
                           if row["positive"] == (stratum == "positive"))
        if dict(observed) != targets:
            raise ValueError(f"SADD {stratum} split counts differ from the frozen quotas")
    for field in ("path", "label_path"):
        if len({row[field] for row in records}) != EXPECTED_IMAGES:
            raise ValueError(f"Duplicate output path: {field}")
    _duplicate_audit(records)


def _create_links(dataset_root, records):
    for role in ("images", "labels"):
        (dataset_root / role).mkdir(exist_ok=False)
        for split in SPLITS:
            (dataset_root / role / split).mkdir(exist_ok=False)
    for row in records:
        os.symlink(row["image_source"], row["path"])
        os.symlink(row["label_source"], row["label_path"])
    for role in ("images", "labels"):
        for directory in [dataset_root / role / split for split in SPLITS] + [dataset_root / role]:
            descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _verify_links_and_sources(dataset_root, source, records, original_pairs):
    if dataset_root.is_symlink() or dataset_root.resolve(strict=True) != dataset_root:
        raise ValueError("Dataset directory or an ancestor changed into a symlink")
    if _source_inventory(source) != original_pairs:
        raise RuntimeError("The complete source pairing changed after the initial audit")
    for role, field in (("images", "path"), ("labels", "label_path")):
        role_root = dataset_root / role
        if role_root.is_symlink() or {path.name for path in role_root.iterdir()} != set(SPLITS):
            raise ValueError(f"Unexpected output directories: {role_root}")
        for split in SPLITS:
            directory = role_root / split
            expected = {Path(row[field]).name for row in records if row["split"] == split}
            if directory.is_symlink() or {path.name for path in directory.iterdir()} != expected:
                raise ValueError(f"Output directory is not exactly the frozen inventory: {directory}")
    for row in records:
        for field, source_field in (("path", "image_source"), ("label_path", "label_source")):
            output, original = Path(row[field]), Path(row[source_field])
            if (not output.is_absolute() or not original.is_absolute()
                    or not output.is_symlink() or os.readlink(output) != str(original)
                    or output.resolve(strict=True) != original or source not in original.parents):
                raise ValueError(f"Output is not an exact absolute link to its approved source: {output}")
        info = image_info(Path(row["path"]))
        for field, info_field in (("image_sha256", "sha256"), ("pixel_sha256", "pixel_sha256"),
                                  ("width", "width"), ("height", "height"),
                                  ("image_bytes", "bytes"), ("source_mode", "source_mode"),
                                  ("image_format", "format")):
            if row[field] != info[info_field]:
                raise RuntimeError(f"Image changed between source and final link audits: {row['name']} ({field})")
        if any(row[field] != value for field, value in _label_info(Path(row["label_path"])).items()):
            raise RuntimeError(f"Label changed between source and final link audits: {row['name']}")
    if _source_inventory(source) != original_pairs:
        raise RuntimeError("The complete source pairing changed during the final audit")


def _content_identity(records, protocol):
    fields = ("name", "source_relative_path", "label_source_relative_path", "split",
              "image_sha256", "label_sha256", "semantic_label_sha256", "pixel_sha256",
              "width", "height", "positive", "box_count")
    portable_records = [{field: row[field] for field in fields}
                        for row in sorted(records, key=lambda row: row["name"])]
    return digest({"schema": SCHEMA, "protocol": protocol, "records": portable_records})


def _readback_json(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary generated JSON file: {path}")
    payload = path.read_bytes()
    if json.loads(payload) != expected or sha_file(path) != hashlib.sha256(payload).hexdigest():
        raise RuntimeError(f"Generated artifact failed the final read-back: {path}")
    return hashlib.sha256(payload).hexdigest()


def prepare_sadd(taskdata_root, source_root=DEFAULT_SOURCE_ROOT):
    """Create only SADD_full; return its verified readiness, or raise on failure.

    A failed invocation preserves its newly created directory and writes a
    dataset-local preparation_failed.json when possible. It never mutates any
    other dataset, retries into an existing SADD_full, or installs dependencies.
    """
    dataset_root = None
    started_at = utc()
    try:
        code_paths = {"preparer_sha256": Path(__file__),
                      "common_helper_sha256": Path(image_info.__code__.co_filename)}
        code_hashes = {name: sha_file(path) for name, path in code_paths.items()}
        requested_source = Path(source_root).expanduser()
        if not requested_source.is_absolute() or ".." in requested_source.parts:
            raise ValueError("--source-root must be an explicit absolute source directory")
        root = ensure_taskdata_root(taskdata_root, readonly_roots=(requested_source,))
        dataset_root = create_dataset_dir(root, DATASET_DIRECTORY)
        source = _real_source_root(requested_source)
        pairs = _source_inventory(source)
        source_records = _audit_source(source, pairs)
        records = _assign_splits(source_records, dataset_root)
        protocol = _protocol()
        content_identity = _content_identity(records, protocol)
        source_identity = digest([
            {key: row[key] for key in (
                "source_relative_path", "label_source_relative_path", "image_sha256",
                "label_sha256", "pixel_sha256", "semantic_label_sha256", "width", "height")}
            for row in sorted(source_records, key=lambda row: row["name"])
        ])
        _create_links(dataset_root, records)
        yaml_path = dataset_root / "SADD.yaml"
        yaml_text = (
            "# Full SADD image-level protocol; not an official or scene-independent split.\n"
            f"path: {json.dumps(str(dataset_root), ensure_ascii=False)}\n"
            "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: aircraft\n"
        )
        write_text_new(yaml_path, yaml_text)
        yaml_sha = sha_file(yaml_path)
        totals = {"images": EXPECTED_IMAGES, **EXPECTED_STRATA,
                  "boxes": sum(row["box_count"] for row in records)}
        shared = {
            "schema": SCHEMA, "task": "detection", "dataset": DATASET,
            "root": str(dataset_root), "source_root": str(source),
            "protocol": PROTOCOL_ID, "protocol_details": protocol, "split_seed": SPLIT_SEED,
            "stratification": "positive/negative by validated box_count",
            "counts": EXPECTED_SPLITS, "total_counts": totals,
            "stratified_counts": STRATIFIED_TARGETS,
            "source_inventory_sha256": source_identity,
            "content_identity_sha256": content_identity,
        }
        manifest = {**shared, "data_yaml": str(yaml_path), "data_yaml_sha256": yaml_sha,
                    "names": ["aircraft"], "records": records}
        files = {**shared, "inventory_kind": "all image/label symlink pairs and original raw hashes",
                 "record_count": len(records), "source_file_count": 2 * len(records),
                 "records": records}
        manifest_path, files_path = dataset_root / "manifest.json", dataset_root / "files.json"
        write_json_new(manifest_path, manifest)
        write_json_new(files_path, files)

        # Full second pass is mandatory, not a sampled or filename-only check.
        _verify_links_and_sources(dataset_root, source, records, pairs)
        _verify_partition(records, source_records)
        if (yaml_path.is_symlink() or yaml_path.read_text(encoding="utf-8") != yaml_text
                or sha_file(yaml_path) != yaml_sha):
            raise RuntimeError("SADD.yaml changed or failed read-back")
        manifest_sha = _readback_json(manifest_path, manifest)
        files_sha = _readback_json(files_path, files)
        audit = {
            **shared, "status": "verified", "started_at": started_at, "verified_at": utc(),
            "data_yaml_sha256": yaml_sha, "manifest_sha256": manifest_sha,
            "files_sha256": files_sha, "duplicates": _duplicate_audit(records),
            "checks": {
                "all_2966_source_pairs_present_exactly_once": True,
                "all_source_images_verified_and_decoded_twice": True,
                "all_raw_image_and_label_hashes_rechecked_through_links": True,
                "all_labels_valid_and_semantically_rechecked": True,
                "source_inventory_unchanged": True,
                "positive_negative_strata_and_exact_split_quotas": True,
                "all_output_links_have_exact_absolute_source_targets": True,
                "no_cross_split_exact_byte_or_rgb_groups": True,
                "no_semantic_conflicts_within_exact_content_groups": True,
                "saved_yaml_manifest_and_files_readback": True,
            },
            "zero_byte_label_count": sum(row["label_bytes"] == 0 for row in records),
            "boundary_overhang_box_count": sum(row["boundary_overhang_box_count"] for row in records),
            "box_counts": {split: sum(row["box_count"] for row in records if row["split"] == split)
                           for split in SPLITS},
            "source_files_modified": 0, "source_files_deleted": 0,
            "images_discarded": 0, "labels_rewritten": 0,
            **code_hashes,
        }
        audit_path = dataset_root / "audit.json"
        write_json_new(audit_path, audit)
        audit_sha = _readback_json(audit_path, audit)
        expected_entries = {"images", "labels", "SADD.yaml", "manifest.json", "files.json", "audit.json"}
        if {path.name for path in dataset_root.iterdir()} != expected_entries:
            raise RuntimeError("Unexpected files appeared in this new dataset directory before publication")
        if (sha_file(manifest_path) != manifest_sha or sha_file(files_path) != files_sha
                or sha_file(yaml_path) != yaml_sha or sha_file(audit_path) != audit_sha):
            raise RuntimeError("A generated artifact changed before readiness publication")
        if any(sha_file(path) != code_hashes[name] for name, path in code_paths.items()):
            raise RuntimeError("Preparation code or the common helper changed during this audit")
        readiness = {
            "schema": SCHEMA, "task": "detection", "dataset": DATASET,
            "protocol": PROTOCOL_ID,
            "ready": True, "status": "verified", "published_at": utc(),
            "root": str(dataset_root), "source_root": str(source),
            "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
            "audit": str(audit_path), "audit_sha256": audit_sha,
            "files": str(files_path), "files_sha256": files_sha,
            "data_yaml": str(yaml_path), "data_yaml_sha256": yaml_sha,
            "counts": EXPECTED_SPLITS, "total_counts": totals,
            "stratified_counts": STRATIFIED_TARGETS, "split_seed": SPLIT_SEED,
            "source_inventory_sha256": source_identity,
            "content_identity_sha256": content_identity,
            "scene_disjoint": False, "official_split": False,
        }
        # This is the only ready publication. There are no subsequent source writes.
        write_json_new(dataset_root / "readiness.json", readiness)
        return readiness
    except Exception as error:
        if dataset_root is not None and not os.path.lexists(dataset_root / "readiness.json"):
            failure = {"schema": SCHEMA, "task": "detection", "dataset": DATASET,
                       "ready": False, "status": "failed", "started_at": started_at,
                       "failed_at": utc(), "root": str(dataset_root),
                       "error_type": type(error).__name__, "error": str(error),
                       "source_files_modified": 0, "source_files_deleted": 0,
                       "partial_output_preserved": True}
            try:
                write_json_new(dataset_root / "preparation_failed.json", failure)
            except Exception as receipt_error:
                print(f"Could not publish dataset-local failure receipt: {receipt_error}", file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskdata-root", required=True,
                        help="New absolute Linux taskdata/taskdata_<id> root, or this campaign's marked root")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT,
                        help="Read-only full SADD source containing images/ and labels/")
    args = parser.parse_args(argv)
    try:
        readiness = prepare_sadd(args.taskdata_root, args.source_root)
    except Exception as error:
        print(json.dumps({"schema": SCHEMA, "dataset": DATASET, "ready": False,
                          "status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(canonical(readiness).decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
