#!/usr/bin/env python3
"""Stage the explicitly approved ShipDataset decontamination, never readiness.

Approval is limited to a new copy with original test retained, then val, then
train. Same-split repeated/conflicting annotations remain unchanged. No AIRcraft
mode exists here because that dataset is blocked on original-test ambiguity.
"""

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket


TASK = Path("/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1")
SPLITS = ("train", "val", "test")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return sha(data)


def compact_groups(rows, key):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    duplicate_groups = [members for members in grouped.values() if len(members) > 1]
    return {"duplicate_groups": len(duplicate_groups),
            "cross_split_groups": sum(len({r["split"] for r in g}) > 1 for g in duplicate_groups),
            "same_split_semantic_disagreement_groups": {
                split: sum(len({r["label_semantic_sha256"] for r in g if r["split"] == split}) > 1
                           for g in duplicate_groups) for split in SPLITS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--reviewed-audit-sha256", required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    if not audit_path.is_relative_to(TASK / "external_inputs/audits"):
        raise ValueError("Audit path must be inside the current independent task")
    audit_bytes = audit_path.read_bytes()
    if sha(audit_bytes) != args.reviewed_audit_sha256:
        raise ValueError("Audited archive record changed after review")
    audit = json.loads(audit_bytes)
    if audit["dataset_key"] != "sar_shipdataset" or audit["dataset_identity"] != "SAR-ShipDataset":
        raise ValueError("Wrong dataset; AIRcraft is not authorized for staging")
    if audit["nc"] != 1 or audit["names"] != ["ship"]:
        raise ValueError("Unexpected class mapping")
    if audit["source_archive_image_label_mismatches"]:
        raise ValueError("Archive differs from the audited existing source")
    if audit["label_stats"].get("invalid_normalized_box", 0):
        raise ValueError("Unresolved invalid normalized annotations")
    if audit["decoder"]["exif_orientation_counts"] != {"1": 39729}:
        raise ValueError("Unexpected image orientation; decoder semantics need review")
    if audit["counts"] != {"train": 31783, "val": 3972, "test": 3974}:
        raise ValueError("Original fixed split changed")
    rows_path = audit_path.parent / "audited_rows.json"
    rows_bytes = rows_path.read_bytes()
    if sha(rows_bytes) != audit["rows_sha256"]:
        raise ValueError("Audited row manifest changed")
    rows = json.loads(rows_bytes)
    byte_excluded = audit["byte_duplicates"]["candidate_exclusions_not_applied"]
    pixel_excluded = audit["pixel_duplicates"]["candidate_exclusions_not_applied"]
    excluded_names = {s: set(byte_excluded[s]) | set(pixel_excluded[s]) for s in SPLITS}
    if excluded_names["test"]:
        raise ValueError("Test exclusion is not authorized")
    kept = [r for r in rows if r["name"] not in excluded_names[r["split"]]]
    removed = [r for r in rows if r["name"] in excluded_names[r["split"]]]
    counts = {s: sum(r["split"] == s for r in kept) for s in SPLITS}
    if counts["test"] != 3974 or any(counts[s] == 0 for s in SPLITS):
        raise ValueError("Unexpected retained counts")
    for key in ("image_sha256", "pixel_sha256"):
        if compact_groups(kept, key)["cross_split_groups"]:
            raise ValueError("Retained data still has cross-split content overlap")
    if {(r["name"], r["image_sha256"], r["label_sha256"]) for r in rows if r["split"] == "test"} != {
            (r["name"], r["image_sha256"], r["label_sha256"]) for r in kept if r["split"] == "test"}:
        raise ValueError("Original test membership/content changed")
    destination = TASK / "data/sar_shipdataset"
    destination.mkdir(exist_ok=False)
    data_root = destination / "data"
    for split in SPLITS:
        (data_root / "images" / split).mkdir(parents=True)
        (data_root / "labels" / split).mkdir(parents=True)
    manifest = []
    for index, row in enumerate(kept):
        original = Path(row["source_image"]).resolve()
        if not original.is_relative_to(audit_path.parent / "extracted"):
            raise ValueError("Source image is outside the independently extracted archive")
        target_image = data_root / "images" / row["split"] / row["name"]
        target_label = data_root / "labels" / row["split"] / (Path(row["name"]).stem + ".txt")
        shutil.copy2(original, target_image)
        shutil.copy2(original.with_suffix(".txt"), target_label)
        if file_hash(target_image) != row["image_sha256"] or file_hash(target_label) != row["label_sha256"]:
            raise ValueError("Copied image/label differs from audited original: " + row["name"])
        manifest.append({k: row[k] for k in ("split", "name", "image_sha256", "label_sha256",
            "pixel_sha256", "source_group", "label_semantic_sha256", "width", "height")})
        if index % 4000 == 0:
            print("verified copy", index, "/", len(kept), flush=True)
    manifest_path = destination / "data_content.json"
    manifest_hash = write_new(manifest_path, manifest)
    excluded_rows = [{"split": r["split"], "name": r["name"],
                      "image_sha256": r["image_sha256"], "pixel_sha256": r["pixel_sha256"],
                      "label_sha256": r["label_sha256"],
                      "byte_rule": r["name"] in set(byte_excluded[r["split"]]),
                      "decoded_pixel_rule": r["name"] in set(pixel_excluded[r["split"]])}
                     for r in removed]
    exclusions_hash = write_new(destination / "exclusions.json", excluded_rows)
    yaml_path = destination / "data.yaml"
    yaml_data = (f"path: {data_root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 "nc: 1\nnames:\n  0: ship\n").encode()
    with yaml_path.open("xb") as stream:
        stream.write(yaml_data)
    descriptor = {"protocol": "ssdd", "root": str(data_root), "yaml": str(yaml_path),
                  "yaml_sha256": sha(yaml_data), "manifest": str(manifest_path),
                  "manifest_sha256": manifest_hash, "nc": 1, "names": ["ship"], "counts": counts}
    descriptor_hash = write_new(destination / "dataset.json", descriptor)
    source_audit = {
        "status": "verified", "dataset_identity": "SAR-ShipDataset", "nc": 1, "names": ["ship"],
        "manifest_sha256": manifest_hash, "counts": counts, "original_counts": audit["counts"],
        "protocol": "existing seed2026 split with test-first byte/decoded-RGB cross-split decontamination",
        "approval_scope": "Parent operator authorized independent-copy exclusions on 2026-10-09; "
            "preserve original test and all remaining label bytes; no resplit search or relabeling.",
        "partition_priority": ["test", "val", "train"],
        "source_archive": audit["archive"], "source_provenance": audit["provenance"],
        "archive_audit_path": str(audit_path), "archive_audit_sha256": sha(audit_bytes),
        "staging_script_sha256": file_hash(Path(__file__)),
        "exclusions_path": str(destination / "exclusions.json"), "exclusions_sha256": exclusions_hash,
        "exclusion_counts": {s: len(excluded_names[s]) for s in SPLITS},
        "byte_exclusion_counts": {s: len(byte_excluded[s]) for s in SPLITS},
        "pixel_exclusion_counts": {s: len(pixel_excluded[s]) for s in SPLITS},
        "pixel_exclusions_cover_byte_exclusions": all(set(byte_excluded[s]) <= set(pixel_excluded[s])
                                                      for s in SPLITS),
        "retained_byte_duplicates": compact_groups(kept, "image_sha256"),
        "retained_pixel_duplicates": compact_groups(kept, "pixel_sha256"),
        "original_test_3974_names_and_image_label_hashes_unchanged": True,
        "source_files_modified_or_deleted": False,
        "new_training_seeds_share_this_manifest": True,
        "decoder": audit["decoder"], "filename_group_audit": audit["group_audit"],
        "limitations": [
            "This is a fixed local image split, not a verified official or acquisition-disjoint protocol.",
            "Sensor/polarization/date filename groups are unverified proxies, not recovered scene IDs.",
            "Cross-split exact byte/RGB duplicates are excluded from lower-priority splits only.",
            "Within-split duplicate images and differing source annotations are preserved, not adjudicated.",
            "No near-duplicate or geographic/acquisition-independence claim is made.",
            "The audit is not a claim that every original label is correct.",
        ],
        "prepared_on": socket.gethostname(),
        "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    audit_hash = write_new(destination / "source_audit.json", source_audit)
    completion = {"status": "staged_awaiting_root_readiness_verification", "dataset_identity": "SAR-ShipDataset",
                  "nc": 1, "counts": counts, "descriptor_sha256": descriptor_hash,
                  "manifest_sha256": manifest_hash, "source_audit_sha256": audit_hash,
                  "readiness_written": False}
    write_new(destination / "stage_complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
