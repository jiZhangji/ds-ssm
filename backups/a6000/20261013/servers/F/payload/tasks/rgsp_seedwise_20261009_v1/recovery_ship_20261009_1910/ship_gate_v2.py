#!/usr/bin/env python3
"""Independent, fail-closed SAR-ShipDataset staging/publication gate (v2).

Server-only and standard-library-only.  Does not modify the frozen plan, runner,
archive auditor, source labels, source split, or original staging script.  A
reviewed all-image archive audit is required; decoded-pixel identities are taken
from that SHA-bound Pillow audit and are not decoded a second time here.
V2 permits only the exact 969 null source-group names already present in the
fixed source snapshot. It does not infer missing scene IDs or change any data.

--validate-only is read-only (including no lock/evidence/candidate creation).
--stage calls the pinned original staging script, then deeply verifies its copy.
--publish additionally requires explicit reviewed stage and selected manifest
hashes. It validates a fully written/fsynced candidate before atomically linking
it to the live readiness path, without replacing any existing file.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import collections
from contextlib import contextmanager
import datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import types
import zipfile


TASK = Path("/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1")
HELPER_ROOT = TASK / "recovery_ship_20261009_1910"
PYTHON = Path("/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python")
DATASET = TASK / "data/sar_shipdataset"
IDENTITY = "SAR-ShipDataset"
SCHEMA = "rgsp-ship-publication-gate-v2"
SPLITS = ("train", "val", "test")
COUNTS = {"train": 31783, "val": 3972, "test": 3974}
UNKNOWN_COUNTS = {"train": 773, "val": 92, "test": 104}
# SHA256 of canonical({split: ordinal-sorted unique unknown names}).
# The fixed source's original per-split order is already this exact order.
UNKNOWN_NAMES_SHA256 = "3ae853c94d02fc793d8da6f35b75c25d23a90164bf0289131dc5f59240d4d40e"
UNKNOWN_LIST_SHA256 = {
    "train": "9078bc56a143acc1cc8d4747c86a19b72cb84e7614cfbe09b42cb580d6c126d7",
    "val": "4d995d9c09fc19b92ff7f348cabd81c05b92a65dba47018320f6c3de1488765a",
    "test": "8ce32232d29d72780e9c5e0ba5d0fb8906f53d47c62008792f6b49690d460bdc",
}
UNKNOWN_FAMILY_COUNTS = {
    "train": {"newship": 586, "ship": 187},
    "val": {"newship": 66, "ship": 26},
    "test": {"newship": 75, "ship": 29},
}
UNKNOWN_NAME_PATTERN = r"(newship|ship)[0-9]+\.jpg"
PLAN_SHA256 = "9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4"
PLAN_FILE_SHA256 = "37e1e2254643541118e4bab29d11cb93f1483847fc73aa69be8d63be4907cbe0"
SNAPSHOT_SHA256 = "aea74cd1d3a55b750004b72249a8896c13a7751025b5398d8eb2e37098bb9b69"
SOURCE_SHA256 = "a69d2acf8fcf2d90cf691b15c007549002be5036c7155493e5cdcd809f25dc4a"
ARCHIVE_SHA256 = "6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e"
ARCHIVE_SIZE = 426990175
SNAPSHOT_SIZE = 5795637
PINNED = {
    "manifest.py": "ff1f6708e735d5bd91d9c341edfcfdcc714829c5ab5ee0c6110df263d5b6fed7",
    "deferred_data.py": "dbfd4cadcdbe6a7533bfb679ed540d2739c8ee6b961b4558bd5e66e6e05a7fb1",
    "audit_external_archives.py": "772fdada1542a3bc2c644859a6cc82ae76dac150ddf2459a83462e6695473b0b",
    "stage_shipdataset.py": "81e621a130f8f6e66c3785c27a411ebe46dc4cb9733f1299411a35d0352f41f8",
}
LIMITATIONS = [
    "This is a fixed local image split, not a verified official or acquisition-disjoint protocol.",
    "Sensor/polarization/date filename groups are unverified proxies, not recovered scene IDs.",
    "Cross-split exact byte/RGB duplicates are excluded from lower-priority splits only.",
    "Within-split duplicate images and differing source annotations are preserved, not adjudicated.",
    "No near-duplicate or geographic/acquisition-independence claim is made.",
    "The audit is not a claim that every original label is correct.",
]
MANIFEST_FIELDS = ("split", "name", "image_sha256", "label_sha256", "pixel_sha256",
                   "source_group", "label_semantic_sha256", "width", "height")
GROUP_FIELDS = ("split", "name", "source_group", "image_sha256", "label_sha256",
                "label_semantic_sha256", "canonical_labels", "width", "height",
                "gray_mean", "gray_std", "gray_min", "gray_max")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def equal(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def checked_hash(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Expected an explicit lowercase SHA256")
    return value


def object_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON object key: " + key)
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def parse_json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs,
                      parse_constant=reject_constant)


def explicit_path(value, parent=None):
    path = Path(value)
    require(path.is_absolute(), "An explicit absolute server path is required")
    resolved = path.resolve()
    require(resolved == path, "Symlink/non-normalized path is not accepted: " + str(path))
    if parent is not None:
        require(resolved != parent and resolved.is_relative_to(parent),
                "Path is outside the assigned independent directory: " + str(path))
    return path


def regular(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "Expected a regular, non-symlink file: " + str(path))
    return before


def fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def file_record(path, expected=None, keep_bytes=False):
    before = regular(path)
    hasher = hashlib.sha256()
    chunks = [] if keep_bytes else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
            if chunks is not None:
                chunks.append(block)
    after = regular(path)
    require(fingerprint(before) == fingerprint(after), "File changed during read: " + str(path))
    result = {"path": str(path), "sha256": hasher.hexdigest(), "size": after.st_size}
    if expected is not None:
        require(result["sha256"] == checked_hash(expected), "SHA mismatch: " + str(path))
    return result, b"".join(chunks) if chunks is not None else None


def read_json(path, expected=None):
    record, data = file_record(path, expected, keep_bytes=True)
    return parse_json(data), record


def recheck(records):
    for record in records:
        current, _ = file_record(Path(record["path"]), record["sha256"])
        require(current["size"] == record["size"], "File size changed: " + record["path"])


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": digest(payload), "size": len(payload)}


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_pinned_module(name, data):
    # Execute the already hashed source bytes, not a possibly stale .pyc cache.
    module = types.ModuleType(name)
    module.__file__ = str(TASK / (name + ".py"))
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(data, module.__file__, "exec"), module.__dict__)
    return module


def context(args):
    require(os.name == "posix", "This gate is server-only; never run it on the workstation")
    require(socket.gethostname() == args.expected_hostname,
            "Actual server hostname does not match the explicitly assigned host")
    require(Path(sys.executable).absolute() == PYTHON, "Use the existing assigned project Python")
    require(Path(__file__).resolve().parent == HELPER_ROOT, "Deploy at the independent helper path")
    require(TASK.resolve() == TASK, "Campaign root must not be redirected by a symlink")
    plan, plan_file = read_json(TASK / "plan.json", PLAN_FILE_SHA256)
    require(digest(canonical(plan)) == PLAN_SHA256, "Canonical frozen plan SHA mismatch")
    files = [plan_file]
    module_source = {}
    for name, expected in PINNED.items():
        record, data = file_record(TASK / name, expected, keep_bytes=True)
        files.append(record)
        module_source[name] = data
    sys.path.insert(0, str(TASK))
    manifest = load_pinned_module("manifest", module_source["manifest.py"])
    deferred = load_pinned_module("deferred_data", module_source["deferred_data.py"])
    require(Path(manifest.__file__).resolve() == TASK / "manifest.py", "Wrong manifest import")
    require(Path(deferred.__file__).resolve() == TASK / "deferred_data.py", "Wrong deferred import")
    manifest.verify_code(plan)
    entry = plan["datasets"]["sar_shipdataset"]
    equal(entry, {
        "audit": str(DATASET / "source_audit.json"), "deferred": True,
        "descriptor": str(DATASET / "dataset.json"), "expected_identity": IDENTITY,
        "expected_nc": 1, "protocol": "ssdd", "readiness": str(DATASET / "readiness.json"),
    }, "Frozen deferred ShipDataset declaration differs")
    own, _ = file_record(Path(__file__).resolve())
    files.append(own)
    return {"plan": plan, "entry": entry, "manifest": manifest, "deferred": deferred,
            "files": files, "helper": own,
            "paths": manifest.validate_deferred_declaration(entry)}


def source_snapshot(audit):
    provenance = audit["provenance"]
    path = explicit_path(provenance["snapshot_archive"], TASK / "external_inputs")
    record, _ = file_record(path, SNAPSHOT_SHA256)
    require(record["size"] == SNAPSHOT_SIZE, "Source snapshot size differs")
    require(provenance["snapshot_archive_sha256"] == SNAPSHOT_SHA256,
            "Archive audit cites a different source snapshot")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Source snapshot has repeated ZIP members")
        meta_names = [name for name in names if name.endswith("/source_snapshot.json")]
        source_names = [name for name in names if name.endswith("/sar_shipdataset_source.json")]
        require(len(meta_names) == len(source_names) == 1, "Ambiguous source snapshot payload")
        metadata_bytes = archive.read(meta_names[0])
        source_bytes = archive.read(source_names[0])
    metadata, source = parse_json(metadata_bytes), parse_json(source_bytes)
    require(digest(source_bytes) == SOURCE_SHA256, "Ship source payload SHA differs")
    require(metadata["datasets"]["sar_shipdataset"]["sha256"] == SOURCE_SHA256,
            "Source snapshot metadata SHA differs")
    equal(provenance, {
        "snapshot_archive": str(path), "snapshot_archive_sha256": SNAPSHOT_SHA256,
        "snapshot_metadata_sha256": digest(metadata_bytes), "source_manifest_sha256": SOURCE_SHA256,
        "source_host": metadata["hostname"], "source_audited_utc": source["audited_utc"],
    }, "Source provenance does not match the fixed snapshot")
    equal(source["counts"], COUNTS, "Original source split counts differ")
    equal(source["names"], ["ship"], "Original source class mapping differs")
    equal(source["nc"], 1, "Original source class count differs")
    equal(source["dataset_identity"], IDENTITY, "Original source identity differs")
    for split in SPLITS:
        equal(source["seed2026_membership_check"]["splits"][split],
              {"equal": True, "only_actual": [], "only_expected": []},
              "Source did not retain its fixed seed2026 membership")
    return source, record


def verify_known_unknown_proxies(source, audit):
    """Bind missing proxy metadata to the exact reviewed fixed-source names.

    This is not a blanket exemption for null groups. The source snapshot is
    SHA-pinned by source_snapshot before this function is used in the real gate.
    Exact counts, canonical list hashes, naming families, and order also remain
    independently fixed here, so same-count substitutions are rejected.
    """
    expected = {split: [] for split in SPLITS}
    for row in source["rows"]:
        if row["source_group"] is None:
            split, name = row["split"], row["name"]
            require(split in SPLITS and isinstance(name, str), "Invalid fixed-source unknown proxy row")
            expected[split].append(name)
    counts = {split: len(names) for split, names in expected.items()}
    equal(counts, UNKNOWN_COUNTS, "Fixed-source unknown proxy counts changed")
    families, list_hashes = {}, {}
    for split, names in expected.items():
        require(names == sorted(set(names)), "Fixed-source unknown names are reordered or repeated")
        family_counts = collections.Counter()
        for name in names:
            match = re.fullmatch(UNKNOWN_NAME_PATTERN, name)
            require(match is not None, "Unreviewed null-group filename family: " + name)
            family_counts[match.group(1)] += 1
        families[split] = dict(family_counts)
        list_hashes[split] = digest(canonical(names))
    equal(families, UNKNOWN_FAMILY_COUNTS, "Fixed-source unknown filename-family counts changed")
    equal(list_hashes, UNKNOWN_LIST_SHA256, "Fixed-source unknown per-split names changed")
    names_hash = digest(canonical(expected))
    require(names_hash == UNKNOWN_NAMES_SHA256, "Fixed-source unknown-name set changed")
    require(len({name for names in expected.values() for name in names}) == 969,
            "Fixed-source unknown names overlap between splits")
    equal(audit["group_audit"]["unparsed"], expected,
          "Archive unparsed names differ from the exact ordered fixed-source null-group names")
    return {
        "status": "exact_known_unknown_source_proxies_preserved",
        "source_manifest_sha256": SOURCE_SHA256,
        "counts": counts, "total": 969, "ordered_names_sha256": names_hash,
        "per_split_names_sha256": list_hashes, "name_family_counts": families,
        "name_family_pattern": UNKNOWN_NAME_PATTERN,
        "hash_encoding": "SHA256 of UTF8 compact sorted-key JSON; ordinal unique name lists; no trailing newline",
        "meaning": "Only existing missing sensor/polarization/date proxies in the fixed source are accepted; "
            "null is preserved, not converted into a scene or acquisition ID. No sample is excluded by this check.",
    }


def canonical_labels(data):
    rows = []
    for line in data.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 5, "Malformed original YOLO label")
        values = [Decimal(field) for field in fields]
        require(all(value.is_finite() for value in values), "Nonfinite original YOLO label")
        cls, x, y, width, height = values
        require(cls == 0 and all(0 <= value <= 1 for value in values[1:])
                and width > 0 and height > 0, "Invalid original ShipDataset annotation")
        require(x - width / 2 >= Decimal("-0.000001")
                and x + width / 2 <= Decimal("1.000001")
                and y - height / 2 >= Decimal("-0.000001")
                and y + height / 2 <= Decimal("1.000001"),
                "Unexpected out-of-image ShipDataset annotation; review before staging")
        rows.append(tuple(format(value.normalize(), "f") if value else "0" for value in values))
    require(rows and len(rows) == len(set(rows)), "Unexpected empty or repeated original annotations")
    ordered = sorted(rows)
    return ordered, digest(json.dumps(ordered, separators=(",", ":")).encode())


def groups_for(rows, key):
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row[key]].append(row)
    groups = []
    removals = {split: [] for split in SPLITS}
    pairs = collections.Counter()
    signatures = collections.Counter()
    for content, members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        splits = sorted({row["split"] for row in members})
        signatures["/".join(splits)] += 1
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1:]:
                pairs[left + "/" + right] += int(left in splits and right in splits)
        # Preserve all members in the highest-priority represented split.
        highest = next(split for split in ("test", "val", "train") if split in splits)
        for row in members:
            if row["split"] != highest:
                removals[row["split"]].append(row["name"])
        groups.append({
            key: content, "splits": splits,
            "label_bytes_differ": len({row["label_sha256"] for row in members}) > 1,
            "label_semantics_differ": len({row["label_semantic_sha256"] for row in members}) > 1,
            "members": [{field: row[field] for field in GROUP_FIELDS} for row in members],
        })
    summary = {
        "key": key, "unique_content": len(buckets), "duplicate_groups": len(groups),
        "cross_split_groups": sum(len(group["splits"]) > 1 for group in groups),
        "semantic_disagreement_groups": sum(group["label_semantics_differ"] for group in groups),
        "test_internal_duplicate_groups": sum(
            sum(row["split"] == "test" for row in group["members"]) > 1 for group in groups),
        "test_internal_semantic_disagreement_groups": sum(
            len({row["label_semantic_sha256"] for row in group["members"]
                 if row["split"] == "test"}) > 1 for group in groups),
        "split_signatures": dict(signatures), "cross_split_pairs": dict(pairs),
        "candidate_exclusions_not_applied": {split: sorted(names) for split, names in removals.items()},
        "candidate_exclusion_counts": {split: len(names) for split, names in removals.items()},
    }
    return summary, groups


def compact_groups(rows, key):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    duplicate_groups = [members for members in grouped.values() if len(members) > 1]
    return {
        "duplicate_groups": len(duplicate_groups),
        "cross_split_groups": sum(len({row["split"] for row in group}) > 1 for group in duplicate_groups),
        "same_split_semantic_disagreement_groups": {
            split: sum(len({row["label_semantic_sha256"] for row in group if row["split"] == split}) > 1
                       for group in duplicate_groups) for split in SPLITS},
    }


def full_audit_gate(args):
    audit_path = explicit_path(args.audit, TASK / "external_inputs/audits")
    audit, audit_file = read_json(audit_path, checked_hash(args.reviewed_audit_sha256))
    for field, expected in {
        "schema": "rgsp-external-archive-audit-v1", "status": "audit_only_not_ready",
        "dataset_key": "sar_shipdataset", "dataset_identity": IDENTITY, "nc": 1,
        "names": ["ship"], "counts": COUNTS, "no_data_changes": True,
        "no_split_changes": True, "source_archive_image_label_mismatches": [],
        "audit_script_sha256": PINNED["audit_external_archives.py"],
    }.items():
        equal(audit[field], expected, "Reviewed archive audit differs: " + field)
    equal(audit["decoder"]["implementation"],
          "Pillow stored raster converted to RGB; no EXIF transpose", "Unreviewed pixel decoder semantics")
    equal(audit["decoder"]["exif_orientation_counts"], {"1": 39729}, "Unexpected image orientations")
    for field in ("invalid_normalized_box", "box_edge_outside_image", "duplicate_rows"):
        equal(audit["label_stats"].get(field, 0), 0, "Unresolved source annotation issue: " + field)
    source, snapshot_file = source_snapshot(audit)
    unknown_metadata = verify_known_unknown_proxies(source, audit)
    archive_path = explicit_path(audit["archive"]["path"], TASK / "external_inputs/raw_archives")
    archive_file, _ = file_record(archive_path, ARCHIVE_SHA256)
    require(archive_file["size"] == ARCHIVE_SIZE, "Original source ZIP is incomplete")
    equal(audit["archive"]["sha256"], ARCHIVE_SHA256, "Audit source archive SHA differs")
    equal(audit["archive"]["size"], ARCHIVE_SIZE, "Audit source archive size differs")
    equal(audit["archive"]["crc"], "passed_all_members", "Full archive CRC audit was not completed")
    equal(audit["archive"]["members"], 79459, "Unexpected archive member count")
    rows, rows_file = read_json(audit_path.parent / "audited_rows.json", audit["rows_sha256"])
    require(isinstance(rows, list) and len(rows) == 39729, "All 39729 audited images are required")
    originals = {(row["split"], row["name"]): row for row in source["rows"]}
    require(len(originals) == len(source["rows"]) == 39729, "Duplicate/missing original source rows")
    seen, observed_counts, observed_boxes = set(), collections.Counter(), collections.Counter()
    raw_root = explicit_path(str(audit_path.parent / "extracted/ship_dataset_v0"), audit_path.parent)
    for row in rows:
        split, name = row["split"], row["name"]
        require(split in SPLITS and Path(name).name == name and name.endswith(".jpg"), "Unsafe audit row")
        key = (split, name)
        require(key in originals and key not in seen, "Audited membership differs from fixed source")
        seen.add(key)
        original = originals[key]
        for field in ("image_sha256", "label_sha256", "pixel_sha256", "label_semantic_sha256"):
            checked_hash(row[field])
        for field in ("image_sha256", "label_sha256", "source_group"):
            equal(row[field], original[field], "Original source binding differs: " + name + " / " + field)
        require(type(row["width"]) is int and type(row["height"]) is int
                and row["width"] > 0 and row["height"] > 0, "Invalid decoded dimensions")
        equal(row["exif_orientation"], 1, "Unexpected row orientation")
        image_path = explicit_path(row["source_image"], raw_root)
        require(image_path == raw_root / name, "Unexpected extracted image path")
        file_record(image_path, row["image_sha256"])
        _, label_bytes = file_record(image_path.with_suffix(".txt"), row["label_sha256"], keep_bytes=True)
        label_rows, semantic_hash = canonical_labels(label_bytes)
        equal(label_rows, row["canonical_labels"], "Audited canonical labels differ: " + name)
        require(semantic_hash == row["label_semantic_sha256"], "Audited semantic label SHA differs: " + name)
        equal(row["class_box_counts"], {"0": len(label_rows)}, "Audited class-box counts differ")
        equal(original["label_boxes"], len(label_rows), "Original source annotation count differs")
        observed_counts[split] += 1
        observed_boxes[split] += len(label_rows)
    equal(dict(observed_counts), COUNTS, "Audited split counts differ")
    require(seen == set(originals), "Audited full-source coverage differs")
    equal(dict(observed_boxes), {split: source["split_stats"][split]["label_stats"]["boxes"]
                                 for split in SPLITS}, "Original source box totals differ")
    expected_raw = {name for _, name in seen} | {Path(name).stem + ".txt" for _, name in seen}
    equal(sorted(path.name for path in raw_root.iterdir()), sorted(expected_raw),
          "Extracted source contains extra/missing files")
    files = [audit_file, snapshot_file, archive_file, rows_file]
    summaries = {}
    for kind, key in (("byte", "image_sha256"), ("pixel", "pixel_sha256")):
        summary, groups = groups_for(rows, key)
        equal(summary, audit[kind + "_duplicates"], "Independent " + kind + " duplicate-summary recomputation differs")
        recorded_groups, group_file = read_json(audit_path.parent / (kind + "_groups.json"),
                                               audit[kind + "_groups_sha256"])
        equal(groups, recorded_groups, "Independent " + kind + " duplicate-group members differ")
        require(summary["test_internal_semantic_disagreement_groups"] == 0,
                "STOP: original test has " + kind + "-identical images with contradictory annotations; "
                "do not stage/publish, relabel, drop test members, or signal failed readiness")
        files.append(group_file)
        summaries[kind] = summary
    byte_excluded = {split: set(summaries["byte"]["candidate_exclusions_not_applied"][split]) for split in SPLITS}
    pixel_excluded = {split: set(summaries["pixel"]["candidate_exclusions_not_applied"][split]) for split in SPLITS}
    excluded = {split: byte_excluded[split] | pixel_excluded[split] for split in SPLITS}
    require(not excluded["test"], "Original test exclusions are not authorized")
    require(all(byte_excluded[split] <= pixel_excluded[split] for split in SPLITS),
            "Decoded equality failed to cover byte equality; decoder review required")
    kept = [row for row in rows if row["name"] not in excluded[row["split"]]]
    removed = [row for row in rows if row["name"] in excluded[row["split"]]]
    counts = dict(collections.Counter(row["split"] for row in kept))
    require(set(counts) == set(SPLITS) and all(value > 0 for value in counts.values())
            and counts["test"] == 3974, "Candidate retained membership is invalid")
    for key in ("image_sha256", "pixel_sha256"):
        require(compact_groups(kept, key)["cross_split_groups"] == 0,
                "Candidate retained copy still has cross-split " + key + " overlap")
    original_test = {(row["name"], row["image_sha256"], row["label_sha256"])
                     for row in source["rows"] if row["split"] == "test"}
    retained_test = {(row["name"], row["image_sha256"], row["label_sha256"])
                     for row in kept if row["split"] == "test"}
    require(original_test == retained_test and len(retained_test) == 3974, "Original test is not preserved")
    predicted_manifest = [{field: row[field] for field in MANIFEST_FIELDS} for row in kept]
    predicted_manifest_sha256 = digest((json.dumps(
        predicted_manifest, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8"))
    if args.expected_manifest_sha256 is not None:
        require(predicted_manifest_sha256 == checked_hash(args.expected_manifest_sha256),
                "Audited candidate differs from the operator-selected cross-host manifest; no stage created")
    return {"audit": audit, "audit_path": audit_path, "files": files, "kept": kept,
            "removed": removed, "counts": counts, "byte_excluded": byte_excluded,
            "pixel_excluded": pixel_excluded, "excluded": excluded,
            "summary": {"status": "audit_gate_verified", "archive_audit": audit_file,
                        "counts": counts, "original_counts": COUNTS,
                        "unknown_source_group_metadata": unknown_metadata,
                        "predicted_manifest_sha256": predicted_manifest_sha256,
                        "byte_exclusion_counts": {split: len(byte_excluded[split]) for split in SPLITS},
                        "pixel_exclusion_counts": {split: len(pixel_excluded[split]) for split in SPLITS},
                        "union_exclusion_counts": {split: len(excluded[split]) for split in SPLITS},
                        "test_internal_duplicate_groups": {
                            kind: summaries[kind]["test_internal_duplicate_groups"] for kind in summaries},
                        "test_internal_semantic_disagreement_groups": {"byte": 0, "pixel": 0},
                        "original_test_preserved": True,
                        "pixel_verification_scope": "Duplicate rules independently recomputed from the reviewed "
                            "SHA-bound full Pillow audit; every corresponding original image/label byte hash "
                            "was rechecked, not a second decoded-pixel implementation.",
                        "limitations": LIMITATIONS}}


def verify_stage(args, ctx, gate):
    require(DATASET.is_dir() and not DATASET.is_symlink(), "No completed independent staged dataset")
    descriptor, descriptor_file = read_json(DATASET / "dataset.json")
    audit, audit_file = read_json(DATASET / "source_audit.json")
    completion, completion_file = read_json(DATASET / "stage_complete.json")
    rows, manifest_file = read_json(DATASET / "data_content.json")
    exclusions, exclusions_file = read_json(DATASET / "exclusions.json")
    yaml_file, yaml_bytes = file_record(DATASET / "data.yaml", keep_bytes=True)
    expected_rows = [{field: row[field] for field in MANIFEST_FIELDS} for row in gate["kept"]]
    equal(rows, expected_rows, "Staged manifest does not exactly preserve the authorized kept row order/content")
    require(manifest_file["sha256"] == gate["summary"]["predicted_manifest_sha256"],
            "Staged manifest bytes differ from the pinned stage serializer and audited kept rows")
    equal(descriptor, {
        "protocol": "ssdd", "root": str(DATASET / "data"), "yaml": str(DATASET / "data.yaml"),
        "yaml_sha256": yaml_file["sha256"], "manifest": str(DATASET / "data_content.json"),
        "manifest_sha256": manifest_file["sha256"], "nc": 1, "names": ["ship"], "counts": gate["counts"],
    }, "Staged descriptor differs from the exact selected copy")
    expected_yaml = (f"path: {DATASET / 'data'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                     "nc: 1\nnames:\n  0: ship\n").encode()
    require(yaml_bytes == expected_yaml, "Staged YAML path/class order differs")
    expected_exclusions = [{
        "split": row["split"], "name": row["name"], "image_sha256": row["image_sha256"],
        "pixel_sha256": row["pixel_sha256"], "label_sha256": row["label_sha256"],
        "byte_rule": row["name"] in gate["byte_excluded"][row["split"]],
        "decoded_pixel_rule": row["name"] in gate["pixel_excluded"][row["split"]],
    } for row in gate["removed"]]
    equal(exclusions, expected_exclusions, "Staged exclusions ledger differs from the authorized exact union")
    expected_audit = {
        "status": "verified", "dataset_identity": IDENTITY, "nc": 1, "names": ["ship"],
        "manifest_sha256": manifest_file["sha256"], "counts": gate["counts"], "original_counts": COUNTS,
        "protocol": "existing seed2026 split with test-first byte/decoded-RGB cross-split decontamination",
        "approval_scope": "Parent operator authorized independent-copy exclusions on 2026-10-09; "
            "preserve original test and all remaining label bytes; no resplit search or relabeling.",
        "partition_priority": ["test", "val", "train"], "source_archive": gate["audit"]["archive"],
        "source_provenance": gate["audit"]["provenance"], "archive_audit_path": str(gate["audit_path"]),
        "archive_audit_sha256": args.reviewed_audit_sha256,
        "staging_script_sha256": PINNED["stage_shipdataset.py"],
        "exclusions_path": str(DATASET / "exclusions.json"), "exclusions_sha256": exclusions_file["sha256"],
        "exclusion_counts": {split: len(gate["excluded"][split]) for split in SPLITS},
        "byte_exclusion_counts": {split: len(gate["byte_excluded"][split]) for split in SPLITS},
        "pixel_exclusion_counts": {split: len(gate["pixel_excluded"][split]) for split in SPLITS},
        "pixel_exclusions_cover_byte_exclusions": True,
        "retained_byte_duplicates": compact_groups(gate["kept"], "image_sha256"),
        "retained_pixel_duplicates": compact_groups(gate["kept"], "pixel_sha256"),
        "original_test_3974_names_and_image_label_hashes_unchanged": True,
        "source_files_modified_or_deleted": False, "new_training_seeds_share_this_manifest": True,
        "decoder": gate["audit"]["decoder"], "filename_group_audit": gate["audit"]["group_audit"],
        "limitations": LIMITATIONS,
    }
    require(set(audit) == set(expected_audit) | {"prepared_on", "prepared_utc"}, "Unexpected source-audit fields")
    for field, value in expected_audit.items():
        equal(audit[field], value, "Staged source audit differs: " + field)
    require(audit["prepared_on"] in {"a6000-c", "powerleader-a6000-f"}, "Unexpected stage-producing host")
    require(isinstance(audit["prepared_utc"], str)
            and datetime.datetime.fromisoformat(audit["prepared_utc"]).tzinfo is not None,
            "Stage-production timestamp must be timezone-aware")
    equal(completion, {
        "status": "staged_awaiting_root_readiness_verification", "dataset_identity": IDENTITY,
        "nc": 1, "counts": gate["counts"], "descriptor_sha256": descriptor_file["sha256"],
        "manifest_sha256": manifest_file["sha256"], "source_audit_sha256": audit_file["sha256"],
        "readiness_written": False,
    }, "Stage completion does not bind the exact staged artifacts")
    if args.reviewed_stage_complete_sha256 is not None:
        require(completion_file["sha256"] == checked_hash(args.reviewed_stage_complete_sha256),
                "Stage completion differs from the operator-reviewed artifact")
    if args.expected_manifest_sha256 is not None:
        require(manifest_file["sha256"] == checked_hash(args.expected_manifest_sha256),
                "Staged content differs from the operator-selected cross-host manifest")
    ctx["manifest"].verify_dataset(descriptor)
    # Frozen verifier checks exact images and all referenced labels, but not extra
    # label filenames; explicitly verify both sets and every non-symlink file.
    for split in SPLITS:
        expected_images = {row["name"] for row in rows if row["split"] == split}
        expected_labels = {Path(name).stem + ".txt" for name in expected_images}
        for kind, names in (("images", expected_images), ("labels", expected_labels)):
            folder = DATASET / "data" / kind / split
            require(folder.resolve() == folder, "Staged content directory is redirected")
            equal(sorted(path.name for path in folder.iterdir()), sorted(names), "Staged exact file set differs")
            for name in names:
                regular(folder / name)
    for key in ("image_sha256", "pixel_sha256"):
        summary = compact_groups(rows, key)
        require(summary["cross_split_groups"] == 0
                and summary["same_split_semantic_disagreement_groups"]["test"] == 0,
                "Staged copy has forbidden cross-split overlap/test semantic conflict")
    files = [descriptor_file, audit_file, completion_file, manifest_file, exclusions_file, yaml_file]
    recheck(files)
    return {"descriptor": descriptor, "source_audit": audit, "files": files,
            "summary": {"status": "staged_verified_not_published", "counts": gate["counts"],
                        "descriptor": descriptor_file, "source_audit": audit_file,
                        "stage_complete": completion_file, "manifest": manifest_file,
                        "exclusions": exclusions_file, "yaml": yaml_file,
                        "original_test_preserved": True,
                        "readiness_present": ctx["paths"]["readiness"].exists()}}


@contextmanager
def writer_lock():
    import fcntl
    # Only the two explicit mutating modes open this lock file.
    lock_path = HELPER_ROOT / "ship_gate.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def new_evidence(args):
    path = explicit_path(args.evidence_dir, HELPER_ROOT / "evidence")
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_live(ctx, gate, staged):
    live = ctx["paths"]["readiness"]
    require(not live.is_symlink(), "Live readiness is a symlink; inspect it without replacement")
    if not live.exists():
        return None
    ready, record = read_json(live)
    expected = {"status": "verified_ready", "dataset_identity": IDENTITY, "nc": 1,
                "descriptor_sha256": staged["summary"]["descriptor"]["sha256"],
                "manifest_sha256": staged["summary"]["manifest"]["sha256"],
                "source_audit_sha256": staged["summary"]["source_audit"]["sha256"]}
    for field, value in expected.items():
        equal(ready[field], value, "Existing live readiness differs; it will not be overwritten")
    equal(ready.get("schema"), SCHEMA, "Existing readiness was not published by this reviewed gate")
    equal(ready.get("plan_sha256"), PLAN_SHA256, "Existing readiness has a different frozen plan binding")
    equal(ready.get("archive_audit_sha256"), gate["summary"]["archive_audit"]["sha256"],
          "Existing readiness has a different reviewed audit binding")
    equal(ready.get("stage_complete_sha256"), staged["summary"]["stage_complete"]["sha256"],
          "Existing readiness has a different reviewed staging binding")
    equal(ready.get("helper_sha256"), ctx["helper"]["sha256"],
          "Existing readiness uses a different gate version; inspect without replacement")
    validation_ref = ready["gate_validation"]
    validation_path = explicit_path(validation_ref["path"], HELPER_ROOT / "evidence")
    validation, validation_file = read_json(validation_path, validation_ref["sha256"])
    equal(validation_file, validation_ref, "Existing readiness validation record size/path differs")
    equal(validation["status"], "candidate_inputs_verified", "Existing readiness lacks gate validation")
    equal(validation["plan_sha256"], PLAN_SHA256, "Existing gate validation plan differs")
    equal(validation["staged"]["stage_complete"]["sha256"],
          staged["summary"]["stage_complete"]["sha256"], "Existing gate validation stage binding differs")
    ctx["deferred"].resolve_ready(ctx["entry"], ctx["paths"], ready, record)
    return record


def run_stage(args, ctx, gate):
    require(not DATASET.exists() and not DATASET.is_symlink(),
            "Staging destination already exists; preserve it and inspect, never retry over it")
    evidence = new_evidence(args)
    invocation = {"schema": SCHEMA, "mode": "stage", "started_utc": utc(),
                  "hostname": socket.gethostname(), "helper": ctx["helper"],
                  "plan_sha256": PLAN_SHA256, "audit_gate": gate["summary"],
                  "readiness_will_not_be_published": True}
    write_new(evidence / "invocation.json", invocation)
    command = [str(PYTHON), "-B", str(TASK / "stage_shipdataset.py"),
               "--audit", str(gate["audit_path"]), "--reviewed-audit-sha256", args.reviewed_audit_sha256]
    recheck(ctx["files"] + gate["files"])
    with (evidence / "stage.stdout.txt").open("xb") as stdout, (evidence / "stage.stderr.txt").open("xb") as stderr:
        process = subprocess.run(command, cwd=str(TASK), stdout=stdout, stderr=stderr, check=False)
        for stream in (stdout, stderr):
            stream.flush()
            os.fsync(stream.fileno())
    write_new(evidence / "stage_process.json", {"argv": command, "returncode": process.returncode,
                                               "finished_utc": utc()})
    require(process.returncode == 0, "Original stage failed; independent outputs/logs preserved; no readiness written")
    staged = verify_stage(args, ctx, gate)
    require(not ctx["paths"]["readiness"].exists(), "Unexpected readiness appeared during staging; inspect it")
    result = {"schema": SCHEMA, "status": "staged_verified_not_published", "finished_utc": utc(),
              "hostname": socket.gethostname(), "helper": ctx["helper"], "plan_sha256": PLAN_SHA256,
              "audit_gate": gate["summary"], "staged": staged["summary"],
              "source_evidence": ctx["files"] + gate["files"], "readiness_written": False}
    result_file = write_new(evidence / "stage_gate_complete.json", result)
    sync_directory(evidence)
    return {"status": result["status"], "evidence": result_file, "staged": staged["summary"]}


def publish(args, ctx, gate, staged):
    existing = validate_live(ctx, gate, staged)
    if existing is not None:
        return {"status": "already_published", "readiness": existing, "readiness_written": False}
    evidence = new_evidence(args)
    validation = {"schema": SCHEMA, "status": "candidate_inputs_verified", "verified_utc": utc(),
                  "hostname": socket.gethostname(), "helper": ctx["helper"], "plan_sha256": PLAN_SHA256,
                  "audit_gate": gate["summary"], "staged": staged["summary"],
                  "source_evidence": ctx["files"] + gate["files"]}
    validation_file = write_new(evidence / "publication_validation.json", validation)
    ready = {"schema": SCHEMA, "status": "verified_ready", "dataset_identity": IDENTITY, "nc": 1,
             "descriptor_sha256": staged["summary"]["descriptor"]["sha256"],
             "manifest_sha256": staged["summary"]["manifest"]["sha256"],
             "source_audit_sha256": staged["summary"]["source_audit"]["sha256"],
             "plan_sha256": PLAN_SHA256, "archive_audit_sha256": args.reviewed_audit_sha256,
             "stage_complete_sha256": staged["summary"]["stage_complete"]["sha256"],
             "gate_validation": validation_file, "helper_sha256": ctx["helper"]["sha256"]}
    candidate_path = evidence / "readiness.candidate.json"
    candidate_file = write_new(candidate_path, ready)
    # resolve_ready accepts the explicit candidate artifact for its stable-file
    # recheck. The frozen declaration still points to the untouched live path.
    _, candidate_dataset_audit, candidate_provenance = ctx["deferred"].resolve_ready(
        ctx["entry"], ctx["paths"], ready, candidate_file)
    candidate_validation = write_new(evidence / "candidate_validation.json", {
        "status": "verified_candidate_not_published", "candidate": candidate_file,
        "dataset_audit": candidate_dataset_audit, "provenance": candidate_provenance,
        "verified_utc": utc(), "live_path": str(ctx["paths"]["readiness"]),
    })
    recheck(ctx["files"] + gate["files"] + staged["files"]
            + [ctx["helper"], validation_file, candidate_file, candidate_validation])
    ctx["manifest"].verify_code(ctx["plan"])
    live_path = ctx["paths"]["readiness"]
    require(candidate_path.stat().st_dev == live_path.parent.stat().st_dev,
            "Atomic no-clobber publication requires the same filesystem")
    require(not live_path.exists() and not live_path.is_symlink(),
            "Live readiness appeared after validation; it will not be replaced")
    sync_directory(evidence)
    # Atomic create: readers see either no live entry or a complete fsynced JSON.
    # Keep the evidence candidate hard link; do not remove or replace any file.
    os.link(candidate_path, live_path, follow_symlinks=False)
    sync_directory(live_path.parent)
    live_record, _ = file_record(live_path, candidate_file["sha256"])
    require(os.path.samefile(candidate_path, live_path), "Publication link identity changed unexpectedly")
    result = {"schema": SCHEMA, "status": "published", "published_utc": utc(),
              "hostname": socket.gethostname(), "readiness": live_record,
              "candidate": candidate_file, "candidate_validation": candidate_validation,
              "stage_complete": staged["summary"]["stage_complete"],
              "manifest": staged["summary"]["manifest"], "publication": "atomic_no_clobber_hard_link",
              "readiness_written": True, "existing_files_overwritten": False}
    result_file = write_new(evidence / "publication_complete.json", result)
    sync_directory(evidence)
    return {"status": "published", "evidence": result_file, "readiness": live_record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--stage", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-hostname", required=True, choices=("a6000-c", "powerleader-a6000-f"))
    parser.add_argument("--audit", required=True)
    parser.add_argument("--reviewed-audit-sha256", required=True)
    parser.add_argument("--reviewed-stage-complete-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--evidence-dir")
    args = parser.parse_args()
    if args.validate_only:
        require(args.evidence_dir is None, "Read-only validation must not request an evidence directory")
    else:
        require(args.evidence_dir is not None, "Explicit mutating modes require a fresh evidence directory")
    if args.publish:
        require(args.reviewed_stage_complete_sha256 is not None and args.expected_manifest_sha256 is not None,
                "Publication requires explicit reviewed stage_complete and selected cross-host manifest SHA256")
    if args.stage:
        require(args.reviewed_stage_complete_sha256 is None,
                "A new stage cannot cite an existing stage_complete; inspect that copy read-only")
    for value in (args.reviewed_audit_sha256, args.reviewed_stage_complete_sha256,
                  args.expected_manifest_sha256):
        if value is not None:
            checked_hash(value)
    ctx = context(args)

    def action():
        gate = full_audit_gate(args)
        if args.stage:
            return run_stage(args, ctx, gate)
        staged = verify_stage(args, ctx, gate) if DATASET.exists() else None
        if args.publish:
            require(staged is not None, "No verified staged copy; readiness will not be published")
            return publish(args, ctx, gate, staged)
        live = validate_live(ctx, gate, staged) if staged is not None else None
        recheck(ctx["files"] + gate["files"] + (staged["files"] if staged is not None else []))
        ctx["manifest"].verify_code(ctx["plan"])
        return {"schema": SCHEMA, "status": "validated_only", "verified_utc": utc(),
                "hostname": socket.gethostname(), "helper": ctx["helper"], "plan_sha256": PLAN_SHA256,
                "audit_gate": gate["summary"], "staged": staged["summary"] if staged else None,
                "live_readiness": live, "files_written": False, "readiness_written": False}

    if args.validate_only:
        result = action()
    else:
        with writer_lock():
            result = action()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No failed/invalid/rejected marker is ever written into a dataset.
        print(json.dumps({"schema": SCHEMA, "status": "gate_stopped",
                          "error_type": type(exc).__name__, "error": str(exc),
                          "live_readiness_present_observed": (DATASET / "readiness.json").exists(),
                          "instruction": "Inspect preserved evidence read-only; do not overwrite/retry blindly. "
                              "If live readiness exists, an atomic publication may already have occurred before "
                              "a completion-record IO failure; validate it instead of replacing it.",
                          "finished_utc": utc()}, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(2)
