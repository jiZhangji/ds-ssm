#!/usr/bin/env python3
"""CPU-only archive/image/label audit in a fresh server task directory.

Never changes source data or a split. Never emits readiness. The output supports
an explicit, later protocol decision, including reporting conflicting test labels.
"""

import argparse
import collections
import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import zipfile

import PIL
from PIL import Image, ImageStat


TASK = Path("/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1")
SPLITS = ("train", "val", "test")
ARCHIVES = {
    "sar_shipdataset": ("ship_dataset_v0.zip", 426990175,
        "6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e"),
    "sar_aircraft10": ("SAR_AIRcraft.zip", 1598574534,
        "32cf1dd8d3a11552916227304bed9e8f86b8c1b8fc4c7b1d24be916d13147069"),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input changed during read: " + str(path))
    return h.hexdigest()


def write_new(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return sha(data)


def safe_members(archive):
    members = archive.infolist()
    names = set()
    total = 0
    for member in members:
        name = member.filename
        path = PurePosixPath(name)
        if (not name or "\\" in name or path.is_absolute() or
                ".." in path.parts or ":" in name or name in names):
            raise ValueError("Unsafe or repeated archive member: " + name)
        if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
            raise ValueError("Symlink/encrypted member not accepted: " + name)
        names.add(name)
        total += member.file_size
    if len(members) > 200000 or total > 10 * 1024 ** 3:
        raise ValueError("Archive exceeds the independently observed dataset size")
    return members, total


def source_snapshot(path, expected_hash, dataset):
    if file_hash(path) != expected_hash:
        raise ValueError("Source snapshot ZIP SHA mismatch")
    with zipfile.ZipFile(path) as archive:
        members, _ = safe_members(archive)
        names = [member.filename for member in members]
        metadata_name = [n for n in names if n.endswith("/source_snapshot.json")]
        source_name = [n for n in names if n.endswith("/" + dataset + "_source.json")]
        if len(metadata_name) != 1 or len(source_name) != 1:
            raise ValueError("Ambiguous source snapshot filenames")
        metadata_bytes = archive.read(metadata_name[0])
        data = archive.read(source_name[0])
    metadata = json.loads(metadata_bytes)
    expected = metadata["datasets"][dataset]["sha256"]
    if sha(data) != expected:
        raise ValueError("Source snapshot payload SHA mismatch")
    source = json.loads(data)
    return source, {"snapshot_archive": str(path), "snapshot_archive_sha256": expected_hash,
                    "snapshot_metadata_sha256": sha(metadata_bytes),
                    "source_manifest_sha256": expected,
                    "source_host": metadata["hostname"],
                    "source_audited_utc": source["audited_utc"]}


def canonical_label(data, nc):
    rows = []
    issues = collections.Counter()
    classes = collections.Counter()
    for line in data.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Not a YOLO five-field label: " + line)
        values = [Decimal(s) for s in fields]
        if not all(v.is_finite() for v in values):
            raise ValueError("Nonfinite label: " + line)
        cls = values[0]
        if cls != int(cls) or not 0 <= cls < nc:
            raise ValueError("Invalid class: " + line)
        x, y, w, h = values[1:]
        if not all(0 <= v <= 1 for v in values[1:]) or w <= 0 or h <= 0:
            issues["invalid_normalized_box"] += 1
        if x - w / 2 < Decimal("-0.000001") or x + w / 2 > Decimal("1.000001") or \
                y - h / 2 < Decimal("-0.000001") or y + h / 2 > Decimal("1.000001"):
            issues["box_edge_outside_image"] += 1
        rows.append(tuple(format(v.normalize(), "f") if v else "0" for v in values))
        classes[int(cls)] += 1
    issues["duplicate_rows"] = len(rows) - len(set(rows))
    # Sorted multiset ignores formatting/order, but retains duplicated annotations.
    canonical = sorted(rows)
    return canonical, sha(json.dumps(canonical, separators=(",", ":")).encode()), dict(issues), dict(classes)


def regenerate_air(raw, source):
    payloads = {s: (raw / "annotations" / (s + ".json")).read_bytes() for s in SPLITS}
    for split, payload in payloads.items():
        if sha(payload) != source["annotation_files"][split + ".json"]["sha256"]:
            raise ValueError("Archive COCO differs from source snapshot: " + split)
    data = {s: json.loads(payloads[s]) for s in SPLITS}
    categories = sorted(data["train"]["categories"], key=lambda c: c["id"])
    if len({c["id"] for c in categories}) != len(categories):
        raise ValueError("Duplicate COCO category IDs")
    names = [c["name"] for c in categories]
    if names != source["names"] or len(names) != source["nc"]:
        raise ValueError("Source and archive category names disagree")
    mapping = {c["id"]: i for i, c in enumerate(categories)}
    labels, report, dimensions = {}, {}, {}
    for split in SPLITS:
        coco = data[split]
        if sorted(coco["categories"], key=lambda c: c["id"]) != categories:
            raise ValueError("Inconsistent split categories")
        images = {i["id"]: i for i in coco["images"]}
        if len(images) != len(coco["images"]):
            raise ValueError("Duplicate COCO image IDs")
        seen_names = set()
        annotations = collections.defaultdict(list)
        stats = collections.Counter(total_annotations=len(coco["annotations"]))
        for ann in coco["annotations"]:
            if ann["image_id"] not in images:
                stats["orphan_annotations"] += 1
                continue
            if ann.get("iscrowd", 0) == 1:
                stats["skipped_iscrowd"] += 1
                continue
            annotations[ann["image_id"]].append(ann)
        for info in coco["images"]:
            name = info["file_name"]
            if PurePosixPath(name).name != name or name in seen_names:
                raise ValueError("Unsafe/repeated flat COCO filename: " + name)
            seen_names.add(name)
            w_img, h_img = float(info["width"]), float(info["height"])
            if w_img <= 0 or h_img <= 0:
                raise ValueError("Invalid image size in COCO")
            dimensions[name] = (int(w_img), int(h_img))
            lines = []
            for ann in annotations[info["id"]]:
                if ann["category_id"] not in mapping:
                    stats["skipped_unknown_category"] += 1
                    continue
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    stats["skipped_nonpositive_bbox"] += 1
                    continue
                stats["raw_box_edge_outside_image"] += int(
                    x < 0 or y < 0 or x + w > w_img or y + h > h_img)
                values = [(x + w / 2) / w_img, (y + h / 2) / h_img, w / w_img, h / h_img]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite annotation")
                stats["component_clamped_boxes"] += int(any(v < 0 or v > 1 for v in values))
                values = [min(max(v, 0.0), 1.0) for v in values]
                lines.append(str(mapping[ann["category_id"]]) + " " +
                             " ".join(f"{v:.6f}" for v in values))
            labels[(split, name)] = ("\n".join(lines) + ("\n" if lines else "")).encode()
            stats["written_boxes"] += len(lines)
            stats["empty_images"] += int(not lines)
        report[split] = dict(stats)
    return labels, report, dimensions


def duplicate_audit(rows, key):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    groups = []
    removals = {s: [] for s in SPLITS}
    pairs = collections.Counter()
    signatures = collections.Counter()
    for value, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        splits = sorted({m["split"] for m in members})
        signatures["/".join(splits)] += 1
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                pairs[a + "/" + b] += int(a in splits and b in splits)
        for row in members:
            if row["split"] == "train" and ("val" in splits or "test" in splits) or \
                    row["split"] == "val" and "test" in splits:
                removals[row["split"]].append(row["name"])
        groups.append({key: value, "splits": splits,
                       "label_bytes_differ": len({m["label_sha256"] for m in members}) > 1,
                       "label_semantics_differ": len({m["label_semantic_sha256"] for m in members}) > 1,
                       "members": [{k: m[k] for k in ("split", "name", "source_group",
                           "image_sha256", "label_sha256", "label_semantic_sha256", "canonical_labels",
                           "width", "height", "gray_mean", "gray_std", "gray_min", "gray_max")}
                           for m in members]})
    return {"key": key, "unique_content": len(grouped), "duplicate_groups": len(groups),
            "cross_split_groups": sum(len(g["splits"]) > 1 for g in groups),
            "semantic_disagreement_groups": sum(g["label_semantics_differ"] for g in groups),
            "test_internal_duplicate_groups": sum(
                sum(m["split"] == "test" for m in g["members"]) > 1 for g in groups),
            "test_internal_semantic_disagreement_groups": sum(
                len({m["label_semantic_sha256"] for m in g["members"] if m["split"] == "test"}) > 1
                for g in groups),
            "split_signatures": dict(signatures), "cross_split_pairs": dict(pairs),
            "candidate_exclusions_not_applied": {s: sorted(v) for s, v in removals.items()},
            "candidate_exclusion_counts": {s: len(v) for s, v in removals.items()},
            "groups": groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ARCHIVES, required=True)
    parser.add_argument("--snapshot-zip", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(TASK / "external_inputs/audits"):
        raise ValueError("Audit output must be a new directory inside the authorized task")
    if socket.gethostname() not in {"powerleader-a6000-b", "a6000-c", "powerleader-a6000-f", "powerleader-PT6620W"}:
        raise ValueError("Unexpected audit host")
    source, provenance = source_snapshot(args.snapshot_zip, args.snapshot_sha256, args.dataset)
    filename, size, expected_hash = ARCHIVES[args.dataset]
    archive_path = TASK / "external_inputs/raw_archives" / filename
    if archive_path.stat().st_size != size or file_hash(archive_path) != expected_hash:
        raise ValueError("Archive incomplete or SHA mismatch; no output data created")
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    extract_root = output / "extracted"
    extract_root.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        members, unpacked_size = safe_members(archive)
        print(utc(), "CRC", args.dataset, flush=True)
        failed = archive.testzip()
        if failed is not None:
            raise ValueError("ZIP CRC failure: " + failed)
        archive.extractall(extract_root)
    print(utc(), "extracted", args.dataset, flush=True)
    if args.dataset == "sar_shipdataset":
        raw = extract_root / "ship_dataset_v0"
        labels, conversion, expected_dimensions = None, None, {}
    else:
        candidates = [p.parent.parent for p in extract_root.rglob("train.json")
                      if p.parent.name == "annotations"]
        if len(candidates) != 1:
            raise ValueError("Ambiguous AIRcraft annotation root")
        raw = candidates[0]
        labels, conversion, expected_dimensions = regenerate_air(raw, source)
    rows, mismatches = [], []
    source_names = [r["name"] for r in source["rows"]]
    raw_image_root = raw if labels is None else raw / "images"
    actual_names = {p.name for p in raw_image_root.iterdir() if p.suffix.lower() == ".jpg"}
    if len(set(source_names)) != len(source_names) or set(source_names) != actual_names:
        raise ValueError("Source and archive filenames are not an exact one-to-one match")
    dimensions, label_stats = collections.Counter(), collections.Counter()
    orientations = collections.Counter()
    parsed_groups = {s: set() for s in SPLITS}
    unparsed = {s: [] for s in SPLITS}
    for index, original in enumerate(source["rows"]):
        split, name = original["split"], original["name"]
        image_path = raw_image_root / name
        label = image_path.with_suffix(".txt").read_bytes() if labels is None else labels[(split, name)]
        image_hash, label_hash = file_hash(image_path), sha(label)
        if image_hash != original["image_sha256"] or label_hash != original["label_sha256"]:
            mismatches.append({"split": split, "name": name,
                               "image_equal": image_hash == original["image_sha256"],
                               "label_equal": label_hash == original["label_sha256"]})
        canonical, semantic_hash, issues, class_counts = canonical_label(label, source["nc"])
        label_stats.update(issues)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            original_mode = image.mode
            orientation = int(image.getexif().get(274, 1))
            orientations[orientation] += 1
            rgb = image.convert("RGB")
            pixel_hash = sha((f"RGB:{width}x{height}:".encode()) + rgb.tobytes())
            gray = rgb.convert("L")
            gray_stats = ImageStat.Stat(gray)
            gray_min, gray_max = gray.getextrema()
        if expected_dimensions and (width, height) != expected_dimensions[name]:
            raise ValueError("Decoded image and COCO size disagree: " + name)
        dimensions[f"{width}x{height}/{original_mode}"] += 1
        if args.dataset == "sar_aircraft10":
            match = re.fullmatch(r"(\d+)__(\d+)__(\d+)___(\d+)\.jpg", name)
            group = match.group(1) if match else None
        else:
            match = re.match(r"^(Gao|Sen)_ship_([a-z]+)_0(\d{8})", name)
            group = "/".join(match.groups()) if match else None
        if group is None:
            unparsed[split].append(name)
        else:
            parsed_groups[split].add(group)
        rows.append({"split": split, "name": name, "image_sha256": image_hash,
                     "label_sha256": label_hash, "pixel_sha256": pixel_hash,
                     "label_semantic_sha256": semantic_hash, "canonical_labels": canonical,
                     "class_box_counts": class_counts, "source_group": group,
                     "source_image": str(image_path), "width": width, "height": height,
                     "source_mode": original_mode, "exif_orientation": orientation,
                     "gray_mean": gray_stats.mean[0],
                     "gray_std": gray_stats.stddev[0], "gray_min": gray_min, "gray_max": gray_max})
        if index % 2000 == 0:
            print(utc(), "image audit", args.dataset, index, "/", len(source["rows"]), flush=True)
    byte_report = duplicate_audit(rows, "image_sha256")
    pixel_report = duplicate_audit(rows, "pixel_sha256")
    report = {"schema": "rgsp-external-archive-audit-v1", "status": "audit_only_not_ready",
              "dataset_identity": source["dataset_identity"], "dataset_key": args.dataset,
              "representation": "512x512 overlapping patch derivative" if labels is not None
                  else "flat ship chips with the existing seed2026 random image split",
              "nc": source["nc"], "names": source["names"], "counts": source["counts"],
              "started_utc": started, "finished_utc": utc(), "hostname": socket.gethostname(),
              "audit_script_sha256": file_hash(Path(__file__)), "provenance": provenance,
              "decoder": {"implementation": "Pillow stored raster converted to RGB; no EXIF transpose",
                          "python": sys.version, "pillow": PIL.__version__,
                          "libjpeg": getattr(Image.core, "jpeglib_version", None),
                          "exif_orientation_counts": dict(orientations),
                          "limitation": "Decoded-RGB equality is additional conservative screening, "
                              "not a near-duplicate or acquisition-independence proof."},
              "archive": {"path": str(archive_path), "sha256": expected_hash, "size": size,
                          "crc": "passed_all_members", "members": len(members),
                          "unpacked_size": unpacked_size},
              "source_archive_image_label_mismatches": mismatches,
              "dimensions": dict(dimensions), "label_stats": dict(label_stats),
              "coco_conversion": conversion,
              "group_audit": {"unparsed": unparsed,
                  "counts": {s: len(parsed_groups[s]) for s in SPLITS},
                  "cross_split_intersections": {a + "/" + b: sorted(parsed_groups[a] & parsed_groups[b])
                      for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]},
                  "limitation": "AIR numeric prefixes are original-image IDs, not verified geographic scenes; "
                     "Ship groups are only unverified sensor/polarization/date filename proxies."},
              "byte_duplicates": {k: v for k, v in byte_report.items() if k != "groups"},
              "pixel_duplicates": {k: v for k, v in pixel_report.items() if k != "groups"},
              "gray_summary": {"min_mean": min(r["gray_mean"] for r in rows),
                               "max_mean": max(r["gray_mean"] for r in rows),
                               "constant_images": sum(r["gray_min"] == r["gray_max"] for r in rows)},
              "no_data_changes": True, "no_split_changes": True,
              "readiness_requires": "Explicit protocol review plus frozen stage/hash verification; "
                  "cross-split overlap and any test-internal semantic disagreement must be disclosed."}
    report["rows_sha256"] = write_new(output / "audited_rows.json", rows)
    report["byte_groups_sha256"] = write_new(output / "byte_groups.json", byte_report["groups"])
    report["pixel_groups_sha256"] = write_new(output / "pixel_groups.json", pixel_report["groups"])
    report_sha = write_new(output / "archive_audit.json", report)
    print(json.dumps({"report": str(output / "archive_audit.json"), "sha256": report_sha,
                      "mismatches": len(mismatches), "byte_duplicate_groups": byte_report["duplicate_groups"],
                      "pixel_duplicate_groups": pixel_report["duplicate_groups"],
                      "byte_candidate_exclusions": byte_report["candidate_exclusion_counts"],
                      "pixel_candidate_exclusions": pixel_report["candidate_exclusion_counts"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
