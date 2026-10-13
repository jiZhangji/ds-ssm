#!/usr/bin/env python3
"""Integrity and no-clobber helpers for this new server-only data protocol.

No model package is imported, no source is rewritten, and no existing dataset
directory can be adopted as a prepared result. Execute preparation on an
explicitly selected Linux server, not on the Windows workstation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "rgsp-unified-data-v1"
SPLIT_SEED = 20261012
CAMPAIGN = "rgsp_unified_a6000_20261012_v2"
MARKER_NAME = ".rgsp_taskdata_identity.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_new(path, payload, allow_identical=False):
    """Publish one complete file with link(2), never replace an old path.

    The only removed file is the private temporary created by this invocation.
    Artifacts in interrupted/failed dataset directories are deliberately kept.
    """
    path = Path(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"An existing real output directory is required: {path.parent}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            if not (allow_identical and path.is_file() and not path.is_symlink()
                    and path.read_bytes() == payload):
                raise FileExistsError(f"No-clobber output already exists: {path}")
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_new(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    _write_bytes_new(path, payload)


def write_text_new(path, value):
    _write_bytes_new(path, value.encode("utf-8"))


def ensure_taskdata_root(raw, readonly_roots=()):
    """Accept only a new/empty taskdata root or this protocol's own marker.

    A pre-existing populated directory without the exact marker is rejected.
    Source/output ancestors must be disjoint; symlinked output paths and broad
    directory names cannot become an output target. No --force mode exists.
    """
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Data preparation is Linux-server-only; no local fallback is permitted")
    requested = Path(raw).expanduser()
    if (not requested.is_absolute() or ".." in requested.parts
            or len(requested.parts) < 4
            or not re.fullmatch(r"taskdata(?:_[A-Za-z0-9_.-]+)?", requested.name)):
        raise ValueError("Use an explicit absolute new taskdata or taskdata_<id> directory")
    root = requested.resolve()
    if root != requested or requested.is_symlink():
        raise ValueError(f"Output root or an ancestor is a symlink: {requested}")
    for source in readonly_roots:
        source = Path(source).resolve()
        if source == root or source in root.parents or root in source.parents:
            raise ValueError(f"New taskdata must be disjoint from the read-only source: {source}")
    marker = root / MARKER_NAME
    expected = {"schema": SCHEMA, "campaign": CAMPAIGN, "split_seed": SPLIT_SEED,
                "purpose": "new-independent-data-preparation", "root": str(root)}
    payload = (json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Taskdata root is not a directory: {root}")
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != payload:
                raise ValueError(f"Taskdata marker mismatch: {marker}")
        elif any(root.iterdir()):
            raise ValueError(f"Refusing to adopt a pre-existing populated directory: {root}")
        else:
            _write_bytes_new(marker, payload, allow_identical=True)
    else:
        root.mkdir(parents=True, exist_ok=False)
        _write_bytes_new(marker, payload, allow_identical=True)
    return root


def create_dataset_dir(root, name):
    root = Path(root)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Unsafe dataset directory name: {name!r}")
    if not (root / MARKER_NAME).is_file():
        raise ValueError("Initialize/verify the new taskdata marker before creating a dataset")
    target = root / name
    target.mkdir(exist_ok=False)
    _sync_directory(root)
    return target


def image_info(path):
    """Hash stable source bytes and exact PIL-decoded RGB including dimensions."""
    from PIL import Image, ImageFile

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing image: {path}")
    before = path.stat()
    before_sha = sha_file(path)
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Multi-frame images are outside this protocol: {path}")
            source_mode, source_format = image.mode, image.format
            rgb = image.convert("RGB")
            header = canonical({"mode": "RGB", "width": rgb.width, "height": rgb.height})
            pixel_sha = hashlib.sha256(header + rgb.tobytes()).hexdigest()
            width, height = rgb.width, rgb.height
    after = path.stat()
    after_sha = sha_file(path)
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or before_sha != after_sha:
        raise RuntimeError(f"Source changed during the image audit: {path}")
    return {"sha256": before_sha, "pixel_sha256": pixel_sha, "width": width,
            "height": height, "source_mode": source_mode, "bytes": before.st_size,
            "format": source_format}


def stable_group_order(groups, namespace):
    """Hash-ranking is fixed across training seeds and independent of host paths."""
    return sorted(groups.items(), key=lambda item: (
        digest([SPLIT_SEED, namespace, item[0]]), item[0]))


def fixed_group_split(records, targets, namespace):
    """Find exact two/three-way image quotas without splitting an RGB group.

    This is an exact bounded subset-sum, not greedy assignment or retrying
    random splits until one looks attractive. For three splits it constrains
    the two smaller bins using bitsets; the largest bin is the complement.
    All-singleton inventories use the equivalent direct hash-ranked slicing.
    An infeasible quota raises instead of deleting data or cutting a group.
    """
    names = list(targets)
    if (len(names) not in (2, 3) or any(type(v) is not int or v < 0 for v in targets.values())
            or sum(targets.values()) != len(records)):
        raise ValueError("Two/three nonnegative exact split quotas must sum to the record count")
    groups = defaultdict(list)
    for row in records:
        groups[row["pixel_sha256"]].append(row)
    ordered = stable_group_order(groups, namespace)
    output = {name: [] for name in names}
    if all(len(rows) == 1 for _, rows in ordered):
        cursor = 0
        for name in names:
            output[name] = [rows[0] for _, rows in ordered[cursor:cursor + targets[name]]]
            cursor += targets[name]
    elif len(names) == 2:
        chosen = min(names, key=lambda name: (targets[name], names.index(name)))
        other = next(name for name in names if name != chosen)
        quota = targets[chosen]
        mask = (1 << (quota + 1)) - 1
        reachable, history = 1, []
        for _, rows in ordered:
            history.append(reachable)
            reachable |= (reachable << len(rows)) & mask
        if not ((reachable >> quota) & 1):
            raise RuntimeError(f"RGB groups cannot meet exact quotas in {namespace}: {targets}")
        remaining = quota
        for (_, rows), previous in zip(reversed(ordered), reversed(history)):
            if (previous >> remaining) & 1:
                output[other].extend(rows)
            else:
                output[chosen].extend(rows)
                remaining -= len(rows)
        if remaining:
            raise AssertionError("Subset-sum reconstruction failed")
    else:
        smaller = sorted(names, key=lambda name: (targets[name], names.index(name)))
        first, second, complement = smaller
        a_max, b_max = targets[first], targets[second]
        b_mask = (1 << (b_max + 1)) - 1
        reachable, history = [1] + [0] * a_max, []
        for _, rows in ordered:
            weight = len(rows)
            history.append(reachable)
            following = reachable.copy()
            for a, bits in enumerate(reachable):
                if not bits:
                    continue
                following[a] |= (bits << weight) & b_mask
                if a + weight <= a_max:
                    following[a + weight] |= bits
            reachable = following
        if not ((reachable[a_max] >> b_max) & 1):
            raise RuntimeError(f"RGB groups cannot meet exact quotas in {namespace}: {targets}")
        a, b = a_max, b_max
        for (_, rows), previous in zip(reversed(ordered), reversed(history)):
            weight = len(rows)
            if (previous[a] >> b) & 1:
                output[complement].extend(rows)
            elif a >= weight and ((previous[a - weight] >> b) & 1):
                output[first].extend(rows)
                a -= weight
            elif b >= weight and ((previous[a] >> (b - weight)) & 1):
                output[second].extend(rows)
                b -= weight
            else:
                raise AssertionError("Joint subset-sum reconstruction failed")
        if a or b:
            raise AssertionError("Joint subset-sum did not reconstruct both quotas")
    for name in names:
        if len(output[name]) != targets[name]:
            raise AssertionError(f"Unexpected reconstructed quota: {name}")
        output[name] = sorted(output[name], key=lambda row: row.get(
            "source_relative_path", row.get("name", row.get("path", ""))))
    membership = defaultdict(set)
    for name, rows in output.items():
        for row in rows:
            membership[row["pixel_sha256"]].add(name)
    if any(len(splits) != 1 for splits in membership.values()):
        raise AssertionError("A decoded RGB group crossed output splits")
    return output
