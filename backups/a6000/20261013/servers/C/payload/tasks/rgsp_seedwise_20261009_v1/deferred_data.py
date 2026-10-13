#!/usr/bin/env python3
"""Wait for one explicitly declared, audited external detection dataset.

Standard library only: no CUDA imports, downloads, source mutations or discovery.
The producer publishes immutable descriptor/audit files and readiness.json last.
Readiness is trusted-coordinator evidence, not a cryptographic signature.
"""
from __future__ import annotations

import hashlib
import json
import time

from manifest import (absolute, canonical, check_file, utc, validate_deferred_declaration,
                      verify_dataset)

MAX_WAIT_SECONDS = 7 * 24 * 60 * 60
POLL_SECONDS = 30


def read_hashed(path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def require_equal(value, key, expected, path):
    actual = value.get(key)
    if type(actual) is not type(expected) or canonical(actual) != canonical(expected):
        raise RuntimeError(f"Deferred data binding differs: {path}, {key}={actual!r}; expected {expected!r}")


def class_mapping(names, nc):
    """Preserve explicit YAML/source class order while normalizing numeric keys."""
    if not isinstance(names, (dict, list)) or len(names) != nc:
        raise RuntimeError("Deferred class names differ from the audited class count")
    mapping = {}
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for key, value in items:
        if type(key) not in (int, str) or not isinstance(value, str) or not value:
            raise RuntimeError("Deferred class names need nonempty strings and explicit integer indices")
        index = int(key)
        if index in mapping:
            raise RuntimeError("Deferred class mapping has duplicate normalized indices")
        mapping[index] = value
    if set(mapping) != set(range(nc)):
        raise RuntimeError("Deferred class mapping must cover exactly zero through nc-1")
    return mapping


def resolve_ready(entry, paths, readiness, readiness_file):
    """Verify the whole staged copy, then recheck that its publication is stable."""
    descriptor, descriptor_file = read_hashed(paths["descriptor"])
    audit, audit_file = read_hashed(paths["source_audit"])
    identity, nc = entry["expected_identity"], entry["expected_nc"]
    for value, path in ((readiness, paths["readiness"]), (audit, paths["source_audit"])):
        require_equal(value, "dataset_identity", identity, path)
        require_equal(value, "nc", nc, path)
    require_equal(readiness, "status", "verified_ready", paths["readiness"])
    require_equal(audit, "status", "verified", paths["source_audit"])
    require_equal(readiness, "descriptor_sha256", descriptor_file["sha256"], paths["readiness"])
    require_equal(readiness, "source_audit_sha256", audit_file["sha256"], paths["readiness"])
    require_equal(descriptor, "protocol", entry["protocol"], paths["descriptor"])
    require_equal(descriptor, "nc", nc, paths["descriptor"])
    source_names = class_mapping(audit.get("names"), nc)
    descriptor_names = class_mapping(descriptor.get("names"), nc)
    if source_names != descriptor_names:
        raise RuntimeError("Deferred descriptor class order differs from its verified source audit")
    for key in ("root", "yaml", "manifest"):
        absolute(descriptor[key])
    counts = descriptor.get("counts")
    if (not isinstance(counts, dict) or set(counts) != {"train", "val", "test"}
            or any(type(value) is not int or value <= 0 for value in counts.values())):
        raise ValueError("Deferred descriptor requires positive train/val/test image counts")
    for value, path in ((readiness, paths["readiness"]), (audit, paths["source_audit"])):
        require_equal(value, "manifest_sha256", descriptor["manifest_sha256"], path)
    require_equal(audit, "counts", counts, paths["source_audit"])
    data_audit = verify_dataset(descriptor)
    for artifact in (readiness_file, descriptor_file, audit_file):
        check_file(artifact)
    provenance = {"dataset_identity": identity, "nc": nc, "protocol": entry["protocol"],
                  "class_names": descriptor_names, "source_class_names": source_names,
                  "readiness": readiness_file, "descriptor": descriptor_file,
                  "source_audit": audit_file,
                  "verification": "ready publication + source audit + full staged image/label manifest hashes",
                  "trust_model": "producer source audit and coordinator publication; not a security signature"}
    return descriptor, data_audit, provenance


def wait_for_dataset(entry):
    paths = validate_deferred_declaration(entry)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    last_status = None
    while True:
        try:
            readiness, readiness_file = read_hashed(paths["readiness"])
        except FileNotFoundError:
            readiness, readiness_file = {"status": "not_published"}, None
        status = readiness.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("Deferred readiness must have an explicit nonempty status")
        if status == "verified_ready":
            result = resolve_ready(entry, paths, readiness, readiness_file)
            print(json.dumps({"state": "deferred_dataset_verified", "dataset_identity": entry["expected_identity"],
                              "readiness": readiness_file, "verified_utc": utc()}, sort_keys=True), flush=True)
            return result
        if status in {"failed", "invalid", "rejected"}:
            raise RuntimeError(f"Deferred data producer reported {status}: {paths['readiness']}")
        if status != last_status:
            print(json.dumps({"state": "waiting_deferred_dataset", "dataset_identity": entry["expected_identity"],
                              "readiness_path": str(paths["readiness"]), "readiness_status": status,
                              "max_wait_seconds": MAX_WAIT_SECONDS, "updated_utc": utc()}, sort_keys=True), flush=True)
            last_status = status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Seven-day finite deferred-data wait expired; no data guessed and no CUDA trial started")
        time.sleep(min(POLL_SECONDS, remaining))
