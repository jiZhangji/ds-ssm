#!/usr/bin/env python3
"""Frozen, explicit manifests for the finite seed-ordered RGSP campaign.

This module is standard-library-only. It never imports torch or probes a GPU.
Absolute paths in a plan refer to the designated server, not this workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "rgsp-seedwise-v1"
CONTROLS = (
    "original", "cnn_yolov8n", "matched_identity", "matched_scale075", "local_energy",
    "local_energy_input", "local_energy_delta", "full", "full_fixed_half",
    "residual_scale075", "energy_only", "guidance_only", "global",
    "full_no_lrc", "const_guidance", "post_gate", "lsgm_trend_only",
    "lsgm_std_only", "lsgm_point_only",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"An explicit absolute server path is required: {value!r}")
    return path


def within(path, root):
    path, root = absolute(path).resolve(), absolute(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Output must be strictly inside campaign root: {path}")
    return path


def frozen(path, value):
    """Create once; never silently replace a pre-existing artifact."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen artifact differs: {path}; use a new plan/output")


def atomic_json(path, value):
    """Only for explicitly mutable status/checkpoint metadata, never plan files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def config_payload(spec):
    """Exactly one real P2/P3/P4 block; P4 is 2 ordinary + 1 RGSP after scaling."""
    control = spec["control"]
    if control not in CONTROLS:
        raise ValueError(f"Unsupported control: {control}")
    if control == "cnn_yolov8n":
        if spec.get("placement", "baseline") != "baseline":
            raise ValueError("The CNN reference has no RGSP placement; use baseline")
        # Exact repository ultralytics/cfg/models/v8/yolov8.yaml architecture,
        # with nc=1 and only the frozen n=[.33,.25,1024] scale retained.
        # Never download/resolve a mutable installed-package model template.
        return {
            "nc": 1, "scales": {"n": [0.33, 0.25, 1024]}, "scale": "n",
            "backbone": [
                [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]],
                [-1, 3, "C2f", [128, True]], [-1, 1, "Conv", [256, 3, 2]],
                [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
                [-1, 6, "C2f", [512, True]], [-1, 1, "Conv", [1024, 3, 2]],
                [-1, 3, "C2f", [1024, True]], [-1, 1, "SPPF", [1024, 5]],
            ],
            "head": [
                [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
                [-1, 3, "C2f", [512]], [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
                [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]],
                [-1, 3, "C2f", [512]], [-1, 1, "Conv", [512, 3, 2]],
                [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
                [[15, 18, 21], 1, "Detect", [1]],
            ],
        }
    placement = spec.get("placement", "baseline" if control == "original" else "p3")
    if placement not in {"baseline", "p2", "p3", "p4"} or (control == "original") != (placement == "baseline"):
        raise ValueError("original must use baseline; RGSP must use exactly one of p2/p3/p4")
    state_dim = spec.get("state_dim", 16)
    if state_dim not in ({0, 8, 16, 32} if control == "original" else {8, 16, 32}):
        raise ValueError("state_dim must be 8, 16 or 32")
    options = {k: spec.get(k, 3) for k in ("trend_kernel_size", "stats_kernel_size")}
    if any(v not in {3, 5, 7} for v in options.values()):
        raise ValueError("Window sizes must be 3, 5 or 7")
    backbone = []

    def add(channels, repeats, stage):
        if stage == placement:
            # Quoted literal is needed for the Python keyword 'global'.
            backbone.append([-1, repeats, "DS_VSSBlockSweep", [channels, state_dim, repr(control), options]])
        else:
            backbone.append([-1, repeats, "VSSBlock", [channels]])
        return len(backbone) - 1

    backbone.append([-1, 1, "SimpleStem", [128, 3]])
    add(128, 3, "p2")
    backbone.append([-1, 1, "VisionClueMerge", [256]])
    p3 = add(256, 3, "p3")
    backbone.append([-1, 1, "VisionClueMerge", [512]])
    if placement == "p4":
        backbone.append([-1, 6, "VSSBlock", [512]])
        p4 = add(512, 3, "p4")
    else:
        p4 = add(512, 9, "p4")
    backbone.append([-1, 1, "VisionClueMerge", [1024]])
    add(1024, 3, "p5")
    backbone.append([-1, 1, "SPPF", [1024, 5]])
    p5 = len(backbone) - 1
    head = []

    def h(source, repeats, module, args):
        head.append([source, repeats, module, args])
        return len(backbone) + len(head) - 1

    h(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    h([-1, p4], 1, "Concat", [1])
    h4 = h(-1, 3, "XSSBlock", [512])
    h(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    h([-1, p3], 1, "Concat", [1])
    h3 = h(-1, 3, "XSSBlock", [256])
    h(-1, 1, "Conv", [256, 3, 2])
    h([-1, h4], 1, "Concat", [1])
    out4 = h(-1, 3, "XSSBlock", [512])
    h(-1, 1, "Conv", [512, 3, 2])
    h([-1, p5], 1, "Concat", [1])
    out5 = h(-1, 3, "XSSBlock", [1024])
    h([h3, out4, out5], 1, "Detect", [1])
    count = sum(max(round(row[1] * 0.33), 1) for row in backbone if row[2] == "DS_VSSBlockSweep")
    if count != (0 if control == "original" else 1):
        raise AssertionError("Scaled model does not contain the intended single controlled block")
    return {"nc": 1, "scales": {"T": [0.33, 0.25, 1024]}, "scale": "T", "backbone": backbone, "head": head}


def load_plan(path):
    plan = read_json(absolute(path))
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"Expected schema {SCHEMA}")
    for key in ("campaign_root", "source_root", "python", "barrier_path", "receipts_dir"):
        absolute(plan[key])
    ids, run_dirs = set(), set()
    trials = plan["trials"]
    for trial in trials:
        tid = trial["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", tid) or tid in ids:
            raise ValueError(f"Invalid/duplicate trial id: {tid}")
        ids.add(tid)
        if trial["seed"] not in {0, 1}:
            raise ValueError("Only seed0 and seed1 are authorized")
        if trial["kind"] not in {"detection", "command", "reuse", "not_ready"}:
            raise ValueError(f"Unknown trial kind: {trial['kind']}")
        if trial["kind"] != "not_ready":
            run_dir = str(within(trial["run_dir"], plan["campaign_root"]))
            if run_dir in run_dirs:
                raise ValueError(f"Shared trial output directory: {run_dir}")
            run_dirs.add(run_dir)
        if trial["kind"] == "detection":
            config_payload(trial["spec"])
            absolute(trial["config"])
            if trial["dataset"] not in plan["datasets"]:
                raise ValueError(f"Missing explicit dataset: {trial['dataset']}")
        if trial["kind"] == "command":
            absolute(trial["cwd"])
            absolute(trial["command"][0])
            if not trial.get("success_artifacts"):
                raise ValueError("Command trials need explicit success_artifacts")
            for artifact in trial["success_artifacts"]:
                within(artifact, trial["run_dir"])
    seed0 = sorted(t["id"] for t in trials if t["seed"] == 0)
    if not seed0 or sorted(plan["seed0_trial_ids"]) != seed0:
        raise ValueError("seed0_trial_ids must cover the ENTIRE intended seed0 matrix, including not_ready trials")
    assigned = set()
    lane_ids = set()
    for lane in plan["lanes"]:
        if lane["id"] in lane_ids:
            raise ValueError("Duplicate lane id")
        lane_ids.add(lane["id"])
        absolute(lane["output_dir"])
        if not lane["gpu_uuid"].startswith("GPU-") or not lane["hostname"]:
            raise ValueError("Each lane requires actual hostname and GPU UUID")
        for tid in lane["trial_ids"]:
            if tid not in ids or tid in assigned:
                raise ValueError(f"Unknown/multiply assigned trial: {tid}")
            assigned.add(tid)
    # Unassigned trials are allowed only to remain visible to the global barrier.
    # They are NOT silently removed from its expected completion set.
    if not isinstance(plan["code_hashes"], list) or not plan["code_hashes"]:
        raise ValueError("Nonempty explicit source/artifact code_hashes list is required")
    return plan


def trial_by_id(plan, tid):
    return next(t for t in plan["trials"] if t["id"] == tid)


def check_file(entry):
    path = absolute(entry["path"])
    if not path.is_file() or sha(path) != entry["sha256"]:
        raise RuntimeError(f"Frozen file missing or hash mismatch: {path}")


def verify_code(plan):
    for item in plan["code_hashes"]:
        check_file(item)


def verify_dataset(dataset):
    """Recheck the staged YOLO copy; never modify/cache the source dataset here."""
    check_file({"path": dataset["yaml"], "sha256": dataset["yaml_sha256"]})
    check_file({"path": dataset["manifest"], "sha256": dataset["manifest_sha256"]})
    root = absolute(dataset["root"])
    inventory = read_json(dataset["manifest"])
    if not isinstance(inventory, list):
        raise ValueError("data manifest must be the explicit list of split/name/image_sha256/label_sha256 rows")
    counts, content, seen = {}, {}, set()
    for row in inventory:
        split, name = row["split"], row["name"]
        if split not in {"train", "val", "test"} or Path(name).name != name:
            raise ValueError("Unsafe data inventory row")
        if (split, name) in seen:
            raise ValueError("Duplicate data inventory row")
        seen.add((split, name))
        for kind, filename, key in (("images", name, "image_sha256"), ("labels", Path(name).stem + ".txt", "label_sha256")):
            check_file({"path": str(root / kind / split / filename), "sha256": row[key]})
        counts[split] = counts.get(split, 0) + 1
        content.setdefault(split, set()).add(row["image_sha256"])
    if counts != dataset["counts"]:
        raise RuntimeError(f"Dataset counts changed: {counts} != {dataset['counts']}")
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm"}
    for split in ("train", "val", "test"):
        actual_names = {p.name for p in (root / "images" / split).iterdir() if p.is_file() and p.suffix.lower() in suffixes}
        expected_names = {name for s, name in seen if s == split}
        if actual_names != expected_names:
            raise RuntimeError(f"Staged image file set differs from manifest: {split}")
    for left in content:
        for right in content:
            if left < right and content[left] & content[right]:
                raise RuntimeError(f"Byte-identical image content crosses {left}/{right}")
    return {"counts": counts, "manifest_sha256": dataset["manifest_sha256"], "verified_utc": utc()}


def receipt_path(plan, trial):
    return absolute(plan["receipts_dir"]) / (trial["id"] + ".json")


def validate_receipt(plan, trial, receipt, verify_files=False):
    for key, expected in (("schema", SCHEMA), ("status", "complete"), ("plan_sha256", digest(plan)),
                          ("trial_id", trial["id"]), ("trial_sha256", digest(trial)), ("seed", trial["seed"])):
        if receipt.get(key) != expected:
            raise RuntimeError(f"Bad completion receipt {trial['id']}: {key}")
    if not receipt.get("artifacts"):
        raise RuntimeError("Completion requires hashed artifact evidence")
    for artifact in receipt["artifacts"]:
        absolute(artifact["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise RuntimeError("Completion artifact must carry a full SHA256")
    if verify_files:
        for artifact in receipt["artifacts"]:
            check_file(artifact)
    return receipt


def write_receipt(plan, trial, artifacts, extra=None):
    receipt = {"schema": SCHEMA, "status": "complete", "plan_sha256": digest(plan),
               "trial_id": trial["id"], "trial_sha256": digest(trial), "seed": trial["seed"],
               "finished_utc": utc(), "artifacts": [{"path": str(absolute(p)), "sha256": sha(p)} for p in artifacts]}
    if extra:
        receipt["details"] = extra
    path = receipt_path(plan, trial)
    if path.exists():
        existing = validate_receipt(plan, trial, read_json(path), verify_files=True)
        if existing["artifacts"] != receipt["artifacts"]:
            raise RuntimeError(f"Completion evidence changed: {path}")
        return existing
    frozen(path, receipt)
    return receipt


def verify_barrier(plan):
    marker = read_json(absolute(plan["barrier_path"]))
    payload = marker["payload"]
    if marker.get("payload_sha256") != digest(payload):
        raise RuntimeError("Global barrier payload checksum mismatch")
    if payload.get("plan_sha256") != digest(plan) or payload.get("schema") != SCHEMA:
        raise RuntimeError("Global barrier is for a different immutable plan")
    if payload.get("state") != "all_seed0_complete" or sorted(payload.get("seed0_trial_ids", [])) != sorted(plan["seed0_trial_ids"]):
        raise RuntimeError("Global barrier does not cover the whole seed0 matrix")
    receipts = payload.get("receipts", {})
    if sorted(receipts) != sorted(plan["seed0_trial_ids"]):
        raise RuntimeError("Global barrier omitted required completion receipts")
    for tid, item in receipts.items():
        validate_receipt(plan, trial_by_id(plan, tid), item)
    return payload


def issue_barrier(plan, receipts_dir):
    """Trusted coordinator only. This is hash-bound, NOT a security signature."""
    receipts = {}
    for tid in plan["seed0_trial_ids"]:
        trial = trial_by_id(plan, tid)
        if trial["kind"] == "not_ready":
            raise RuntimeError(f"Global seed0 is incomplete/not ready: {tid}")
        receipts[tid] = validate_receipt(plan, trial, read_json(absolute(receipts_dir) / (tid + ".json")))
    payload = {"schema": SCHEMA, "state": "all_seed0_complete", "plan_sha256": digest(plan),
               "seed0_trial_ids": sorted(plan["seed0_trial_ids"]), "receipts": receipts,
               "issued_utc": utc(), "trust_model": "trusted coordinator, hash-bound evidence; no cryptographic signature"}
    frozen(plan["barrier_path"], {"payload": payload, "payload_sha256": digest(payload)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--issue-barrier", action="store_true")
    parser.add_argument("--receipts-dir")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.write_configs:
        import yaml
        for trial in plan["trials"]:
            if trial["kind"] != "detection":
                continue
            path = within(trial["config"], plan["campaign_root"])
            rendered = yaml.safe_dump(config_payload(trial["spec"]), sort_keys=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding="utf-8") != rendered:
                    raise RuntimeError(f"Frozen model config differs: {path}")
            else:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
    if args.issue_barrier:
        issue_barrier(plan, args.receipts_dir or plan["receipts_dir"])
    print(json.dumps({"schema": SCHEMA, "plan_sha256": digest(plan), "trials": len(plan["trials"]),
                      "seed0_trials": len(plan["seed0_trial_ids"]), "configs_written": args.write_configs}))


if __name__ == "__main__":
    main()
