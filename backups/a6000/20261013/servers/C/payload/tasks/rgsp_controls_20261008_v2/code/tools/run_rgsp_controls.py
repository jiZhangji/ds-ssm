#!/usr/bin/env python3
"""Finite, fail-closed SSDD control campaign. Run only on a designated server."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_sarmix_proxy import model_payload

# Order frozen before observing new test results. No validation-driven expansion.
VARIANTS = (
    "original", "matched_identity", "matched_scale075", "local_energy",
    "local_energy_input", "local_energy_delta", "full", "full_fixed_half",
    "residual_scale075", "energy_only", "guidance_only", "global",
)
SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen(path, value):
    path = Path(path)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Frozen artifact differs: {path}; create a new campaign")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def code_hashes():
    files = [p for p in (ROOT / "ultralytics").rglob("*") if p.suffix in {".py", ".yaml", ".yml"}]
    files += [ROOT / "mbyolo_train_sarair.py"]
    files += [ROOT / "tools" / name for name in (
        "run_rgsp_controls.py", "train_rgsp_control.py", "validate_rgsp_controls.py",
        "run_sarmix_proxy.py", "evaluate_sarmix.py", "validate_selective_scan_backend.py",
        "audit_ssdd_content.py", "launch_rgsp_lane.sh",
    )]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def config(variant):
    result = model_payload("baseline" if variant == "original" else "p3", 16, "full")
    result["nc"] = 1
    result["head"][-1][3] = [1]
    if variant != "original":
        assert result["backbone"][3][2] == "DS_VSSBlockSweep"
        # Ultralytics literal-evaluates string arguments; "global" is a Python
        # keyword and raises SyntaxError unless passed as a quoted literal.
        result["backbone"][3][3].append(repr(variant))
    return result


def prepare(source, lane, seed, smoke):
    """Copy small SSDD dataset, avoiding writes/caches in the source dataset."""
    originals = {}
    for split, count in (("train", 928), ("test", 232)):
        paths = sorted(p for p in (source / "images" / split).iterdir() if p.suffix.lower() in SUFFIXES)
        if len(paths) != count:
            raise RuntimeError(f"Expected {count} SSDD {split} images; got {len(paths)}")
        if len({p.stem for p in paths}) != len(paths):
            raise RuntimeError(f"Nonunique image stems: {split}")
        originals[split] = paths
    # Preserve official test images/annotations. Exclude only development
    # copies with exactly the same bytes as test; do not alter source files.
    source_inventory = []
    for split, paths in originals.items():
        for image in paths:
            label = source / "labels" / split / (image.stem + ".txt")
            source_inventory.append({"split": split, "name": image.name,
                                     "image_sha256": sha(image), "label_sha256": sha(label)})
    frozen(lane / "manifest" / "source_inventory.json", source_inventory)
    test_hashes = {row["image_sha256"] for row in source_inventory if row["split"] == "test"}
    train_hashes = {row["name"]: row["image_sha256"] for row in source_inventory if row["split"] == "train"}
    excluded, grouped = [], defaultdict(list)
    for image in originals["train"]:
        digest = train_hashes[image.name]
        if digest in test_hashes:
            excluded.append({"image": image.name, "sha256": digest, "reason": "identical to official test content"})
        else:
            grouped[digest].append(image)
    # Shuffle content groups, never individual duplicated files. Greedy fill
    # produces exactly 186 validation files here because sufficient singletons
    # exist; fail explicitly if that invariant changes on a different source.
    groups = sorted(grouped.values(), key=lambda group: group[0].name)
    random.Random(seed).shuffle(groups)
    train_images, val_images = [], []
    for group in groups:
        if len(val_images) + len(group) <= 186:
            val_images.extend(group)
        else:
            train_images.extend(group)
    if len(val_images) != 186:
        raise RuntimeError("Could not create an exact 186-image grouped validation set")
    selected = {"train": sorted(train_images), "val": sorted(val_images), "test": originals["test"]}
    frozen(lane / "manifest" / "split_policy.json", {
        "policy": "official-test-preserving byte-content-grouped development split v2",
        "seed": seed, "official_train": 928, "official_test": 232,
        "excluded_from_development": excluded,
        "counts_before_smoke_subsetting": {k: len(v) for k, v in selected.items()},
        "development_unique_content": len(groups),
        "annotation_policy": "retain every included original annotation; duplicated content stays in one split",
        "limitations": "byte identity is not a near-duplicate or scene-disjoint audit",
    })
    if smoke:
        selected = {"train": selected["train"][:32], "val": selected["val"][:16]}
    rows, image_hashes = [], {}
    for split, paths in selected.items():
        image_hashes[split] = set()
        for src in paths:
            label = source / "labels" / src.parent.name / (src.stem + ".txt")
            if not label.is_file():
                raise RuntimeError(f"Missing annotation: {label}")
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise RuntimeError(f"Invalid SSDD class/box: {label}")
                coords = list(map(float, fields[1:]))
                if not all(0 <= x <= 1 for x in coords) or min(coords[2:]) <= 0:
                    raise RuntimeError(f"Invalid normalized box: {label}")
            entry = {"split": split, "name": src.name, "image_sha256": sha(src), "label_sha256": sha(label)}
            image_hashes[split].add(entry["image_sha256"])
            for kind, original in (("images", src), ("labels", label)):
                target = lane / "data" / kind / split / original.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(original, target)
                if sha(target) != entry["image_sha256" if kind == "images" else "label_sha256"]:
                    raise RuntimeError(f"Staged data differs: {target}")
            rows.append(entry)
    for left in image_hashes:
        for right in image_hashes:
            if left < right and image_hashes[left] & image_hashes[right]:
                raise RuntimeError(f"Identical image content across {left}/{right}")
    frozen(lane / "manifest" / "data_content.json", rows)
    data = {"path": str(lane / "data"), "train": "images/train", "val": "images/val", "names": {0: "ship"}}
    if not smoke:
        data["test"] = "images/test"
    frozen(lane / "manifest" / "data.yaml", yaml.safe_dump(data, sort_keys=False))
    return lane / "manifest" / "data.yaml"


def gpu_idle():
    target = os.environ.get("CUDA_VISIBLE_DEVICES")
    if target is None or "," in target:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU")
    out = subprocess.check_output([
        "nvidia-smi", "-i", target, "--query-compute-apps=pid,process_name", "--format=csv,noheader"
    ], text=True).strip()
    if out:
        raise RuntimeError(f"Assigned GPU already has a compute task; not taking over:\n{out}")


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("a") as f:
        f.write("\n" + datetime.now(timezone.utc).isoformat() + "\n" + repr(command) + "\n")
        f.flush()
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def summarize(lane):
    rows = []
    for variant in VARIANTS:
        path = lane / "runs" / variant / "eval_test.json"
        if path.exists():
            item = json.loads(path.read_text())
            rows.append({"variant": variant, "seed": item["seed"], **item["overall"],
                         "weights_sha256": item.get("weights_sha256"), "evidence": str(path)})
    (lane / "results_partial.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (lane / "results_partial.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--mode", choices=("prepare", "run", "smoke"), default="prepare")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--resume", action="store_true", help="Explicitly resume interrupted training, not bitwise continuation")
    args = p.parse_args()
    lane = args.output.resolve()
    lane.mkdir(parents=True, exist_ok=True)
    lock = (lane / "lane.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    smoke = args.mode == "smoke"
    data = prepare(args.source_data.resolve(), lane, args.seed, smoke)
    configs = {}
    for variant in args.variants:
        configs[variant] = lane / "manifest" / "configs" / (variant + ".yaml")
        frozen(configs[variant], yaml.safe_dump(config(variant), sort_keys=False))
    frozen(lane / "manifest" / "protocol.json", {
        "campaign": "rgsp_controls_20261008_v2", "seed": args.seed, "smoke_only": smoke,
        "variants": args.variants, "epochs": 2 if smoke else 150, "patience": 30,
        "imgsz": 640, "batch": 16, "weights": "none", "amp": False, "workers": 4,
        "split": "exclude development content identical to official test; group remaining duplicates; 730/186/232 on audited source",
        "fitness": "0.1*AP50+0.9*AP50_95", "train_val_nms_iou": 0.7, "test_nms_iou": 0.6,
        "code": code_hashes(), "data_manifest_sha256": sha(lane / "manifest" / "data_content.json"),
        "source_inventory_sha256": sha(lane / "manifest" / "source_inventory.json"),
        "split_policy_sha256": sha(lane / "manifest" / "split_policy.json"),
        "config_sha256": {v: sha(c) for v, c in configs.items()},
        "hyperparameters_source": "tools/train_rgsp_control.py (all explicit, source hashed)",
        "resume_policy": "explicit only; EMA/FP16 optimizer legacy resume is not bitwise identical",
        "initialization_policy": "same seeded construction for all matched controls; reset all BN running stats after stride probe",
    })
    if args.mode == "prepare":
        print(f"PREPARED {lane}", flush=True)
        return
    status = {"host": socket.gethostname(), "pid": os.getpid(), "seed": args.seed,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": datetime.now(timezone.utc).isoformat(),
              "state": "running", "completed": []}
    status_path = lane / "status.json"
    try:
        for variant in args.variants:
            status["current"] = variant
            status_path.write_text(json.dumps(status, indent=2))
            run_dir = lane / "runs" / variant
            done = run_dir / "train_complete.json"
            if not done.exists():
                gpu_idle()
                command = [sys.executable, "-u", str(ROOT / "tools/train_rgsp_control.py"),
                           "--config", str(configs[variant]), "--data", str(data),
                           "--project", str(lane / "runs"), "--variant", variant, "--seed", str(args.seed)]
                if smoke:
                    command.append("--smoke")
                if args.resume:
                    command.append("--resume")
                execute(command, lane / "logs" / (variant + ".train.log"))
                if not done.exists():
                    raise RuntimeError(f"Training did not emit completion evidence: {run_dir}")
            if not smoke and not (run_dir / "eval_test.json").exists():
                gpu_idle()
                execute([
                    sys.executable, "-u", str(ROOT / "tools/evaluate_sarmix.py"),
                    "--weights", str(run_dir / "weights/best.pt"), "--data", str(data),
                    "--experiment", f"rgsp_controls_seed{args.seed}_{variant}",
                    "--placement", "baseline" if variant == "original" else "p3",
                    "--variant", variant, "--state-dim", "0" if variant == "original" else "16",
                    "--seed", str(args.seed), "--subset", "SSDD", "--split", "test",
                    "--output-json", str(run_dir / "eval_test.json"), "--imgsz", "640",
                    "--batch", "16", "--device", "0", "--workers", "4",
                    "--conf", "0.001", "--iou", "0.6",
                    "--project", str(lane / "evaluations"), "--name", variant,
                ], lane / "logs" / (variant + ".eval.log"))
            status["completed"].append(variant)
            summarize(lane)
        status["state"] = "complete"
    except Exception as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        raise
    finally:
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
