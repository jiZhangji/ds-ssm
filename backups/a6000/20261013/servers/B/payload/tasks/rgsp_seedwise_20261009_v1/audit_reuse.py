#!/usr/bin/env python3
"""Read-only verification of continuous historical runs; derive new reuse evidence."""
import argparse
import json
import math
from pathlib import Path

from manifest import frozen, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--dataset", choices=("ssdd", "sadd"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run, protocol_path, code, output = map(Path, (args.run, args.protocol, args.code, args.output))
    if not all(x.is_absolute() for x in (run, protocol_path, code, output)):
        raise ValueError("All evidence and output paths must be absolute")
    expected_manifest = {"ssdd": "cd2f5314ecc8139f8e51a8c283d309f522c12abd58320003ea86bacbaac853de",
                         "sadd": "23077fb38483e161a3b7675b1d0629b19cbe334482588a0bdebf7eba88372e10"}[args.dataset]
    protocol = json.loads(protocol_path.read_text())
    if protocol["data_manifest_sha256"] != expected_manifest or protocol["seed"] != args.seed:
        raise RuntimeError("Seed or fixed dataset split is not compatible with this campaign")
    for name, expected in protocol["code"].items():
        if sha(code / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    train = json.loads((run / "train_complete.json").read_text())
    evaluation = json.loads((run / "eval_test.json").read_text())
    runtime = json.loads((run / "runtime_audit.json").read_text())
    if (run / "runtime_audit_resume.json").exists() or runtime.get("resumed"):
        raise RuntimeError("Interrupted/resumed training is preserved as history, not reused in the clean matrix")
    if train.get("smoke_only") or runtime.get("smoke_only"):
        raise RuntimeError("Smoke results are not formal results")
    best = run / "weights/best.pt"
    best_sha = sha(best)
    if train["best_sha256"] != best_sha or evaluation["weights_sha256"] != best_sha:
        raise RuntimeError("Training/test checkpoint hash chain is broken")
    if evaluation["seed"] != args.seed or evaluation["variant"] != run.name or evaluation["split"] != "test":
        raise RuntimeError("Wrong seed/variant/evaluation split")
    for key, expected in {"imgsz": 640, "batch": 16, "conf": 0.001, "iou": 0.6, "precision_mode": "fp32"}.items():
        if evaluation[key] != expected:
            raise RuntimeError(f"Evaluation protocol differs: {key}")
    import yaml
    training_args = yaml.safe_load((run / "args.yaml").read_text())
    expected_args = {"epochs": 150 if args.dataset == "ssdd" else 200,
                     "patience": 30 if args.dataset == "ssdd" else 40,
                     "seed": args.seed, "batch": 16, "imgsz": 640, "amp": False,
                     "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.01,
                     "weight_decay": 0.0005, "nbs": 64, "cos_lr": False,
                     "warmup_epochs": 3.0, "module_warmup_epochs": 0,
                     "pretrained": False}
    for key, expected in expected_args.items():
        if training_args.get(key) != expected:
            raise RuntimeError(f"Training protocol differs: {key}: {training_args.get(key)!r}")
    controls_path = run / "controls_by_epoch.jsonl"
    rows = [json.loads(line) for line in controls_path.read_text().splitlines() if line.strip()]
    final_epoch = train["last_epoch_observed"]
    if [row["epoch"] for row in rows] != list(range(1, final_epoch + 1)):
        raise RuntimeError("Missing/non-contiguous epoch controls: cannot certify a continuous run")
    if any(not math.isfinite(float(v)) for row in rows for v in row["loss"]):
        raise RuntimeError("Historical nonfinite training loss")
    if any(not math.isfinite(float(v)) for v in evaluation["overall"].values()):
        raise RuntimeError("Nonfinite historical test metric")
    if run.name != "original":
        for row in rows:
            if len(row["controls"]) != 1 or row["controls"][0]["control"] != run.name:
                raise RuntimeError("Historical control identity mismatch")
        for c in (runtime["controls"] + train["reloaded_controls"]):
            if c["control"] != run.name:
                raise RuntimeError("Reload control identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    marker = {"status": "complete", "best_sha256": best_sha, "eval_test_sha256": sha(run / "eval_test.json"),
              "source_train_complete_sha256": sha(run / "train_complete.json"),
              "derivation": "Verified complete legacy train/test chain; no new training/evaluation performed.",
              "smoke_only": False}
    old_marker = run / "evaluation_complete.json"
    if old_marker.exists():
        old = json.loads(old_marker.read_text())
        if old.get("exit_code") != 0 or old.get("eval_sha256") != marker["eval_test_sha256"]:
            raise RuntimeError("Existing historical evaluation marker differs")
        marker["source_evaluation_complete"] = {"path": str(old_marker), "sha256": sha(old_marker)}
    frozen(output / "evaluation_complete.json", marker)
    audit = {"status": "verified", "trial_id": args.trial_id, "protocol_compatible": True,
             "origin_run": str(run), "source_code_files_verified": len(protocol["code"]),
             "data_manifest_sha256": expected_manifest, "seed": args.seed,
             "continuous_epochs_verified": len(rows), "training_args_verified": expected_args,
             "runtime_audit": {"path": str(run / "runtime_audit.json"), "sha256": sha(run / "runtime_audit.json")},
             "epoch_audit": {"path": str(controls_path), "sha256": sha(controls_path)},
             "limitations": ["Historical continuous seeded-scratch trial, not rerun under the new launcher.",
                             "Same-shape RGSP controls share initialized tensors; different architecture/state/position do not guarantee tensor-matched initialization.",
                             "Dataset remains exact-content-cleaned image split, not a verified unseen-scene split.",
                             "Legacy EMA/FP16 checkpoint storage and test metric protocol retained."]}
    frozen(output / "audit.json", audit)
    paths = {"train_complete": run / "train_complete.json", "eval_test": run / "eval_test.json",
             "evaluation_complete": output / "evaluation_complete.json", "best": best,
             "protocol": protocol_path, "audit": output / "audit.json"}
    entry = {k: {"path": str(path), "sha256": sha(path)} for k, path in paths.items()}
    entry["origin"] = str(run)
    frozen(output / "entry.json", entry)
    print(json.dumps({"status": "verified", "trial_id": args.trial_id, "epochs": len(rows),
                      "entry": entry, "metric": evaluation["overall"]}))


if __name__ == "__main__":
    main()
