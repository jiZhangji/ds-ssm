#!/usr/bin/env python3
"""Run the resumable nine-experiment SAR-Mix proxy ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "mbyolo_train_sarair.py"
EVALUATE = ROOT / "tools" / "evaluate_sarmix.py"
SUMMARIZE = ROOT / "tools" / "summarize_sarmix.py"
PLACEMENTS = ("p2", "p3", "p4", "p5", "p2p3")
MODULES = {
    "full": "DS_VSSBlockSweep",
    "ds_only": "DS_VSSBlockSweepNoPFIM",
    "pfim_only": "DS_VSSBlockSweepPFIMOnly",
    "global": "DS_VSSBlockSweepPFIMGlobal",
}
SUBSETS = ("SSDD", "OGSOD", "SAR-Ship", "SAR-Aircraft")


@dataclass(frozen=True)
class Spec:
    key: str
    variant: str
    placement: str
    state_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", default="output_dir/sarmix_proxy")
    parser.add_argument("--experiment-id", default="sarmix_proxy_v1")
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--ablation-placement", choices=(*PLACEMENTS, "auto"), default="p2")
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def add_layer(items: list, source, repeats: int, module: str, args: list) -> int:
    index = len(items)
    items.append([source, repeats, module, args])
    return index


def model_payload(placement: str, state_dim: int, variant: str) -> dict:
    placement_suffix = placement[1:] if placement.startswith("p") else placement
    selected = set() if placement == "baseline" else {
        f"p{value}" for value in placement_suffix.split("p")
    }
    ds_module = MODULES[variant]
    backbone = []
    add_layer(backbone, -1, 1, "SimpleStem", [128, 3])

    p2_module = ds_module if "p2" in selected else "VSSBlock"
    p2_args = [128, state_dim] if "p2" in selected else [128]
    add_layer(backbone, -1, 3, p2_module, p2_args)
    add_layer(backbone, -1, 1, "VisionClueMerge", [256])

    p3_module = ds_module if "p3" in selected else "VSSBlock"
    p3_args = [256, state_dim] if "p3" in selected else [256]
    p3 = add_layer(backbone, -1, 3, p3_module, p3_args)
    add_layer(backbone, -1, 1, "VisionClueMerge", [512])

    if "p4" in selected:
        add_layer(backbone, -1, 6, "VSSBlock", [512])
        p4 = add_layer(backbone, -1, 3, ds_module, [512, state_dim])
    else:
        p4 = add_layer(backbone, -1, 9, "VSSBlock", [512])
    add_layer(backbone, -1, 1, "VisionClueMerge", [1024])

    p5_module = ds_module if "p5" in selected else "VSSBlock"
    p5_args = [1024, state_dim] if "p5" in selected else [1024]
    add_layer(backbone, -1, 3, p5_module, p5_args)
    p5_sppf = add_layer(backbone, -1, 1, "SPPF", [1024, 5])

    head = []

    def add_head(source, repeats: int, module: str, args: list) -> int:
        index = len(backbone) + len(head)
        head.append([source, repeats, module, args])
        return index

    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p4], 1, "Concat", [1])
    head_p4 = add_head(-1, 3, "XSSBlock", [512])
    add_head(-1, 1, "nn.Upsample", [None, 2, "nearest"])
    add_head([-1, p3], 1, "Concat", [1])
    head_p3 = add_head(-1, 3, "XSSBlock", [256])
    add_head(-1, 1, "Conv", [256, 3, 2])
    add_head([-1, head_p4], 1, "Concat", [1])
    head_p4_out = add_head(-1, 3, "XSSBlock", [512])
    add_head(-1, 1, "Conv", [512, 3, 2])
    add_head([-1, p5_sppf], 1, "Concat", [1])
    head_p5 = add_head(-1, 3, "XSSBlock", [1024])
    add_head([head_p3, head_p4_out, head_p5], 1, "Detect", [11])
    return {
        "nc": 11,
        "scales": {"T": [0.33, 0.25, 1024]},
        "scale": "T",
        "backbone": backbone,
        "head": head,
    }


def write_config(config_dir: Path, spec: Spec) -> Path:
    output = config_dir / f"{spec.key}_sd{spec.state_dim}.yaml"
    variant = "full" if spec.variant == "baseline" else spec.variant
    rendered = yaml.safe_dump(model_payload(spec.placement, spec.state_dim, variant), sort_keys=False)
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise SystemExit(f"Frozen generated config changed: {output}; use a new experiment ID")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log: Path, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}")
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {datetime.now().isoformat()} =====\n$ {printable}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    try:
        with results_csv.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except OSError:
        return 0


def placement_specs(state_dim: int) -> list[Spec]:
    return [Spec("baseline", "baseline", "baseline", 0)] + [
        Spec(f"full_{placement}", "full", placement, state_dim) for placement in PLACEMENTS
    ]


def module_specs(placement: str, state_dim: int) -> list[Spec]:
    return [
        Spec(f"ds_only_{placement}", "ds_only", placement, state_dim),
        Spec(f"pfim_only_{placement}", "pfim_only", placement, state_dim),
        Spec(f"global_{placement}", "global", placement, state_dim),
    ]


def best_full_placement(project: Path, experiment_id: str) -> str:
    candidates = []
    for placement in PLACEMENTS:
        path = project / f"{experiment_id}_full_{placement}" / "eval_val_overall.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((float(item["overall"]["ap50_95"]), placement))
    if len(candidates) != len(PLACEMENTS):
        raise RuntimeError("Automatic ablation placement requires all five full-placement evaluations")
    return max(candidates)[1]


def freeze_protocol(args, data: Path, manifest: Path, configs: dict[str, Path]) -> None:
    snapshot = data.parent / "dataset_snapshot.json"
    payload = {
        "experiment_id": args.experiment_id,
        "data_yaml_sha256": sha256(data),
        "dataset_snapshot_sha256": sha256(snapshot) if snapshot.exists() else None,
        "state_dim": args.state_dim,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "AdamW",
        "lr0": 0.0002,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "nbs": 64,
        "optimizer_warmup_epochs": 3.0,
        "module_warmup_epochs": 0,
        "seed": args.seed,
        "weights": "none",
        "amp": args.amp,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "ultralytics/nn/modules/mamba_yolo.py",
                ROOT / "ultralytics/nn/tasks.py",
                TRAIN,
                EVALUATE,
                Path(__file__).resolve(),
            )
        },
        "config_sha256": {key: sha256(path) for key, path in sorted(configs.items())},
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    protocol = manifest / "protocol.json"
    manifest.mkdir(parents=True, exist_ok=True)
    if protocol.exists() and protocol.read_text(encoding="utf-8") != rendered:
        raise SystemExit(f"Frozen protocol changed: {protocol}; use a new experiment ID")
    protocol.write_text(rendered, encoding="utf-8")


def train_and_evaluate(args, spec: Spec, config: Path, data: Path, project: Path, log_dir: Path) -> None:
    run_name = f"{args.experiment_id}_{spec.key}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = run_dir / "train_complete.json"
    if not done.exists():
        if last_pt.exists():
            command = [
                sys.executable,
                "-u",
                str(TRAIN),
                "--task", "train",
                "--data", str(data),
                "--config", str(config),
                "--resume", str(last_pt),
                "--resume_batch_size", str(args.batch),
                "--resume_imgsz", str(args.imgsz),
                "--resume_device", args.device,
            ]
            action = "RESUME"
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Refusing to overwrite incomplete run without last.pt: {run_dir}")
            command = [
                sys.executable,
                "-u",
                str(TRAIN),
                "--task", "train",
                "--data", str(data),
                "--config", str(config),
                "--weights", "none",
                "--imgsz", str(args.imgsz),
                "--batch_size", str(args.batch),
                "--device", args.device,
            ]
            action = "START"
        command.extend(
            [
                "--epochs", str(args.epochs),
                "--workers", str(args.workers),
                "--optimizer", "AdamW",
                "--lr0", "0.0002",
                "--lrf", "0.01",
                "--weight_decay", "0.0005",
                "--nbs", "64",
                "--warmup_epochs", "3.0",
                "--module_warmup_epochs", "0",
                "--patience", str(args.patience),
                "--cos_lr",
                "--mosaic", "0.3",
                "--close_mosaic", "15",
                "--degrees", "180",
                "--translate", "0.1",
                "--scale", "0.3",
                "--flipud", "0.5",
                "--fliplr", "0.5",
                "--seed", str(args.seed),
                "--deterministic",
                "--project", str(project),
                "--name", run_name,
            ]
        )
        if args.cache:
            command.append("--cache")
        if args.amp:
            command.append("--amp")
        print(f"{action}: {spec.key}; completed epochs={completed_epochs(run_dir / 'results.csv')}")
        run(command, log_dir / f"{run_name}.train.log", args.dry_run)
        if not args.dry_run:
            done.write_text(
                json.dumps(
                    {
                        "finished_at": datetime.now().isoformat(),
                        "epochs_recorded": completed_epochs(run_dir / "results.csv"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    else:
        print(f"SKIP TRAIN: {spec.key}; completion marker exists")

    if args.dry_run:
        return
    if not best_pt.exists():
        raise RuntimeError(f"Missing validation-selected checkpoint: {best_pt}")

    data_root = data.parent
    evaluations = [("overall", data)] + [
        (source, data_root / "subsets" / f"{source}.yaml") for source in SUBSETS
    ]
    for subset, subset_yaml in evaluations:
        output = run_dir / f"eval_val_{subset.lower().replace('-', '_')}.json"
        if output.exists():
            print(f"SKIP EVAL: {spec.key} {subset}")
            continue
        command = [
            sys.executable,
            "-u",
            str(EVALUATE),
            "--weights", str(best_pt),
            "--data", str(subset_yaml),
            "--experiment", run_name,
            "--placement", spec.placement,
            "--variant", spec.variant,
            "--state-dim", str(spec.state_dim),
            "--seed", str(args.seed),
            "--subset", subset,
            "--output-json", str(output),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.eval_batch),
            "--device", args.device,
            "--workers", str(args.workers),
            "--project", str(project / "evaluations"),
            "--name", f"{run_name}_{subset}_val",
        ]
        run(command, log_dir / f"{run_name}.eval.log", False)


def selected(specs: list[Spec], only: list[str]) -> list[Spec]:
    if not only:
        return specs
    return [spec for spec in specs if spec.key in only]


def run_summary(project: Path, experiment_id: str, log: Path) -> None:
    command = [sys.executable, str(SUMMARIZE), "--project", str(project), "--experiment-id", experiment_id]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(result.stdout)
        handle.write(result.stderr)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        raise RuntimeError(f"Metric summary failed ({result.returncode}); inspect {log}")


def main() -> None:
    args = parse_args()
    data = Path(args.data).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()
    if not data.exists():
        raise SystemExit(f"Missing data YAML: {data}")
    project.mkdir(parents=True, exist_ok=True)
    manifest = project / f"{args.experiment_id}-manifest"
    config_dir = manifest / "configs"
    log_dir = ROOT / "logs" / args.experiment_id

    all_specs = placement_specs(args.state_dim)
    fixed_ablation = args.ablation_placement if args.ablation_placement != "auto" else "p2"
    # Generate every possible module-control config so an automatic placement
    # choice remains covered by the frozen protocol.
    config_specs = list(all_specs)
    for placement in PLACEMENTS:
        config_specs.extend(module_specs(placement, args.state_dim))
    configs = {spec.key: write_config(config_dir, spec) for spec in config_specs}
    all_specs.extend(module_specs(fixed_ablation, args.state_dim))
    known = {spec.key for spec in config_specs}
    unknown = set(args.only) - known
    if unknown:
        raise SystemExit(f"Unknown --only experiments: {sorted(unknown)}; known: {sorted(known)}")
    freeze_protocol(args, data, manifest, configs)

    print(f"Experiment: {args.experiment_id}")
    print(f"Data:       {data}")
    print(f"Scratch:    True")
    print(f"Module warmup epochs: 0")
    print(f"Runs:       {len(selected(all_specs, args.only))}")

    placement_only = set(spec.key for spec in placement_specs(args.state_dim))
    requested = set(args.only)
    run_placements = not requested or bool(requested & placement_only)
    if run_placements:
        for spec in selected(placement_specs(args.state_dim), args.only or list(placement_only)):
            try:
                train_and_evaluate(args, spec, configs[spec.key], data, project, log_dir)
            except Exception as error:
                if not args.continue_on_error:
                    raise
                print(f"FAILED {spec.key}: {error}")
            if not args.dry_run:
                run_summary(project, args.experiment_id, log_dir / "summary.log")

    if args.ablation_placement == "auto" and not args.dry_run:
        ablation_placement = best_full_placement(project, args.experiment_id)
        ablation_specs = module_specs(ablation_placement, args.state_dim)
        for spec in ablation_specs:
            configs[spec.key] = write_config(config_dir, spec)
    else:
        ablation_placement = fixed_ablation
        ablation_specs = module_specs(ablation_placement, args.state_dim)

    module_keys = {spec.key for spec in ablation_specs}
    run_modules = not requested or bool(requested & module_keys)
    if run_modules:
        for spec in selected(ablation_specs, args.only or list(module_keys)):
            try:
                train_and_evaluate(args, spec, configs[spec.key], data, project, log_dir)
            except Exception as error:
                if not args.continue_on_error:
                    raise
                print(f"FAILED {spec.key}: {error}")
            if not args.dry_run:
                run_summary(project, args.experiment_id, log_dir / "summary.log")

    if not args.dry_run:
        run_summary(project, args.experiment_id, log_dir / "summary.log")
    print(f"Queue finished: {args.experiment_id}")


if __name__ == "__main__":
    main()
