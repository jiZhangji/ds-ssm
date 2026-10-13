#!/usr/bin/env python3
"""Run resumable Baseline versus Full-P3 experiments on the detection matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "mbyolo_train_sarair.py"
EVALUATE = ROOT / "tools" / "evaluate_sarmix.py"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    folder: str
    yaml_name: str


@dataclass(frozen=True)
class VariantSpec:
    key: str
    variant: str
    placement: str


DATASETS = (
    DatasetSpec("SSDD", "SSDD", "SSDD.yaml"),
    DatasetSpec("SAR-AIRcraft1.0", "SAR_AIRcraft", "SAR_AIRcraft.yaml"),
    DatasetSpec("AIR-SARShip-2.0", "AIR_SARShip_2", "AIR_SARShip_2.yaml"),
    DatasetSpec("SRSDD-v1.0", "SRSDD_v1", "SRSDD_v1.yaml"),
    DatasetSpec("SADD", "SADD", "SADD.yaml"),
)
VARIANTS = (
    VariantSpec("baseline", "baseline", "baseline"),
    VariantSpec("full_p3", "full", "p3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing the four prepared datasets")
    parser.add_argument("--project", default="output_dir/sar4_full")
    parser.add_argument("--experiment-id", default="sar4_scratch_p3_sd16_v1")
    parser.add_argument("--datasets", nargs="+", choices=[item.key for item in DATASETS], default=None)
    parser.add_argument("--variants", nargs="+", choices=[item.key for item in VARIANTS], default=None)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ssdd-val-count", type=int, default=186)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(".", "")


def write_frozen(path: Path, content: str, description: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise SystemExit(f"Frozen {description} changed: {path}; use a new experiment ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_dataset_root(source_yaml: Path, payload: dict) -> Path:
    value = Path(payload.get("path", source_yaml.parent))
    if not value.is_absolute():
        value = source_yaml.parent / value
    return value.expanduser().resolve()


def prepare_ssdd_split(source_yaml: Path, output: Path, seed: int, val_count: int) -> Path:
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root(source_yaml, payload)
    official_train = image_files(dataset_root / "images" / "train")
    official_test = image_files(dataset_root / "images" / "test")
    if not official_train or not official_test:
        raise SystemExit(f"SSDD requires non-empty official train/test directories under {dataset_root}")
    if not 0 < val_count < len(official_train):
        raise SystemExit(f"Invalid SSDD validation count {val_count} for {len(official_train)} train images")

    shuffled = list(official_train)
    random.Random(seed).shuffle(shuffled)
    val_images = sorted(shuffled[:val_count])
    train_images = sorted(shuffled[val_count:])
    split_dir = output.parent / "ssdd_split"
    train_list = split_dir / "train.txt"
    val_list = split_dir / "val.txt"
    write_frozen(train_list, "\n".join(map(str, train_images)) + "\n", "SSDD train split")
    write_frozen(val_list, "\n".join(map(str, val_images)) + "\n", "SSDD validation split")

    generated = {
        "path": str(dataset_root),
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "test": payload.get("test", "images/test"),
        "names": payload["names"],
    }
    rendered = yaml.safe_dump(generated, sort_keys=False, allow_unicode=True)
    write_frozen(output, rendered, "SSDD data YAML")
    snapshot = {
        "official_train": len(official_train),
        "development_train": len(train_images),
        "development_val": len(val_images),
        "official_test": len(official_test),
        "seed": seed,
    }
    write_frozen(
        split_dir / "snapshot.json",
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        "SSDD split snapshot",
    )
    return output


def class_count(data_yaml: Path) -> int:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names")
    if not isinstance(names, (dict, list)) or not names:
        raise SystemExit(f"Missing class names in {data_yaml}")
    return len(names)


def model_config(nc: int, placement: str, state_dim: int, variant: str) -> dict:
    from run_sarmix_proxy import model_payload

    payload = model_payload(placement, state_dim, "full" if variant == "baseline" else variant)
    payload["nc"] = nc
    payload["head"][-1][3] = [nc]
    return payload


def write_config(path: Path, nc: int, variant: VariantSpec, state_dim: int) -> Path:
    config_state_dim = 0 if variant.key == "baseline" else state_dim
    payload = model_config(nc, variant.placement, config_state_dim, variant.variant)
    rendered = yaml.safe_dump(payload, sort_keys=False)
    write_frozen(path, rendered, "model config")
    return path


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    try:
        with results_csv.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except OSError:
        return 0


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


def train_command(args, data: Path, config: Path, run_name: str, run_dir: Path) -> tuple[str, list[str]]:
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        action = "RESUME"
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
    else:
        action = "START"
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
            "--project", str(Path(args.project).expanduser().resolve()),
            "--name", run_name,
        ]
    )
    if args.cache:
        command.append("--cache")
    if args.amp:
        command.append("--amp")
    return action, command


def evaluation_command(
    args,
    dataset: DatasetSpec,
    variant: VariantSpec,
    data: Path,
    run_name: str,
    run_dir: Path,
) -> list[str]:
    state_dim = 0 if variant.key == "baseline" else args.state_dim
    return [
        sys.executable,
        "-u",
        str(EVALUATE),
        "--weights", str(run_dir / "weights" / "best.pt"),
        "--data", str(data),
        "--experiment", run_name,
        "--placement", variant.placement,
        "--variant", variant.variant,
        "--state-dim", str(state_dim),
        "--seed", str(args.seed),
        "--subset", dataset.key,
        "--split", "test",
        "--output-json", str(run_dir / "eval_test.json"),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.eval_batch),
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", str(Path(args.project).expanduser().resolve() / "evaluations"),
        "--name", f"{run_name}_test",
    ]


def run_one(args, dataset: DatasetSpec, variant: VariantSpec, data: Path, config: Path, log_dir: Path) -> None:
    project = Path(args.project).expanduser().resolve()
    run_name = f"{args.experiment_id}_{slug(dataset.key)}_{variant.key}"
    run_dir = project / run_name
    done = run_dir / "train_complete.json"
    if not done.exists():
        if run_dir.exists() and any(run_dir.iterdir()) and not (run_dir / "weights" / "last.pt").exists():
            raise RuntimeError(f"Refusing to overwrite incomplete run without last.pt: {run_dir}")
        action, command = train_command(args, data, config, run_name, run_dir)
        print(f"{action}: {dataset.key} / {variant.key}; epochs={completed_epochs(run_dir / 'results.csv')}")
        run(command, log_dir / f"{run_name}.train.log", args.dry_run)
        if not args.dry_run:
            best_pt = run_dir / "weights" / "best.pt"
            epochs_recorded = completed_epochs(run_dir / "results.csv")
            if not best_pt.exists() or not epochs_recorded:
                raise RuntimeError(f"Training exited without a usable checkpoint: {run_dir}")
            done.write_text(
                json.dumps(
                    {
                        "finished_at": datetime.now().isoformat(),
                        "epochs_recorded": epochs_recorded,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    else:
        print(f"SKIP TRAIN: {dataset.key} / {variant.key}")

    evaluate = evaluation_command(args, dataset, variant, data, run_name, run_dir)
    if args.dry_run:
        run(evaluate, log_dir / f"{run_name}.eval.log", True)
        return
    if not (run_dir / "weights" / "best.pt").exists():
        raise RuntimeError(f"Missing validation-selected checkpoint: {run_dir / 'weights' / 'best.pt'}")
    if (run_dir / "eval_test.json").exists():
        print(f"SKIP EVAL: {dataset.key} / {variant.key}")
    else:
        run(evaluate, log_dir / f"{run_name}.eval.log", False)


def summarize(project: Path, experiment_id: str) -> None:
    rows = []
    for path in sorted(project.glob(f"{experiment_id}_*/eval_test.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        metric = item["overall"]
        rows.append(
            {
                "dataset": item["subset"],
                "experiment": item["experiment"],
                "variant": item["variant"],
                "placement": item["placement"],
                "state_dim": item["state_dim"],
                **metric,
                "inference_ms": item.get("speed_ms", {}).get("inference", ""),
                "weights": item["weights"],
            }
        )
    if not rows:
        return
    rows.sort(key=lambda item: (item["dataset"], item["variant"] != "baseline"))
    fields = tuple(rows[0])
    csv_path = project / f"{experiment_id}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = project / f"{experiment_id}_summary.json"
    json_path.write_text(json.dumps({"experiment_id": experiment_id, "records": rows}, indent=2), encoding="utf-8")

    lines = [
        f"# Four-Dataset SAR Results: {experiment_id}",
        "",
        "All rows use random initialization and report the official test split after validation-set checkpoint selection.",
        "",
        "| Dataset | Variant | Placement | P | R | AP50 | AP75 | AP50:95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['dataset']} | {item['variant']} | {item['placement']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['ap50']:.4f} | "
            f"{item['ap75']:.4f} | {item['ap50_95']:.4f} |"
        )
    indexed = {(item["dataset"], item["variant"]): item for item in rows}
    gains = []
    for dataset in DATASETS:
        baseline = indexed.get((dataset.key, "baseline"))
        full = indexed.get((dataset.key, "full"))
        if baseline and full:
            gains.append(
                {
                    "dataset": dataset.key,
                    "ap50": full["ap50"] - baseline["ap50"],
                    "ap75": full["ap75"] - baseline["ap75"],
                    "ap50_95": full["ap50_95"] - baseline["ap50_95"],
                }
            )
    if gains:
        lines.extend(
            [
                "",
                "## Full-P3 Gain over Baseline",
                "",
                "| Dataset | Delta AP50 | Delta AP75 | Delta AP50:95 |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in gains:
            lines.append(
                f"| {item['dataset']} | {item['ap50']:+.4f} | "
                f"{item['ap75']:+.4f} | {item['ap50_95']:+.4f} |"
            )
    md_path = project / f"{experiment_id}_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {md_path}")


def freeze_protocol(args, manifest: Path, data_files: dict[str, Path], configs: dict[str, Path]) -> None:
    payload = {
        "experiment_id": args.experiment_id,
        "datasets": {key: {"path": str(path), "sha256": sha256(path)} for key, path in data_files.items()},
        "state_dim": args.state_dim,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "optimizer": "AdamW",
        "lr0": 0.0002,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "nbs": 64,
        "optimizer_warmup_epochs": 3.0,
        "module_warmup_epochs": 0,
        "weights": "none",
        "seed": args.seed,
        "amp": args.amp,
        "configs": {key: sha256(path) for key, path in configs.items()},
        "code": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (TRAIN, EVALUATE, Path(__file__).resolve(), ROOT / "ultralytics/nn/modules/mamba_yolo.py")
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_frozen(manifest / "protocol.json", rendered, "protocol")


def main() -> None:
    args = parse_args()
    args.project = str(Path(args.project).expanduser().resolve())
    data_root = Path(args.data_root).expanduser().resolve()
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    manifest = project / f"{args.experiment_id}-manifest"
    config_dir = manifest / "configs"
    data_dir = manifest / "datasets"
    log_dir = ROOT / "logs" / args.experiment_id

    selected_datasets = set(args.datasets or [item.key for item in DATASETS])
    selected_variants = set(args.variants or [item.key for item in VARIANTS])
    known_datasets = {item.key for item in DATASETS}
    known_variants = {item.key for item in VARIANTS}
    unknown_datasets = selected_datasets - known_datasets
    unknown_variants = selected_variants - known_variants
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {sorted(unknown_datasets)}")
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {sorted(unknown_variants)}")

    data_files = {}
    configs = {}
    for dataset in DATASETS:
        if dataset.key not in selected_datasets:
            continue
        source = data_root / dataset.folder / dataset.yaml_name
        if not source.exists():
            raise SystemExit(f"Missing dataset YAML: {source}")
        if dataset.key == "SSDD":
            data = prepare_ssdd_split(source, data_dir / "SSDD.yaml", args.seed, args.ssdd_val_count)
        else:
            data = source.resolve()
        data_files[dataset.key] = data
        nc = class_count(data)
        for variant in VARIANTS:
            key = f"{dataset.key}:{variant.key}"
            configs[key] = write_config(
                config_dir / f"{slug(dataset.key)}_{variant.key}.yaml",
                nc,
                variant,
                args.state_dim,
            )
    freeze_protocol(args, manifest, data_files, configs)

    queue = [
        (dataset, variant)
        for dataset in DATASETS
        for variant in VARIANTS
        if dataset.key in selected_datasets and variant.key in selected_variants
    ]
    print(f"Experiment: {args.experiment_id}")
    print(f"Runs:       {len(queue)}")
    print("Scratch:    True")
    print("Placement:  P3 for Full; original Mamba-YOLO for Baseline")
    print("Eval split: test")

    for dataset, variant in queue:
        try:
            run_one(
                args,
                dataset,
                variant,
                data_files[dataset.key],
                configs[f"{dataset.key}:{variant.key}"],
                log_dir,
            )
        except Exception as error:
            if not args.continue_on_error:
                raise
            print(f"FAILED {dataset.key} / {variant.key}: {error}")
        if not args.dry_run:
            summarize(project, args.experiment_id)
    if not args.dry_run:
        summarize(project, args.experiment_id)
    print(f"Queue finished: {args.experiment_id}")


if __name__ == "__main__":
    main()
