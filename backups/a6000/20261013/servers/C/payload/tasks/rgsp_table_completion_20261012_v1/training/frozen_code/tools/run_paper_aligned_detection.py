#!/usr/bin/env python3
"""Run resumable paper-aligned AIR-SARShip-2.0 and SRSDD-v1.0 HBB experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from run_sar4_full_suite import VARIANTS, class_count, slug, write_config


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "mbyolo_train_sarair.py"
EVALUATE = ROOT / "tools" / "evaluate_sarmix.py"


@dataclass(frozen=True)
class Protocol:
    key: str
    folder: str
    yaml_name: str
    imgsz: int
    batch: int
    eval_batch: int
    epochs: int
    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    nbs: int
    warmup_epochs: float
    lr_schedule: str
    lr_milestones: tuple[int, ...]
    mosaic: float
    close_mosaic: int
    degrees: float
    translate: float
    scale: float
    flipud: float
    fliplr: float
    hsv_v: float
    multi_scale: bool
    evidence: str


PROTOCOLS = {
    "AIR-SARShip-2.0": Protocol(
        key="AIR-SARShip-2.0",
        folder="AIR_SARShip_2_Paper",
        yaml_name="AIR_SARShip_2_Paper.yaml",
        imgsz=416,
        batch=12,
        eval_batch=24,
        epochs=240,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        nbs=12,
        warmup_epochs=3.2,
        lr_schedule="multistep",
        lr_milestones=(160, 200),
        mosaic=0.0,
        close_mosaic=0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_v=0.0,
        multi_scale=True,
        evidence="Bao et al. 2021: 210/90, 512 crops with 256 overlap, YOLOv3 416, SGD 240e, batch 12, lr 1e-3, milestones 160/200, warmup 500 iterations",
    ),
    "SRSDD-v1.0": Protocol(
        key="SRSDD-v1.0",
        folder="SRSDD_v1_Paper_HBB",
        yaml_name="SRSDD_v1_Paper_HBB.yaml",
        imgsz=640,
        batch=16,
        eval_batch=16,
        epochs=300,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nbs=64,
        warmup_epochs=3.0,
        lr_schedule="auto",
        lr_milestones=(),
        mosaic=1.0,
        close_mosaic=10,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        hsv_v=0.4,
        multi_scale=False,
        evidence="GDB-YOLOv5s 2024 official 532/134 HBB split; YOLOv5s reference training recipe where the article does not publish a complete executable config",
    ),
}


def completed_epochs(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def execute(command: list[str], log: Path, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {datetime.now().isoformat()} =====\n$ {printable}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); inspect {log}")


def train_command(args, protocol: Protocol, data: Path, config: Path, run_name: str, run_dir: Path) -> list[str]:
    last = run_dir / "weights" / "last.pt"
    command = [sys.executable, "-u", str(TRAIN), "--task", "train", "--data", str(data)]
    if last.exists():
        command.extend(
            [
                "--config", str(config),
                "--resume", str(last),
                "--resume_batch_size", str(protocol.batch),
                "--resume_imgsz", str(protocol.imgsz),
                "--resume_device", args.device,
            ]
        )
    else:
        command.extend(
            [
                "--config", str(config),
                "--weights", args.weights,
                "--batch_size", str(protocol.batch),
                "--imgsz", str(protocol.imgsz),
                "--device", args.device,
            ]
        )
    command.extend(
        [
            "--epochs", str(protocol.epochs),
            "--workers", str(args.workers),
            "--optimizer", protocol.optimizer,
            "--lr0", str(protocol.lr0),
            "--lrf", str(protocol.lrf),
            "--momentum", str(protocol.momentum),
            "--weight_decay", str(protocol.weight_decay),
            "--nbs", str(protocol.nbs),
            "--warmup_epochs", str(protocol.warmup_epochs),
            "--module_warmup_epochs", "0",
            "--patience", str(protocol.epochs),
            "--lr_schedule", protocol.lr_schedule,
            "--lr_gamma", "0.1",
            "--mosaic", str(protocol.mosaic),
            "--close_mosaic", str(protocol.close_mosaic),
            "--degrees", str(protocol.degrees),
            "--translate", str(protocol.translate),
            "--scale", str(protocol.scale),
            "--flipud", str(protocol.flipud),
            "--fliplr", str(protocol.fliplr),
            "--hsv_h", "0.0",
            "--hsv_s", "0.0",
            "--hsv_v", str(protocol.hsv_v),
            "--seed", str(args.seed),
            "--deterministic",
            "--no-val-during-training",
            "--save_period", "10",
            "--project", str(Path(args.project).resolve()),
            "--name", run_name,
        ]
    )
    if protocol.lr_milestones:
        command.extend(["--lr_milestones", *(str(value) for value in protocol.lr_milestones)])
    if protocol.multi_scale:
        command.append("--multi_scale")
    if args.cache:
        command.append("--cache")
    if args.amp:
        command.append("--amp")
    return command


def eval_command(args, protocol: Protocol, variant, data: Path, run_name: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(EVALUATE),
        "--weights", str(run_dir / "weights" / "last.pt"),
        "--data", str(data),
        "--experiment", run_name,
        "--placement", variant.placement,
        "--variant", variant.variant,
        "--state-dim", str(0 if variant.key == "baseline" else args.state_dim),
        "--seed", str(args.seed),
        "--subset", protocol.key,
        "--split", "test",
        "--output-json", str(run_dir / "eval_paper_test.json"),
        "--imgsz", str(protocol.imgsz),
        "--batch", str(protocol.eval_batch),
        "--device", args.eval_device or args.device,
        "--workers", str(args.workers),
        "--project", str(Path(args.project).resolve() / "evaluations"),
        "--name", f"{run_name}_paper_test",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--project", default="output_dir/paper_aligned_detection")
    parser.add_argument("--experiment-id", default="paper_aligned_p3_d16_v1")
    parser.add_argument("--datasets", nargs="+", choices=sorted(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--variants", nargs="+", choices=("baseline", "full_p3"), default=["baseline", "full_p3"])
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="none")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    project = Path(args.project).resolve()
    manifest_dir = project / f"{args.experiment_id}-manifest"
    config_dir = manifest_dir / "configs"
    log_dir = project / "logs" / args.experiment_id
    selected_variants = [item for item in VARIANTS if item.key in args.variants]
    manifest = {
        "experiment_id": args.experiment_id,
        "selection_policy": "fixed-last-epoch; official test evaluated once after training",
        "initialization": args.weights,
        "seed": args.seed,
        "state_dim": args.state_dim,
        "protocols": {},
    }

    for dataset_key in args.datasets:
        protocol = PROTOCOLS[dataset_key]
        data = data_root / "detection" / protocol.folder / protocol.yaml_name
        if not data.exists():
            raise SystemExit(f"Missing prepared dataset: {data}")
        manifest["protocols"][dataset_key] = {**protocol.__dict__, "lr_milestones": list(protocol.lr_milestones)}
        for variant in selected_variants:
            config = write_config(
                config_dir / f"{slug(dataset_key)}_{variant.key}_sd{args.state_dim}.yaml",
                class_count(data),
                variant,
                args.state_dim,
            )
            run_name = f"{args.experiment_id}_{slug(dataset_key)}_{variant.key}"
            run_dir = project / run_name
            result = run_dir / "eval_paper_test.json"
            if result.exists():
                print(f"SKIP COMPLETE: {result}", flush=True)
                continue
            if completed_epochs(run_dir / "results.csv") < protocol.epochs:
                execute(
                    train_command(args, protocol, data, config, run_name, run_dir),
                    log_dir / f"{run_name}.train.log",
                    args.dry_run,
                )
            checkpoint = run_dir / "weights" / "last.pt"
            if not args.dry_run and not checkpoint.exists():
                raise RuntimeError(f"Training did not produce {checkpoint}")
            execute(
                eval_command(args, protocol, variant, data, run_name, run_dir),
                log_dir / f"{run_name}.eval.log",
                args.dry_run,
            )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
