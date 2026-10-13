#!/usr/bin/env python3
"""Run paper-aligned MSTAR-SOC classification and FUSAR-Map segmentation pairs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_shared_sar_tasks_v2.py"


@dataclass(frozen=True)
class Protocol:
    key: str
    task_dir: str
    folder: str
    imgsz: int
    batch: int
    epochs: int
    patience: int
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    scheduler: str
    lr_milestones: tuple[int, ...]
    lr_step_size: int
    lr_gamma: float
    label_smoothing: float
    mixed_precision: bool
    evidence: str


PROTOCOLS = {
    "MSTAR-SOC": Protocol(
        key="MSTAR-SOC",
        task_dir="classification",
        folder="MSTAR_SOC",
        imgsz=88,
        batch=100,
        epochs=100,
        patience=100,
        optimizer="SGD",
        lr=0.001,
        weight_decay=0.004,
        momentum=0.9,
        scheduler="multistep",
        lr_milestones=(50,),
        lr_step_size=2,
        lr_gamma=0.1,
        label_smoothing=0.0,
        mixed_precision=False,
        evidence="A-ConvNet SOC public implementation: official 17/15 split, 88 crop, batch 100, SGD 100e, lr 1e-3, milestone 50, momentum .9, weight decay .004",
    ),
    "FUSAR-Map": Protocol(
        key="FUSAR-Map",
        task_dir="segmentation",
        folder="FUSAR_Map_Paper_Code",
        imgsz=256,
        batch=10,
        epochs=150,
        patience=150,
        optimizer="Adam",
        lr=0.001,
        weight_decay=0.0,
        momentum=0.9,
        scheduler="step",
        lr_milestones=(),
        lr_step_size=2,
        lr_gamma=0.93,
        label_smoothing=0.0,
        mixed_precision=False,
        evidence="FUSAR-Map paper and released code: WuHan_03_02 test scene, 4 classes with unknown merged into vegetation, 17560/4390 train/val patches plus 1280 test patches, 256 input, batch 10, Adam 1e-3, lr x0.93 every 2e, weighted categorical CE, 150e",
    ),
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--project", default="output_dir/paper_aligned_shared")
    parser.add_argument("--experiment-id", default="paper_aligned_shared_p3_d16_v1")
    parser.add_argument("--datasets", nargs="+", choices=sorted(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--variants", nargs="+", choices=("baseline", "ours"), default=["baseline", "ours"])
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    project = Path(args.project).resolve()
    manifest = {
        "experiment_id": args.experiment_id,
        "selection_policy": "fixed-last-epoch; test evaluated once after training",
        "state_dim": args.state_dim,
        "seed": args.seed,
        "protocols": {key: {**asdict(value), "lr_milestones": list(value.lr_milestones)} for key, value in PROTOCOLS.items()},
    }
    manifest_dir = project / f"{args.experiment_id}-manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for dataset_key in args.datasets:
        protocol = PROTOCOLS[dataset_key]
        manifest_path = data_root / protocol.task_dir / protocol.folder / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing prepared manifest: {manifest_path}")
        for variant in args.variants:
            output = project / args.experiment_id / dataset_key.replace("-", "_") / variant / f"seed_{args.seed}"
            result = output / "result.json"
            if result.exists():
                print(f"SKIP COMPLETE: {result}", flush=True)
                continue
            command = [
                sys.executable,
                "-u",
                str(RUNNER),
                "--manifest", str(manifest_path),
                "--dataset", dataset_key,
                "--variant", variant,
                "--output", str(output),
                "--epochs", str(protocol.epochs),
                "--patience", str(protocol.patience),
                "--batch", str(protocol.batch),
                "--imgsz", str(protocol.imgsz),
                "--workers", str(args.workers),
                "--lr", str(protocol.lr),
                "--weight-decay", str(protocol.weight_decay),
                "--optimizer", protocol.optimizer,
                "--momentum", str(protocol.momentum),
                "--scheduler", protocol.scheduler,
                "--lr-step-size", str(protocol.lr_step_size),
                "--lr-gamma", str(protocol.lr_gamma),
                "--label-smoothing", str(protocol.label_smoothing),
                "--state-dim", str(args.state_dim),
                "--seed", str(args.seed),
                "--device", args.device,
            ]
            if protocol.lr_milestones:
                command.extend(["--lr-milestones", *(str(value) for value in protocol.lr_milestones)])
            if args.amp and protocol.mixed_precision:
                command.append("--amp")
            elif args.amp:
                print(f"AMP DISABLED BY PAPER PROTOCOL: {dataset_key}", flush=True)
            execute(
                command,
                project / "logs" / args.experiment_id / f"{dataset_key}_{variant}.log",
                args.dry_run,
            )


if __name__ == "__main__":
    main()
