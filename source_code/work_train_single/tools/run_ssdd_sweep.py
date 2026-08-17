#!/usr/bin/env python3
"""Train and evaluate a resumable SSDD model/recipe sweep.

The sweep generates explicit model YAMLs for placement comparisons, keeps the
state dimension fixed by using DS_VSSBlockSweep, trains each run sequentially,
and evaluates validation-selected best.pt on VAL with standard inference and
optional TTA. TEST evaluation must be explicitly requested after model
selection.

Examples:
  python tools/run_ssdd_sweep.py --preset core --dry-run
  python tools/run_ssdd_sweep.py --preset placement --devices 0,1 --tta
  python tools/run_ssdd_sweep.py --preset optimizer --seeds 0 1 2
  python tools/run_ssdd_sweep.py --preset extended --allow-large-sweep
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = ROOT / "mbyolo_train_sarair.py"
EVAL_ENTRY = ROOT / "tools" / "evaluate_ssdd_sweep.py"
BASE_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "SSDD_HF.yaml"


@dataclass(frozen=True)
class Recipe:
    name: str
    optimizer: str
    lr0: float
    lrf: float
    weight_decay: float
    nbs: int
    warmup_epochs: float
    module_warmup_epochs: int
    cos_lr: bool
    mosaic: float
    close_mosaic: int
    degrees: float
    translate: float
    scale: float
    flipud: float
    fliplr: float
    batch: int = 32
    amp: bool = False


RECIPES = {
    "sgd_ref": Recipe(
        "sgd_ref", "SGD", 0.01, 0.01, 0.0005, 64, 3.0, 0, False,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "sgd_cos": Recipe(
        "sgd_cos", "SGD", 0.01, 0.01, 0.0005, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "sgd_bs64": Recipe(
        "sgd_bs64", "SGD", 0.01, 0.01, 0.0005, 64, 3.0, 0, False,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5, batch=64,
    ),
    "adamw_1e4": Recipe(
        "adamw_1e4", "AdamW", 0.0001, 0.01, 0.000125, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "adamw_2e4": Recipe(
        "adamw_2e4", "AdamW", 0.0002, 0.01, 0.0005, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "adamw_bs64": Recipe(
        "adamw_bs64", "AdamW", 0.0002, 0.01, 0.0005, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5, batch=64,
    ),
    "adamw_3e4": Recipe(
        "adamw_3e4", "AdamW", 0.0003, 0.01, 0.0005, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "adamw_lowwd": Recipe(
        "adamw_lowwd", "AdamW", 0.0002, 0.01, 0.000125, 64, 3.0, 0, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "adamw_nomosaic": Recipe(
        "adamw_nomosaic", "AdamW", 0.0002, 0.01, 0.0005, 64, 3.0, 0, True,
        0.0, 0, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
    "adamw_mildaug": Recipe(
        "adamw_mildaug", "AdamW", 0.0002, 0.01, 0.0005, 64, 3.0, 0, True,
        0.2, 20, 90.0, 0.1, 0.2, 0.5, 0.5,
    ),
    "adamw_warmup20": Recipe(
        "adamw_warmup20", "AdamW", 0.0002, 0.01, 0.0005, 64, 3.0, 20, True,
        0.3, 30, 180.0, 0.1, 0.3, 0.5, 0.5,
    ),
}

PLACEMENTS = (
    "baseline",
    "p2",
    "p3",
    "p4",
    "p5",
    "p2p3",
    "p2p4",
    "p3p4",
    "p2p3p4",
    "p2p3p4p5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("core", "placement", "optimizer", "extended", "custom"), default="core")
    parser.add_argument("--data", default=str(BASE_DATA))
    parser.add_argument("--project", default="output_dir/ssdd_sweep")
    parser.add_argument("--sweep-id", default="")
    parser.add_argument("--devices", default="0", help="physical GPUs, e.g. 0 or 0,1")
    parser.add_argument("--eval-device", default="0", help="logical evaluation GPU after CUDA masking")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="Use val for sweep selection; reserve test for the frozen winner",
    )
    parser.add_argument("--eval-conf", type=float, default=0.001)
    parser.add_argument("--eval-iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--placements", nargs="+", default=[])
    parser.add_argument("--recipes", nargs="+", default=[])
    parser.add_argument("--state-dims", nargs="+", type=int, default=[16])
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--weights", default="none")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-large-sweep", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def logical_devices(physical: str) -> str:
    count = len([item for item in physical.split(",") if item.strip()])
    return ",".join(str(index) for index in range(count)) or "0"


def add_layer(items: list, from_value, repeats: int, module: str, args: list) -> int:
    index = len(items)
    items.append([from_value, repeats, module, args])
    return index


def make_model_yaml(placement: str, state_dim: int, output: Path) -> None:
    if placement not in PLACEMENTS:
        raise ValueError(f"Unknown placement: {placement}")

    selected = set() if placement == "baseline" else {
        f"p{value}" for value in placement.removeprefix("p").split("p")
    }
    ds = "DS_VSSBlockSweep"

    backbone: list = []
    add_layer(backbone, -1, 1, "SimpleStem", [128, 3])

    p2_module = ds if "p2" in selected else "VSSBlock"
    p2_args = [128, state_dim] if p2_module == ds else [128]
    p2 = add_layer(backbone, -1, 3, p2_module, p2_args)

    add_layer(backbone, -1, 1, "VisionClueMerge", [256])
    p3_module = ds if "p3" in selected else "VSSBlock"
    p3_args = [256, state_dim] if p3_module == ds else [256]
    p3 = add_layer(backbone, -1, 3, p3_module, p3_args)

    add_layer(backbone, -1, 1, "VisionClueMerge", [512])
    if "p4" in selected:
        # T depth scaling turns repeats 6 + 3 into 2 ordinary VSS blocks
        # followed by one DS block, preserving the original three-block P4 stage.
        add_layer(backbone, -1, 6, "VSSBlock", [512])
        p4 = add_layer(backbone, -1, 3, ds, [512, state_dim])
    else:
        p4 = add_layer(backbone, -1, 9, "VSSBlock", [512])

    add_layer(backbone, -1, 1, "VisionClueMerge", [1024])
    p5_module = ds if "p5" in selected else "VSSBlock"
    p5_args = [1024, state_dim] if p5_module == ds else [1024]
    p5 = add_layer(backbone, -1, 3, p5_module, p5_args)
    p5_sppf = add_layer(backbone, -1, 1, "SPPF", [1024, 5])

    head: list = []

    def add_head(from_value, repeats: int, module: str, args: list) -> int:
        index = len(backbone) + len(head)
        head.append([from_value, repeats, module, args])
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

    add_head([head_p3, head_p4_out, head_p5], 1, "Detect", [1])

    payload = {
        "nc": 1,
        "scales": {"T": [0.33, 0.25, 1024]},
        "scale": "T",
        "backbone": backbone,
        "head": head,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False)
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(
                f"Generated config changed for frozen sweep: {output}; use a new --sweep-id"
            )
        return
    output.write_text(rendered, encoding="utf-8")


def experiments_for(args: argparse.Namespace) -> list[dict]:
    def add(placement: str, recipe: str, state_dim: int) -> dict:
        return {
            "placement": placement,
            "recipe": recipe,
            "state_dim": 0 if placement == "baseline" else state_dim,
        }

    if args.preset == "placement":
        return [add(placement, "sgd_ref", 16) for placement in PLACEMENTS]

    if args.preset == "optimizer":
        return [add("p2", recipe, 16) for recipe in RECIPES]

    if args.preset == "extended":
        selected_placements = PLACEMENTS
        selected_recipes = ("sgd_ref", "sgd_cos", "adamw_1e4", "adamw_2e4")
        runs = [
            add(placement, recipe, 16)
            for placement in selected_placements
            for recipe in selected_recipes
        ]
        runs.extend(add("p2", "sgd_ref", state_dim) for state_dim in (8, 32, 64))
        return runs

    if args.preset == "custom":
        placements = args.placements or ["p2"]
        recipes = args.recipes or ["sgd_ref"]
        state_dims = args.state_dims or [16]
        runs = []
        for placement in placements:
            for recipe in recipes:
                for state_dim in ([0] if placement == "baseline" else state_dims):
                    runs.append(add(placement, recipe, state_dim))
        return runs

    # Core: placement screening, selected optimizer settings, and explicit
    # state-dimension checks for the current P2 location.
    runs = [
        add(placement, "sgd_ref", 16)
        for placement in ("baseline", "p2", "p3", "p4", "p5", "p2p3")
    ]
    runs.extend(
        add("p2", recipe, 16)
        for recipe in (
            "sgd_bs64", "adamw_1e4", "adamw_2e4", "adamw_bs64",
            "adamw_3e4", "adamw_lowwd", "adamw_nomosaic", "adamw_warmup20",
        )
    )
    runs.extend(add("p2", "sgd_ref", state_dim) for state_dim in (8, 32))
    return runs


def count_completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    values = []
    for row in rows:
        try:
            values.append(int(float(row.get("epoch", 0))))
        except (TypeError, ValueError):
            continue
    return max(values, default=0)


def json_is_current(path: Path, weights: Path, mode: str, split: str) -> bool:
    if not path.exists() or not weights.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("mode") == mode
            and payload.get("split") == split
            and payload.get("weights_sha256") == sha256(weights)
        )
    except (OSError, json.JSONDecodeError):
        return False


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(subprocess.list2cmdline([part]) for part in command)
    print(f"\n$ {printable}")
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now().isoformat()} =====\n$ {printable}\n")
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {log_path}")


def build_train_command(args: argparse.Namespace, spec: dict, config: Path, run_name: str, run_dir: Path, resume: Path | None) -> list[str]:
    recipe = RECIPES[spec["recipe"]]
    command = [
        sys.executable, "-u", str(TRAIN_ENTRY),
        "--task", "train",
        "--data", str(Path(args.data).resolve()),
        "--config", str(config),
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch_size", str(recipe.batch),
        "--device", logical_devices(args.devices),
        "--workers", str(args.workers),
        "--optimizer", recipe.optimizer,
        "--lr0", str(recipe.lr0),
        "--lrf", str(recipe.lrf),
        "--weight_decay", str(recipe.weight_decay),
        "--nbs", str(recipe.nbs),
        "--warmup_epochs", str(recipe.warmup_epochs),
        "--module_warmup_epochs", str(recipe.module_warmup_epochs),
        "--patience", str(args.patience),
        "--mosaic", str(recipe.mosaic),
        "--close_mosaic", str(recipe.close_mosaic),
        "--degrees", str(recipe.degrees),
        "--translate", str(recipe.translate),
        "--scale", str(recipe.scale),
        "--flipud", str(recipe.flipud),
        "--fliplr", str(recipe.fliplr),
        "--project", str(Path(args.project).resolve()),
        "--name", run_name,
        "--deterministic",
    ]
    if recipe.cos_lr:
        command.append("--cos_lr")
    if recipe.amp:
        command.append("--amp")
    if args.cache:
        command.append("--cache")
    if resume is not None:
        command.extend([
            "--resume", str(resume),
            "--resume_batch_size", str(recipe.batch),
            "--resume_imgsz", str(args.imgsz),
            "--resume_device", logical_devices(args.devices),
        ])
    return command


def build_eval_command(args: argparse.Namespace, spec: dict, run_name: str, weights: Path, mode: str, output_json: Path, eval_project: Path) -> list[str]:
    return [
        sys.executable, "-u", str(EVAL_ENTRY),
        "--weights", str(weights),
        "--data", str(Path(args.data).resolve()),
        "--run-name", run_name,
        "--placement", spec["placement"],
        "--recipe", spec["recipe"],
        "--state-dim", str(spec["state_dim"]),
        "--seed", str(spec["seed"]),
        "--mode", mode,
        "--split", args.eval_split,
        "--output-json", str(output_json),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.eval_batch),
        "--device", args.eval_device,
        "--workers", str(args.workers),
        "--conf", str(args.eval_conf),
        "--iou", str(args.eval_iou),
        "--project", str(eval_project),
        "--name", f"{run_name}_{mode}_{args.eval_split}",
    ]


def write_summary(project: Path, sweep_id: str) -> None:
    records = []
    for path in sorted(project.glob(f"{sweep_id}_*/eval_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not records:
        return

    fields = [
        "run_name", "placement", "recipe", "state_dim", "seed", "split", "mode",
        "precision", "recall", "f1", "ap50", "ap75", "ap50_95",
        "parameters", "gflops", "result_dir",
    ]
    csv_path = project / f"{sweep_id}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(records, key=lambda row: (-row["overall"]["ap50_95"], row["mode"], row["run_name"])):
            writer.writerow({
                "run_name": item["run_name"],
                "placement": item["placement"],
                "recipe": item["recipe"],
                "state_dim": item["state_dim"],
                "seed": item["seed"],
                "split": item["split"],
                "mode": item["mode"],
                **item["overall"],
                "parameters": item["model"]["parameters"],
                "gflops": item["model"]["gflops"],
                "result_dir": item["result_dir"],
            })

    lines = [
        f"# SSDD Sweep: {sweep_id}",
        "",
        f"{records[0]['split'].upper()} results from validation-selected best.pt checkpoints. Standard and TTA are listed separately.",
        "",
        "| Rank | Placement | Recipe | State | Seed | Split | Mode | P | R | F1 | AP50 | AP75 | AP50:95 | Params | GFLOPs |",
        "|---:|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(sorted(records, key=lambda row: (-row["overall"]["ap50_95"], row["mode"], row["run_name"])), 1):
        overall = item["overall"]
        lines.append(
            f"| {rank} | {item['placement']} | {item['recipe']} | {item['state_dim']} | "
            f"{item['seed']} | {item['split']} | {item['mode']} | {100*overall['precision']:.3f} | "
            f"{100*overall['recall']:.3f} | {100*overall['f1']:.3f} | "
            f"{100*overall['ap50']:.3f} | {100*overall['ap75']:.3f} | "
            f"{100*overall['ap50_95']:.3f} | {item['model']['parameters']} | "
            f"{item['model']['gflops']:.3f} |"
        )

    by_run_mode = {(item["run_name"], item["mode"]): item for item in records}
    paired_runs = sorted(
        run_name
        for run_name, mode in by_run_mode
        if mode == "standard" and (run_name, "tta") in by_run_mode
    )
    if paired_runs:
        lines.extend(
            [
                "",
                "## TTA Delta vs Standard",
                "",
                "| Run | dP | dR | dAP50 | dAP75 | dAP50:95 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for run_name in paired_runs:
            standard = by_run_mode[(run_name, "standard")]["overall"]
            tta = by_run_mode[(run_name, "tta")]["overall"]
            lines.append(
                f"| {run_name} | {100*(tta['precision']-standard['precision']):+.3f} | "
                f"{100*(tta['recall']-standard['recall']):+.3f} | "
                f"{100*(tta['ap50']-standard['ap50']):+.3f} | "
                f"{100*(tta['ap75']-standard['ap75']):+.3f} | "
                f"{100*(tta['ap50_95']-standard['ap50_95']):+.3f} |"
            )
    (project / f"{sweep_id}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary MD : {project / f'{sweep_id}_summary.md'}")


def main() -> None:
    args = parse_args()
    if args.sweep_id == "":
        args.sweep_id = f"ssdd_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.preset == "extended" and not args.allow_large_sweep:
        raise SystemExit("extended preset requires --allow-large-sweep")

    data_path = Path(args.data).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()
    manifest_dir = project / f"{args.sweep_id}-manifest"
    config_dir = manifest_dir / "configs"
    log_dir = ROOT / "logs" / args.sweep_id
    eval_project = project / "evaluations"
    data_path.exists() or raise_missing(data_path)

    specs = experiments_for(args)
    unique = {}
    for spec in specs:
        key = (spec["placement"], spec["recipe"], spec["state_dim"])
        unique[key] = spec
    specs = list(unique.values())
    if not specs:
        raise SystemExit("No experiments selected")
    if len(specs) > 60 and not args.allow_large_sweep:
        raise SystemExit(f"Selected {len(specs)} experiments; add --allow-large-sweep")

    config_paths = {}
    for spec in specs:
        placement = spec["placement"]
        state_dim = spec["state_dim"]
        key = (placement, state_dim)
        if key not in config_paths:
            config_path = config_dir / f"{placement}_sd{state_dim}.yaml"
            if not args.dry_run or not config_path.exists():
                make_model_yaml(placement, state_dim, config_path)
            config_paths[key] = config_path
        spec["config"] = str(config_paths[key])

    for spec in specs:
        for seed in args.seeds:
            spec_name = f"{spec['placement']}_sd{spec['state_dim']}_{spec['recipe']}_seed{seed}"
            spec.setdefault("expanded", []).append({"seed": seed, "run_name": f"{args.sweep_id}_{spec_name}"})

    protocol = {
        "sweep_id": args.sweep_id,
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "devices": args.devices,
        "logical_train_devices": logical_devices(args.devices),
        "eval_device": args.eval_device,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "patience": args.patience,
        "workers": args.workers,
        "eval_batch": args.eval_batch,
        "eval_split": args.eval_split,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,
        "seeds": args.seeds,
        "tta": args.tta,
        "cache": args.cache,
        "weights": args.weights,
        "experiments": [
            {key: value for key, value in spec.items() if key not in ("expanded", "config")}
            for spec in specs
        ],
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                TRAIN_ENTRY,
                EVAL_ENTRY,
                ROOT / "tools" / "run_ssdd_sweep.py",
                ROOT / "ultralytics" / "nn" / "modules" / "mamba_yolo.py",
                ROOT / "ultralytics" / "nn" / "modules" / "__init__.py",
                ROOT / "ultralytics" / "nn" / "tasks.py",
                ROOT / "ultralytics" / "engine" / "trainer.py",
                ROOT / "ultralytics" / "engine" / "validator.py",
                ROOT / "ultralytics" / "utils" / "metrics.py",
            )
        },
        "config_sha256": {
            f"{placement}_sd{state_dim}": sha256(path)
            for (placement, state_dim), path in sorted(config_paths.items())
        },
    }
    protocol_text = json.dumps(protocol, indent=2, sort_keys=True)
    manifest_file = manifest_dir / "sweep_manifest.json"
    if manifest_file.exists() and not args.dry_run:
        old = manifest_file.read_text(encoding="utf-8")
        if old != protocol_text:
            raise SystemExit(f"Frozen sweep protocol changed: {manifest_file}; use a new --sweep-id")
    elif not args.dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(protocol_text, encoding="utf-8")

    print(f"Sweep: {args.sweep_id}")
    print(f"Data : {data_path}")
    print(f"Runs : {len(specs) * len(args.seeds)} training runs")
    print(f"TTA  : {args.tta}")
    for spec in specs:
        print(f"  - {spec['placement']} / {spec['recipe']} / state_dim={spec['state_dim']}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    expanded = []
    for spec in specs:
        for seed in args.seeds:
            expanded.append({**spec, "seed": seed})

    if args.dry_run:
        print("DRY RUN COMMANDS:")
        for spec in expanded:
            run_name = f"{args.sweep_id}_{spec['placement']}_sd{spec['state_dim']}_{spec['recipe']}_seed{spec['seed']}"
            run_dir = project / run_name
            best_pt = run_dir / "weights" / "best.pt"
            train = build_train_command(args, spec, Path(spec["config"]), run_name, run_dir, None)
            train.extend(["--seed", str(spec["seed"])])
            run_logged(train, ROOT, env, log_dir / f"{run_name}.train.log", True)
            for mode in ("standard", "tta") if args.tta else ("standard",):
                evaluate = build_eval_command(
                    args,
                    spec,
                    run_name,
                    best_pt,
                    mode,
                    run_dir / f"eval_{args.eval_split}_{mode}.json",
                    eval_project,
                )
                run_logged(evaluate, ROOT, env, log_dir / f"{run_name}.eval.log", True)
        print("DRY RUN: no training or evaluation started")
        return

    for spec in expanded:
        run_name = f"{args.sweep_id}_{spec['placement']}_sd{spec['state_dim']}_{spec['recipe']}_seed{spec['seed']}"
        run_dir = project / run_name
        weights_dir = run_dir / "weights"
        last_pt = weights_dir / "last.pt"
        best_pt = weights_dir / "best.pt"
        results_csv = run_dir / "results.csv"
        train_log = log_dir / f"{run_name}.train.log"
        eval_log = log_dir / f"{run_name}.eval.log"
        completed = count_completed_epochs(results_csv)

        if completed < args.epochs:
            resume = last_pt if last_pt.exists() else None
            command = build_train_command(args, spec, Path(spec["config"]), run_name, run_dir, resume)
            command.extend(["--seed", str(spec["seed"])])
            try:
                run_logged(command, ROOT, env, train_log, args.dry_run)
            except Exception:
                if not args.continue_on_error:
                    raise
                print(f"Training failed for {run_name}; continuing because --continue-on-error was set")
                continue
        else:
            print(f"\nSKIP TRAIN {run_name}: results.csv reaches epoch {completed}")

        if not best_pt.exists():
            print(f"SKIP EVAL {run_name}: missing {best_pt}")
            continue

        for mode in ("standard", "tta") if args.tta else ("standard",):
            output_json = run_dir / f"eval_{args.eval_split}_{mode}.json"
            if json_is_current(output_json, best_pt, mode, args.eval_split):
                print(f"SKIP EVAL {run_name} {mode}: current JSON exists")
                continue
            command = build_eval_command(args, spec, run_name, best_pt, mode, output_json, eval_project)
            try:
                run_logged(command, ROOT, env, eval_log, args.dry_run)
            except Exception:
                if not args.continue_on_error:
                    raise
                print(f"Evaluation failed for {run_name} {mode}; continuing because --continue-on-error was set")

        write_summary(project, args.sweep_id)

    write_summary(project, args.sweep_id)


def raise_missing(path: Path) -> None:
    raise SystemExit(f"Missing dataset YAML: {path}")


if __name__ == "__main__":
    main()
