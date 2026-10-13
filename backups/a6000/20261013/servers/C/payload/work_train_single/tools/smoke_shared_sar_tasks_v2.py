#!/usr/bin/env python3
"""GPU smoke test for the shared SAR task runner using synthetic data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb(path: Path, value: int, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), value, dtype=np.uint8)
    image[16:48, 16:48] = np.asarray([(value + 80) % 255, value, 255 - value], dtype=np.uint8)
    Image.fromarray(image).save(path)


def write_manifest(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_classification(root: Path) -> Path:
    records = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for index in range(count):
            label = index % 2
            image = root / "classification" / split / f"{index}.png"
            save_rgb(image, 32 + label * 160 + index)
            records.append({"path": str(image), "label": label, "split": split})
    return write_manifest(
        root / "classification" / "manifest.json",
        {"task": "classification", "classes": ["class_0", "class_1"], "protocol": "synthetic_gpu_smoke", "records": records},
    )


def prepare_segmentation(root: Path) -> Path:
    records = []
    palette = [[0, 0, 0], [255, 0, 0]]
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for index in range(count):
            image = root / "segmentation" / split / f"image_{index}.png"
            mask = root / "segmentation" / split / f"mask_{index}.png"
            save_rgb(image, 48 + index)
            target = np.zeros((64, 64, 3), dtype=np.uint8)
            target[16:48, 16:48] = np.asarray(palette[1], dtype=np.uint8)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target).save(mask)
            records.append({"image": str(image), "mask": str(mask), "split": split})
    return write_manifest(
        root / "segmentation" / "manifest.json",
        {"task": "segmentation", "classes": 2, "palette": palette, "protocol": "synthetic_gpu_smoke", "records": records},
    )


def prepare_change_detection(root: Path) -> Path:
    records = []
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for index in range(count):
            directory = root / "change_detection" / split
            image1 = directory / f"image1_{index}.png"
            image2 = directory / f"image2_{index}.png"
            mask = directory / f"mask_{index}.png"
            save_rgb(image1, 40 + index)
            save_rgb(image2, 60 + index)
            target = np.zeros((64, 64), dtype=np.uint8)
            target[20:44, 20:44] = 255
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target).save(mask)
            records.append(
                {
                    "image1": str(image1),
                    "image2": str(image2),
                    "mask": str(mask),
                    "crop": [0, 0, 64, 64],
                    "split": split,
                }
            )
    return write_manifest(
        root / "change_detection" / "manifest.json",
        {"task": "change_detection", "classes": 1, "protocol": "synthetic_gpu_smoke", "records": records},
    )


def run_once(runner: Path, manifest: Path, task: str, args, epochs: int, log: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(runner),
        "--manifest",
        str(manifest),
        "--dataset",
        f"smoke_{task}",
        "--variant",
        args.variant,
        "--output",
        str(log.parent),
        "--epochs",
        str(epochs),
        "--patience",
        "3",
        "--batch",
        "2",
        "--imgsz",
        "64",
        "--workers",
        "0",
        "--state-dim",
        str(args.state_dim),
        "--device",
        args.device,
        "--amp",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.visible_device
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{task} failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/dev/shm/ig_dsssm_shared_v2_smoke")
    parser.add_argument("--variant", choices=("baseline", "ours"), required=True)
    parser.add_argument("--visible-device", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--tasks", nargs="+", choices=("classification", "segmentation", "change_detection"), default=("classification", "segmentation", "change_detection"))
    args = parser.parse_args()

    root = Path(args.root).resolve() / args.variant
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().with_name("run_shared_sar_tasks_v2.py")
    preparers = {
        "classification": prepare_classification,
        "segmentation": prepare_segmentation,
        "change_detection": prepare_change_detection,
    }
    summary = {}
    for task in args.tasks:
        manifest = preparers[task](root / "data")
        output = root / "runs" / task
        output.mkdir(parents=True, exist_ok=True)
        first_log = output / "epoch1.log"
        resume_log = output / "resume_epoch2.log"
        run_once(runner, manifest, task, args, epochs=1, log=first_log)
        result = output / "result.json"
        if not result.exists() or not (output / "last.pt").exists() or not (output / "best.pt").exists():
            raise RuntimeError(f"{task} did not write the expected artifacts")
        result.unlink()
        run_once(runner, manifest, task, args, epochs=2, log=resume_log)
        if "RESUME: epoch 1/2" not in resume_log.read_text(encoding="utf-8"):
            raise RuntimeError(f"{task} did not resume from its checkpoint")
        payload = json.loads(result.read_text(encoding="utf-8"))
        summary[task] = {
            "result": str(result),
            "metric": payload["metric"],
            "params": payload["params"],
            "gflops": payload["gflops"],
            "fps": payload["fps"],
        }
        print(f"PASS {args.variant} {task}: {result}", flush=True)

    summary_path = root / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS SUMMARY: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
