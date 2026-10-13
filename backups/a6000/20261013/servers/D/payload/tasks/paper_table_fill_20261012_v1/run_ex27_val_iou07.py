#!/usr/bin/env python3
"""Re-evaluate the archived PFIM ex27 checkpoint under its recorded val setup."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


CHECKPOINT = Path("/home/powerleader/ig-ds-ssm/weights/historical_ssdd/ex27_best.pt")
DATA_YAML = Path(
    "/home/powerleader/ig-ds-ssm/experiments/ssdd_historical_checkpoint_diagnostic_20260828/"
    "SSDD_official_test_diagnostic.yaml"
)
SOURCE = Path("/home/powerleader/ig-ds-ssm/work_train_single/ultralytics")
TASK_ROOT = Path("/home/powerleader/ig-ds-ssm/tasks/paper_table_fill_20261012_v1")
PROJECT = TASK_ROOT / "val_ex27_iou07"
RESULT_PATH = PROJECT / "ex27_val_metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    if PROJECT.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {PROJECT}")
    PROJECT.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    record = {
        "run_id": "paper-table-fill-ssdd-ex27-val-iou07-20261012",
        "status": "RUNNING",
        "started_at_utc": started,
        "host": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "source_path": str(SOURCE),
        "source_git_revision": git_revision(SOURCE),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "data_yaml": str(DATA_YAML),
        "data_yaml_sha256": sha256(DATA_YAML),
        "split": "val (maps to official images/test; historical TEST-as-VAL)",
        "settings": {
            "imgsz": 640,
            "batch": 32,
            "device": 0,
            "half": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "workers": 0,
            "plots": False,
            "augment": False,
        },
        "metrics_fraction": None,
        "metrics_percent": None,
        "completed_at_utc": None,
        "error": None,
    }
    try:
        model = YOLO(str(CHECKPOINT))
        result = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=640,
            batch=32,
            device=0,
            half=False,
            conf=0.001,
            iou=0.7,
            max_det=300,
            workers=0,
            plots=False,
            augment=False,
            verbose=False,
            project=str(PROJECT),
            name="evaluation",
            exist_ok=False,
        )
        fractions = {
            "precision": float(result.box.mp),
            "recall": float(result.box.mr),
            "AP50": float(result.box.map50),
            "AP75": float(result.box.map75),
            "AP50_95": float(result.box.map),
        }
        record["metrics_fraction"] = fractions
        record["metrics_percent"] = {key: 100.0 * value for key, value in fractions.items()}
        record["status"] = "COMPLETED"
    except Exception:
        record["status"] = "FAILED"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["completed_at_utc"] = utc_now()
        temporary = RESULT_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, RESULT_PATH)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
