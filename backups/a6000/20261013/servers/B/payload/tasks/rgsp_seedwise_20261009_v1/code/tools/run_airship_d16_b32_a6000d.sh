#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
RAW="$ROOT/data/cross_task_raw"
PREPARED="$ROOT/data/cross_task_prepared_v2"

"$PYTHON_BIN" -u "$ROOT/work_train_single/tools/prepare_cross_task_v2.py" \
  --raw-root "$RAW" \
  --extra-root "$RAW" \
  --out "$PREPARED" \
  --datasets AIR-SARShip-2.0

DATASET="AIR-SARShip-2.0" \
STATE_DIM=16 \
BATCH=32 \
EVAL_BATCH=32 \
EPOCHS=200 \
PATIENCE=40 \
IMGSZ=640 \
DEVICE=0 \
bash "$ROOT/work_train_single/tools/run_detection_v2_queue.sh"
