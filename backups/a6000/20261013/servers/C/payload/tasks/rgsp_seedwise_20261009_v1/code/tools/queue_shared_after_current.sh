#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?Usage: $0 WAIT_PID DATASET...}"
shift
DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "No datasets supplied" >&2; exit 2; }

BASE=/home/powerleader/ig-ds-ssm
CODE="$BASE/work_train_single"
PYTHON_BIN="$BASE/envs/ig-ds-ssm-env/bin/python"
RAW_ROOT="$BASE/data/cross_task_raw"
PREPARED_ROOT="$BASE/data/cross_task_prepared"
PROJECT="$BASE/experiments/cross_task"

echo "Waiting for existing experiment PID $WAIT_PID: ${DATASETS[*]}"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

# Let the existing runner finish evaluation and release its CUDA context.
while pgrep -f 'mbyolo_train_sarair.py|tools/run_sar4_full_suite.py' >/dev/null; do
  sleep 30
done

export ROOT="$CODE" PYTHON_BIN RAW_ROOT PREPARED_ROOT PROJECT
export DEVICE=cuda EPOCHS=100 PATIENCE=20 BATCH=8 IMGSZ=256 WORKERS=4 STATE_DIM=16
export WANDB_MODE=disabled

echo "Starting shared SAR tasks: ${DATASETS[*]}"
exec bash "$CODE/tools/run_a6000_small_tasks.sh" "${DATASETS[@]}"
