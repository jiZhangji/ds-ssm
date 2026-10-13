#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?Usage: $0 c|d|a}"
ROOT=/home/powerleader/ig-ds-ssm
CODE="$ROOT/work_train_single"
RAW="$ROOT/data/cross_task_raw"
PREPARED="$ROOT/data/cross_task_prepared"
PROJECT="$ROOT/experiments/cross_task"
RELEASE=https://github.com/jiZhangji/ds-ssm/releases/download/cross-task-small-data-v1
PYTHON_BIN="$ROOT/envs/ig-ds-ssm-env/bin/python"
mkdir -p "$RAW" "$PREPARED" "$ROOT/logs" "$PROJECT"

case "$ROLE" in
  a) ASSETS=(SAR_ACD.zip); DATASETS=(SAR-ACD) ;;
  b) ASSETS=(SAMPLE_dataset_public.zip); DATASETS=(SAMPLE) ;;
  c) ASSETS=(Ottawa_data.zip Yellowriver_data.zip); DATASETS=(Ottawa "Yellow River") ;;
  d) ASSETS=(Farmland_data.zip Sulzberger_data.zip); DATASETS=(Farmland Sulzberger) ;;
  *) echo "Unknown role: $ROLE" >&2; exit 2 ;;
esac

download() {
  local name="$1" target="$2"
  if [[ -s "$target" ]]; then
    echo "SKIP PRESENT: $target"
    return
  fi
  curl -fL --retry 20 --retry-delay 5 --connect-timeout 20 \
    "$RELEASE/$name" -o "$target.part"
  mv "$target.part" "$target"
}

download work_train_single_small_tasks.tar.gz "$ROOT/work_train_single_small_tasks.tar.gz"
download SHA256SUMS.txt "$RAW/SHA256SUMS.txt"
for asset in "${ASSETS[@]}"; do download "$asset" "$RAW/$asset"; done
(
  cd "$RAW"
  sha256sum -c --ignore-missing SHA256SUMS.txt
)
tar -C "$ROOT" -xzf "$ROOT/work_train_single_small_tasks.tar.gz"
rm -f "$ROOT/work_train_single_small_tasks.tar.gz"
test -x "$PYTHON_BIN"
test -f "$ROOT/work_train_single/ultralytics/nn/modules/mamba_yolo.py"

export ROOT="$CODE"
export PYTHON_BIN RAW_ROOT="$RAW"
export PREPARED_ROOT="$PREPARED"
export PROJECT="$PROJECT"
export DEVICE=cuda EPOCHS=100 PATIENCE=20 BATCH=8 IMGSZ=256 WORKERS=4 STATE_DIM=16
export WANDB_MODE=disabled
exec bash "$CODE/tools/run_a6000_small_tasks.sh" "${DATASETS[@]}"
