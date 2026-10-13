#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/cross_task_raw/cross_task_small}"
PREPARED="${PREPARED:-$ROOT/data/cross_task_prepared_v2}"
PROJECT="${PROJECT:-$ROOT/experiments/fusar_count_aligned_sd16_20260828}"
VARIANT="${VARIANT:-ours}"
STATE_DIM="${STATE_DIM:-16}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-32}"

if [[ "$VARIANT" != "baseline" && "$VARIANT" != "ours" ]]; then
  echo "VARIANT must be baseline or ours" >&2
  exit 2
fi

MANIFEST="$PREPARED/segmentation/FUSAR_Map_Count_Aligned/manifest.json"
if [[ ! -s "$MANIFEST" ]]; then
  "$PYTHON_BIN" -u "$CODE/tools/prepare_cross_task_v2.py" \
    --raw-root "$RAW_ROOT" \
    --out "$PREPARED" \
    --datasets FUSAR-Map-Count-Aligned
fi

mkdir -p "$PROJECT"
"$PYTHON_BIN" -u "$CODE/tools/run_shared_sar_tasks_v2.py" \
  --manifest "$MANIFEST" \
  --dataset FUSAR-Map-Count-Aligned \
  --variant "$VARIANT" \
  --output "$PROJECT/$VARIANT" \
  --epochs "$EPOCHS" \
  --patience "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 256 \
  --workers 8 \
  --lr 0.0002 \
  --state-dim "$STATE_DIM" \
  --seed 0 \
  --device "$DEVICE" \
  --amp
