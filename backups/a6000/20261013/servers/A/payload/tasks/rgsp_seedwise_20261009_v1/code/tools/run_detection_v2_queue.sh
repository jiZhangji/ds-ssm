#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/cross_task_prepared_v2/detection}"
PROJECT="${PROJECT:-$ROOT/experiments/corrected_detection_v2}"
DATASET="${DATASET:?Set DATASET to SADD, AIR-SARShip-2.0, or SRSDD-v1.0}"
STATE_DIM="${STATE_DIM:-32}"
BATCH="${BATCH:-32}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-40}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"

case "$DATASET" in
  SADD) ID="corrected_v2_sadd_sd${STATE_DIM}_b${BATCH}" ;;
  AIR-SARShip-2.0) ID="corrected_v2_airship_tiles_sd${STATE_DIM}_b${BATCH}" ;;
  SRSDD-v1.0) ID="corrected_v2_srsdd_hbb_clipped_sd${STATE_DIM}_b${BATCH}" ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

cd "$CODE"
exec "$PYTHON_BIN" -u tools/run_sar4_full_suite.py \
  --data-root "$DATA_ROOT" \
  --project "$PROJECT" \
  --experiment-id "$ID" \
  --datasets "$DATASET" \
  --variants baseline full_p3 \
  --state-dim "$STATE_DIM" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --eval-batch "$EVAL_BATCH" \
  --device "$DEVICE" \
  --workers 8 \
  --seed 0 \
  --cache \
  --continue-on-error
