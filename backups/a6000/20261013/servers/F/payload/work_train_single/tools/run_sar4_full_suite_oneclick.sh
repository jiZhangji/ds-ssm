#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/mnt/disk1/zhangji/ds_ssm_sarmix/prepared}"
PROJECT="${PROJECT:-/mnt/disk1/zhangji/ds_ssm_sar4_full/output}"
EXPERIMENT_ID="${EXPERIMENT_ID:-sar4_scratch_p3_sd16_v1}"
DEVICE="${DEVICE:-0}"
DATASETS="${DATASETS:-SSDD OGSOD SAR-Ship SAR-Aircraft}"
VARIANTS="${VARIANTS:-baseline full_p3}"
STATE_DIM="${STATE_DIM:-16}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-30}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
EVAL_BATCH="${EVAL_BATCH:-16}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
SSDD_VAL_COUNT="${SSDD_VAL_COUNT:-186}"
CACHE="${CACHE:-0}"
AMP="${AMP:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a dataset_args <<< "$DATASETS"
read -r -a variant_args <<< "$VARIANTS"
extra_args=()
[[ "$CACHE" == "1" ]] && extra_args+=(--cache)
[[ "$AMP" == "1" ]] && extra_args+=(--amp)
[[ "$CONTINUE_ON_ERROR" == "1" ]] && extra_args+=(--continue-on-error)
[[ "$DRY_RUN" == "1" ]] && extra_args+=(--dry-run)

exec "$PYTHON_BIN" -u "$ROOT/tools/run_sar4_full_suite.py" \
  --data-root "$DATA_ROOT" \
  --project "$PROJECT" \
  --experiment-id "$EXPERIMENT_ID" \
  --datasets "${dataset_args[@]}" \
  --variants "${variant_args[@]}" \
  --device "$DEVICE" \
  --state-dim "$STATE_DIM" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --eval-batch "$EVAL_BATCH" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --ssdd-val-count "$SSDD_VAL_COUNT" \
  "${extra_args[@]}"
