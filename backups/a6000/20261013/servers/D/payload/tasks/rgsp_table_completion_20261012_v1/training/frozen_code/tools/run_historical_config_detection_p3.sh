#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/cross_task_prepared_v2/detection}"
PROJECT="${PROJECT:-$ROOT/experiments/historical_protocol_detection}"
EXPERIMENT_ID="${EXPERIMENT_ID:-det_histcfg_p3_d16_fp32_v1}"
DEVICE="${DEVICE:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
WORKERS="${WORKERS:-4}"
EVAL_BATCH="${EVAL_BATCH:-64}"
MODULE_WARMUP_EPOCHS="${MODULE_WARMUP_EPOCHS:-20}"

if [[ $# -eq 0 ]]; then
  set -- SSDD SAR-AIRcraft1.0 SADD AIR-SARShip-2.0 SRSDD-v1.0
fi

cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export ULTRALYTICS_SKIP_TENSORBOARD_GRAPH="${ULTRALYTICS_SKIP_TENSORBOARD_GRAPH:-1}"

for dataset in "$@"; do
  extra=()
  if [[ "$dataset" == "SSDD" ]]; then
    extra+=(--ssdd-full-train)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    extra+=(--dry-run)
  fi

  ULTRALYTICS_SKIP_AMP_CHECK=1 "$PYTHON_BIN" -u tools/run_sar4_full_suite.py \
    --data-root "$DATA_ROOT" \
    --project "$PROJECT" \
    --experiment-id "$EXPERIMENT_ID" \
    --datasets "$dataset" \
    --variants full_p3 \
    --state-dim 16 \
    --epochs 300 \
    --patience 100 \
    --imgsz 640 \
    --batch 32 \
    --eval-batch "$EVAL_BATCH" \
    --device "$DEVICE" \
    --eval-device "$EVAL_DEVICE" \
    --workers "$WORKERS" \
    --seed 0 \
    --cache \
    --continue-on-error \
    --optimizer AdamW \
    --lr0 0.001 \
    --warmup-epochs 5 \
    --warmup-bias-lr 0.001 \
    --module-warmup-epochs "$MODULE_WARMUP_EPOCHS" \
    --nbs 64 \
    --weights "$ROOT/yolov8n.pt" \
    --hsv-h 0.015 \
    --hsv-s 0.7 \
    --hsv-v 0.4 \
    --mosaic 1.0 \
    --close-mosaic 10 \
    --degrees 0 \
    --translate 0.1 \
    --scale 0.5 \
    --flipud 0 \
    --fliplr 0.5 \
    "${extra[@]}"
done
