#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
PY="${PY:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
CODE="${CODE:-$ROOT/work_train_single}"
DATA="${DATA:-$ROOT/data/cross_task_prepared_v2}"
PROJECT="${PROJECT:-$ROOT/experiments/priority_optimization_20260831}"
DEVICE="${DEVICE:-cuda:0}"
RUNNER="$CODE/tools/run_shared_sar_tasks_v2.py"
LOG_DIR="$PROJECT/logs"
LOCK_FILE="$PROJECT/queue.lock"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another priority optimization queue is already running: $LOCK_FILE"
  exit 0
fi

run_job() {
  local name="$1"
  local result="$2"
  shift 2
  if [[ -f "$result" ]]; then
    echo "SKIP COMPLETE: $result"
    return
  fi
  echo "===== START $name $(date --iso-8601=seconds) ====="
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "===== END $name $(date --iso-8601=seconds) ====="
}

COMMON=(
  "$PY" -u "$RUNNER"
  --state-dim 16
  --seed 0
  --device "$DEVICE"
  --workers 8
  --amp
)

# Classification diagnostics: the current P3 runs underperform the matched
# baseline. Lower LR and no label smoothing test whether the small datasets are
# being over-regularized, while preserving the same split, batch, and model.
OUT="$PROJECT/sar-acd/ours_lr1e4_ls0"
run_job "sar_acd_ours_lr1e4_ls0" "$OUT/result.json" \
  "${COMMON[@]}" \
  --manifest "$DATA/classification/SAR_ACD/manifest.json" \
  --dataset SAR-ACD --variant ours --output "$OUT" \
  --epochs 160 --patience 35 --batch 64 --imgsz 224 \
  --optimizer AdamW --scheduler cosine --lr 0.0001 \
  --weight-decay 0.0005 --label-smoothing 0.0

OUT="$PROJECT/sample/ours_lr1e4_ls0"
run_job "sample_ours_lr1e4_ls0" "$OUT/result.json" \
  "${COMMON[@]}" \
  --manifest "$DATA/classification/SAMPLE/manifest.json" \
  --dataset SAMPLE --variant ours --output "$OUT" \
  --epochs 160 --patience 35 --batch 64 --imgsz 224 \
  --optimizer AdamW --scheduler cosine --lr 0.0001 \
  --weight-decay 0.0005 --label-smoothing 0.0

# Change-detection diagnostics: P3 currently over-predicts change on
# Sulzberger and loses recall on Yellow River. A lower LR tests optimization
# stability without changing data, threshold selection, loss, or architecture.
OUT="$PROJECT/sulzberger/ours_lr1e4"
run_job "sulzberger_ours_lr1e4" "$OUT/result.json" \
  "${COMMON[@]}" \
  --manifest "$DATA/change_detection/Sulzberger/manifest.json" \
  --dataset Sulzberger --variant ours --output "$OUT" \
  --epochs 120 --patience 25 --batch 16 --imgsz 192 \
  --optimizer AdamW --scheduler cosine --lr 0.0001 \
  --weight-decay 0.0005 --label-smoothing 0.0

OUT="$PROJECT/yellow-river/ours_lr1e4"
run_job "yellow_river_ours_lr1e4" "$OUT/result.json" \
  "${COMMON[@]}" \
  --manifest "$DATA/change_detection/Yellow_River/manifest.json" \
  --dataset "Yellow River" --variant ours --output "$OUT" \
  --epochs 120 --patience 25 --batch 16 --imgsz 192 \
  --optimizer AdamW --scheduler cosine --lr 0.0001 \
  --weight-decay 0.0005 --label-smoothing 0.0

echo "===== PRIORITY OPTIMIZATION QUEUE COMPLETE $(date --iso-8601=seconds) ====="
