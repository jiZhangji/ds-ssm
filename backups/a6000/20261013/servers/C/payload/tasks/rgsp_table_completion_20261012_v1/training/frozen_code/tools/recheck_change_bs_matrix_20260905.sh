#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/powerleader/ig-ds-ssm
PY=$ROOT/envs/ig-ds-ssm-env/bin/python
CODE=$ROOT/work_train_single
RUNNER=$CODE/tools/run_shared_sar_tasks_v2.py
BASE=$ROOT/data/cross_task_prepared_v2/change_detection
BS=${1:?batch size required}
DEVICE=${2:-cuda:0}
TAG=${3:?output tag required}
shift 3

EXP=$ROOT/experiments/recheck_change_${TAG}_20260905
LOG=$ROOT/logs/recheck_change_${TAG}_20260905.log
mkdir -p "$EXP" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] start change-detection BS=$BS device=$DEVICE tag=$TAG"

run_one() {
  local dataset="$1" variant="$2"
  local slug=${dataset// /_}
  local manifest="$BASE/$slug/manifest.json"
  local out="$EXP/${slug}/${variant}_bs${BS}"
  [[ -f "$out/result.json" ]] && { echo "[skip] $dataset $variant BS$BS"; return; }
  "$PY" -u "$RUNNER" \
    --manifest "$manifest" --dataset "$dataset" --variant "$variant" --output "$out" \
    --backbone-layout legacy --decoder multiscale \
    --epochs 150 --patience 35 --batch "$BS" --imgsz 192 --eval-batch "$BS" \
    --workers 8 --lr 0.0002 --state-dim 16 --seed 0 --device "$DEVICE"
}

for dataset in "$@"; do
  run_one "$dataset" baseline
  run_one "$dataset" ours
done

echo "[$(date '+%F %T')] completed change-detection BS=$BS tag=$TAG"
