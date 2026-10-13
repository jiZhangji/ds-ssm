#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/powerleader/ig-ds-ssm
PY=$ROOT/envs/ig-ds-ssm-env/bin/python
CODE=$ROOT/work_train_single
RUNNER=$CODE/tools/run_shared_sar_tasks_v2.py
PREPARED=$ROOT/data/cross_task_prepared_v2
EXP=$ROOT/experiments/recheck_change_bs_20260904
LOG=$ROOT/logs/recheck_change_bs_20260904.log
WAIT_PID=${WAIT_PID:-103681}
mkdir -p "$EXP" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] waiting for A OpenEarthMap queue PID $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
while pgrep -f "$RUNNER" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] starting change-detection BS sweep"

run_one() {
  local dataset="$1" slug="$2" variant="$3" bs="$4"
  local out="$EXP/${slug,,}/${variant}_bs${bs}"
  [[ -f "$out/result.json" ]] && { echo "[skip] $dataset $variant BS$bs"; return; }
  "$PY" -u "$RUNNER" \
    --manifest "$PREPARED/change_detection/$slug/manifest.json" \
    --dataset "$dataset" --variant "$variant" --output "$out" \
    --epochs 150 --patience 25 --batch "$bs" --imgsz 192 --eval-batch "$bs" \
    --workers 6 --lr 0.0002 --state-dim 16 --seed 0 --device cuda:0
}

for pair in "Ottawa|Ottawa" "Yellow River|Yellow_River" "Farmland|Farmland"; do
  IFS='|' read -r dataset slug <<< "$pair"
  for bs in 32 64; do
    run_one "$dataset" "$slug" baseline "$bs"
    run_one "$dataset" "$slug" ours "$bs"
  done
done
echo "[$(date '+%F %T')] change-detection BS sweep completed"
