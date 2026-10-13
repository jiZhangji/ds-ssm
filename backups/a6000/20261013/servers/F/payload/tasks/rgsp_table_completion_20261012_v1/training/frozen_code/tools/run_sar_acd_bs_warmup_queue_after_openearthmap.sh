#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/powerleader/ig-ds-ssm
PY=$ROOT/envs/ig-ds-ssm-env/bin/python
CODE=$ROOT/work_train_single
RUNNER=$CODE/tools/run_shared_sar_tasks_v2.py
MANIFEST=$ROOT/data/cross_task_prepared_v2/classification/SAR_ACD/manifest.json
WEIGHTS=$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth
EXP=$ROOT/experiments/recheck_sar_acd_bs_warmup_20260904
LOG=$ROOT/logs/recheck_sar_acd_bs_warmup_20260904.log
WAIT_PID=${WAIT_PID:-119326}
mkdir -p "$EXP" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] waiting for B OpenEarthMap queue PID $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
while pgrep -f "$RUNNER" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] starting SAR-ACD BS/warmup sweep"

run_one() {
  local tag="$1" variant="$2" bs="$3" warmup="$4"
  local out="$EXP/$tag"
  [[ -f "$out/result.json" ]] && { echo "[skip] SAR-ACD $tag"; return; }
  local extra=()
  [[ "$warmup" != 0 ]] && extra+=(--module-warmup-epochs "$warmup")
  "$PY" -u "$RUNNER" \
    --manifest "$MANIFEST" --dataset SAR-ACD --variant "$variant" --output "$out" \
    --backbone-layout vmamba_tiny --pretrained "$WEIGHTS" \
    --epochs 200 --patience 35 --batch "$bs" --imgsz 224 --eval-batch "$bs" \
    --workers 8 --lr 0.0002 --state-dim 16 --seed 0 --device cuda:0 "${extra[@]}"
}

run_one baseline_bs64 baseline 64 0
run_one baseline_bs128 baseline 128 0
run_one ours_bs32 ours 32 0
run_one ours_bs64 ours 64 0
run_one ours_bs128 ours 128 0
run_one ours_warmup20_bs64 ours 64 20
run_one ours_warmup50_bs64 ours 64 50
echo "[$(date '+%F %T')] SAR-ACD BS/warmup sweep completed"
