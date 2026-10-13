#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/powerleader/ig-ds-ssm
PY=$ROOT/envs/ig-ds-ssm-env/bin/python
CODE=$ROOT/work_train_single
RUNNER=$CODE/tools/run_shared_sar_tasks_v2.py
MANIFEST=$ROOT/data/openearthmap_sar/sar_only_p/manifest.json
WEIGHTS=$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth
DEVICE=${1:?device required}
TAG=${2:?queue tag required}
shift 2

EXP=$ROOT/experiments/openearthmap_saronly_p_vmamba_tiny_pretrained_p3_d16_effective_bs64_${TAG}
LOG=$ROOT/logs/openearthmap_saronly_p_bs64_${TAG}.log
mkdir -p "$EXP" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] start OpenEarthMap-SAR physical BS64 queue on $DEVICE"

run_one() {
  local name="$1" variant="$2" warmup="$3"
  local out="$EXP/$name"
  [[ -f "$out/result.json" ]] && { echo "[skip] $name"; return; }
  local extra=()
  [[ "$warmup" != 0 ]] && extra+=(--module-warmup-epochs "$warmup")
  "$PY" -u "$RUNNER" \
    --manifest "$MANIFEST" --dataset OpenEarthMap-SAR --variant "$variant" --output "$out" \
    --backbone-layout vmamba_tiny --decoder upernet_lite --pretrained "$WEIGHTS" \
    --epochs 150 --patience 30 --batch 32 --grad-accum 2 \
    --imgsz 512 --eval-imgsz 1024 --eval-batch 1 --workers 8 \
    --optimizer Adam --lr 0.0001 --weight-decay 0.0 --scheduler fixed \
    --label-smoothing 0.0 --segmentation-loss weighted_cross_entropy_plus_dice \
    --selection-metric miou_all --state-dim 16 --seed 0 --device "$DEVICE" "${extra[@]}"
}

for spec in "$@"; do
  IFS=':' read -r name variant warmup <<< "$spec"
  run_one "$name" "$variant" "$warmup"
done

echo "[$(date '+%F %T')] completed OpenEarthMap-SAR physical BS64 queue"
