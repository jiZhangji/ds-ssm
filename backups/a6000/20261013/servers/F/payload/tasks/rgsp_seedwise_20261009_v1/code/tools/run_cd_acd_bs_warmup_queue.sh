#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangji/ig_ds_ssm_cross_task_v1
PY=/home/zhangji/miniconda3/envs/mamba-yolo/bin/python
CODE=$ROOT/work_train_single
RUNNER=$CODE/tools/run_shared_sar_tasks_v2.py
PREPARED=$ROOT/data/cross_task_prepared_v2
WEIGHTS=$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth
EXP=$ROOT/experiments
mkdir -p "$ROOT/logs" "$EXP"

run_cd() {
  local dataset="$1" slug="$2" bs="$3" variant="$4"
  local manifest="$PREPARED/change_detection/$slug/manifest.json"
  local out="$EXP/bs_sweep_change/${slug,,}/${variant}_bs${bs}"
  if [[ -f "$out/result.json" ]]; then
    echo "[skip] $dataset $variant BS$bs"
    return
  fi
  "$PY" -u "$RUNNER" \
    --manifest "$manifest" --dataset "$dataset" --variant "$variant" \
    --output "$out" --epochs 150 --patience 25 --batch "$bs" --imgsz 192 \
    --eval-batch "$bs" --workers 2 --lr 0.0002 --state-dim 16 --seed 0 \
    --device cuda:0
}

run_acd() {
  local tag="$1" bs="$2" variant="$3" warmup="$4" layout="$5"
  local manifest="$PREPARED/classification/SAR_ACD/manifest.json"
  local out="$EXP/bs_warmup_sar_acd/${tag}"
  if [[ -f "$out/result.json" ]]; then
    echo "[skip] SAR-ACD $tag"
    return
  fi
  local extra=()
  [[ "$warmup" != 0 ]] && extra+=(--module-warmup-epochs "$warmup")
  [[ "$layout" == vmamba_tiny ]] && extra+=(--backbone-layout vmamba_tiny --pretrained "$WEIGHTS")
  "$PY" -u "$RUNNER" \
    --manifest "$manifest" --dataset SAR-ACD --variant "$variant" \
    --output "$out" --epochs 200 --patience 35 --batch "$bs" --imgsz 224 \
    --eval-batch "$bs" --workers 2 --lr 0.0002 --state-dim 16 --seed 0 \
    --device cuda:1 "${extra[@]}"
}

run_change_gpu() {
  for dataset in Ottawa Yellow_River Farmland; do
    case "$dataset" in
      Ottawa) name="Ottawa";;
      Yellow_River) name="Yellow River";;
      Farmland) name="Farmland";;
    esac
    for bs in 32 64; do
      run_cd "$name" "$dataset" "$bs" baseline
      run_cd "$name" "$dataset" "$bs" ours
    done
  done
}

run_acd_gpu() {
  run_acd baseline_bs64 64 baseline 0 vmamba_tiny
  run_acd baseline_bs128 128 baseline 0 vmamba_tiny
  run_acd ours_bs32 32 ours 0 vmamba_tiny
  run_acd ours_bs64 64 ours 0 vmamba_tiny
  run_acd ours_bs128 128 ours 0 vmamba_tiny
  run_acd ours_warmup20_bs64 64 ours 20 vmamba_tiny
  run_acd ours_warmup50_bs64 64 ours 50 vmamba_tiny
}

case "${1:-all}" in
  change) run_change_gpu;;
  acd) run_acd_gpu;;
  all) run_change_gpu & p1=$!; run_acd_gpu & p2=$!; wait "$p1" "$p2";;
  *) echo "usage: $0 [change|acd|all]" >&2; exit 2;;
esac
