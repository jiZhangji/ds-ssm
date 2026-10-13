#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/powerleader/ig-ds-ssm
PY=$ROOT/envs/ig-ds-ssm-env/bin/python
RUNNER=$ROOT/work_train_single/tools/run_shared_sar_tasks_v2.py
MANIFEST=$ROOT/data/openearthmap_sar/sar_only_p/manifest.json
WEIGHTS=$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth
LOGDIR=$ROOT/logs
EXP=$ROOT/experiments

mkdir -p "$LOGDIR" "$EXP"
exec > >(tee -a "$LOGDIR/openearthmap_bs64_f_queue.log") 2>&1

echo "[$(date '+%F %T')] BS64 queue started on F"
echo "[queue] waiting until both A6000 GPUs are free"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; do
  sleep 60
done

run_one() {
  local tag=$1
  shift
  local out="$EXP/$tag/ours"
  if [[ -f "$out/result.json" ]]; then
    echo "[queue] skip completed: $tag"
    return
  fi
  echo "[$(date '+%F %T')] start $tag"
  "$PY" -u "$RUNNER" \
    --manifest "$MANIFEST" \
    --dataset OpenEarthMap-SAR \
    --variant ours \
    --backbone-layout vmamba_tiny \
    --decoder upernet_lite \
    --output "$out" \
    --epochs 150 \
    --patience 30 \
    --batch 64 \
    --imgsz 512 \
    --eval-batch 1 \
    --eval-imgsz 1024 \
    --workers 8 \
    --optimizer Adam \
    --lr 1e-4 \
    --weight-decay 0.0 \
    --scheduler fixed \
    --label-smoothing 0.0 \
    --segmentation-loss weighted_cross_entropy_plus_dice \
    --selection-metric miou_all \
    --state-dim 16 \
    --seed 0 \
    --device cuda:0,cuda:1 \
    "$@"
  echo "[$(date '+%F %T')] finished $tag"
}

run_one openearthmap_saronly_vmamba_tiny_p3_d16_bs64_scratch_f_v1
run_one openearthmap_saronly_vmamba_tiny_pretrained_p3_d16_bs64_f_v1 --pretrained "$WEIGHTS"
run_one openearthmap_saronly_vmamba_tiny_pretrained_p3_d16_warmup50_bs64_f_v1 --pretrained "$WEIGHTS" --module-warmup-epochs 50
run_one openearthmap_saronly_vmamba_tiny_pretrained_p3_d16_warmup20_bs64_f_v1 --pretrained "$WEIGHTS" --module-warmup-epochs 20

echo "[$(date '+%F %T')] BS64 queue completed"
