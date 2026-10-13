#!/usr/bin/env bash
set -euo pipefail

ROOT=${IG_DS_SSM_ROOT:-/home/powerleader/ig-ds-ssm}
PY="$ROOT/envs/ig-ds-ssm-env/bin/python"
WORK="$ROOT/work_train_single"
MANIFEST="$ROOT/data/openearthmap_sar/sar_only_p/manifest.json"
CKPT="$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth"
EXPERIMENT="openearthmap_saronly_p_vmamba_tiny_pretrained_p3_d16_decoder_warmup20_bs32_v1"
OUTPUT="$ROOT/experiments/$EXPERIMENT/ours"

test -x "$PY"
test -f "$MANIFEST"
test -f "$CKPT"
mkdir -p "$OUTPUT"

exec "$PY" -u "$WORK/tools/run_shared_sar_tasks_v2.py" \
  --manifest "$MANIFEST" \
  --dataset OpenEarthMap-SAR \
  --variant ours \
  --backbone-layout vmamba_tiny \
  --decoder upernet_lite \
  --pretrained "$CKPT" \
  --output "$OUTPUT" \
  --module-warmup-epochs 20 \
  --epochs 150 \
  --patience 30 \
  --batch 32 \
  --imgsz 512 \
  --eval-batch 1 \
  --eval-imgsz 1024 \
  --workers 8 \
  --optimizer Adam \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --scheduler fixed \
  --label-smoothing 0.0 \
  --segmentation-loss weighted_cross_entropy_plus_dice \
  --selection-metric miou_all \
  --state-dim 16 \
  --seed 0 \
  --device cuda:0
