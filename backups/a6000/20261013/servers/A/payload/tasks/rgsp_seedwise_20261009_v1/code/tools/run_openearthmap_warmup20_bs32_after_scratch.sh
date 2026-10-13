#!/usr/bin/env bash
set -euo pipefail

BASE=/home/powerleader/ig-ds-ssm
PY=$BASE/envs/ig-ds-ssm-env/bin/python
RUN=$BASE/work_train_single/tools/run_shared_sar_tasks_v2.py
MAN=$BASE/data/openearthmap_sar/sar_only_p/manifest.json
PRE=$BASE/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth

echo "[queue] waiting for scratch BS32 experiment PID 102177"
while kill -0 102177 2>/dev/null; do sleep 60; done
echo "[queue] scratch experiment finished: $(date '+%F %T')"

OUT=$BASE/experiments/openearthmap_saronly_p_vmamba_tiny_pretrained_p3_d16_warmup20_bs32_queue_v1/ours
mkdir -p "$OUT"
echo "[queue] start warmup20: $(date '+%F %T')"
"$PY" -u "$RUN" \
  --manifest "$MAN" \
  --dataset OpenEarthMap-SAR \
  --variant ours \
  --backbone-layout vmamba_tiny \
  --decoder upernet_lite \
  --pretrained "$PRE" \
  --output "$OUT" \
  --module-warmup-epochs 20 \
  --epochs 150 \
  --patience 30 \
  --batch 32 \
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
  --device cuda:0 \
  > "$BASE/logs/openearthmap_p3_warmup20_bs32_queue.log" 2>&1
echo "[queue] warmup20 finished: $(date '+%F %T')"
