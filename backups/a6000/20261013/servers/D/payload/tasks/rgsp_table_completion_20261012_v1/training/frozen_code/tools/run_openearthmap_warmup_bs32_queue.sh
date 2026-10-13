#!/usr/bin/env bash
set -euo pipefail

BASE=/home/powerleader/ig-ds-ssm
PY=$BASE/envs/ig-ds-ssm-env/bin/python
RUN=$BASE/work_train_single/tools/run_shared_sar_tasks_v2.py
MAN=$BASE/data/openearthmap_sar/sar_only_p/manifest.json
PRE=$BASE/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth
COMMON="--manifest $MAN --dataset OpenEarthMap-SAR --variant ours --backbone-layout vmamba_tiny --decoder upernet_lite --pretrained $PRE --epochs 150 --patience 30 --batch 32 --imgsz 512 --eval-batch 1 --eval-imgsz 1024 --workers 8 --optimizer Adam --lr 1e-4 --weight-decay 0.0 --scheduler fixed --label-smoothing 0.0 --segmentation-loss weighted_cross_entropy_plus_dice --selection-metric miou_all --state-dim 16 --seed 0 --device cuda:0"

echo "[queue] waiting for current BS32 experiment PID 116982"
while kill -0 116982 2>/dev/null; do sleep 60; done
echo "[queue] current experiment finished: $(date '+%F %T')"

OUT=$BASE/experiments/openearthmap_saronly_p_vmamba_tiny_pretrained_p3_d16_warmup50_bs32_queue_v1/ours
mkdir -p "$OUT"
echo "[queue] start warmup50: $(date '+%F %T')"
"$PY" -u "$RUN" $COMMON --output "$OUT" --module-warmup-epochs 50 > "$BASE/logs/openearthmap_p3_warmup50_bs32_queue.log" 2>&1
echo "[queue] warmup50 finished: $(date '+%F %T')"
echo "[queue] all finished: $(date '+%F %T')"
