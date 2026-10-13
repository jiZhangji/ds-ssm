#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_fusar_map_net_head.sh baseline|ours [device]}"
DEVICE="${2:-cuda}"
if [[ "$VARIANT" != "baseline" && "$VARIANT" != "ours" ]]; then
  echo "variant must be baseline or ours" >&2
  exit 2
fi

ROOT="/home/powerleader/ig-ds-ssm/work_train_single"
PYTHON="/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python"
MANIFEST="/home/powerleader/ig-ds-ssm/data/cross_task_prepared_v2/segmentation/FUSAR_Map_Paper_Code/manifest.json"
EXPERIMENT="fusar_mambayolo_t_fusarmap_head_d16_v1"
OUTPUT="/home/powerleader/ig-ds-ssm/experiments/${EXPERIMENT}/${VARIANT}/seed_0"

cd "$ROOT"
mkdir -p "$OUTPUT"

exec "$PYTHON" -u tools/run_shared_sar_tasks_v2.py \
  --manifest "$MANIFEST" \
  --dataset FUSAR-Map \
  --variant "$VARIANT" \
  --backbone-layout full_t \
  --decoder fusar_map_net \
  --output "$OUTPUT" \
  --epochs 150 \
  --patience 150 \
  --batch 10 \
  --imgsz 256 \
  --workers 4 \
  --optimizer Adam \
  --lr 0.001 \
  --weight-decay 0 \
  --scheduler step \
  --lr-step-size 2 \
  --lr-gamma 0.93 \
  --label-smoothing 0 \
  --segmentation-loss weighted_cross_entropy_plus_dice \
  --selection-metric miou_all \
  --state-dim 16 \
  --seed 0 \
  --device "$DEVICE"
