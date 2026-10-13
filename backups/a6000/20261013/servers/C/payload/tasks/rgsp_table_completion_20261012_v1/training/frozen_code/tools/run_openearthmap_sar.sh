#!/usr/bin/env bash
set -euo pipefail

ROOT=${IG_DS_SSM_ROOT:-/home/powerleader/ig-ds-ssm}
VARIANT=${1:?usage: run_openearthmap_sar.sh baseline|ours}
if [[ "$VARIANT" != "baseline" && "$VARIANT" != "ours" ]]; then
  echo "variant must be baseline or ours" >&2
  exit 2
fi

PY="$ROOT/envs/ig-ds-ssm-env/bin/python"
WORK="$ROOT/work_train_single"
MANIFEST="$ROOT/data/openearthmap_sar/sar_only_p/manifest.json"
EXPERIMENT="openearthmap_saronly_p_p3_d16_v1"
OUTPUT="$ROOT/experiments/$EXPERIMENT/$VARIANT"

test -x "$PY"
test -f "$MANIFEST"
mkdir -p "$OUTPUT"

exec "$PY" -u "$WORK/tools/run_shared_sar_tasks_v2.py" \
  --manifest "$MANIFEST" \
  --dataset OpenEarthMap-SAR \
  --variant "$VARIANT" \
  --backbone-layout full_t \
  --decoder upernet_lite \
  --output "$OUTPUT" \
  --epochs 150 \
  --patience 30 \
  --batch 8 \
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
