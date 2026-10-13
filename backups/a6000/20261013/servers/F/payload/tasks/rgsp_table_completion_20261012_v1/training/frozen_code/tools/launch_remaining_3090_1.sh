#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/zhangji/ds_ssm_sarmix
CODE="$ROOT/work_train_single"
export ROOT CODE
export PYTHON_BIN=/mnt/disk1/zhangji/ig_dsssm_cross_task/envs/ig-ds-ssm-env/bin/python
export DATA_ROOT=/mnt/disk1/zhangji/ig_dsssm_cross_task/data/cross_task_prepared/detection
export PROJECT="$ROOT/experiments/cross_task_detection"
export WANDB_MODE=disabled

cd "$CODE"
exec bash tools/queue_a6000_detection.sh \
  --experiment-id ig_dsssm_p3_remaining_3090_1_v2 \
  --datasets SRSDD-v1.0 AIR-SARShip-2.0 \
  --state-dim 16 \
  --batch 16 \
  --eval-batch 16 \
  --epochs 150 \
  --patience 30 \
  --workers 8
