#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: launch_sadd_lane.sh GPU SEED TASK_ROOT SOURCE_SADD_ROOT INVENTORY_SHA256" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
seed="$2"
task_root="$3"
source_data="$4"
inventory="$5"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export YOLO_AUTOINSTALL=false WANDB_MODE=disabled
export YOLO_CONFIG_DIR="$task_root/runtime-config-seed$seed"
export MPLCONFIGDIR="$task_root/matplotlib-seed$seed"
export PYTHONPATH="$task_root/code"
cd "$task_root/code"
exec /home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python -u tools/run_sadd_controls.py \
  --source-data "$source_data" --output "$task_root/seed$seed" --seed "$seed" \
  --expected-inventory "$inventory" --mode run
