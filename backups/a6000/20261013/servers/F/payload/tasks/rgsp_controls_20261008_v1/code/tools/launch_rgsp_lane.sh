#!/usr/bin/env bash
# Foreground finite queue; caller may use nohup. Does not stop other processes.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "usage: launch_rgsp_lane.sh GPU SEED TASK_ROOT SOURCE_SSDD_ROOT" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
seed="$2"
task_root="$3"
source_data="$4"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export YOLO_AUTOINSTALL=false
export WANDB_MODE=disabled
export YOLO_CONFIG_DIR="$task_root/runtime-config-seed$seed"
export MPLCONFIGDIR="$task_root/matplotlib-seed$seed"
export PYTHONPATH="$task_root/code"
cd "$task_root/code"
exec /home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python -u tools/run_rgsp_controls.py \
  --source-data "$source_data" --output "$task_root/seed$seed" --seed "$seed" --mode run
