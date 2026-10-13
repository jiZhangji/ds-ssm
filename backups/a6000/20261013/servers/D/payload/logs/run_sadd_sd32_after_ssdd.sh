#!/usr/bin/env bash
set -euo pipefail
root=/home/powerleader/ig-ds-ssm
raw=$root/data/cross_task_raw
prepared=$root/data/cross_task_prepared
while kill -0 77890 2>/dev/null; do sleep 30; done
until [ -f "$raw/SAR-aircraft.zip" ] && [ "$(stat -c %s "$raw/SAR-aircraft.zip")" -eq 186704906 ]; do sleep 30; done
cd $root/work_train_single
$root/envs/ig-ds-ssm-env/bin/python -u tools/prepare_cross_task_data.py --raw-root "$raw" --out "$prepared" --seed 0 --datasets SADD
$root/envs/ig-ds-ssm-env/bin/python -u tools/run_sar4_full_suite.py \
  --data-root "$prepared/detection" --project "$root/experiments/cross_task_detection" \
  --experiment-id ig_dsssm_p3_sadd_a6000_sd32_seed0 --datasets SADD --variants full_p3 \
  --device 0 --state-dim 32 --epochs 150 --patience 30 --imgsz 640 \
  --batch 8 --eval-batch 8 --workers 8 --seed 0 --continue-on-error
