#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/disk1/zhangji/ds_ssm_sarmix_fast/code_cda7901/work_train_single
PYTHON=/home/zhangji/miniconda3/envs/mamba-yolo/bin/python
DATA=/mnt/disk1/zhangji/ds_ssm_sarmix_fast/SAR-Mix-1000-400/SAR-Mix.yaml
PROJECT=/mnt/disk1/zhangji/ds_ssm_sarmix_fast/output
EXPERIMENT=sarmix_scratch_d16_v1
GPU=${GPU:-0}
LOG=${LOG:-/mnt/disk1/zhangji/ds_ssm_sarmix_fast/logs/${EXPERIMENT}.p3_queue.log}

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "===== queued at $(date -Is) ====="
echo "Host: $(hostname) GPU: $GPU"

# Do not disturb another user's job. Require two consecutive quiet samples.
quiet=0
while true; do
  read -r used util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  echo "$(date -Is) used=${used}MiB util=${util}%"
  if [ "$used" -lt 1500 ] && [ "$util" -le 5 ]; then
    quiet=$((quiet + 1))
  else
    quiet=0
  fi
  if [ "$quiet" -ge 2 ]; then
    break
  fi
  sleep 60
done

cd "$ROOT"
exec "$PYTHON" -u tools/run_sarmix_proxy.py \
  --data "$DATA" \
  --project "$PROJECT" \
  --experiment-id "$EXPERIMENT" \
  --ablation-placement p3 \
  --state-dim 16 \
  --epochs 150 \
  --patience 25 \
  --imgsz 512 \
  --batch 16 \
  --eval-batch 32 \
  --device "$GPU" \
  --workers 8 \
  --seed 0 \
  --only baseline full_p3 \
  --continue-on-error
