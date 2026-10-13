#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/cross_task_prepared/detection}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task_detection}"
DEVICE="${DEVICE:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_MEMORY_MIB="${IDLE_MEMORY_MIB:-1500}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"

DATASETS=()
EXPERIMENT_ID=""
STATE_DIM=16
BATCH=16
EVAL_BATCH=16
EPOCHS=150
PATIENCE=30
WORKERS=8
SEED=0

usage() {
  echo "Usage: $0 --experiment-id ID --datasets DATASET... [--state-dim 16] [--batch 16]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done
      ;;
    --state-dim) STATE_DIM="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --eval-batch) EVAL_BATCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -n "$EXPERIMENT_ID" && ${#DATASETS[@]} -gt 0 ]] || { usage; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CODE/tools/run_sar4_full_suite.py" ]] || { echo "Missing runner under $CODE" >&2; exit 1; }

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    SSDD) yaml="$DATA_ROOT/SSDD/SSDD.yaml" ;;
    SAR-AIRcraft1.0) yaml="$DATA_ROOT/SAR_AIRcraft/SAR_AIRcraft.yaml" ;;
    AIR-SARShip-2.0) yaml="$DATA_ROOT/AIR_SARShip_2/AIR_SARShip_2.yaml" ;;
    SRSDD-v1.0) yaml="$DATA_ROOT/SRSDD_v1/SRSDD_v1.yaml" ;;
    SADD) yaml="$DATA_ROOT/SADD/SADD.yaml" ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
  esac
  [[ -f "$yaml" ]] || { echo "Dataset is not prepared: $yaml" >&2; exit 1; }
done

mkdir -p "$ROOT/logs" "$PROJECT"
LOG="$ROOT/logs/${EXPERIMENT_ID}.queue.log"
LOCK="$ROOT/logs/${EXPERIMENT_ID}.queue.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "Queue already active: $EXPERIMENT_ID"; exit 0; }

gpu_is_idle() {
  local used compute_pids
  used=$(nvidia-smi --id="$DEVICE" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] || return 1
  (( used < IDLE_MEMORY_MIB )) || return 1
  compute_pids=$(nvidia-smi --id="$DEVICE" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ -z "$compute_pids" ]]
}

echo "[$(date -Is)] queued $EXPERIMENT_ID on $(hostname)" | tee -a "$LOG"
echo "datasets=${DATASETS[*]} state_dim=$STATE_DIM batch=$BATCH" | tee -a "$LOG"
if ! gpu_is_idle; then
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    echo "[$(date -Is)] selected GPU $DEVICE is busy; queue will not start a competing job" | tee -a "$LOG"
    exit 3
  fi
  while ! gpu_is_idle; do
    sleep "$POLL_SECONDS"
  done
fi

echo "[$(date -Is)] GPU is idle; starting $EXPERIMENT_ID" | tee -a "$LOG"
cd "$CODE"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -u tools/run_sar4_full_suite.py \
  --data-root "$DATA_ROOT" \
  --project "$PROJECT" \
  --experiment-id "$EXPERIMENT_ID" \
  --datasets "${DATASETS[@]}" \
  --variants baseline full_p3 \
  --device "$DEVICE" \
  --state-dim "$STATE_DIM" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --imgsz 640 \
  --batch "$BATCH" \
  --eval-batch "$EVAL_BATCH" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --continue-on-error >>"$LOG" 2>&1
echo "[$(date -Is)] queue finished: $EXPERIMENT_ID" | tee -a "$LOG"
