#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/cross_task_raw}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/data/cross_task_prepared}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-256}"
WORKERS="${WORKERS:-4}"
STATE_DIM="${STATE_DIM:-16}"
SEED="${SEED:-0}"

[[ $# -gt 0 ]] || { echo "Usage: $0 DATASET..." >&2; exit 2; }
DATASETS=("$@")

mkdir -p "$PREPARED_ROOT" "$PROJECT"
"$PYTHON_BIN" -u "$ROOT/tools/prepare_cross_task_data.py" \
  --raw-root "$RAW_ROOT" --out "$PREPARED_ROOT" --seed "$SEED" \
  --datasets "${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    SAR-ACD) safe="SAR_ACD"; task="classification" ;;
    SAMPLE) safe="SAMPLE"; task="classification" ;;
    FUSAR-Map) safe="FUSAR_Map"; task="segmentation" ;;
    Ottawa) safe="Ottawa"; task="change_detection" ;;
    "Yellow River") safe="Yellow_River"; task="change_detection" ;;
    Farmland) safe="Farmland"; task="change_detection" ;;
    Sulzberger) safe="Sulzberger"; task="change_detection" ;;
    *) echo "Unsupported shared dataset: $dataset" >&2; exit 2 ;;
  esac
  manifest="$PREPARED_ROOT/$task/$safe/manifest.json"
  for variant in baseline ours; do
    "$PYTHON_BIN" -u "$ROOT/tools/run_shared_sar_tasks.py" \
      --manifest "$manifest" --dataset "$dataset" --variant "$variant" \
      --output "$PROJECT/$task/$safe/$variant" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --batch "$BATCH" --imgsz "$IMGSZ" \
      --workers "$WORKERS" --state-dim "$STATE_DIM" --device "$DEVICE"
  done
done

"$PYTHON_BIN" -u "$ROOT/tools/summarize_cross_task_results.py" \
  --project "$PROJECT" --output "$PROJECT/final_results.json"
