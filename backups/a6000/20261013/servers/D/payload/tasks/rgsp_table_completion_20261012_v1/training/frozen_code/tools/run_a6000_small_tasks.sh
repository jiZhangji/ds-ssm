#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/cross_task_raw}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/data/cross_task_prepared}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task}"
DEVICE="${DEVICE:-cuda}"
DATASETS=("$@")

[[ ${#DATASETS[@]} -gt 0 ]] || { echo "Usage: $0 DATASET..." >&2; exit 2; }
mkdir -p "$RAW_ROOT" "$PREPARED_ROOT" "$PROJECT"

"$PYTHON_BIN" -u "$ROOT/tools/prepare_cross_task_data.py" \
  --raw-root "$RAW_ROOT" --out "$PREPARED_ROOT" --seed "${SEED:-0}" \
  --datasets "${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    SAR-ACD) safe="SAR_ACD"; task="classification" ;;
    SAMPLE) safe="SAMPLE"; task="classification" ;;
    Ottawa) safe="Ottawa"; task="change_detection" ;;
    "Yellow River") safe="Yellow_River"; task="change_detection" ;;
    Farmland) safe="Farmland"; task="change_detection" ;;
    Sulzberger) safe="Sulzberger"; task="change_detection" ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
  esac
  manifest="$PREPARED_ROOT/$task/$safe/manifest.json"
  for variant in baseline ours; do
    "$PYTHON_BIN" -u "$ROOT/tools/run_shared_sar_tasks.py" \
      --manifest "$manifest" --dataset "$dataset" --variant "$variant" \
      --output "$PROJECT/$task/$safe/$variant" --epochs "${EPOCHS:-100}" \
      --patience "${PATIENCE:-20}" --batch "${BATCH:-8}" \
      --imgsz "${IMGSZ:-256}" --workers "${WORKERS:-4}" \
      --state-dim "${STATE_DIM:-16}" --device "$DEVICE"
  done
done

"$PYTHON_BIN" -u "$ROOT/tools/summarize_cross_task_results.py" \
  --project "$PROJECT" --output "$PROJECT/final_results.json"
