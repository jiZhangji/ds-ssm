#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/data/cross_task_prepared}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task}"
DEVICE="${DEVICE:-cuda:1}"

for spec in \
  "classification|SAR_ACD|SAR-ACD" \
  "classification|SAMPLE|SAMPLE" \
  "segmentation|FUSAR_Map|FUSAR-Map" \
  "change_detection|Ottawa|Ottawa" \
  "change_detection|Yellow_River|Yellow River" \
  "change_detection|Farmland|Farmland" \
  "change_detection|Sulzberger|Sulzberger"; do
  IFS='|' read -r task safe dataset <<< "$spec"
  "$PYTHON_BIN" -u "$ROOT/tools/run_shared_sar_tasks.py" \
    --manifest "$PREPARED_ROOT/$task/$safe/manifest.json" \
    --dataset "$dataset" --variant ours --state-dim 32 \
    --output "$PROJECT/$task/$safe/ours_sd32" \
    --epochs 100 --patience 20 --batch 8 --imgsz 256 --workers 4 \
    --device "$DEVICE"
done

"$PYTHON_BIN" -u "$ROOT/tools/summarize_cross_task_results.py" \
  --project "$PROJECT" --output "$PROJECT/final_results.json"
