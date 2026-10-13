#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOT="${RAW_ROOT:-$ROOT/data/cross_task_raw}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/data/cross_task_prepared}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/cross_task_remaining}"
SEED="${SEED:-0}"

mkdir -p "$PREPARED_ROOT" "$PROJECT" "$LOG_ROOT"

"$PYTHON_BIN" -u "$ROOT/tools/prepare_cross_task_data.py" \
  --raw-root "$RAW_ROOT" --out "$PREPARED_ROOT" --seed "$SEED" \
  --datasets AIR-SARShip-2.0 SRSDD-v1.0 SADD SAR-ACD SAMPLE FUSAR-Map \
    Ottawa "Yellow River" Farmland Sulzberger \
  > "$LOG_ROOT/prepare.log" 2>&1

"$PYTHON_BIN" -u "$ROOT/tools/run_sar4_full_suite.py" \
  --data-root "$PREPARED_ROOT/detection" --project "$PROJECT/detection" \
  --experiment-id "ig_dsssm_p3_remaining_detection_seed${SEED}" \
  --datasets AIR-SARShip-2.0 SRSDD-v1.0 SADD \
  --variants baseline full_p3 --device 0 --state-dim 16 \
  --epochs 150 --patience 30 --imgsz 640 --batch 8 --eval-batch 8 \
  --workers 4 --seed "$SEED" --continue-on-error \
  > "$LOG_ROOT/gpu0_detection.log" 2>&1 &
GPU0_PID=$!
echo "$GPU0_PID" > "$LOG_ROOT/gpu0_detection.pid"

(
  for spec in \
    "classification|SAR_ACD|SAR-ACD" \
    "classification|SAMPLE|SAMPLE" \
    "segmentation|FUSAR_Map|FUSAR-Map" \
    "change_detection|Ottawa|Ottawa" \
    "change_detection|Yellow_River|Yellow River" \
    "change_detection|Farmland|Farmland" \
    "change_detection|Sulzberger|Sulzberger"; do
    IFS='|' read -r task safe dataset <<< "$spec"
    manifest="$PREPARED_ROOT/$task/$safe/manifest.json"
    for variant in baseline ours; do
      "$PYTHON_BIN" -u "$ROOT/tools/run_shared_sar_tasks.py" \
        --manifest "$manifest" --dataset "$dataset" --variant "$variant" \
        --output "$PROJECT/$task/$safe/$variant" --epochs 100 --patience 20 \
        --batch 8 --imgsz 256 --workers 4 --device cuda:1
    done
  done
  "$PYTHON_BIN" -u "$ROOT/tools/summarize_cross_task_results.py" \
    --project "$PROJECT" --output "$PROJECT/final_results.json"
) > "$LOG_ROOT/gpu1_shared.log" 2>&1 &
GPU1_PID=$!
echo "$GPU1_PID" > "$LOG_ROOT/gpu1_shared.pid"

echo "GPU0 detection PID=$GPU0_PID"
echo "GPU1 shared PID=$GPU1_PID"
echo "Logs: $LOG_ROOT"
