#!/usr/bin/env bash
set -euo pipefail

# Authoritative matrix from the user's Phase 0-4 protocol.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_ROOTS="${RAW_ROOTS:-${RAW_ROOT:-$ROOT/data/cross_task_raw}}"
PREPARED_ROOT="${PREPARED_ROOT:-$ROOT/data/cross_task_prepared}"
PROJECT="${PROJECT:-$ROOT/experiments/cross_task}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-256}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-0}"
MODE="${MODE:-plan}"

DETECTION_DATASETS=(SSDD SAR-AIRcraft1.0 AIR-SARShip-2.0 SRSDD-v1.0 SADD)
SHARED_DATASETS=(SAR-ACD SAMPLE FUSAR-Map Ottawa "Yellow River" Farmland Sulzberger)

if [[ "$MODE" != "plan" && "$MODE" != "prepare" && "$MODE" != "run" && "$MODE" != "all" ]]; then
  echo "MODE must be plan, prepare, run, or all" >&2
  exit 2
fi

echo "===== IG-DS-SSM/PFIM authoritative cross-task suite ====="
echo "Detection : SSDD SAR-AIRcraft1.0 AIR-SARShip-2.0 SRSDD-v1.0 SADD"
echo "Classify  : SAR-ACD SAMPLE"
echo "Segment   : FUSAR-Map"
echo "Change    : Ottawa Yellow River Farmland Sulzberger"
echo "Protocol  : scratch, seed=$SEED, P3 state_dim=16, module_warmup=0"
echo "Mode      : $MODE"

if [[ "$MODE" == "plan" ]]; then
  echo "Planned runs: 24 (10 detection + 14 classification/segmentation/change detection)"
  echo "PLAN ONLY: no extraction or training started."
  exit 0
fi

if [[ "$MODE" == "prepare" || "$MODE" == "all" ]]; then
  read -r -a RAW_ARGS <<< "$RAW_ROOTS"
  "$PYTHON_BIN" -u "$ROOT/tools/prepare_cross_task_data.py" \
    --raw-root "${RAW_ARGS[@]}" --out "$PREPARED_ROOT" --seed "$SEED"
fi

if [[ "$MODE" == "prepare" ]]; then
  echo "PREPARE ONLY: no training started. Prepared data is under $PREPARED_ROOT"
  exit 0
fi

# Detection uses the already validated Mamba-YOLO runner. Existing Phase 1
# output directories are reused, so completed runs are skipped and last.pt is resumed.
DETECT_DATA_ROOT="${DETECT_DATA_ROOT:-$PREPARED_ROOT/detection}"
"$PYTHON_BIN" -u "$ROOT/tools/run_sar4_full_suite.py" \
  --data-root "$DETECT_DATA_ROOT" --project "$PROJECT/detection" \
  --experiment-id "ig_dsssm_p3_all_detection_seed${SEED}" \
  --datasets "${DETECTION_DATASETS[@]}" --variants baseline full_p3 \
  --device "$DEVICE" --state-dim 16 --epochs 150 --patience 30 \
  --imgsz 640 --batch 16 --eval-batch 16 --workers 8 --seed "$SEED" \
  --continue-on-error

for dataset in "${SHARED_DATASETS[@]}"; do
  case "$dataset" in
    SAR-ACD) safe="SAR_ACD"; task_dir="classification" ;;
    SAMPLE) safe="SAMPLE"; task_dir="classification" ;;
    FUSAR-Map) safe="FUSAR_Map"; task_dir="segmentation" ;;
    Ottawa) safe="Ottawa"; task_dir="change_detection" ;;
    "Yellow River") safe="Yellow_River"; task_dir="change_detection" ;;
    Farmland) safe="Farmland"; task_dir="change_detection" ;;
    Sulzberger) safe="Sulzberger"; task_dir="change_detection" ;;
  esac
  manifest="$PREPARED_ROOT/$task_dir/$safe/manifest.json"
  [[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; continue; }
  task=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["task"])' "$manifest")
  for variant in baseline ours; do
    "$PYTHON_BIN" -u "$ROOT/tools/run_shared_sar_tasks.py" \
      --manifest "$manifest" --dataset "$dataset" --variant "$variant" \
      --output "$PROJECT/$task/$safe/$variant" --epochs "$EPOCHS" \
      --patience "$PATIENCE" --batch "$BATCH" --imgsz "$IMGSZ" --workers "$WORKERS" --device "$DEVICE"
  done
done

"$PYTHON_BIN" -u "$ROOT/tools/summarize_cross_task_results.py" --project "$PROJECT" --output "$PROJECT/final_results.json"
