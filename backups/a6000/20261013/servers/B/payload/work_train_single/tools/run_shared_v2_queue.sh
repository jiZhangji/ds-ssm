#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
PREPARED="${PREPARED:-$ROOT/data/cross_task_prepared_v2}"
PROJECT="${PROJECT:-$ROOT/experiments/corrected_shared_v2}"
STATE_DIM="${STATE_DIM:-32}"
DEVICE="${DEVICE:-cuda:0}"

run_pair() {
  local dataset="$1" manifest="$2" imgsz="$3" batch="$4" epochs="$5" patience="$6"
  local slug
  slug="$(printf '%s' "$dataset" | tr ' /' '__' | tr '[:upper:]' '[:lower:]')"
  for variant in baseline ours; do
    "$PYTHON_BIN" -u "$CODE/tools/run_shared_sar_tasks_v2.py" \
      --manifest "$manifest" \
      --dataset "$dataset" \
      --variant "$variant" \
      --output "$PROJECT/$slug/$variant" \
      --epochs "$epochs" \
      --patience "$patience" \
      --batch "$batch" \
      --imgsz "$imgsz" \
      --workers 6 \
      --lr 0.0002 \
      --state-dim "$STATE_DIM" \
      --seed 0 \
      --device "$DEVICE" \
      --amp
  done
}

for dataset in "$@"; do
  case "$dataset" in
    SAR-ACD)
      run_pair "$dataset" "$PREPARED/classification/SAR_ACD/manifest.json" 224 64 200 35
      ;;
    SAMPLE)
      run_pair "$dataset" "$PREPARED/classification/SAMPLE/manifest.json" 224 64 200 35
      ;;
    FUSAR-Map)
      run_pair "$dataset" "$PREPARED/segmentation/FUSAR_Map/manifest.json" 512 4 160 30
      ;;
    Ottawa)
      run_pair "$dataset" "$PREPARED/change_detection/Ottawa/manifest.json" 192 16 150 25
      ;;
    "Yellow River")
      run_pair "$dataset" "$PREPARED/change_detection/Yellow_River/manifest.json" 192 16 150 25
      ;;
    Farmland)
      run_pair "$dataset" "$PREPARED/change_detection/Farmland/manifest.json" 192 16 150 25
      ;;
    Sulzberger)
      run_pair "$dataset" "$PREPARED/change_detection/Sulzberger/manifest.json" 192 16 150 25
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac
done
