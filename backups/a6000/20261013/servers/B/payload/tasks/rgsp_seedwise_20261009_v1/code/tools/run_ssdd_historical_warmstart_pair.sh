#!/usr/bin/env bash
set -euo pipefail

# Target-domain warm-start diagnostic. The recovered checkpoints were already
# trained on SSDD, so these runs must never be reported as external pretraining.

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="$ROOT/work_train_single"
PYTHON_BIN="$ROOT/envs/ig-ds-ssm-env/bin/python"
MANIFEST="$ROOT/experiments/sar4/sar4_scratch_p3_sd16_v1-manifest"
DATA_YAML="$MANIFEST/datasets/SSDD.yaml"
PROJECT="$ROOT/experiments/ssdd_historical_warmstart_pair_20260828"
WEIGHT_DIR="$ROOT/weights/historical_ssdd"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
USE_AMP="${USE_AMP:-0}"

AMP_ARGS=()
if [[ "$USE_AMP" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

mkdir -p "$PROJECT" "$PROJECT/evaluations" "$ROOT/logs"

run_one() {
  local label="$1"
  local config="$2"
  local initial_weights="$3"
  local placement="$4"
  local variant="$5"
  local state_dim="$6"
  local run_name="ssdd_${label}_b${BATCH}_e${EPOCHS}"
  local run_dir="$PROJECT/$run_name"
  local train_log="$ROOT/logs/${run_name}.train.log"
  local eval_log="$ROOT/logs/${run_name}.eval.log"

  if [[ ! -s "$run_dir/train_complete.json" ]]; then
    if [[ -f "$run_dir/weights/last.pt" ]]; then
      "$PYTHON_BIN" -u "$CODE/mbyolo_train_sarair.py" \
        --task train \
        --data "$DATA_YAML" \
        --config "$config" \
        --resume "$run_dir/weights/last.pt" \
        --resume_batch_size "$BATCH" \
        --resume_imgsz 640 \
        --resume_device "$DEVICE" \
        --epochs "$EPOCHS" \
        --workers "$WORKERS" \
        >> "$train_log" 2>&1
    else
      "$PYTHON_BIN" -u "$CODE/mbyolo_train_sarair.py" \
        --task train \
        --data "$DATA_YAML" \
        --config "$config" \
        --weights "$initial_weights" \
        --epochs "$EPOCHS" \
        --imgsz 640 \
        --batch_size "$BATCH" \
        --device "$DEVICE" \
        --workers "$WORKERS" \
        --optimizer AdamW \
        --lr0 0.0002 \
        --lrf 0.01 \
        --weight_decay 0.0005 \
        --nbs 64 \
        --warmup_epochs 3.0 \
        --module_warmup_epochs 0 \
        --patience 150 \
        --cos_lr \
        --mosaic 0.3 \
        --close_mosaic 15 \
        --degrees 180 \
        --translate 0.1 \
        --scale 0.3 \
        --flipud 0.5 \
        --fliplr 0.5 \
        --seed 0 \
        --deterministic \
        --cache \
        "${AMP_ARGS[@]}" \
        --project "$PROJECT" \
        --name "$run_name" \
        >> "$train_log" 2>&1
    fi
    test -s "$run_dir/weights/best.pt"
    printf '{"status":"complete","evidence":"target-domain-warmstart-diagnostic"}\n' > "$run_dir/train_complete.json"
  fi

  if [[ ! -s "$run_dir/eval_test.json" ]]; then
    "$PYTHON_BIN" -u "$CODE/tools/evaluate_sarmix.py" \
      --weights "$run_dir/weights/best.pt" \
      --data "$DATA_YAML" \
      --experiment "$run_name" \
      --placement "$placement" \
      --variant "$variant" \
      --state-dim "$state_dim" \
      --seed 0 \
      --subset SSDD \
      --split test \
      --output-json "$run_dir/eval_test.json" \
      --imgsz 640 \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --workers "$WORKERS" \
      --project "$PROJECT/evaluations" \
      --name "${run_name}_official_test" \
      >> "$eval_log" 2>&1
  fi
}

# Run ours first so the most relevant compatibility diagnostic finishes first.
run_one \
  "p3_d16_from_historical_ex27" \
  "$MANIFEST/configs/ssdd_full_p3.yaml" \
  "$WEIGHT_DIR/ex27_best.pt" \
  "P3" \
  "full" \
  16

run_one \
  "baseline_from_historical_baseline" \
  "$MANIFEST/configs/ssdd_baseline.yaml" \
  "$WEIGHT_DIR/baseline_best.pt" \
  "baseline" \
  "baseline" \
  0

echo "Warm-start pair complete: $PROJECT"
