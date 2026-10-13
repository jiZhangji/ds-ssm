#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate recovered SSDD checkpoints under one frozen evaluator. These
# checkpoints were selected on the historical SSDD validation protocol, so the
# outputs are diagnostic evidence and must not be mixed with scratch results.

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
CODE="${CODE:-$ROOT/work_train_single}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/ig-ds-ssm-env/bin/python}"
DATA_YAML="${DATA_YAML:-$ROOT/data/ssdd_historical_prepared/SSDD/SSDD.yaml}"
WEIGHT_DIR="${WEIGHT_DIR:-$ROOT/weights/historical_ssdd}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/experiments/ssdd_historical_checkpoint_diagnostic_20260828}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"

if [[ ! -f "$DATA_YAML" ]]; then
  echo "Missing SSDD YAML: $DATA_YAML" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/evaluations"

EFFECTIVE_DATA_YAML="$DATA_YAML"
DIAGNOSTIC_DATA_YAML="$OUTPUT_ROOT/SSDD_official_test_diagnostic.yaml"
"$PYTHON_BIN" - "$DATA_YAML" "$DIAGNOSTIC_DATA_YAML" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
root = Path(payload.get("path", source.parent))
if not root.is_absolute():
    root = (source.parent / root).resolve()
val = Path(payload.get("val", ""))
val = val if val.is_absolute() else root / val
if not val.exists():
    payload["val"] = payload["test"]
payload["path"] = str(root)
output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
EFFECTIVE_DATA_YAML="$DIAGNOSTIC_DATA_YAML"

evaluate_one() {
  local label="$1"
  local weights="$2"
  local placement="$3"
  local variant="$4"
  local state_dim="$5"
  local output="$OUTPUT_ROOT/${label}_official_test.json"

  if [[ -s "$output" ]]; then
    echo "SKIP COMPLETE: $output"
    return
  fi
  if [[ ! -f "$weights" ]]; then
    echo "Missing checkpoint: $weights" >&2
    exit 2
  fi

  "$PYTHON_BIN" -u "$CODE/tools/evaluate_sarmix.py" \
    --weights "$weights" \
    --data "$EFFECTIVE_DATA_YAML" \
    --experiment "historical_${label}_fixed_recheck" \
    --placement "$placement" \
    --variant "$variant" \
    --state-dim "$state_dim" \
    --seed 0 \
    --subset SSDD \
    --split test \
    --output-json "$output" \
    --imgsz 640 \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$OUTPUT_ROOT/evaluations" \
    --name "historical_${label}_official_test"
}

evaluate_one \
  "mambayolo_t_baseline" \
  "$WEIGHT_DIR/baseline_best.pt" \
  "baseline" \
  "baseline" \
  0

evaluate_one \
  "pfim_ex27" \
  "$WEIGHT_DIR/ex27_best.pt" \
  "historical_backbone" \
  "historical_pfim_ex27" \
  16

echo "Historical checkpoint diagnostics complete: $OUTPUT_ROOT"
