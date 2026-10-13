#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
WEIGHT_DIR="$ROOT/weights/historical_ssdd"
EX27="$WEIGHT_DIR/ex27_best.pt"
BASELINE="$WEIGHT_DIR/baseline_best.pt"
EX27_SIZE=12269861
BASELINE_SIZE=12315576

while [[ ! -f "$EX27" || ! -f "$BASELINE" ]] || \
      [[ "$(stat -c %s "$EX27" 2>/dev/null || echo 0)" -ne "$EX27_SIZE" ]] || \
      [[ "$(stat -c %s "$BASELINE" 2>/dev/null || echo 0)" -ne "$BASELINE_SIZE" ]]; do
  sleep 10
done

echo "1cebf8735311e5f0461b079d4e50b8b5d957dd6f171532e832821857b240aa09  $EX27" | sha256sum -c -
echo "c70f6a023adc51dbeaa892976fdcfba542a4b0c867a1bcb9a7ba3c726210d52e  $BASELINE" | sha256sum -c -

export DATA_YAML="${DATA_YAML:-$ROOT/data/sar4_prepared/SSDD/SSDD.yaml}"
export DEVICE="${DEVICE:-0}"
export BATCH="${BATCH:-32}"
export WORKERS="${WORKERS:-8}"

exec "$ROOT/work_train_single/tools/run_ssdd_historical_checkpoint_diagnostic.sh"
