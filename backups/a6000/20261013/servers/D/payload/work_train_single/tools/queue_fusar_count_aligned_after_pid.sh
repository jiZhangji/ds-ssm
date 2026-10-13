#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
WAIT_PID="${WAIT_PID:?Set WAIT_PID to the active queue process that must finish first}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUNNER="$ROOT/work_train_single/tools/run_fusar_count_aligned.sh"

echo "Waiting for PID $WAIT_PID before FUSAR count-aligned experiments"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

for variant in ours baseline; do
  echo "Starting FUSAR count-aligned variant=$variant"
  env ROOT="$ROOT" VARIANT="$variant" STATE_DIM=16 DEVICE=cuda:0 "$RUNNER"
done

echo "FUSAR count-aligned pair complete"
