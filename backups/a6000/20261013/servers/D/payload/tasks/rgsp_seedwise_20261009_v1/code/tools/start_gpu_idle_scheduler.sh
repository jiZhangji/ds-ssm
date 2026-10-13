#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/powerleader/ig-ds-ssm}"
SCHEDULER="$ROOT/work_train_single/tools/gpu_idle_scheduler.py"
CONFIG="$ROOT/work_train_single/tools/ig_ds_ssm_experiment_queue.json"
RUNTIME="$ROOT/scheduler"

mkdir -p "$RUNTIME"
nohup python3 -u "$SCHEDULER" \
  --config "$CONFIG" \
  --state "$RUNTIME/state.json" \
  --status "$RUNTIME/status.json" \
  --lock "$RUNTIME/scheduler.lock" \
  --loop-seconds 60 \
  >> "$RUNTIME/scheduler.log" 2>&1 </dev/null &
echo "$!" > "$RUNTIME/launcher.pid"
echo "scheduler launcher pid=$!"
echo "status=$RUNTIME/status.json"
echo "log=$RUNTIME/scheduler.log"
