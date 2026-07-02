#!/usr/bin/env bash
set -euo pipefail

# Run from Mamba-YOLO project root on the server.

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-1}"

"${PYTHON_BIN}" complexity_benchmark/model_complexity_benchmark.py \
  --configs \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --warmup 20 \
  --runs 100 \
  --out complexity_benchmark/results/model_complexity.csv

echo "Done. Results are in complexity_benchmark/results/model_complexity.csv"
