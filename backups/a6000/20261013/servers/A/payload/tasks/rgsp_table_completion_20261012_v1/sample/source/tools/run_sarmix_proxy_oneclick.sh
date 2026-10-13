#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK_ROOT="${WORK_ROOT:-$ROOT/sarmix_workspace}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$WORK_ROOT/data_archives}"
PREPARED_ROOT="${PREPARED_ROOT:-$WORK_ROOT/prepared}"
SARMIX_ROOT="${SARMIX_ROOT:-$WORK_ROOT/SAR-Mix-1000-400}"
PROJECT="${PROJECT:-$WORK_ROOT/output}"
EXPERIMENT_ID="${EXPERIMENT_ID:-sarmix_scratch_d16_v1}"
DEVICE="${DEVICE:-0}"
ONLY="${ONLY:-}"

mkdir -p "$WORK_ROOT" "$PROJECT" "$ROOT/logs"

echo "===== SAR-Mix proxy protocol ====="
echo "root:       $ROOT"
echo "workspace:  $WORK_ROOT"
echo "experiment: $EXPERIMENT_ID"
echo "device:     $DEVICE"
echo "only:       ${ONLY:-all nine experiments}"

"$PYTHON" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
import selective_scan_cuda_core  # noqa: F401
print("selective_scan_cuda_core=OK")
PY

if [[ ! -f "$SARMIX_ROOT/SAR-Mix.yaml" ]]; then
    declare -A YAMLS=(
        [OGSOD]="$PREPARED_ROOT/OGSOD/OGSOD.yaml"
        [SAR_AIRcraft]="$PREPARED_ROOT/SAR_AIRcraft/SAR_AIRcraft.yaml"
        [SSDD]="$PREPARED_ROOT/SSDD/SSDD.yaml"
        [ship_dataset_v0]="$PREPARED_ROOT/ship_dataset_v0/ship_dataset_v0.yaml"
    )
    for dataset in OGSOD SAR_AIRcraft SSDD ship_dataset_v0; do
        if [[ ! -f "${YAMLS[$dataset]}" ]]; then
            echo "===== Preparing $dataset ====="
            "$PYTHON" "$ROOT/tools/prepare_hf_sar_det_datasets.py" \
                --dataset-root "$ARCHIVE_ROOT" \
                --out "$PREPARED_ROOT" \
                --datasets "$dataset"
        fi
    done

    echo "===== Building deterministic mixed subset ====="
    "$PYTHON" "$ROOT/tools/prepare_sarmix_proxy.py" \
        --ssdd "${YAMLS[SSDD]}" \
        --ogsod "${YAMLS[OGSOD]}" \
        --sar-ship "${YAMLS[ship_dataset_v0]}" \
        --sar-aircraft "${YAMLS[SAR_AIRcraft]}" \
        --out "$SARMIX_ROOT" \
        --copy-mode hardlink
fi

"$PYTHON" - "$SARMIX_ROOT/dataset_snapshot.json" <<'PY'
import json
import sys
from pathlib import Path

snapshot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert snapshot["total_train_images"] == 1000, snapshot
assert snapshot["total_val_images"] == 400, snapshot
assert len(snapshot["classes"]) == 11, snapshot
print("dataset check: 1000 train / 400 val / 11 classes")
for source, item in snapshot["sources"].items():
    print(f"  {source}: {item['train_images']} train / {item['val_images']} val")
PY

ONLY_ARGS=()
if [[ -n "$ONLY" ]]; then
    read -r -a REQUESTED <<< "$ONLY"
    ONLY_ARGS=(--only "${REQUESTED[@]}")
fi

echo "===== Starting resumable serial queue ====="
"$PYTHON" -u "$ROOT/tools/run_sarmix_proxy.py" \
    --data "$SARMIX_ROOT/SAR-Mix.yaml" \
    --project "$PROJECT" \
    --experiment-id "$EXPERIMENT_ID" \
    --ablation-placement p2 \
    --state-dim 16 \
    --epochs 150 \
    --patience 25 \
    --imgsz 512 \
    --batch 16 \
    --eval-batch 32 \
    --device "$DEVICE" \
    --workers 8 \
    --seed 0 \
    --cache \
    "${ONLY_ARGS[@]}"

echo "===== Final metrics ====="
"$PYTHON" "$ROOT/tools/summarize_sarmix.py" \
    --project "$PROJECT" \
    --experiment-id "$EXPERIMENT_ID"
