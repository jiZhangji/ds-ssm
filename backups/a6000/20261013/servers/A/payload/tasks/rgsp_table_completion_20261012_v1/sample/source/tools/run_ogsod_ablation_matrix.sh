#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_YAML="${DATA_YAML:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/det/OGSOD/OGSOD.yaml}"
PROJECT="${PROJECT:-$ROOT_DIR/output_dir/ogsod_ablation}"
MATRIX_ID="${MATRIX_ID:-ogsod_t_ablation_v2}"
SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-baseline no_pfim global full}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
EVAL_DEVICE="${EVAL_DEVICE:-0}"
EVAL_BATCH="${EVAL_BATCH:-32}"
DRY_RUN="${DRY_RUN:-0}"
DATA_FINGERPRINT="${DATA_FINGERPRINT:-0}"
VERIFY_FALLBACK="${VERIFY_FALLBACK:-1}"

: "${CONDA_PREFIX:?Activate the sar-pretrain conda environment before running this matrix}"

CONFIG_DIR="$ROOT_DIR/ultralytics/cfg/models/mamba-yolo"
LOG_DIR="$ROOT_DIR/logs/$MATRIX_ID"
MANIFEST_DIR="$PROJECT/$MATRIX_ID-manifest"
PROTOCOL_FILE="$MANIFEST_DIR/protocol.txt"
SHA_FILE="$MANIFEST_DIR/sha256.txt"
DATASET_FILE="$MANIFEST_DIR/dataset_snapshot.json"

declare -A CONFIGS=(
  [baseline]="$CONFIG_DIR/Mamba-YOLO-T-OGSOD-baseline.yaml"
  [no_pfim]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-no-pfim.yaml"
  [gamma]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-gamma.yaml"
  [write]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-write.yaml"
  [delta]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-delta.yaml"
  [gamma_write]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-gamma-write.yaml"
  [global]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-global.yaml"
  [no_2dstd]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-no-2dstd.yaml"
  [full]="$CONFIG_DIR/Mamba-YOLO-T-DS-SSM-OGSOD-pfim-full.yaml"
)

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p "$LOG_DIR" "$PROJECT" "$MANIFEST_DIR"

test -f "$DATA_YAML" || {
  echo "Missing dataset config: $DATA_YAML"
  exit 1
}

for variant in $VARIANTS; do
  test -n "${CONFIGS[$variant]:-}" || {
    echo "Unknown variant: $variant"
    exit 1
  }
  test -f "${CONFIGS[$variant]}" || {
    echo "Missing model config: ${CONFIGS[$variant]}"
    exit 1
  }
done

if [ "$VERIFY_FALLBACK" = "1" ]; then
  python tools/check_dsssm_cpu_fallback.py
fi

config_check_args=()
for variant in $VARIANTS; do
  config_check_args+=(--config "$variant=${CONFIGS[$variant]}")
done
python tools/check_ogsod_ablation_configs.py "${config_check_args[@]}"

if [ "$DATA_FINGERPRINT" = "1" ]; then
  dataset_snapshot_tmp="$(mktemp "$MANIFEST_DIR/dataset_snapshot.XXXXXX.json")"
  python tools/snapshot_yolo_dataset.py \
    --data "$DATA_YAML" \
    --output "$dataset_snapshot_tmp"
  if [ -f "$DATASET_FILE" ]; then
    cmp -s "$dataset_snapshot_tmp" "$DATASET_FILE" || {
      rm -f "$dataset_snapshot_tmp"
      echo "Dataset fingerprint changed for MATRIX_ID=$MATRIX_ID. Start a new MATRIX_ID."
      exit 1
    }
    rm -f "$dataset_snapshot_tmp"
  else
    mv "$dataset_snapshot_tmp" "$DATASET_FILE"
  fi
elif [ "$DRY_RUN" != "1" ]; then
  echo "Formal controlled runs require DATA_FINGERPRINT=1."
  echo "This records image-list and label-content hashes before training."
  exit 1
else
  echo "Dataset fingerprint deferred for DRY_RUN."
fi

if [ "$DRY_RUN" != "1" ] && pgrep -af '[m]byolo_train_sarair.py' >/dev/null; then
  echo "Another training process is already running:"
  pgrep -af '[m]byolo_train_sarair.py'
  exit 1
fi

protocol_tmp="$(mktemp "$MANIFEST_DIR/protocol.XXXXXX.txt")"
{
  echo "matrix_id=$MATRIX_ID"
  echo "root=$ROOT_DIR"
  echo "data=$DATA_YAML"
  echo "seeds=$SEEDS"
  echo "variants=$VARIANTS"
  echo "gpu_devices=$GPU_DEVICES"
  echo "eval_device=$EVAL_DEVICE"
  echo "eval_batch=$EVAL_BATCH"
  echo "epochs=300"
  echo "global_batch=64"
  echo "optimizer=AdamW"
  echo "lr0=0.0002"
  echo "lrf=0.01"
  echo "weight_decay=0.0005"
  echo "nbs=64"
  echo "warmup_epochs=3.0"
  echo "module_warmup_epochs=0"
  echo "amp=false"
  echo "cos_lr=true"
  echo "mosaic=0.3"
  echo "close_mosaic=30"
  echo "degrees=180"
  echo "translate=0.1"
  echo "scale=0.3"
  echo "flipud=0.5"
  echo "fliplr=0.5"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} > "$protocol_tmp"

sha_tmp="$(mktemp "$MANIFEST_DIR/sha256.XXXXXX.txt")"
{
  sha256sum "$DATA_YAML"
  sha256sum "$ROOT_DIR/tools/run_ogsod_ablation_matrix.sh"
  sha256sum "$ROOT_DIR/mbyolo_train_sarair.py"
  sha256sum "$ROOT_DIR/ultralytics/nn/modules/mamba_yolo.py"
  sha256sum "$ROOT_DIR/ultralytics/nn/tasks.py"
  sha256sum "$ROOT_DIR/ultralytics/nn/modules/common_utils_mbyolo.py"
  sha256sum "$ROOT_DIR/ultralytics/engine/trainer.py"
  sha256sum "$ROOT_DIR/ultralytics/engine/validator.py"
  sha256sum "$ROOT_DIR/ultralytics/engine/model.py"
  sha256sum "$ROOT_DIR/ultralytics/utils/metrics.py"
  sha256sum "$ROOT_DIR/tools/evaluate_ogsod_ablation.py"
  sha256sum "$ROOT_DIR/tools/summarize_ogsod_ablation_matrix.py"
  sha256sum "$ROOT_DIR/tools/snapshot_yolo_dataset.py"
  sha256sum "$ROOT_DIR/tools/evaluate_ogsod_strata.py"
  sha256sum "$ROOT_DIR/tools/visualize_pfim_importance.py"
  sha256sum "$ROOT_DIR/tools/snapshot_ogsod_model.py"
  sha256sum "$ROOT_DIR/tools/check_dsssm_cpu_fallback.py"
  sha256sum "$ROOT_DIR/tools/check_ogsod_ablation_configs.py"
  for variant in $VARIANTS; do
    sha256sum "${CONFIGS[$variant]}"
  done
} > "$sha_tmp"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN protocol preview:"
  cat "$protocol_tmp"
  echo "DRY_RUN SHA-256 preview:"
  cat "$sha_tmp"
  rm -f "$protocol_tmp" "$sha_tmp"
else
  if [ -f "$PROTOCOL_FILE" ]; then
    cmp -s "$protocol_tmp" "$PROTOCOL_FILE" || {
      rm -f "$protocol_tmp" "$sha_tmp"
      echo "Frozen protocol changed for MATRIX_ID=$MATRIX_ID. Start a new MATRIX_ID."
      exit 1
    }
    rm -f "$protocol_tmp"
  else
    mv "$protocol_tmp" "$PROTOCOL_FILE"
  fi

  if [ -f "$SHA_FILE" ]; then
    cmp -s "$sha_tmp" "$SHA_FILE" || {
      rm -f "$sha_tmp"
      echo "Code/config hashes changed for MATRIX_ID=$MATRIX_ID. Start a new MATRIX_ID."
      exit 1
    }
    rm -f "$sha_tmp"
  else
    mv "$sha_tmp" "$SHA_FILE"
  fi
fi

if [ "$DRY_RUN" != "1" ] && [ ! -f "$MANIFEST_DIR/run_manifest.txt" ]; then
  {
    echo "created_at=$(date --iso-8601=seconds)"
    echo "git_dirty=$(test -n "$(git status --short 2>/dev/null)" && echo yes || echo no)"
    cat "$PROTOCOL_FILE"
  } > "$MANIFEST_DIR/run_manifest.txt"
  git status --short > "$MANIFEST_DIR/git_status.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv > "$MANIFEST_DIR/gpus.csv" 2>/dev/null || true
fi

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    run_name="${MATRIX_ID}_${variant}_seed${seed}"
    run_dir="$PROJECT/$run_name"
    last_pt="$run_dir/weights/last.pt"
    best_pt="$run_dir/weights/best.pt"
    results_csv="$run_dir/results.csv"
    test_json="$run_dir/test_metrics.json"
    completed=0

    if [ ! -f "$run_dir/model_snapshot.json" ]; then
      python tools/snapshot_ogsod_model.py \
        --config "${CONFIGS[$variant]}" \
        --data "$DATA_YAML" \
        --variant "$variant" \
        --run-name "$run_name" \
        --imgsz 1024 \
        --output "$run_dir/model_snapshot.json"
    fi

    if [ -f "$results_csv" ]; then
      completed=$(($(wc -l < "$results_csv") - 1))
    fi

    if [ "$completed" -lt 300 ]; then
      timestamp="$(date +%Y%m%d_%H%M%S)"
      if [ -f "$last_pt" ]; then
        log_file="$LOG_DIR/${run_name}_resume_${timestamp}.log"
        command=(
          python -u mbyolo_train_sarair.py
          --task train
          --data "$DATA_YAML"
          --config "${CONFIGS[$variant]}"
          --resume "$last_pt"
          --resume_batch_size 64
          --resume_imgsz 1024
          --resume_device "$GPU_DEVICES"
        )
        action="RESUME"
      else
        if [ -d "$run_dir" ]; then
          unexpected_files="$(find "$run_dir" -mindepth 1 -maxdepth 1 ! -name model_snapshot.json -print -quit 2>/dev/null)"
          if [ -n "$unexpected_files" ]; then
            echo "Refusing to overwrite incomplete directory without last.pt: $run_dir"
            echo "Unexpected existing entry: $unexpected_files"
            exit 1
          fi
        fi
        log_file="$LOG_DIR/${run_name}_${timestamp}.log"
        command=(
          python -u mbyolo_train_sarair.py
          --task train
          --data "$DATA_YAML"
          --config "${CONFIGS[$variant]}"
          --weights none
          --imgsz 1024
          --batch_size 64
          --device "$GPU_DEVICES"
        )
        action="START"
      fi

      command+=(
        --epochs 300
        --workers 16
        --optimizer AdamW
        --lr0 0.0002
        --lrf 0.01
        --weight_decay 0.0005
        --nbs 64
        --warmup_epochs 3.0
        --patience 300
        --module_warmup_epochs 0
        --cos_lr
        --cache
        --mosaic 0.3
        --close_mosaic 30
        --degrees 180
        --translate 0.1
        --scale 0.3
        --flipud 0.5
        --fliplr 0.5
        --seed "$seed"
        --deterministic
        --project "$PROJECT"
        --name "$run_name"
      )

      echo "$action: variant=$variant seed=$seed completed=$completed/300"
      echo "CONFIG=${CONFIGS[$variant]}"
      echo "RUN_DIR=$run_dir"
      echo "LOG=$log_file"

      if [ "$DRY_RUN" = "1" ]; then
        printf 'TRAIN_COMMAND='
        printf '%q ' "${command[@]}"
        printf '\n'
      elif ! "${command[@]}" > "$log_file" 2>&1; then
        echo "Training failed: $run_name"
        echo "Inspect: $log_file"
        exit 1
      else
        echo "TRAIN COMPLETE: $run_name"
      fi
    else
      echo "TRAIN ALREADY COMPLETE: $run_name ($completed/300)"
    fi

    eval_name="${run_name}_best_test_1024_fp32"
    eval_command=(
      python -u tools/evaluate_ogsod_ablation.py
      --weights "$best_pt"
      --data "$DATA_YAML"
      --variant "$variant"
      --seed "$seed"
      --run-name "$run_name"
      --output-json "$test_json"
      --imgsz 1024
      --batch "$EVAL_BATCH"
      --device "$EVAL_DEVICE"
      --workers 8
      --project "$PROJECT/eval"
      --name "$eval_name"
    )

    if [ -f "$test_json" ]; then
      echo "TEST ALREADY COMPLETE: $run_name"
    elif [ "$DRY_RUN" = "1" ]; then
      printf 'EVAL_COMMAND='
      printf '%q ' "${eval_command[@]}"
      printf '\n'
    else
      test -f "$best_pt" || {
        echo "Missing validation-selected checkpoint: $best_pt"
        exit 1
      }
      eval_log="$LOG_DIR/${run_name}_best_test.log"
      if ! "${eval_command[@]}" > "$eval_log" 2>&1; then
        echo "TEST evaluation failed: $run_name"
        echo "Inspect: $eval_log"
        exit 1
      fi
      echo "TEST COMPLETE: $run_name"
      python tools/summarize_ogsod_ablation_matrix.py \
        --project "$PROJECT" \
        --matrix-id "$MATRIX_ID" \
        --expected-seeds "$SEEDS"
    fi
  done
done

if [ "$DRY_RUN" != "1" ] && find "$PROJECT" -maxdepth 2 -name test_metrics.json -print -quit | grep -q .; then
  python tools/summarize_ogsod_ablation_matrix.py \
    --project "$PROJECT" \
    --matrix-id "$MATRIX_ID" \
    --expected-seeds "$SEEDS"
fi

echo "Matrix queue finished: $MATRIX_ID"
