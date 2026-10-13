#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$ROOT/data/sar4_raw}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/sar4_prepared}"
PREPARE_DATA="${PREPARE_DATA:-1}"

required_raw=(OGSOD SAR_AIRcraft SSDD ship_dataset_v0)
required_yaml=(
  "$DATA_ROOT/OGSOD/OGSOD.yaml"
  "$DATA_ROOT/SAR_AIRcraft/SAR_AIRcraft.yaml"
  "$DATA_ROOT/SSDD/SSDD.yaml"
  "$DATA_ROOT/ship_dataset_v0/ship_dataset_v0.yaml"
)

missing_prepared=()
for yaml_path in "${required_yaml[@]}"; do
  [[ -f "$yaml_path" ]] || missing_prepared+=("$yaml_path")
done

if ((${#missing_prepared[@]})); then
  if [[ "$PREPARE_DATA" != "1" ]]; then
    printf 'Missing prepared dataset: %s\n' "${missing_prepared[@]}" >&2
    exit 2
  fi

  for dataset in "${required_raw[@]}"; do
    yaml_name="$dataset"
    [[ "$dataset" == "ship_dataset_v0" ]] && yaml_name="ship_dataset_v0"
    target="$DATA_ROOT/$dataset/$yaml_name.yaml"
    [[ -f "$target" ]] && continue
    if [[ ! -d "$RAW_DATA_ROOT/$dataset" && ! -f "$RAW_DATA_ROOT/$dataset.zip" ]]; then
      echo "Missing raw dataset: $RAW_DATA_ROOT/$dataset or $RAW_DATA_ROOT/$dataset.zip" >&2
      exit 2
    fi
    "$PYTHON_BIN" -u "$ROOT/tools/prepare_hf_sar_det_datasets.py" \
      --dataset-root "$RAW_DATA_ROOT" \
      --out "$DATA_ROOT" \
      --datasets "$dataset" \
      --seed 0
  done
fi

for yaml_path in "${required_yaml[@]}"; do
  [[ -s "$yaml_path" ]] || { echo "Prepared YAML is missing or empty: $yaml_path" >&2; exit 2; }
done

export PYTHON_BIN DATA_ROOT
exec "$ROOT/tools/run_sar4_full_suite_oneclick.sh"
