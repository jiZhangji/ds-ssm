#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/cross_task_raw}"
REPO="${GITHUB_REPO:-jiZhangji/ds-ssm}"
TAG="${GITHUB_DATA_TAG:-cross-task-small-data-v1}"
BASE_URL="https://github.com/$REPO/releases/download/$TAG"

ASSETS=(
  SAR_ACD.zip
  SAMPLE_dataset_public.zip
  Ottawa_data.zip
  Yellowriver_data.zip
  Farmland_data.zip
  Sulzberger_data.zip
  SHA256SUMS.txt
)

mkdir -p "$RAW_ROOT"
for asset in "${ASSETS[@]}"; do
  target="$RAW_ROOT/$asset"
  if [[ -s "$target" ]]; then
    echo "SKIP PRESENT: $target"
    continue
  fi
  echo "DOWNLOAD: $asset"
  curl -fL --retry 20 --retry-delay 5 --connect-timeout 20 \
    "$BASE_URL/$asset" -o "$target.part"
  mv "$target.part" "$target"
done

(
  cd "$RAW_ROOT"
  sha256sum -c SHA256SUMS.txt
)
echo "READY: $RAW_ROOT"
