#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$ROOT/sarmix_workspace}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$WORK_ROOT/data_archives}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ARCHIVES=(OGSOD.zip SAR_AIRcraft.zip SSDD.zip ship_dataset_v0.zip)

while true; do
    downloading=0
    for archive in "${ARCHIVES[@]}"; do
        if pgrep -f "(wget|curl).*${archive}" >/dev/null; then
            downloading=1
        fi
    done
    if [[ "$downloading" -eq 0 ]]; then
        break
    fi
    echo "[$(date '+%F %T')] waiting for dataset downloads"
    ls -lh "${ARCHIVES[@]/#/$ARCHIVE_ROOT/}" 2>/dev/null || true
    sleep "$POLL_SECONDS"
done

for archive in "${ARCHIVES[@]}"; do
    path="$ARCHIVE_ROOT/$archive"
    if [[ ! -f "$path" ]]; then
        echo "missing archive: $path" >&2
        exit 1
    fi
    echo "checking: $path"
    unzip -tq "$path" >/dev/null
done

exec bash "$ROOT/tools/run_sarmix_proxy_oneclick.sh"
