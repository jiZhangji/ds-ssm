#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 WAIT_PID DATASET [DATASET ...]" >&2
  exit 2
fi

WAIT_PID="$1"
shift

POLL_SECONDS="${POLL_SECONDS:-300}"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

cd "${CODE:?CODE must point to work_train_single}"
exec bash tools/run_historical_config_detection_p3.sh "$@"
