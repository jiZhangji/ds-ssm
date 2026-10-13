#!/usr/bin/env bash
set -euo pipefail

ROOT=${IG_DS_SSM_ROOT:-/home/powerleader/ig-ds-ssm}
PREP_PID_FILE="$ROOT/logs/openearthmap_prepare.pid"
PREP_LOG="$ROOT/logs/openearthmap_prepare.log"
DATA="$ROOT/data/openearthmap_sar/sar_only_p"
MANIFEST="$DATA/manifest.json"
REMOTE=powerleader@10.6.2.4
SSH_KEY="$HOME/.ssh/ig_ds_ssm_ab"
PAIR_LOG="$ROOT/logs/openearthmap_pair_queue.log"

exec >>"$PAIR_LOG" 2>&1
echo "[$(date -Is)] queue started"

prep_pid=$(cat "$PREP_PID_FILE")
while kill -0 "$prep_pid" 2>/dev/null; do
  sleep 60
done
if [[ ! -s "$MANIFEST" ]]; then
  echo "[$(date -Is)] preparation failed"
  tail -n 80 "$PREP_LOG"
  exit 1
fi

"$ROOT/envs/ig-ds-ssm-env/bin/python" - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = Counter(row["split"] for row in manifest["records"])
expected = {"train": 4333, "val": 210, "test": 490}
if dict(counts) != expected:
    raise SystemExit(f"split mismatch: {dict(counts)} != {expected}")
missing = [value for row in manifest["records"] for value in (row["image"], row["mask"]) if not Path(value).is_file()]
if missing:
    raise SystemExit(f"missing materialized files: {missing[:5]}")
print(f"manifest audit passed: {dict(counts)}")
PY

echo "[$(date -Is)] syncing data B -> A over 10.6.2.x LAN"
ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "mkdir -p '$ROOT/data/openearthmap_sar' '$ROOT/logs'"
rsync -a --partial --info=stats2 -e "ssh -i $SSH_KEY -o BatchMode=yes" \
  "$DATA/" "$REMOTE:$DATA/"

echo "[$(date -Is)] launching baseline on B"
if ! pgrep -af "run_shared_sar_tasks_v2.py.*OpenEarthMap-SAR.*--variant baseline" >/dev/null; then
  nohup "$ROOT/work_train_single/tools/run_openearthmap_sar.sh" baseline \
    >"$ROOT/logs/openearthmap_baseline.log" 2>&1 &
  echo $! >"$ROOT/logs/openearthmap_baseline.pid"
fi

echo "[$(date -Is)] launching P3-d16 on A"
ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "
  if ! pgrep -af 'run_shared_sar_tasks_v2.py.*OpenEarthMap-SAR.*--variant ours' >/dev/null; then
    nohup '$ROOT/work_train_single/tools/run_openearthmap_sar.sh' ours \
      >'$ROOT/logs/openearthmap_ours.log' 2>&1 &
    echo \$! >'$ROOT/logs/openearthmap_ours.pid'
  fi
"
echo "[$(date -Is)] both jobs dispatched"
