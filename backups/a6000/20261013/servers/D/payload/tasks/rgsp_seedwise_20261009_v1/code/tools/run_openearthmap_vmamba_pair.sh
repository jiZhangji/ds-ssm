#!/usr/bin/env bash
set -euo pipefail

ROOT=${IG_DS_SSM_ROOT:-/home/powerleader/ig-ds-ssm}
WORK="$ROOT/work_train_single"
PY="$ROOT/envs/ig-ds-ssm-env/bin/python"
MANIFEST="$ROOT/data/openearthmap_sar/sar_only_p/manifest.json"
CKPT="$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth"
EXPERIMENT="openearthmap_saronly_p_vmamba_tiny_pretrained_p3_d16_v1"
REMOTE=powerleader@10.6.2.4
SSH_KEY="$HOME/.ssh/ig_ds_ssm_ab"

test -x "$PY"
test -f "$MANIFEST"
test -f "$CKPT"
mkdir -p "$ROOT/experiments/$EXPERIMENT" "$ROOT/logs"

if ! pgrep -af "run_shared_sar_tasks_v2.py.*$EXPERIMENT.*--variant baseline" >/dev/null; then
  nohup "$PY" -u "$WORK/tools/run_shared_sar_tasks_v2.py" \
    --manifest "$MANIFEST" \
    --dataset OpenEarthMap-SAR \
    --variant baseline \
    --backbone-layout vmamba_tiny \
    --decoder upernet_lite \
    --pretrained "$CKPT" \
    --output "$ROOT/experiments/$EXPERIMENT/baseline" \
    --epochs 150 \
    --patience 30 \
    --batch 8 \
    --imgsz 512 \
    --eval-batch 1 \
    --eval-imgsz 1024 \
    --workers 8 \
    --optimizer Adam \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --scheduler fixed \
    --label-smoothing 0.0 \
    --segmentation-loss weighted_cross_entropy_plus_dice \
    --selection-metric miou_all \
    --state-dim 16 \
    --seed 0 \
    --device cuda:0 \
    >"$ROOT/logs/openearthmap_vmamba_baseline.log" 2>&1 &
  echo $! >"$ROOT/logs/openearthmap_vmamba_baseline.pid"
fi

ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "
  set -euo pipefail
  ROOT='$ROOT'
  PY=\"\$ROOT/envs/ig-ds-ssm-env/bin/python\"
  WORK=\"\$ROOT/work_train_single\"
  MANIFEST=\"\$ROOT/data/openearthmap_sar/sar_only_p/manifest.json\"
  CKPT=\"\$ROOT/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth\"
  EXPERIMENT='$EXPERIMENT'
  test -x \"\$PY\"
  test -f \"\$MANIFEST\"
  test -f \"\$CKPT\"
  mkdir -p \"\$ROOT/experiments/\$EXPERIMENT\" \"\$ROOT/logs\"
  if ! pgrep -af \"run_shared_sar_tasks_v2.py.*\$EXPERIMENT.*--variant ours\" >/dev/null; then
    nohup \"\$PY\" -u \"\$WORK/tools/run_shared_sar_tasks_v2.py\" \\
      --manifest \"\$MANIFEST\" \\
      --dataset OpenEarthMap-SAR \\
      --variant ours \\
      --backbone-layout vmamba_tiny \\
      --decoder upernet_lite \\
      --pretrained \"\$CKPT\" \\
      --output \"\$ROOT/experiments/\$EXPERIMENT/ours\" \\
      --epochs 150 \\
      --patience 30 \\
      --batch 8 \\
      --imgsz 512 \\
      --eval-batch 1 \\
      --eval-imgsz 1024 \\
      --workers 8 \\
      --optimizer Adam \\
      --lr 0.0001 \\
      --weight-decay 0.0 \\
      --scheduler fixed \\
      --label-smoothing 0.0 \\
      --segmentation-loss weighted_cross_entropy_plus_dice \\
      --selection-metric miou_all \\
      --state-dim 16 \\
      --seed 0 \\
      --device cuda:0 \\
      >\"\$ROOT/logs/openearthmap_vmamba_ours.log\" 2>&1 &
    echo \$! >\"\$ROOT/logs/openearthmap_vmamba_ours.pid\"
  fi
"

echo "VMamba-T baseline dispatched on A6000-B"
echo "VMamba-T + IG-DS-SSM/PFIM P3 dispatched on A6000-A"
