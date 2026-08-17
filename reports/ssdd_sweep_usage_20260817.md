# SSDD Placement and Hyperparameter Sweep

## What the sweep does

The sweep runner:

1. Generates explicit Mamba-YOLO-T YAML files for multiple DS-SSM/PFIM placements.
2. Keeps `state_dim` explicit through `DS_VSSBlockSweep` instead of letting stage width change it silently.
3. Trains every selected experiment sequentially.
4. Resumes incomplete runs from `weights/last.pt`.
5. Evaluates validation-selected `weights/best.pt` on TEST in FP32.
6. Runs both standard inference and TTA unless `--no-tta` is supplied.
7. Writes JSON evidence, per-class CSV files, and a ranked CSV/Markdown summary.

Main files:

- `tools/run_ssdd_sweep.py`
- `tools/evaluate_ssdd_sweep.py`
- `ultralytics/nn/modules/mamba_yolo.py` (`DS_VSSBlockSweep`)

## Presets

| Preset | Training runs per seed | Contents |
|---|---:|---|
| `core` | 16 | Six primary placements, eight optimizer/batch/augmentation variants, and P2 state dimensions 8/16/32 |
| `placement` | 10 | Baseline plus P2/P3/P4/P5 and multi-stage placement combinations |
| `optimizer` | 11 | Current P2 placement with SGD/AdamW, LR, weight decay, batch, mosaic, and warmup variants |
| `extended` | 43 | Broad placement-recipe sweep; requires `--allow-large-sweep` |
| `custom` | Variable | User-selected placements, recipes, state dimensions, and seeds |

The default `core` preset includes:

- baseline
- P2, P3, P4, P5, P2+P3 with reference SGD
- P2 with SGD batch 64
- P2 with AdamW LR `1e-4`, `2e-4`, and `3e-4`
- P2 with AdamW batch 64
- P2 with lower weight decay
- P2 without mosaic
- P2 with 20-epoch module warmup
- P2 with state dimensions 8, 16, and 32

`P2 state_dim=32` matches the parsed parameter count of the historical ex7 model. The fixed-16 variants are intended for fairer cross-stage placement screening.

## Prepare the server dataset

The server SSDD release uses COCO JSON annotations and has an official
928-image train set plus a 232-image test set, but no validation split. Prepare
a non-destructive YOLO copy by splitting only the official train set and
keeping the official test set untouched:

```bash
python tools/prepare_ssdd_coco_threeway.py \
  --root /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/SSDD/SSDD \
  --out /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/det/SSDD \
  --val-ratio 0.2 \
  --seed 42 \
  --link-mode hardlink
```

This creates deterministic train/val/test splits, a `split_manifest.json`, and
separate `SSDD_inshore.yaml` and `SSDD_offshore.yaml` evaluation configs when
the corresponding official subsets are present. The source dataset is not
modified. The default `SSDD_HF.yaml` points to this prepared output.

## Server dry run

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sar-pretrain

cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/ds-ssm/source_code/work_train_single

python tools/run_ssdd_sweep.py \
  --preset core \
  --sweep-id ssdd_core_sweep_v1 \
  --devices 0 \
  --dry-run
```

Review the generated run list before starting. The SSDD default data config is `ultralytics/cfg/datasets/SSDD_HF.yaml`.

## Recommended first run

Run the ten placement models first. This isolates location before mixing optimizer changes:

```bash
mkdir -p logs

nohup python -u tools/run_ssdd_sweep.py \
  --preset placement \
  --sweep-id ssdd_placement_sgd_3seed_v1 \
  --devices 0 \
  --seeds 0 1 2 \
  --imgsz 640 \
  --epochs 300 \
  --tta \
  > logs/ssdd_placement_sgd_3seed_v1.queue.log 2>&1 &

echo $!
```

This is 30 training runs. For exploratory screening, omit `1 2` and run only seed 0 first.

## Core mixed sweep

```bash
nohup python -u tools/run_ssdd_sweep.py \
  --preset core \
  --sweep-id ssdd_core_sweep_v1 \
  --devices 0 \
  --seeds 0 \
  --imgsz 640 \
  --epochs 300 \
  --tta \
  > logs/ssdd_core_sweep_v1.queue.log 2>&1 &
```

The command can be executed again with the same arguments. Completed runs and current evaluation JSON files are skipped; incomplete training resumes from `last.pt`.

## Custom sweep

```bash
python tools/run_ssdd_sweep.py \
  --preset custom \
  --sweep-id ssdd_custom_v1 \
  --devices 0 \
  --placements p2 p3 p2p3 \
  --recipes sgd_ref adamw_1e4 adamw_2e4 \
  --state-dims 16 32 \
  --seeds 0 \
  --tta
```

Available recipe names:

- `sgd_ref`
- `sgd_cos`
- `sgd_bs64`
- `adamw_1e4`
- `adamw_2e4`
- `adamw_bs64`
- `adamw_3e4`
- `adamw_lowwd`
- `adamw_nomosaic`
- `adamw_mildaug`
- `adamw_warmup20`

Available placements:

- `baseline`
- `p2`
- `p3`
- `p4`
- `p5`
- `p2p3`
- `p2p4`
- `p3p4`
- `p2p3p4`
- `p2p3p4p5`

## Monitoring

```bash
tail -f logs/ssdd_core_sweep_v1.queue.log
```

Individual training and evaluation logs are saved below:

```text
logs/<sweep-id>/<run-name>.train.log
logs/<sweep-id>/<run-name>.eval.log
```

Check active jobs and GPU status:

```bash
pgrep -af '[r]un_ssdd_sweep.py|[m]byolo_train_sarair.py'
nvidia-smi
```

## Results

Default output root:

```text
output_dir/ssdd_sweep/
```

Important files:

```text
output_dir/ssdd_sweep/<run-name>/weights/best.pt
output_dir/ssdd_sweep/<run-name>/eval_standard.json
output_dir/ssdd_sweep/<run-name>/eval_tta.json
output_dir/ssdd_sweep/<sweep-id>_summary.csv
output_dir/ssdd_sweep/<sweep-id>_summary.md
output_dir/ssdd_sweep/<sweep-id>-manifest/sweep_manifest.json
```

The Markdown summary contains the standard/TTA ranking and the TTA delta relative to standard inference.

## Publication rule

Use validation metrics to select the placement and recipe. Do not repeatedly use TEST to choose among all sweep variants. For publication-quality reporting, rerun the selected baseline/current-P2/winner configurations with seeds 0/1/2 under a new frozen sweep ID, then evaluate their validation-selected `best.pt` once on TEST.
