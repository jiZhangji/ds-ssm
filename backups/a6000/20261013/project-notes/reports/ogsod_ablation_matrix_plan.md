# OGSOD Controlled Ablation Matrix

## Purpose

This matrix resolves the paper's two largest evidence gaps under one frozen protocol:

1. Three-seed `Mamba-YOLO-T -> DS-SSM without PFIM -> spatially unstructured PFIM -> full PFIM` comparison.
2. Single-seed PFIM gamma/write/delta and local-statistic diagnostics.

All runs use the recovered OGSOD local split, 1024 input, 300 epochs, global batch 64, two GPUs, FP32 AdamW, `lr0=2e-4`, cosine scheduling, and the same augmentation settings used by the completed OGSOD T runs.

## Added Model Entrypoints

| Variant | YAML module | Interpretation |
|---|---|---|
| baseline | `VSSBlock` | Original Mamba-YOLO-T |
| no_pfim | `DS_VSSBlockNoPFIM` | Residual suppression and energy damping without PFIM |
| gamma | `DS_VSSBlockPFIMGamma` | PFIM affects the energy-conditioned drive factor only |
| write | `DS_VSSBlockPFIMWrite` | PFIM affects the importance-centered drive factor only |
| delta | `DS_VSSBlockPFIMDelta` | PFIM affects transition step only |
| gamma_write | `DS_VSSBlockPFIMGammaWrite` | PFIM affects both factors on the scan drive |
| global | `DS_VSSBlockPFIMGlobal` | Parameter-matched PFIM with one image-level importance value |
| no_2dstd | `DS_VSSBlockPFIMNo2DStd` | Parameter-matched PFIM using 1-D, not 2-D, local variation |
| full | `DS_VSSBlock` | Both drive factors and transition adaptation active |

Disabled branch scalars are frozen so two-GPU DDP does not encounter unused trainable parameters. The `global` and `no_2dstd` controls have exactly the same parameters and trainable-parameter count as `full`. `global` removes token-wise spatial alignment by averaging importance over the sequence; `no_2dstd` retains token-wise importance but replaces two-dimensional local variation with its sequence-only counterpart. The existing full `DS_VSSBlock` computation is unchanged. In the CUDA path, gamma and write are two factors on the same scan drive; branch runs diagnose this factorization rather than independent recurrent operator locations.

## Preflight

Upload the revised repository to the server, activate `sar-pretrain`, and run:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/ds-ssm/source_code/work_train_single

DRY_RUN=1 \
MATRIX_ID="ogsod_t_primary_3seed_v2" \
SEEDS="0 1 2" \
VARIANTS="baseline no_pfim global full" \
bash tools/run_ogsod_ablation_matrix.sh
```

The dry run validates paths, parses every requested YAML, verifies parameter alignment for any requested `full/global/no_2dstd` controls, runs a CPU fallback semantics check, and prints every train and TEST command without starting training. It omits the full image/label fingerprint. Formal controlled training requires `DATA_FINGERPRINT=1`, because it hashes every image byte stream and every label content as part of the evidence contract.

## Primary Three-Seed Matrix

```bash
mkdir -p logs

MATRIX_ID="ogsod_t_primary_3seed_v2" \
DATA_FINGERPRINT=1 \
SEEDS="0 1 2" \
VARIANTS="baseline no_pfim global full" \
nohup bash tools/run_ogsod_ablation_matrix.sh \
  > logs/ogsod_t_primary_3seed_v2.queue.log 2>&1 &

echo "QUEUE_PID=$!"
tail -f logs/ogsod_t_primary_3seed_v2.queue.log
```

This runs twelve experiments sequentially. Stable names allow the same command to skip completed runs and resume incomplete runs from `last.pt`. Each completed run is automatically evaluated on TEST from its validation-selected `best.pt` in FP32, and the script updates raw-run, mean/std, and seed-paired-delta summaries.

## Diagnostic Branch Matrix

Run only after the primary matrix:

```bash
MATRIX_ID="ogsod_t_branches_seed0_v2" \
DATA_FINGERPRINT=1 \
SEEDS="0" \
VARIANTS="gamma write delta gamma_write no_2dstd" \
nohup bash tools/run_ogsod_ablation_matrix.sh \
  > logs/ogsod_t_branches_seed0_v2.queue.log 2>&1 &

echo "QUEUE_PID=$!"
tail -f logs/ogsod_t_branches_seed0_v2.queue.log
```

## Evidence Rules

- Do not select the best seed for the paper. Report mean and standard deviation across seeds 0/1/2.
- Evaluate every run's `best.pt` on the same TEST split in FP32 at 1024.
- Preserve `run_manifest.txt`, `protocol.txt`, `sha256.txt`, `dataset_snapshot.json`, `git_status.txt`, and `gpus.csv` with the result archive.
- The queue refuses to reuse a `MATRIX_ID` if its frozen training protocol, code/config hash, or full data fingerprint changes. Start a new matrix ID rather than silently mixing states.

## Density, Scale, and Spacing Analysis

After seed 0 of the four primary variants is complete, run the fixed-checkpoint diagnostic below. It partitions TEST by per-image instance count, median normalized box area, and median nearest-neighbor center distance, then reports overall and per-class AP50/AP75/AP50:95 for each stratum.

```bash
PROJECT="$PWD/output_dir/ogsod_ablation"
MATRIX_ID="ogsod_t_primary_3seed_v2"
DATA_YAML="${DATA_YAML:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/det/OGSOD/OGSOD.yaml}"

python tools/evaluate_ogsod_strata.py \
  --data "$DATA_YAML" \
  --model "baseline=$PROJECT/${MATRIX_ID}_baseline_seed0/weights/best.pt" \
  --model "no_pfim=$PROJECT/${MATRIX_ID}_no_pfim_seed0/weights/best.pt" \
  --model "global=$PROJECT/${MATRIX_ID}_global_seed0/weights/best.pt" \
  --model "full=$PROJECT/${MATRIX_ID}_full_seed0/weights/best.pt" \
  --split test \
  --criteria density area nearest \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --output "$PROJECT/${MATRIX_ID}_strata_seed0"
```

This is a post-hoc diagnostic of frozen checkpoints, not a model-selection procedure. Keep the generated subset lists and `strata_manifest.json` with the reported tables.

Generate the fixed-threshold per-image audit before selecting any qualitative example. It matches class-aware predictions to ground truth at IoU 0.75 after the stated confidence/NMS settings and writes the highest improvement and regression candidates for every class:

```bash
python tools/compare_ogsod_per_image.py \
  --data "$DATA_YAML" \
  --baseline "$PROJECT/${MATRIX_ID}_baseline_seed0/weights/best.pt" \
  --full "$PROJECT/${MATRIX_ID}_full_seed0/weights/best.pt" \
  --split test \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --conf 0.25 \
  --nms-iou 0.7 \
  --output "$PROJECT/${MATRIX_ID}_per_image_seed0"
```

Then pass candidates from `candidate_examples.md` to the full seed-0 checkpoint:

```bash
python tools/visualize_pfim_importance.py \
  --weights "$PROJECT/${MATRIX_ID}_full_seed0/weights/best.pt" \
  --images \
    /absolute/path/to/bridge_success.png \
    /absolute/path/to/harbor_success.png \
    /absolute/path/to/tank_failure_1.png \
    /absolute/path/to/tank_failure_2.png \
  --imgsz 1024 \
  --device 0 \
  --output "$PROJECT/${MATRIX_ID}_importance_examples"
```

The tool saves the raw PFIM array and a four-panel GT/prediction/importance/overlay image. The candidate audit is not an AP estimate; it is a deterministic selection rule that prevents visually searching only for favorable cases.
