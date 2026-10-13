# Priority Optimization Plan

Updated: 2026-08-31 02:29 Asia/Shanghai

Published results below are context only unless the split, preprocessing,
initialization, model size, and evaluator are matched.

## Current gap assessment

| Priority | Dataset | Current result | External reference | Numerical gap | Main diagnosis |
|---|---|---:|---:|---:|---|
| P0 | AIR-SARShip-2.0 | old AP50:95 19.291%; paper-aligned baseline 22.947% | 46.49%-54.78% | -23.54 to -31.83 pp using the new baseline | Protocol is now aligned, but the baseline remains weak; wait for the matched P3 result before tuning. |
| P0 | SRSDD-v1.0 | old HBB AP50:95 22.147% | 42.4%-47.6% | -20.25 to -25.45 pp | Conversion, class mapping, polygon-to-HBB clipping, and official split must be audited before another long run. |
| P0 | FUSAR-Map | old incompatible mIoU 21.489% | 38.52%-52.31% | invalid direct comparison | Four-class paper-aligned baseline/P3 is running; do not tune against the old five-class result. |
| P1 | Farmland | P3 F1 60.805% | best listed 89.5% | -28.70 pp | Single-image protocol and macroblock split differ from literature; the local P3 gain over baseline is already +34.76 pp. |
| P1 | Sulzberger | P3 F1 66.180% | best listed 96.9% | -30.72 pp | P3 has excessive recall and low precision; test lower-LR stability before changing the loss or split. |
| P1 | SAR-ACD | P3 accuracy 89.370% | MDB-DPL 98.85% | -9.48 pp | P3 is 5.51 pp below the matched local baseline; small-data over-regularization is a direct hypothesis. |
| P2 | SAMPLE | P3 accuracy 97.037% | no matched public protocol | n/a | P3 is 1.48 pp below the matched local baseline; use the same low-LR/no-smoothing diagnostic as SAR-ACD. |
| P2 | Yellow River | P3 F1 86.308% | best listed 90.0% | -3.69 pp | Close enough for a bounded optimization diagnostic; avoid broad sweeps. |
| P2 | Ottawa | P3 F1 92.173% | best listed 94.9% | -2.73 pp | Competitive range; no immediate rerun. |
| P2 | SSDD | P3 AP50:95 72.261% | YOLOv8n 74.0%; strongest listed 76.8% | -1.74 to -4.54 pp | Already near lightweight references; matched baseline and final protocol matter more than more tuning. |

## Experiments launched on A6000-A

All jobs use P3, `state_dim=16`, seed 0, scratch initialization, the existing
fixed split, and independent output directories. They run sequentially so one
GPU cannot oversubscribe memory.

| Order | Dataset | Change from current P3 | Purpose |
|---:|---|---|---|
| 1 | SAR-ACD | LR `2e-4 -> 1e-4`, label smoothing `0.1 -> 0` | Test small-data over-regularization. |
| 2 | SAMPLE | LR `2e-4 -> 1e-4`, label smoothing `0.1 -> 0` | Test whether the same correction restores the small negative delta. |
| 3 | Sulzberger | LR `2e-4 -> 1e-4` | Reduce unstable early over-prediction while preserving loss and protocol. |
| 4 | Yellow River | LR `2e-4 -> 1e-4` | Recover recall without introducing a broad hyperparameter sweep. |

Remote project:

`/home/powerleader/ig-ds-ssm/experiments/priority_optimization_20260831`

Queue script:

`/home/powerleader/ig-ds-ssm/work_train_single/tools/run_priority_optimization_queue.sh`

## Formal runs left untouched

| Host | Experiment | Status at launch |
|---|---|---|
| A6000-B | FUSAR-Map paper-aligned baseline then P3 | Running |
| A6000-C | MSTAR-SOC baseline then P3 | Running |
| A6000-D | AIR-SARShip-2.0 paper-aligned P3 | Epoch 206/240; baseline AP50:95 22.947% |

## Next decision gates

1. Compare AIR baseline/P3 only after P3 reaches epoch 240 and final test evaluation completes.
2. Do not launch SRSDD training until both official archives are present and the HBB conversion audit passes.
3. If low LR/no smoothing improves SAR-ACD or SAMPLE, run the matched baseline under the same optimized recipe before using the result in a paper table.
4. If Sulzberger remains precision-limited, the next experiment should expose the positive-class weight and BCE/Dice mixture as explicit controlled parameters rather than adding more epochs.
