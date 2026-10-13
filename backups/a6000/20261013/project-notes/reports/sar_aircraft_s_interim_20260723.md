# SAR-Aircraft PFIM-S Interim Record

Updated: 2026-07-24

## Experiment Identity

| Field | Value |
|---|---|
| Run | `sarair_dsssm_pfim_s_1024_2xh200_bs64_adamw_20260722_180047` |
| Dataset | SAR-Aircraft 512x512 overlapping-patch derivative |
| Model | Mamba-YOLO-S + DS-SSM + PFIM |
| Hardware | 2 x NVIDIA H200 |
| Input size | 1024 |
| Global batch | 64 |
| Optimizer | AdamW |
| Initial LR | 0.0002 |
| Final LR ratio | 0.01 |
| Weight decay | 0.0005 |
| Scheduler | Cosine |
| Training precision | FP32 (`amp=False`) |
| Module-only warmup | Disabled |
| Parameters | 21,741,555 |
| Complexity | 48.0 GFLOPs |

## Validation Precision Incident

The model itself did not collapse. The training validator forced every GPU validation pass to FP16 even when training used `amp=False`. The S-scale custom selective-scan model became numerically invalid in FP16, while its FP32 weights continued to improve.

The same recovered checkpoint produced:

| Checkpoint/evaluation | P | R | AP50 | AP50:95 |
|---|---:|---:|---:|---:|
| Old `best.pt`, FP32 | 0.841366 | 0.785869 | 0.831061 | 0.583770 |
| `epoch30.pt`, FP32 | 0.871043 | 0.810369 | 0.853149 | 0.629767 |
| Recovered epoch-136 `last.pt`, FP32 | 0.898546 | 0.823537 | 0.888438 | 0.705877 |
| Recovered epoch-136 `last.pt`, FP16 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Consequently, the repeated `0.1202/0.0842` AP50/AP50:95 values in the historical training CSV are invalid validation artifacts, not model performance. Rows before the FP32-validator resume must not be used to select a final checkpoint or draw a learning curve.

The validator now follows the trainer AMP setting. FP32 validation also uses the per-rank training batch rather than the original doubled validation batch to avoid OOM at 1024 input.

## Latest Verified FP32 Validation

The last user-reported state was epoch 234 of 300:

| Selection | Epoch | P | R | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| Latest | 234 | 0.919670 | 0.824410 | 0.890540 | 0.715840 |
| Best by AP50:95 | 232 | 0.920740 | 0.822640 | 0.890570 | 0.716230 |

This is an interim capacity result. It must not replace the completed PFIM-T main result until training reaches 300 epochs and the final `best.pt` is evaluated on both validation and test splits in FP32.

## Comparison with Completed PFIM-T

| Model | Params | GFLOPs | P | R | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| PFIM-T, completed validation | 5.9646M | 13.2 | 0.909612 | 0.796257 | 0.888252 | 0.705129 |
| PFIM-S, interim best validation | 21.7416M | 48.0 | 0.920740 | 0.822640 | 0.890570 | 0.716230 |

PFIM-S improves AP50:95 by 1.11 percentage points but uses about 3.64x the parameters and computation. PFIM-T remains the appropriate main model for a lightweight paper; PFIM-S is a scale/capacity study.

## Remaining Actions

1. Finish 300 epochs.
2. Evaluate the final FP32 `best.pt` on validation and test splits.
3. Export per-class validation and test metrics.
4. Preserve the final checkpoint epoch and corrected post-resume learning curve separately from invalid historical FP16 rows.

