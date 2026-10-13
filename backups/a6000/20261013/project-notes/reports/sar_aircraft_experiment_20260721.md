# SAR_AIRcraft Current Experiment Record

Updated: 2026-07-23

## Experiment Identity

| Field | Value |
|---|---|
| Run | `sarair_dsssm_pfim_t_1024_2xh100_bs48_adamw_20260721_201418` |
| Dataset | SAR_AIRcraft 512x512 overlapping-patch derivative |
| Dataset YAML | `/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/dataset/det/SAR_AIRcraft/SAR_AIRcraft.yaml` |
| Model | `Mamba-YOLO-T-DS-SSM_Backbone-SAR_AIRcraft-ex7.yaml` |
| Method | Mamba-YOLO-T + DS-SSM + PFIM |
| Weights | `output_dir/sar_aircraft/sarair_dsssm_pfim_t_1024_2xh100_bs48_adamw_20260721_201418/weights/best.pt` |
| Server hardware | 2 x NVIDIA H100 80GB |
| Parameters | 5,964,583 |
| Complexity | 13.2 GFLOPs |

## Training Configuration

- Epochs: 300
- Image size: 1024
- Global batch size: 48 (24 images per GPU)
- Precision: FP32 (`amp=False`)
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Final learning-rate ratio: 0.01
- Weight decay: 0.0005
- Scheduler: cosine
- Module-only warmup: disabled
- External pretrained weights: none
- Mosaic: 0.6; close mosaic: final 20 epochs
- Rotation: 10 degrees
- Translation: 0.1
- Scale: 0.4
- Vertical flip: 0.5
- Horizontal flip: 0.5

## Dataset Protocol

The converted split contains 13,976 train patches, 1,923 validation patches, and 2,989 test patches. The splits have no filename overlap. They correspond to approximately 3,047, 442, and 879 original-image IDs respectively.

This is not the original-image protocol described by MaDiNet. The current data are 512x512 overlapping patches derived from 4,368 original SAR_AIRcraft images. Consequently, numerical comparison with MaDiNet's reported 90.5 AP50 is contextual only, not a protocol-aligned SOTA comparison.

## Overall Metrics

| Split | Images | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1,923 | 4,631 | 0.909612 | 0.796257 | 0.849169 | 0.888252 | 0.705129 |
| Test | 2,989 | 5,996 | 0.792764 | 0.637326 | 0.706597 | 0.708220 | 0.555516 |

## Per-Class Validation Metrics

| Class | Images | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boeing737 | 455 | 610 | 0.892604 | 0.636066 | 0.742809 | 0.781014 | 0.663288 |
| Boeing787 | 442 | 613 | 0.905086 | 0.827080 | 0.864327 | 0.904806 | 0.752538 |
| A330 | 49 | 49 | 0.930653 | 0.821732 | 0.872807 | 0.971584 | 0.802131 |
| A220 | 495 | 566 | 0.865044 | 0.787986 | 0.824719 | 0.835269 | 0.612766 |
| A320/321 | 197 | 197 | 0.958883 | 0.947057 | 0.952933 | 0.988195 | 0.799291 |
| ARJ21 | 624 | 920 | 0.941639 | 0.771660 | 0.848218 | 0.869621 | 0.643623 |
| other | 952 | 1,676 | 0.873378 | 0.782220 | 0.825289 | 0.867272 | 0.662264 |

## Per-Class Test Metrics

| Class | Images | Instances | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boeing737 | 916 | 1,083 | 0.859769 | 0.226448 | 0.358479 | 0.395605 | 0.296472 |
| Boeing787 | 703 | 849 | 0.626541 | 0.661955 | 0.643762 | 0.589946 | 0.473767 |
| A330 | 76 | 76 | 0.946102 | 0.923903 | 0.934871 | 0.983017 | 0.855120 |
| A220 | 856 | 968 | 0.835544 | 0.505165 | 0.629649 | 0.672724 | 0.510614 |
| A320/321 | 80 | 80 | 0.730667 | 0.949513 | 0.825838 | 0.938870 | 0.761258 |
| ARJ21 | 629 | 778 | 0.857575 | 0.634629 | 0.729447 | 0.747603 | 0.519307 |
| other | 1,309 | 2,162 | 0.693147 | 0.559667 | 0.619296 | 0.629778 | 0.472071 |

## Baseline Context

The available new-server Mamba-YOLO-T reference was evaluated at image size 1024 using its best checkpoint:

| Split | Method | Precision | Recall | mAP50 | mAP50-95 | Delta mAP50 | Delta mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | Mamba-YOLO-T reference | 0.836599 | 0.833884 | 0.877509 | 0.699035 | - | - |
| Validation | DS-SSM + PFIM | 0.909612 | 0.796257 | 0.888252 | 0.705129 | +0.010743 | +0.006094 |
| Test | Mamba-YOLO-T reference | 0.740022 | 0.660376 | 0.704018 | 0.545131 | - | - |
| Test | DS-SSM + PFIM | 0.792764 | 0.637326 | 0.708220 | 0.555516 | +0.004202 | +0.010385 |

This comparison is preliminary rather than publication-grade. The reference run had reached about 137 epochs when inspected and used SGD, batch 16, and a different augmentation/optimization setup. The DS-SSM + PFIM run completed 300 epochs with AdamW and global batch 48. A controlled architecture ablation must use identical initialization, optimizer, batch size, augmentation, seed, and training duration.

## Analysis

1. DS-SSM + PFIM is precision-oriented in this run. Compared with the available reference, validation precision increases by 7.30 percentage points while recall decreases by 3.76 points. On test, precision increases by 5.27 points and recall decreases by 2.31 points.
2. The preliminary AP gains are positive on both splits. Test mAP50-95 improves by 1.04 percentage points, larger than the 0.42-point mAP50 gain, suggesting better localization at stricter IoU thresholds.
3. The validation-to-test gap remains large: 18.00 points in mAP50 and 14.96 points in mAP50-95. This is primarily a split/domain issue rather than a global numerical failure.
4. Boeing737 is the dominant failure mode. Its recall falls from 0.636 on validation to 0.226 on test, and mAP50 falls from 0.781 to 0.396. Boeing787 and `other` also show substantial test degradation.
5. A330 and A320/321 have strong AP, but their test sets contain only 76 and 80 instances. These values should be reported with a small-sample caveat.
6. The completed model is lightweight: 5.965M parameters and 13.2 GFLOPs. The evidence supports a lightweight cross-dataset result, but the optimization-confounded baseline does not yet isolate the DS-SSM/PFIM contribution.

## Required Follow-Up

1. Train a 300-epoch Mamba-YOLO-T baseline with exactly the same AdamW, global batch 48, image size 1024, augmentation, seed, FP32 setting, and split.
2. Repeat the matched baseline and method with at least three seeds if SAR_AIRcraft is used in the main paper table.
3. Audit Boeing737 train/validation/test acquisition conditions and annotation characteristics before changing the architecture.
4. Keep MaDiNet comparison labeled as non-aligned until the original-image evaluation protocol is reconstructed.

