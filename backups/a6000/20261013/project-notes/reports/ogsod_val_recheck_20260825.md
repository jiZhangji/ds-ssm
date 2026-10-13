# OGSOD-1.0 validation recheck (2026-08-25)

## Protocol

- Dataset: OGSOD-1.0 SAR-only HBB, 3 classes
- Split: validation (1,834 images, 4,840 instances)
- Checkpoint: validation-selected `best.pt` from each archived run
- Evaluation: 1024 pixels, FP32, confidence 0.001, NMS IoU 0.7
- Training split sizes: 14,664 train / 1,834 validation / 1,833 test

## Validation results

| Model | Precision | Recall | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| Mamba-YOLO-T | 0.805017 | 0.715730 | 0.785269 | 0.393904 | 0.423449 |
| DS-SSM/PFIM-T | 0.801210 | 0.714372 | 0.779901 | 0.412115 | 0.435427 |
| Delta | -0.003807 | -0.001358 | -0.005369 | +0.018211 | +0.011978 |

## Previously archived test results

| Model | Precision | Recall | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| Mamba-YOLO-T | 0.801434 | 0.711545 | 0.771733 | 0.402109 | 0.423018 |
| DS-SSM/PFIM-T | 0.797140 | 0.716924 | 0.772630 | 0.409711 | 0.436021 |
| Delta | -0.004294 | +0.005379 | +0.000897 | +0.007602 | +0.013003 |

The earlier paper-table result is from the held-out test split, not validation. The
fresh validation evaluation independently reproduces the training record and shows
the same main trend: DS-SSM/PFIM improves stricter localization metrics (AP75 and
AP50:95), while AP50 does not consistently improve.

## Recovered artifacts on gpu3090-1

Root:

`/mnt/disk1/zhangji/ogsod_recovered_backup/ds-ssm/source_code/work_train_single`

Runs:

- `output_dir/ogsod/ogsod_mambayolo_t_1024_2xh200_bs64_adamw_20260724_082717`
- `output_dir/ogsod/ogsod_dsssm_pfim_t_1024_2xh200_bs64_adamw_20260724_082717`

Each run contains `weights/best.pt`, `weights/last.pt`, `results.csv`, and
`args.yaml`. The `logs` directory contains the original and resumed training logs
for both models.

Fresh evaluation JSON files:

- `/mnt/disk1/zhangji/ogsod_recovered_backup/eval_baseline_val_iou07.json`
- `/mnt/disk1/zhangji/ogsod_recovered_backup/eval_pfim_t_val_iou07.json`
