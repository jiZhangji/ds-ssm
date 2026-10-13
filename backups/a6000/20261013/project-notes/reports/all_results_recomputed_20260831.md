# IG-DS-SSM/PFIM Recomputed Results

Updated: 2026-08-31 14:48 Asia/Shanghai

This document separates local matched comparisons, historical evidence,
paper-aligned runs, and author-reported external references. Values from
different splits, initialization, HBB/OBB formats, or evaluators are not a
strict leaderboard.

## Detection

| Dataset/protocol | External references | Local baseline | Local P3/ours | Delta/status |
|---|---|---|---|---|
| SSDD official test | Faster R-CNN 71.3; YOLOv9-tiny 72.2; YOLOv10n 72.8; YOLOv8n 74.0; YOLO-SRBD 74.5; CCAI-YOLO 74.9; ARGO-Net 79.3 AP50:95 | Mamba-YOLO-T AP50/AP75/AP50:95 98.605/83.613/70.348 | P3-d16 98.549/85.552/72.261 | +1.913 pp AP50:95 versus historical baseline; protocol not fully matched |
| SAR-AIRcraft patch VAL | DiffDet4SAR AP50 88.4; SFS-CNet 89.7; MaDiNet 90.5; PGD-YOLOv8 90.7 | 87.751 AP50; 69.904 AP50:95 | 88.825 AP50; 70.513 AP50:95 | +0.609 pp AP50:95, historical matched pair |
| SAR-AIRcraft patch TEST | External references predominantly use VAL/original images | 70.402 AP50; 54.513 AP50:95 | 70.822 AP50; 55.552 AP50:95 | +1.039 pp AP50:95, historical matched pair |
| SADD scratch 200e | YOLO-SAD 57.1; SARDet-MIM 55.3 AP50:95 | 98.335/57.030/55.018 AP50/AP75/AP50:95 | 97.760/56.985/54.597 | -0.421 pp AP50:95 |
| SADD partial-transfer 300e | Same external context, different protocol | Missing matched 300e baseline | 98.297/82.999/72.113 | Strong standalone result; no causal delta |
| SAR-ShipDataset | LRTransDet AP50 95.1; SRDet AP50:95 65.9; MaDiNet AP50 97.5 | 94.616 AP50; 49.568 AP50:95 | 94.844 AP50; 57.818 AP50:95 | Historical, not a strict controlled pair |

## Classification

| Dataset | External references | Local baseline | Local P3/ours | Status |
|---|---|---|---|---|
| SAR-ACD | MDB-DPL Accuracy 98.85 under a different six-class protocol | Accuracy 94.882; Macro-F1 94.815 | Accuracy 89.370; Macro-F1 89.181 | -5.512 pp Accuracy |
| SAR-ACD optimization | Same split as current local pair | Baseline above | LR 1e-4/no smoothing P3: best VAL Accuracy 72.510, Macro-F1 72.024 | Running, epoch 35/160; not a test result |
| SAMPLE | RSPC AIA 93.30 is class-incremental, not closed-set Accuracy | Accuracy 98.519; Macro-F1 98.454 | Accuracy 97.037; Macro-F1 96.441 | -1.481 pp Accuracy |
| MSTAR-SOC | A-ConvNet Accuracy about 99.13 | Formal baseline running | P3 queued after baseline | Baseline 18/100; fixed final test only |

## Segmentation

| Dataset/protocol | External references | Local baseline | Local P3/ours | Status |
|---|---|---|---|---|
| FUSAR-Map old five-class | SegNet 38.52; U-Net 43.43; DeepLabv3+ 47.62; MASA-SegNet 50.70; FUSAR-Map Net 52.31 mIoU, all four-class | mIoU/mPA/Dice 21.084/28.223/27.308 | 21.489/28.637/27.491 | Incompatible with literature; diagnostic only |
| FUSAR-Map paper four-class | Same external references | Baseline running | P3 queued after baseline | No completed epoch/result yet |

## Change Detection

| Dataset | External F1 range/best | Local baseline P/R/F1/IoU/OA | Local P3 P/R/F1/IoU/OA | Delta F1 |
|---|---|---|---|---|
| Ottawa | 93.9-94.9; SARCDNet 94.9 | 90.990/94.597/92.758/86.495/98.491 | 89.238/95.308/92.173/85.483/98.346 | -0.585 pp |
| Yellow River | 85.2-90.0; SARCDNet 90.0 | 90.666/85.635/88.079/78.697/97.758 | 90.885/82.170/86.308/75.914/97.478 | -1.771 pp |
| Farmland | 86.7-89.5; SARCDNet 89.5 | 87.724/15.296/26.050/14.976/93.039 | 90.798/45.706/60.805/43.683/95.277 | +34.755 pp |

## Current conclusion

1. Positive local evidence exists on SSDD, SAR-AIRcraft, historical SAR-ShipDataset, and Farmland.
2. Matched P3 is neutral or negative on SADD, SAR-ACD, SAMPLE, Ottawa, and Yellow River.
3. FUSAR-Map still lacks a valid final matched pair.
4. Current evidence supports a lightweight SAR detection contribution more strongly than a universal cross-task backbone claim.
