# Current Results vs External Methods

Updated: 2026-08-30

All gaps are numerical differences in percentage points. Author-reported results may use different splits, preprocessing, initialization, image sizes, HBB/OBB definitions, or evaluators. They are background comparisons, not strict rerankings.

## Detection

| Dataset | Current ours | External reference | Reference result | Gap | Assessment |
|---|---:|---|---:|---:|---|
| SSDD | AP50:95 72.261 | YOLOv9-tiny | 72.2 | +0.061 | essentially tied |
| SSDD | AP50:95 72.261 | YOLOv8n | 74.0 | -1.739 | close |
| SSDD | AP50:95 72.261 | YOLOv8s | 75.4 | -3.139 | behind |
| SSDD | AP50:95 72.261 | Dynamic multi-scale | 76.8 | -4.539 | behind strongest listed method |
| SAR-AIRcraft VAL, historical valid run | AP50 88.825 | DiffDet4SAR | 88.4 | +0.425 | slightly ahead |
| SAR-AIRcraft VAL, historical valid run | AP50 88.825 | SFS-CNet | 89.7 | -0.875 | close |
| SAR-AIRcraft VAL, historical valid run | AP50 88.825 | MaDiNet | 90.5 | -1.675 | close |
| SAR-AIRcraft VAL, historical valid run | AP50 88.825 | PGD-YOLOv8 | 90.7 | -1.875 | close |
| SADD | AP50:95 72.113 | YOLO-SAD | 57.1 | +15.013 | numerically ahead; protocol must be audited |
| SADD | AP50:95 72.113 | SARDet-MIM | 55.3 | +16.813 | numerically ahead; pretraining/protocol differs |
| AIR-SARShip-2.0 | AP50:95 19.291 | ImageNet-SSD | 46.49 | -27.199 | far behind |
| AIR-SARShip-2.0 | AP50:95 19.291 | WBF-DM | 54.78 | -35.489 | far behind strongest listed method |
| SRSDD-v1.0 HBB | AP50:95 22.147 | YOLOv5s | 42.4 | -20.253 | far behind |
| SRSDD-v1.0 HBB | AP50:95 22.147 | GDB-YOLOv5s | 47.6 | -25.453 | far behind strongest HBB reference |
| OGSOD SAR-only | AP50:95 43.602 | ATSS | 42.7 | +0.902 | slightly ahead |
| OGSOD SAR-only | AP50:95 43.602 | YOLOv8n | 49.3 | -5.698 | behind lightweight reference |
| OGSOD SAR-only | AP50:95 43.602 | YOLOv10n | 49.8 | -6.198 | behind strongest listed lightweight reference |
| SAR-ShipDataset, historical | AP50 94.844 | YOLOv5s | 94.8 | +0.044 | tied |
| SAR-ShipDataset, historical | AP50 94.844 | LRTransDet / DiffDet4SAR | 95.1 | -0.256 | close |
| SAR-ShipDataset, historical | AP50 94.844 | MaDiNet | 97.5 | -2.656 | behind |
| SAR-ShipDataset, historical | AP50:95 57.818 | SRDet | 65.9 | -8.082 | behind strict localization result |

## Classification

| Dataset | Current ours | External reference | Reference result | Gap | Assessment |
|---|---:|---|---:|---:|---|
| SAR-ACD | Accuracy 89.370 | MDB-DPL | Accuracy 98.85 | -9.480 | clearly behind; protocol also differs |
| SAR-ACD | Accuracy 89.370 | Other listed methods | mostly mAP, not Accuracy | not comparable | do not mix metrics |
| SAMPLE | Accuracy 97.037 | No directly matched public protocol in current evidence set | - | - | compare only after protocol-specific literature audit |
| MSTAR-SOC | no current ours result | A-ConvNet about 99.13 Accuracy | - | - | experiment missing |

## Segmentation

The only completed local pair uses the old five-class protocol and is not publication-aligned. Current corrected FUSAR runs are incomplete.

| Dataset | Current completed ours | External method | External mIoU | Gap | Assessment |
|---|---:|---|---:|---:|---|
| FUSAR-Map | old-protocol mIoU 21.489 | SegNet | 38.52 | -17.031 | not competitive |
| FUSAR-Map | old-protocol mIoU 21.489 | U-Net | 43.43 | -21.941 | not competitive |
| FUSAR-Map | old-protocol mIoU 21.489 | DeepLabv3+ | 47.62 | -26.131 | not competitive |
| FUSAR-Map | old-protocol mIoU 21.489 | MASA-SegNet | 50.70 | -29.211 | not competitive |
| FUSAR-Map | old-protocol mIoU 21.489 | FUSAR-Map Net | 52.31 | -30.821 | not competitive |

These gaps must not be used as final paper claims because the old local evaluator included black unknown pixels as a fifth class, unlike the four-class literature protocol.

## Change Detection

| Dataset | Current ours F1 | Representative published range | Best listed F1 | Gap to best | Assessment |
|---|---:|---:|---:|---:|---|
| Ottawa | 92.173 | 93.9-94.9 | 94.9, SARCDNet | -2.727 | moderately close |
| Yellow River | 86.308 | 85.2-90.0 | 90.0, SARCDNet | -3.692 | middle of published range |
| Farmland | 60.805 | 86.7-89.5 | 89.5, SARCDNet | -28.695 | far behind |
| Sulzberger | 66.180 | 95.7-96.9 | 96.9, SARCDNet | -30.720 | far behind |

## Overall Assessment

1. SSDD and SAR-AIRcraft are near the competitive range but not the strongest listed results.
2. SADD is numerically much higher than the listed literature, but its small split and different training/initialization protocol must be audited before claiming superiority.
3. AIR-SARShip and SRSDD are currently the largest detection weaknesses and likely require data/protocol correction rather than cosmetic tuning.
4. Classification results are below the strongest published result; SAMPLE lacks a directly matched literature protocol.
5. The completed FUSAR result is not comparable to the four-class literature and is far below it numerically.
6. Change detection is mixed: Ottawa and Yellow River are relatively close, while Farmland and Sulzberger are far behind published methods.
7. The evidence currently supports a lightweight SAR object-detection contribution, not a universal cross-task SOTA claim.

