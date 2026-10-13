# IG-DS-SSM / PFIM Current Experiment Data

Updated: 2026-08-30

This report separates formal local results, controlled diagnostics, historical archives, and incomplete runs. Values from different splits, initialization schemes, input sizes, or training protocols must not be ranked as one table.

## 1. Dataset inventory

| Task | Dataset | Local scale | Classes / labels | Current status |
|---|---|---:|---|---|
| Detection | SSDD | 928 train, 232 test | 1 ship class | Available and evaluated |
| Detection | SAR-AIRcraft-1.0 | 13,976 train, 1,923 val, 2,989 test | 7 aircraft classes | Available; new rerun failed |
| Detection | SAR-ShipDataset | 39,729 images | 1 ship class | Historical result available |
| Detection | SADD | 2,966 images | 1 aircraft class | Available and evaluated |
| Detection | AIR-SARShip-2.0 | 300 annotated images, 2,040 instances | 1 ship class | Converted matched result available |
| Detection | SRSDD-v1.0 | 532 train, 134 original test | 6 HBB-converted ship classes in current prepared set | P3 evaluated; matched baseline missing |
| Detection | OGSOD-1.0 | about 18,331 SAR images | bridge, harbor, tank | Historical controlled result available |
| Classification | SAR-ACD | 2,522 images | 5 aircraft classes | Corrected baseline/P3 result available |
| Classification | SAMPLE | 5,380 source images; current protocol uses 1,345 selected samples | target classes under selected real/synthetic protocol | Corrected baseline/P3 result available |
| Classification | MSTAR-SOC | 2,761 train, 2,439 test | 10 classes | Available, no current formal pair |
| Segmentation | FUSAR-Map | 610 image-mask pairs | 4 reported terrain classes plus unknown pixels | Corrected runs in progress |
| Change detection | Ottawa | 1 image pair, 350x290 | binary change | Corrected result available |
| Change detection | Yellow River | 1 image pair, 289x257 | binary change | Corrected result available |
| Change detection | Farmland | 1 image pair, 291x306 | binary change | Corrected result available |
| Change detection | Sulzberger | 1 image pair, 256x256 | binary change | Corrected result available |

## 2. Current formal detection results

### 2.1 Historical-style 300-epoch P3 reruns

Common configuration: P3, `state_dim=16`, 640 input, global batch 32, AdamW, `lr0=0.001`, seed 0, FP32, `yolov8n.pt` nominal partial transfer, standard inference.

| Dataset | Split | Epoch status | P | R | AP50 | AP75 | AP50:95 | Evidence note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SSDD | official test | 300/300 | 97.302% | 95.421% | 98.549% | 85.552% | 72.261% | Formal `eval_test.json` |
| SADD | test | 300/300 | 96.082% | 98.039% | 98.297% | 82.999% | 72.113% | Formal `eval_test.json`; final CSV val-loss fields need audit |
| SRSDD-v1.0 | HBB test | early stop at 276 | 57.259% | 47.150% | 48.339% | 16.784% | 22.147% | Formal `eval_test.json`; no matched baseline yet |
| SAR-AIRcraft-1.0 | validation only | failed during epoch 15 | 77.512% | 75.071% | 78.859% | not recorded | 53.902% | Best completed row at epoch 14; not a final result |
| AIR-SARShip-2.0 | test | not started in this protocol | - | - | - | - | - | Queue stopped on frozen-protocol conflict |

SAR-AIRcraft failure: both DDP ranks detected a non-finite loss at epoch 15, batch 166. The 14 completed rows are diagnostic only; no final test JSON exists.

### 2.2 Matched scratch detection pairs

| Dataset / protocol | Baseline AP50 | Baseline AP75 | Baseline AP50:95 | P3-d16 AP50 | P3-d16 AP75 | P3-d16 AP50:95 | Delta AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSDD, batch 16, 150 epochs | 83.439% | 19.393% | 35.239% | 88.137% | 33.756% | 42.991% | **+7.752 pp** |
| SSDD, batch 32, 150 epochs | 88.629% | 42.497% | 45.683% | 88.484% | 32.438% | 42.486% | **-3.197 pp** |
| SADD, batch 32, 200 epochs, seed 0 | 98.335% | 57.030% | 55.018% | 97.760% | 56.985% | 54.597% | **-0.421 pp** |
| AIR-SARShip-2.0, batch 32, 200 epochs | 47.590% | 7.463% | 17.375% | 51.559% | 9.087% | 19.291% | **+1.916 pp** |

The SSDD batch-16 and batch-32 outcomes have opposite signs. Batch size and the associated optimization trajectory are therefore material variables and must remain fixed in the final baseline-versus-P3 table.

### 2.3 Historical SSDD evidence

| Evidence | Method | Split | AP50 | AP75 | AP50:95 |
|---|---|---|---:|---:|---:|
| Historical fixed recheck | Mamba-YOLO-T | official test | 98.605% | 83.613% | 70.348% |
| Historical fixed recheck | PFIM ex27 | official test | 98.688% | 84.005% | 71.445% |
| Historical archive best row | PFIM ex27 | historical val | 98.704% | not archived | 74.022% |
| Current 300-epoch P3 rerun | P3-d16 | official test | 98.549% | 85.552% | 72.261% |

The current SSDD rerun is 0.816 pp above the historical PFIM official-test recheck in AP50:95, but 1.761 pp below the historical best validation row. The latter is a different split and is not a direct regression target.

### 2.4 Historical supplementary detection datasets

| Dataset | Method | Split | AP50 | AP75 | AP50:95 | Evidence |
|---|---|---|---:|---:|---:|---|
| OGSOD-1.0 | Mamba-YOLO-T | test | 77.173% | 40.211% | 42.302% | Matched local run |
| OGSOD-1.0 | IG-DS-SSM/PFIM-T | test | 77.263% | 40.971% | 43.602% | Matched local run, +1.300 pp |
| SAR-ShipDataset | Mamba-YOLO-T | test | 94.616% | - | 49.568% | Historical, batch changed during training |
| SAR-ShipDataset | IG-DS-SSM/PFIM-T | test | 94.844% | - | 57.818% | Historical, not a strict controlled pair |
| SAR-AIRcraft patch | Mamba-YOLO-T | val | 87.751% | - | 69.904% | Historical reference |
| SAR-AIRcraft patch | PFIM-T | val | 88.825% | - | 70.513% | Historical reference, +0.609 pp |
| SAR-AIRcraft patch | Mamba-YOLO-T | test | 70.402% | - | 54.513% | Historical reference |
| SAR-AIRcraft patch | PFIM-T | test | 70.822% | - | 55.552% | Historical reference, +1.039 pp |

## 3. Classification results

Corrected protocol: scratch initialization, 224 input, batch 64, up to 200 epochs, same per-dataset split for baseline and P3-d16.

| Dataset | Method | Accuracy | Macro F1 | Params | GFLOPs | Result |
|---|---|---:|---:|---:|---:|---|
| SAR-ACD | Baseline | 94.882% | 94.815% | 1.976M | 0.273 | reference |
| SAR-ACD | P3-d16 | 89.370% | 89.181% | 1.767M | 0.245 | **-5.512 pp accuracy** |
| SAMPLE | Baseline | 98.519% | 98.454% | 1.977M | 0.273 | reference |
| SAMPLE | P3-d16 | 97.037% | 96.441% | 1.769M | 0.245 | **-1.481 pp accuracy** |

Older 100-epoch smoke-like results reached 99.219%/98.438% on SAR-ACD and 100%/100% on SAMPLE, but they use an older protocol and should not replace the corrected table above.

## 4. Segmentation results

| Protocol | Method | mIoU | mPA | Dice | Status |
|---|---|---:|---:|---:|---|
| Old five-class, 100 epochs | Baseline | 21.084% | 28.223% | 27.308% | Completed but not publication-aligned |
| Old five-class, 100 epochs | P3-d16 | 21.489% | 28.637% | 27.491% | Completed but not publication-aligned |
| Corrected FUSAR run | Baseline | latest val mIoU-all 13.462% at epoch 17 | 27.822% | 20.245% | Running; not final |
| Corrected FUSAR run | P3-d16 | latest val mIoU-all 17.095% at epoch 13 | 30.896% | 25.615% | Running; not final |

The old FUSAR evaluator treated black unknown pixels as a fifth class, while published work reports four terrain classes. Its 21% values must not be compared directly with published 43-52% mIoU. The corrected pair is still running and has no `result.json` yet.

## 5. Change-detection results

Corrected protocol: dense spatial macroblocks, scratch initialization, 192 input, batch 16, up to 150 epochs. Baseline and P3 use the same dataset split. P3-d16 has about 2.138M parameters and 1.136 GFLOPs; baseline has about 2.346M parameters and 1.177 GFLOPs.

| Dataset | Method | Precision | Recall | F1 | IoU | OA | Delta F1 vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Ottawa | Baseline | 90.990% | 94.597% | 92.758% | 86.495% | 98.491% | - |
| Ottawa | P3-d16 | 89.238% | 95.308% | 92.173% | 85.483% | 98.346% | **-0.585 pp** |
| Yellow River | Baseline | 90.666% | 85.635% | 88.079% | 78.697% | 97.758% | - |
| Yellow River | P3-d16 | 90.885% | 82.170% | 86.308% | 75.914% | 97.478% | **-1.771 pp** |
| Farmland | Baseline | 87.724% | 15.296% | 26.050% | 14.976% | 93.039% | - |
| Farmland | P3-d16 | 90.798% | 45.706% | 60.805% | 43.683% | 95.277% | **+34.755 pp** |
| Sulzberger | Baseline | 75.922% | 69.540% | 72.591% | 56.974% | 90.461% | - |
| Sulzberger | P3-d16 | 51.656% | 92.066% | 66.180% | 49.454% | 82.908% | **-6.411 pp** |

The change-detection result is strongly dataset-dependent. Farmland improves substantially, while Ottawa, Yellow River, and Sulzberger do not improve in F1. This does not currently support a universal cross-task gain claim.

## 6. Core ablation data

### State dimension on SSDD, batch 16

| Configuration | Recall | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|
| Baseline | 75.543% | 83.439% | 19.393% | 35.239% |
| P3-d16 | 83.516% | 88.137% | 33.756% | 42.991% |
| P3-d32 | 80.220% | 86.266% | 28.646% | 40.596% |

Current controlled evidence favors d16 over d32 by 2.395 pp AP50:95.

### Partial SSDD placement sweep, development validation

| Placement | State dim | AP50:95 | Delta vs baseline |
|---|---:|---:|---:|
| P2 | 16 | 56.673% | +1.834 pp |
| P3 | 8 | 56.118% | +1.280 pp |
| P3 | 32 | 56.016% | +1.177 pp |
| P4 | 8 | 55.817% | +0.979 pp |
| P2 | 32 | 55.773% | +0.934 pp |
| P4 | 16 | 54.978% | +0.140 pp |
| Baseline | 0 | 54.838% | 0 |
| P3 | 16 | 53.840% | -0.999 pp |
| P2 | 8 | 52.134% | -2.704 pp |

This sweep completed only 9 of 16 runs. It does not prove P3 is globally optimal. P4-d32, all P5 settings, and all P2+P3 settings are missing.

### Module decomposition status

| Configuration | Status |
|---|---|
| Mamba-YOLO baseline | Available on several datasets |
| DS-SSM only / no PFIM | Unified final result missing |
| Parameter-matched global importance | Missing |
| Full token PFIM | Available on several datasets |
| Gamma-only / write-only / delta-only | Missing |
| Gamma+write / full combination | Missing |

## 7. Runtime status at collection

| Host | State |
|---|---|
| A6000-A | Idle |
| A6000-B | Running corrected FUSAR-Map jobs, about 22.9 GiB used |
| A6000-C | Idle |
| A6000-D | Idle; SADD finished, AIR-SARShip historical-style queue did not start |
| A6000-E | Unreachable by SSH |
| 3090-3 | Two GPUs idle after SAR-AIRcraft non-finite-loss failure |
| 3090-4 | Idle; SRSDD evaluation complete |

The automatic scheduler's last status lists 18 completed jobs and 6 pending jobs, but its controller log currently ends with `AttributeError: 'str' object has no attribute 'get'`. Scheduler state should be repaired before relying on automatic dispatch.

## 8. Evidence ranking and current conclusion

1. Strongest positive controlled detection evidence: SSDD batch-16 (+7.752 pp AP50:95), OGSOD (+1.300 pp), and AIR-SARShip (+1.916 pp).
2. Negative or neutral controlled evidence: SSDD batch-32 (-3.197 pp), SADD 200-epoch scratch (-0.421 pp), SAR-ACD (-5.512 pp accuracy), SAMPLE (-1.481 pp), and three of four change datasets in F1.
3. Strong standalone scores: SSDD current P3 test AP50:95 72.261% and SADD current P3 test AP50:95 72.113%. These need matched baselines under the same 300-epoch partial-transfer protocol before causal claims.
4. Historical best SSDD val AP50:95 remains 74.022%, but the directly comparable historical official-test value is 71.445%.
5. The current evidence supports a lightweight SAR detection paper more strongly than a universal SAR backbone claim.

## 9. Primary evidence paths

- `dataset_inventory.json`
- `reports/detection_historical_config_run_status_20260829.md`
- `reports/ssdd_historical_recheck_and_fusar_audit_20260828.md`
- `reports/current_d16_vs_sota_20260826.md`
- `reports/ogsod_test_results_20260731.md`
- `reports/sar_shipdataset_current_results.md`
- `reports/sar_aircraft_experiment_20260721.md`
- `database/ogsod_test_results_20260731.csv`
- `database/sar_aircraft_experiment_20260721_per_class.csv`
- `database/sar_shipdataset_current_results.csv`

