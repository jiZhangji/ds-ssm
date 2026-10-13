# Current Cross-Task Results

更新日期：2026-09-04

说明：本表只记录已经生成最终 `result.json` 的实验。验证集过程值不作为最终测试结果。当前新增的 BS32/64 变化检测队列和 SAR-ACD BS/预热队列仍在等待 A/B 上的 OpenEarthMap 任务完成，暂未产生新结果。

## Change Detection

指标均为测试集结果；百分比显示。`BS16` 为此前已完成的受控实验，`BS32/64` 为本次新增队列，尚未完成。

| Dataset | Method | State dim | BS | F1 (%) | IoU (%) | OA (%) | Precision (%) | Recall (%) | Params (M) | GFLOPs | FPS | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ottawa | Baseline | 32 | 16 | 92.758 | 86.495 | 98.491 | 90.990 | 94.597 | 2.346 | 1.177 | 2.158 | Complete |
| Ottawa | IG-DS-SSM/PFIM | 32 | 16 | 92.110 | 85.374 | 98.350 | 90.061 | 94.254 | 2.144 | 1.137 | 2.266 | Complete |
| Yellow River | Baseline | 32 | 16 | 88.079 | 78.697 | 97.758 | 90.666 | 85.635 | 2.346 | 1.177 | 2.186 | Complete |
| Yellow River | IG-DS-SSM/PFIM | 32 | 16 | 87.689 | 78.077 | 97.683 | 90.227 | 85.290 | 2.144 | 1.137 | 2.272 | Complete |
| Farmland | Baseline | 32 | 16 | 26.050 | 14.976 | 93.039 | 87.724 | 15.296 | 2.346 | 1.177 | 2.174 | Complete |
| Farmland | IG-DS-SSM/PFIM | 32 | 16 | 73.104 | 57.610 | 96.456 | 93.342 | 60.078 | 2.144 | 1.137 | 2.309 | Complete |
| Ottawa | IG-DS-SSM/PFIM | 16 | 16 | 92.173 | 85.483 | 98.346 | 89.238 | 95.308 | 2.138 | 1.136 | 2.273 | Complete |
| Yellow River | IG-DS-SSM/PFIM | 16 | 16 | 86.308 | 75.914 | 97.478 | 90.885 | 82.170 | 2.138 | 1.136 | 2.286 | Complete |
| Farmland | IG-DS-SSM/PFIM | 16 | 16 | 60.805 | 43.683 | 95.277 | 90.798 | 45.706 | 2.138 | 1.136 | 2.272 | Complete |
| Ottawa | Baseline | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |
| Ottawa | IG-DS-SSM/PFIM | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |
| Yellow River | Baseline | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |
| Yellow River | IG-DS-SSM/PFIM | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |
| Farmland | Baseline | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |
| Farmland | IG-DS-SSM/PFIM | 32 | 32/64 | - | - | - | - | - | - | - | - | Queued |

## SAR-ACD Classification

| Dataset | Method | State dim | BS | Accuracy (%) | Macro-Precision (%) | Macro-Recall (%) | Macro-F1 (%) | Params (M) | GFLOPs | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAR-ACD | Baseline | - | - | 99.219 | - | - | - | 1.975 | - | Complete, legacy result |
| SAR-ACD | IG-DS-SSM/PFIM | - | - | 98.438 | - | - | - | 1.767 | - | Complete, legacy result |
| SAR-ACD | IG-DS-SSM/PFIM | 16 | 64 | 89.370 | 89.291 | 89.226 | 89.181 | 1.767 | 0.245 | Complete, corrected protocol |
| SAR-ACD | Baseline | 16 | 64/128 | - | - | - | - | - | - | Queued |
| SAR-ACD | IG-DS-SSM/PFIM | 16 | 32/64/128 | - | - | - | - | - | - | Queued |
| SAR-ACD | IG-DS-SSM/PFIM | 16 | 64 | - | - | - | - | - | - | Queued, warmup20 |
| SAR-ACD | IG-DS-SSM/PFIM | 16 | 64 | - | - | - | - | - | - | Queued, warmup50 |

## Queue Status

- A6000-A: current OpenEarthMap task followed by the change-detection BS32/64 queue.
- A6000-B: current OpenEarthMap/BS32 warmup task followed by the SAR-ACD BS and warmup queue.
- A6000-F: occupied by another SRSDD task; not used for this batch.
- New completed results: `0` from the newly queued BS32/64 and SAR-ACD experiments at this check.
