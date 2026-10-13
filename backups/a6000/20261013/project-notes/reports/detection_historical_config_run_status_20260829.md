# P3 Historical-Configuration Detection Runs

Updated: 2026-08-29

## Formal protocol

- Model: IG-DS-SSM/PFIM at P3
- `state_dim`: 16
- Initialization: `yolov8n.pt` partial transfer
- Epochs: 300
- Image size: 640
- Global batch: 32
- Optimizer: AdamW
- `lr0`: 0.001
- `lrf`: 0.01
- Weight decay: 0.0005
- Nominal batch size: 64
- Optimizer warmup: 5 epochs
- DS-SSM/PFIM module warmup: 20 epochs for the SSDD/SADD/SRSDD protocol; SAR-AIRcraft uses 1 epoch to match the approximate warmup-step budget because its training set is much larger.
- Mosaic: 1.0, closed for the last 10 epochs
- Augmentation: HSV 0.015/0.7/0.4, translate 0.1, scale 0.5, horizontal flip 0.5
- Seed: 0
- Precision: FP32. AMP is disabled because the exact P3/lr=0.001 diagnostic produced NaN values.

## Run allocation

| Dataset | Host | GPUs | State | Run ID |
|---|---|---:|---|---|
| SSDD | A6000-D | 1 x A6000 | Completed; formal test written | `det_histcfg_p3_d16_fp32_v1_ssdd_full_p3` |
| SRSDD-v1.0 | 3090-4 | 1 x RTX 3090 | Completed; formal test written | `det_histcfg_p3_d16_fp32_v4_srsdd_v10_full_p3` |
| SAR-AIRcraft1.0 | 3090-3 | 2 x RTX 3090 | Failed at epoch 15 batch 166 with non-finite loss; no final test | `det_histcfg_p3_d16_fp32_aircraft_stepwarm_v1_sar_aircraft10_full_p3` |
| SADD | A6000-D | 1 x A6000 | Completed; formal test written | `det_histcfg_p3_d16_fp32_remaining_v1_sadd_full_p3` |
| AIR-SARShip-2.0 | A6000-D | 1 x A6000 | Not started; frozen-protocol conflict after SADD | `det_histcfg_p3_d16_fp32_remaining_v1_air_sarship_20_full_p3` |

SSDD uses the full 928-image training set and the 232-image official test set. The other datasets use their frozen train/val/test manifests.

## Required runtime fixes

1. TensorBoard model graph tracing is skipped for these runs. Scalar logging remains enabled. Tracing the custom selective scan at 640 pixels expanded a very large JIT graph before the first batch and consumed host memory; it does not affect training or evaluation.
2. The original Mamba-YOLO CUDA dispatch now treats `selective_scan_cuda_core` as the required fast kernel and the other scan kernels as optional. Previously, one missing unused `nrow` extension disabled the installed core kernel and forced a per-token PyTorch fallback, causing an avoidable GPU OOM.

## Main paths

- Local launcher: `source_code/work_train_single/tools/run_historical_config_detection_p3.sh`
- A6000-D project: `/home/powerleader/ig-ds-ssm/experiments/historical_protocol_detection`
- 3090-4 project: `/mnt/disk1/zhangji/ig-ds-ssm-detection/experiments/historical_protocol_detection`
- 3090-3 project: `/mnt/data5/zhangji/ig-ds-ssm-detection/experiments/historical_protocol_detection`

Only completed `eval_test.json` files are final results. Intermediate validation rows must not be reported as test performance. A non-finite training loss invalidates a run. Non-finite auxiliary validation-loss fields with finite detection metrics and a successful independent test evaluation require audit but are not silently treated as equivalent to a training-loss divergence.

## Completed formal test results

| Dataset | P | R | AP50 | AP75 | AP50:95 | Inference |
|---|---:|---:|---:|---:|---:|---:|
| SSDD | 97.302% | 95.421% | 98.549% | 85.552% | 72.261% | 7.10 ms/img |
| SRSDD-v1.0 | 57.259% | 47.150% | 48.339% | 16.784% | 22.147% | 8.98 ms/img |
| SADD | 96.082% | 98.039% | 98.297% | 82.999% | 72.113% | 9.08 ms/img |

The SSDD batch-32 scratch pair (`45.683%` baseline vs `42.486%` P3 AP50:95) is a separate controlled scratch experiment and must not be mixed with the historical partial-transfer protocol above.
