# OGSOD-1.0 Controlled Test Results

## Protocol

The recovered OGSOD archive contains 18,331 SAR images and supplied COCO-style splits of 14,664 training, 1,834 validation, and 1,833 test images. The converted YOLO test split contains 4,769 valid boxes: 3,165 bridges, 399 harbors, and 1,205 tanks. One source annotation was filtered during conversion, so the converted count is one lower than the source JSON count of 4,770.

Both final T-scale runs use 1024 input, 300 epochs, global batch 64, two H200 GPUs, AdamW, and the same train/validation/test files. The reported values are FP32 TEST evaluations of each run's `best.pt`. This is the cleanest matched baseline-to-method comparison currently available in the project.

The dataset originates from Wang et al., "Category-Oriented Localization Distillation for SAR Object Detection and a Unified Benchmark," IEEE TGRS 2023, DOI `10.1109/TGRS.2023.3291356`. The original paper describes 14,665 paired optical-SAR training images and 3,666 SAR test images. The local three-way split therefore must be reported as a recovered local protocol rather than the original published benchmark split.

## Overall Test Metrics

| Model | P | R | AP50 | AP75 | AP50:95 | Params (M) | GFLOPs (640 ref.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO-T | 0.801434 | 0.711545 | 0.771733 | 0.402109 | 0.423018 | 5.984593 | 13.6 |
| IG-DS-SSM/PFIM-T | 0.797140 | 0.716924 | 0.772630 | 0.409711 | 0.436021 | 5.963803 | 13.2 |
| PFIM minus baseline | -0.004294 | +0.005379 | +0.000897 | +0.007602 | +0.013003 | -0.020790 | -0.4 |

In percentage points, PFIM-T changes precision by -0.43, recall by +0.54, AP50 by +0.09, AP75 by +0.76, and AP50:95 by +1.30. The main controlled gain is localization quality under stricter IoU thresholds, not AP50 or precision.

## Per-Class Test Metrics

| Class | Baseline AP50 | Baseline AP75 | Baseline AP50:95 | PFIM AP50 | PFIM AP75 | PFIM AP50:95 | AP50:95 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| bridge | 0.811348 | 0.260497 | 0.362657 | 0.822717 | 0.287746 | 0.382717 | +0.020060 |
| harbor | 0.981278 | 0.882131 | 0.735285 | 0.993148 | 0.886652 | 0.753875 | +0.018590 |
| tank | 0.522573 | 0.063699 | 0.171112 | 0.502027 | 0.054735 | 0.171470 | +0.000358 |

The aggregate gain is driven by bridge and harbor. Tank AP50 and AP75 decrease even though AP50:95 is effectively unchanged. The test split places 1,205 tanks in only 71 positive images, or approximately 17 targets per positive image. This supports a narrow limitation concerning densely packed repetitive objects; it does not support a broad claim that PFIM is ineffective for all small targets.

## Publication Use

Use OGSOD as the primary matched architecture comparison. Use SSDD, SAR-ShipDataset, and SAR-AIRcraft as cross-dataset evidence with their existing protocol caveats. Do not report the incomplete PFIM-S OGSOD run as a final result.
