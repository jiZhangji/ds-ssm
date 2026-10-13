# Current d16 Result vs Published References

## Scope

This report separates the current locally measured result from historical or author-reported values. The current SSDD pair uses the same code/data protocol, `seed=0`, random initialization, `imgsz=640`, `batch=16`, and held-out `test` evaluation.

## Matched Local SSDD Comparison

| Method | Variant | State dim | Batch | Split | Precision | Recall | AP50 | AP75 | AP50:95 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Mamba-YOLO baseline | baseline | 0 | 16 | test | 89.576% | 75.543% | 83.439% | 19.393% | 35.239% |
| IG-DS-SSM/PFIM | P3 / full | 16 | 16 | test | 88.231% | 83.516% | 88.137% | 33.756% | 42.991% |
| **Difference** | P3 - baseline | - | - | - | -1.345 pp | **+7.973 pp** | **+4.698 pp** | **+14.363 pp** | **+7.752 pp** |

Interpretation: under this matched local protocol, P3 improves recall and localization-sensitive AP substantially. Precision decreases by 1.345 percentage points, so the result supports a recall/localization improvement claim rather than a blanket precision improvement claim.

## State-Dimension Diagnostic at Batch 16

The `d16` and `d32` runs use identical train/validation image lists (matching SHA256 hashes), the same official test set, and the same training protocol. The dataset YAML hashes differ only because each YAML embeds its own experiment-manifest path.

| Method | State dim | Batch | Precision | Recall | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO baseline | 0 | 16 | 89.576% | 75.543% | 83.439% | 19.393% | 35.239% |
| IG-DS-SSM/PFIM P3 | 16 | 16 | 88.231% | 83.516% | **88.137%** | **33.756%** | **42.991%** |
| IG-DS-SSM/PFIM P3 | 32 | 16 | **90.511%** | 80.220% | 86.266% | 28.646% | 40.596% |

Relative to the matched baseline, `d32` still improves AP50:95 by `+5.357 pp`. However, compared with `d16`, it loses `2.395 pp` AP50:95, `5.110 pp` AP75, and `3.297 pp` recall. The current controlled evidence therefore favors `state_dim=16`; a larger hidden state should not be assumed better.

## Batch-32 Controlled Runs

These runs started on 2026-08-26 and are still in progress. All use the same deterministic `742 train / 186 val / 232 test` split, random initialization, `150` epochs, seed `0`, and no module warmup.

| Host | Variant | State dim | Batch | Status |
|---|---|---:|---:|---|
| A6000-B | Mamba-YOLO baseline | 0 | 32 | Running |
| A6000-C | IG-DS-SSM/PFIM P3 | 16 | 32 | Running |
| A6000-D | IG-DS-SSM/PFIM P3 | 32 | 32 | Running |

The three final test results will separate the effect of batch size from the effect of state dimension. They must not be compared using intermediate validation epochs.

## Published or Historical SSDD References

These values must not be ranked directly with the local test result because the papers differ in split, initialization, augmentation, training schedule, and sometimes metric definition.

| Method | AP50 | AP50:95 | Params | GFLOPs | Evidence type |
|---|---:|---:|---:|---:|---|
| YOLOv8n | 98.1% | Not reported uniformly | approx. 3M | approx. 8.7 | Author/literature report |
| YOLOv5s | 98.3% | Not reported uniformly | approx. 11M | approx. 28.6 | Author/literature report |
| AMANet | 98.5% | 74.2% | Not reported | Not reported | Author report |
| CCAI-YOLO | 98.8% | 74.9% | 4.754M | 10.1 | Author report |
| MaDiNet | 99.0% | Not reported as a matched value | Not unified | Not unified | Author report |
| SSD-YOLO | 99.0% | 74.3% | Not reported | Not reported | Author report |
| YOLO-SRBD | 98.3% | 74.5% | 7.72M | 21.5 | Author report |
| RSNet | Not reported | 72.5% | 1.49M | Not reported | Author report |
| Historical PFIM archive | 98.704% | 74.022% | approx. 5.96M | approx. 13.21 | Historical archive |

## What Can Be Claimed

1. The current d16 experiment provides a controlled baseline-versus-P3 comparison on SSDD.
2. P3 improves local AP50:95 by `+7.752 pp` and AP75 by `+14.363 pp` under the current test protocol.
3. The current result cannot be called SOTA against the published table until the split and evaluation protocol are aligned.
4. The historical PFIM archive is useful as a reference, but it is not interchangeable with the current scratch/random-initialization experiment.
5. Before the final paper table, Params/GFLOPs/FPS should be measured again with the same profiler and the exact d16/d32 checkpoint family.
6. The completed batch-16 diagnostic currently favors `state_dim=16` over `state_dim=32`; the batch-32 runs are a confirmation check, not evidence that `d32` is superior.

## Source Files

- `.codex-transfer/d16-results/eval_test.json`
- `.codex-transfer/d16-results/sar4_scratch_p3_sd16_v1_summary.md`
- `reports/four_dataset_methods_metrics_opensource_audit_20260823.md`
- `reports/ssdd_comparison_reproducibility_audit_20260823.md`
- `reports/lightweight_detection_comparison_madinet_datasets.md`
