# AAAI Bilingual Paper Status

Updated: 2026-08-12

## Current Manuscripts

| Language | Source | Compiled PDF | Pages |
|---|---|---|---:|
| English AAAI-style manuscript | `paper_draft/aaai_ds_ssm_paper_en.tex` | `paper_draft/aaai_ds_ssm_paper_en.pdf` | 7 |
| Chinese companion manuscript | `paper_draft/aaai_ds_ssm_paper_zh.tex` | `paper_draft/aaai_ds_ssm_paper_zh.pdf` | 5 |

Stable delivery copies are stored under `output/pdf/` after final verification.

## Revision Completed on 2026-08-07

1. Added OGSOD-1.0 as the primary matched single-run architecture comparison.
2. Corrected the PFIM initialization description: gamma/write receive zero centered importance at initialization, while delta receives a small positive `eta/2` offset.
3. Reframed the main claim from universal AP50 improvement to localization-sensitive AP75/AP50:95 improvement at nearly unchanged complexity.
4. Added final SAR-Aircraft PFIM-S validation results as a capacity reference.
5. Added the dense tank failure mode and separated observation from causal explanation.
6. Added an evidence hierarchy table that distinguishes matched OGSOD evidence from protocol-confounded archives.
7. Added a code-aligned method overview figure showing shallow-stage placement, residual suppression, PFIM, the three control paths, and the executed FP32 CUDA scan.
8. Replaced three oversized full-width literature tables with compact selected-baseline tables. The English manuscript decreased from 9 sparse pages after the first OGSOD insertion to 7 balanced pages with the method figure included.
9. Corrected the executable-path description: gamma and write are factorized terms on the same CUDA scan drive, while delta is the second independent operator location. The paper no longer presents them as three independent recurrent update sites.
10. Added an execution-ready OGSOD matrix with three-seed baseline/no-PFIM/global-PFIM/full controls, single-seed factor diagnostics, split/label fingerprints, automatic best-checkpoint TEST evaluation, and paired mean/std aggregation.

## Revision Completed on 2026-08-12

1. Tightened both manuscripts so one OGSOD run per architecture is explicitly a single-run observation, not evidence of a stable improvement.
2. Downgraded SSDD, SAR-ShipDataset, and SAR-AIRcraft from a claimed cross-dataset ``same tendency'' to protocol-limited archive context; no cross-dataset causal or stability claim remains.
3. Documented the checkpoint rule in the paper: `best.pt` is selected on validation fitness `0.1 AP50 + 0.9 AP50:95`, then evaluated once in FP32 on held-out OGSOD TEST.
4. Reframed the dense-tank result as an observed failure mode with density, scale, appearance, and NMS confounded, rather than attributing it to shared importance control before stratified analysis.
5. Recompiled both manuscripts. The English manuscript is 7 letter pages and the Chinese companion is 5 A4 pages; compilation logs contain no undefined citation/reference, overfull-box, or fatal-error warning.
6. Hardened the pending OGSOD matrix as `v2`: image byte streams and labels are hashed, the protocol/code/config fingerprint is immutable per matrix ID, parser-level parameter matching is verified for `full/global/no_2dstd`, and a CPU fallback semantics check confirms the same `gamma * write * u` scan input as the CUDA path.

## Primary Controlled Result

OGSOD-1.0 local three-way protocol:

- Train/validation/test images: 14,664 / 1,834 / 1,833.
- Converted test boxes: 4,769.
- Both T-scale runs: 1024 input, 300 epochs, global batch 64, two H200 GPUs, AdamW, final TEST evaluation from `best.pt`.

| Model | P | R | AP50 | AP75 | AP50:95 | Params (M) | GFLOPs (640 ref.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO-T | 0.801434 | 0.711545 | 0.771733 | 0.402109 | 0.423018 | 5.984593 | 13.6 |
| IG-DS-SSM/PFIM-T | 0.797140 | 0.716924 | 0.772630 | 0.409711 | 0.436021 | 5.963803 | 13.2 |
| PFIM delta | -0.004294 | +0.005379 | +0.000897 | +0.007602 | +0.013003 | -0.020790 | -0.4 |

In percentage points, PFIM-T changes AP50 by +0.09, AP75 by +0.76, and AP50:95 by +1.30. This is a matched protocol but still only one completed run per model.

## Cross-Dataset Evidence Hierarchy

| Dataset/evaluation | Delta AP50 | Delta AP50:95 | Evidence status |
|---|---:|---:|---|
| OGSOD-1.0 TEST | +0.090 pp | +1.300 pp | Matched protocol, single run |
| SSDD archive validation | +0.219 pp | +1.170 pp | Recovered code states |
| SAR-Ship local TEST | +0.228 pp | +8.250 pp | Baseline batch changed 256 to 128 |
| SAR-Aircraft patch validation | +1.074 pp | +0.609 pp | Optimizer/batch/schedule mismatch |
| SAR-Aircraft patch TEST | +0.420 pp | +1.039 pp | Optimizer/batch/schedule mismatch |

Only OGSOD is presented as the primary controlled comparison. The other rows are archive context, not evidence of a replicated cross-dataset trend.

## OGSOD Per-Class Limitation

| Class | Baseline AP50/AP75/AP50:95 | PFIM-T AP50/AP75/AP50:95 |
|---|---|---|
| bridge | 0.811348 / 0.260497 / 0.362657 | 0.822717 / 0.287746 / 0.382717 |
| harbor | 0.981278 / 0.882131 / 0.735285 | 0.993148 / 0.886652 / 0.753875 |
| tank | 0.522573 / 0.063699 / 0.171112 | 0.502027 / 0.054735 / 0.171470 |

Tank contains 1,205 instances in 71 positive test images, about 17 per positive image. The paper describes this as a limitation on densely packed repetitive objects, not a universal small-target failure.

## Remaining Publication Blockers

1. No three-seed matched statistics for the primary OGSOD result.
2. No immutable-protocol `Mamba-YOLO-T -> DS-SSM without PFIM -> full PFIM` ablation.
3. No completed gamma-only, write-only, delta-only, local-statistic, or branch-combination result; runnable entries now exist.
4. No completed parameter-matched alternative result; parameter-identical global-importance and no-2D-std controls are now implemented but still require training.
5. No density/scale/nearest-neighbor stratification or PFIM importance-map failure analysis for dense tanks.
6. The SSDD, SAR-Ship, and SAR-Aircraft baseline deltas remain protocol-confounded and cannot become causal evidence without rerunning baselines.

## Strict Review Status

- Round 1 before revision: `5/10`, borderline reject.
- Round 2 after evidence, claim, and layout revision: `6/10`, borderline/weak reject.
- Round 3 after executable-path correction and experiment-harness audit: `6/10`, weak reject until the prepared runs produce evidence.
- Round 4 after protocol/claim tightening: `6/10`, weak reject. Writing and reproducibility are stronger, but the score cannot increase without completed controlled results.
- Credible path to `7/10+`: complete the three-seed OGSOD baseline/no-PFIM/global/full matrix and branch-level ablations under one frozen code/data protocol.

Review artifacts:

- `reports/top_conference_review_round1_20260807.md`
- `reports/top_conference_review_round2_20260807.md`
- `reports/top_conference_review_round3_20260807.md`
- `reports/top_conference_review_round4_20260812.md`

OGSOD evidence artifacts:

- `database/ogsod_test_results_20260731.csv`
- `database/ogsod_test_per_class_20260731.csv`
- `reports/ogsod_test_results_20260731.md`

Execution artifacts prepared for the missing evidence:

- `source_code/work_train_single/tools/run_ogsod_ablation_matrix.sh`
- `source_code/work_train_single/tools/evaluate_ogsod_ablation.py`
- `source_code/work_train_single/tools/summarize_ogsod_ablation_matrix.py`
- `source_code/work_train_single/tools/snapshot_yolo_dataset.py`
- `source_code/work_train_single/tools/evaluate_ogsod_strata.py`
- `source_code/work_train_single/tools/visualize_pfim_importance.py`
- `source_code/work_train_single/tools/compare_ogsod_per_image.py`
- `reports/ogsod_ablation_matrix_plan.md`
