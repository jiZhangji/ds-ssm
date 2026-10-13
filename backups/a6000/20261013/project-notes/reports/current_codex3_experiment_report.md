# Current Codex3 Experiment Report

- Source server: `server_current_3090x2` (`172.26.151.143`)
- Remote train dir: `/mnt/data5/zhangji/Mamba-YOLO/output_dir/ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3`
- Local archive: `experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3`
- Weights: `weights/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3__best.pt`, `weights/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3__last.pt`
- Downloaded files: `23`

## Metrics

| Mode | Epoch | P | R | mAP50 | mAP50-95 | Source |
|---|---:|---:|---:|---:|---:|---|
| `train_best_val` | 296 | 0.97310 | 0.93040 | 0.97976 | 0.73028 | Best row recomputed from training results.csv |
| `train_last_val` | 300 | 0.94292 | 0.95971 | 0.97806 | 0.72844 | Last epoch from training results.csv |
| `standard_test` |  | 0.98300 | 0.92600 | 0.97800 | 0.72900 | Parsed from user pasted test output |
| `tta_iou0.55_conf0.001` |  | 0.96900 | 0.95200 | 0.97800 | 0.71500 | Parsed from user pasted test output |

## Analysis

- The standard test result is close to the final validation row, so the recovered `best.pt` is consistent with the training record.
- TTA increased recall from `0.926` to `0.952`, but precision and mAP50-95 decreased. This suggests extra augmented predictions recovered more ships but introduced localization/duplicate-quality penalties under COCO-style IoU averaging.
- Compared with the previous best recovered ex27 run (`mAP50-95 0.74022`), this DDP 80 batch run is slightly lower on mAP50-95. The most likely contributors are larger global batch size changing optimization dynamics, the 20-epoch freeze schedule, and DDP/find-unused behavior changing which parameters receive synchronized gradients during the frozen phase.
- The cuDNN warning during test is a non-fatal plan-selection warning; inference completed normally.

## Table Updates

- Updated `database/current_codex3_performance_comparison.csv` and `.xlsx`.
- Appended this run to `database/SSDD_experiment_results_summary_recomputed.csv` and `.xlsx`.
- Appended this run to `database/experiment_ranking_recomputed.csv` and `.xlsx`.
