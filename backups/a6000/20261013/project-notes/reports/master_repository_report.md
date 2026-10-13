# Master Repository Report

Note: this report preserves an earlier repository recovery view. For current model-performance ranking, prefer `reports/experiment_ranking_recomputed.md` and `database/experiment_ranking_recomputed.csv`. The current recomputed Top 1 is `server_3090_b / PFIM...ex27` with AP50:95 `0.74022`.

## Structure

```text
source_code/
experiments/
weights/
logs/
database/
reports/
inventory/
archives/
```

## Counts

- Code versions: `3`
- Ranked experiments: `172`
- Recovered experiments: `40`
- Recovered weight files: `80`
- Recovered log files: `0`
- Recovered config files: `40`

## Top Experiments

| Rank | Server | Experiment | Score | Recovered |
|---:|---|---|---:|---|
| 1 | `server_1080ti` | `ssdd/ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602:15:40` | `0.71534` | `True` |
| 2 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `0.71214` | `True` |
| 3 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `0.71214` | `True` |
| 4 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex2-gemini-20260520:0:18` | `0.71214` | `True` |
| 5 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `0.71082` | `True` |
| 6 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `0.71082` | `True` |
| 7 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_300e_16bs` | `0.71082` | `True` |
| 8 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `0.7097` | `True` |
| 9 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `0.7097` | `True` |
| 10 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex8-gemini-20260521:10:122` | `0.7097` | `True` |
| 11 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27-gemini-20260603:17:45` | `0.70964` | `True` |
| 12 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex24-gemini-20260603:11:15` | `0.70904` | `True` |
| 13 | `server_3090_b` | `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_36bs-ex29-gemini-20260604:14:55` | `0.70893` | `True` |
| 14 | `server_3090_a` | `sar_aircraft/mambayolo_t_sar_aircraft_backbone_ds-ssm_300e_16bs_1024_ex11ssdd_ex2_202606012307` | `0.70824` | `True` |
| 15 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `0.7068` | `True` |
| 16 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `0.7068` | `True` |
| 17 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7-gemini-20260521:00:07` | `0.7068` | `True` |
| 18 | `server_1080ti` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `0.70556` | `True` |
| 19 | `server_3090_a` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `0.70556` | `True` |
| 20 | `server_3090_b` | `ssdd/mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex4-gemini-20260520:14:42` | `0.70556` | `True` |

## Still Need Manual Confirmation

- Resolve same-path code conflicts in `reports/code_conflict_report.md` before creating a single merged code tree.
- Confirm exact experiment-to-uncommitted-code mapping for PFIM/ex-series experiments.
- Confirm whether `weights/` subdirectories indexed as experiment-like rows should be collapsed in future database views.
- Validate ranking metrics manually where `results.csv` column naming differs.
