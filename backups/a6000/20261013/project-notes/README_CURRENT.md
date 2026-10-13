# Current Project Entry

This repository is a recovered research workspace for Mamba-YOLO, DS-SSM, and PFIM experiments on SAR target detection. It contains active working code, historical server snapshots, recovered experiments, weights, reports, and paper drafts.

Use this file as the first entry point before broad searching.

## Current Mainline

The active research mainline is:

**IG-DS-SSM / PFIM for lightweight SAR target detection.**

Core idea:

- `DS-SSM` attenuates local sequence residuals before selective scan.
- `PFIM` estimates token-level importance from local trend and two-dimensional variation.
- In the executed CUDA path, `gamma` and `write_gate` are two factors on the same scan drive; `delta` is the separate transition-control location.

Current primary controlled evidence is the single-seed OGSOD-1.0 TEST comparison:

| Model | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| Mamba-YOLO-T | 0.771733 | 0.402109 | 0.423018 |
| IG-DS-SSM/PFIM-T | 0.772630 | 0.409711 | 0.436021 |

This is a matched protocol but remains a single run. The defensible observation is `+0.760` AP75 and `+1.300` AP50:95 percentage points, not universal AP50 superiority.

Current best recovered SSDD result:

| Experiment | Source | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| `PFIM...ex27` | `server_3090_b` | 0.97378 | 0.95219 | 0.98704 | 0.74022 | 5.9648M | 13.2079 |

The DDP reproduction run `ddp_PFIM...ex27-codex3` completed successfully, but its best validation AP50:95 is `0.73028`, below the recovered `server_3090_b` ex27 result.

## Read First

For research or paper work:

1. `README_CURRENT.md`
2. `reports/thread_019f2406_integrated_context.md`
3. `HANDOFF_FOR_NEXT_MODEL.md`
4. `reports/aaai_paper_current_status.md`
5. `reports/top_conference_review_round3_20260807.md`
6. `reports/ogsod_ablation_matrix_plan.md`
7. `reports/experiment_ranking_recomputed.md`
8. `reports/madinet_ds_ssm_complexity_report.md`
9. `reports/lightweight_detection_comparison_madinet_datasets.md`
10. `paper_draft/aaai_ds_ssm_paper_en.tex`
11. `paper_draft/aaai_ds_ssm_paper_en.pdf`
12. `paper_draft/aaai_ds_ssm_paper_zh.tex`
13. `paper_draft/aaai_ds_ssm_paper_zh.pdf`

For code work:

1. `source_code/README.md`
2. `source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py`
3. `source_code/work_train_single/ultralytics/nn/modules/__init__.py`
4. `source_code/work_train_single/ultralytics/nn/tasks.py`
5. `source_code/work_train_single/ultralytics/cfg/models/mamba-yolo/`
6. `source_code/work_train_single/tools/run_ogsod_ablation_matrix.sh`

For experiment lookup:

1. `database/experiment_ranking_recomputed.csv`
2. `reports/experiment_ranking_recomputed.md`
3. `database/current_codex3_performance_comparison.csv`
4. `reports/current_codex3_experiment_report.md`
5. `reports/sar_aircraft_experiment_20260721.md`
6. `database/sar_aircraft_experiment_20260721_per_class.csv`
7. `reports/sar_shipdataset_current_results.md`
8. `database/sar_shipdataset_current_results.csv`
9. `reports/sar_aircraft_s_interim_20260723.md`
10. `database/sar_aircraft_s_interim_20260723.csv`

## Directory Zones

Active / high-signal:

- `reports/`
- `database/`
- `paper_draft/`
- `source_code/work_train_single/`
- `source_code/work_train_ddp/`
- `complexity_benchmark/`
- `madinet_three_dataset_result_summary/`

Evidence archive / search only when needed:

- `weights/`
- `experiments/`
- `downloads/`
- `archives/`
- `logs/`
- `source_code/server_1080ti/`
- `source_code/server_3090_a/`
- `source_code/server_3090_b/`

Do not move or delete archive directories unless explicitly planning a path migration. Many reports and recovered experiment records refer to these paths.

## Search Policy

The root `.rgignore` excludes large archive and binary-heavy directories from default `rg` searches. This keeps routine searches focused on the active code, reports, paper drafts, and databases.

Examples:

```powershell
rg "PFIM|DeSpeckleSSM|DS_VSSBlock"
rg -u "PFIM" downloads source_code/server_3090_b
rg --files -u weights experiments
```

Use `-u` or search a specific ignored directory when historical evidence is needed.

## Safe Claims

- Current strongest recovered SSDD result: `PFIM...ex27` from `server_3090_b`, AP50:95 `0.74022`.
- Current strongest controlled evidence: OGSOD single-run AP75 `+0.760` pp and AP50:95 `+1.300` pp under a matched protocol.
- Current paper mainline: IG-DS-SSM / PFIM, not DG-DS-SSM.
- MaDiNet remains a strong AP50 reference; this project should emphasize lightweight SAR-aware state-space modeling and localization-sensitive efficiency.
- Full MaDiNet detector complexity has not been measured end-to-end in this workspace; only the official MambaSAR core module was profiled.
- The current manuscript is still `6/10` under strict review because multi-seed and core ablation results are missing.

## Pending Work

- Review `reports/thread_019f2406_integrated_context.md` before changing the experiment plan; it preserves the former main thread's H100 SSDD sweep, initialization, comparison-method, and proxy-dataset decisions.
- Run the frozen OGSOD seeds 0/1/2 matrix: baseline, no-PFIM, parameter-matched global PFIM, and full PFIM.
- Run seed-0 gamma/write/delta/gamma+write diagnostics.
- Run density, area, and nearest-neighbor stratified TEST analysis for dense tanks.
- Integrate mean/std, paired deltas, and per-stratum results into the paper before raising the acceptance claim.
