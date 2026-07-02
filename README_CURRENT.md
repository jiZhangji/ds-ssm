# Current Project Entry

This repository is a recovered research workspace for Mamba-YOLO, DS-SSM, and PFIM experiments on SAR target detection. It contains active working code, historical server snapshots, recovered experiments, weights, reports, and paper drafts.

Use this file as the first entry point before broad searching.

## Current Mainline

The active research mainline is:

**IG-DS-SSM / PFIM for lightweight SAR ship detection.**

Core idea:

- `DS-SSM` injects SAR-aware de-speckling and energy-aware damping into the state-space update.
- `PFIM` estimates token-level importance.
- PFIM modulates `gamma`, `write_gate`, and `delta` inside the DS-SSM update.

Current best recovered SSDD result:

| Experiment | Source | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| `PFIM...ex27` | `server_3090_b` | 0.97378 | 0.95219 | 0.98704 | 0.74022 | 5.9648M | 13.2079 |

The DDP reproduction run `ddp_PFIM...ex27-codex3` completed successfully, but its best validation AP50:95 is `0.73028`, below the recovered `server_3090_b` ex27 result.

## Read First

For research or paper work:

1. `README_CURRENT.md`
2. `HANDOFF_FOR_NEXT_MODEL.md`
3. `reports/current_research_mainline.md`
4. `reports/experiment_ranking_recomputed.md`
5. `reports/madinet_ds_ssm_complexity_report.md`
6. `reports/lightweight_detection_comparison_madinet_datasets.md`
7. `paper_draft/aaai_ds_ssm_paper_en.tex`

For code work:

1. `source_code/README.md`
2. `source_code/work_train_ddp/ultralytics/nn/modules/mamba_yolo.py`
3. `source_code/work_train_ddp/ultralytics/nn/modules/__init__.py`
4. `source_code/work_train_ddp/ultralytics/nn/tasks.py`
5. `source_code/work_train_ddp/ultralytics/cfg/models/mamba-yolo/`

For experiment lookup:

1. `database/experiment_ranking_recomputed.csv`
2. `reports/experiment_ranking_recomputed.md`
3. `database/current_codex3_performance_comparison.csv`
4. `reports/current_codex3_experiment_report.md`

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
- Current paper mainline: IG-DS-SSM / PFIM, not DG-DS-SSM.
- MaDiNet remains a strong AP50 reference; this project should emphasize lightweight SAR-aware state-space modeling and near-SOTA accuracy.
- Full MaDiNet detector complexity has not been measured end-to-end in this workspace; only the official MambaSAR core module was profiled.

## Pending Work

- Confirm exact code/config mapping for `PFIM...ex27`.
- Align `work_train_single` and `work_train_ddp` model modules before publication-grade DDP experiments.
- Run PFIM branch ablations: gamma only, write gate only, delta only, and full PFIM.
- Re-run a clean fixed test command for the ex27 best checkpoint and save the command/log.
