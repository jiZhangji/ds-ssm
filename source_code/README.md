# Source Code Layout

This directory keeps both historical source snapshots and future training work copies.

## Historical Snapshots

Do not edit these directly unless intentionally auditing history.

- `server_1080ti/`: recovered 1080Ti server snapshot. Important early DS-SSM and DDP experiment lineage.
- `server_3090_a/`: recovered 3090-A snapshot. Important SAR-AIRcraft and SSDD experiment lineage.
- `server_3090_b/`: recovered 3090-B snapshot. Important later PFIM/ex-series lineage and metrics changes.

Current recomputed Top 1 result comes from `server_3090_b`, experiment `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27-gemini-20260603:17:45`, with AP50:95 `0.74022`. The older 1080Ti Top 1 note belongs to the previous ranking pass.

## Working Copies

Use these for future experiments.

- `work_train_single/`: normal single-process training work copy. Initialized from `server_3090_a`.
- `work_train_ddp/`: distributed/DDP training work copy. Initialized from `server_1080ti`.

Planned Git branch names when Git is available:

- `train-single` for `work_train_single/`
- `train-ddp` for `work_train_ddp/`

## Rule

Historical snapshots preserve evidence. Working copies are where future code changes and training preparation should happen.

Default root searches ignore historical snapshots through `.rgignore`. Use `rg -u` when auditing old server source code.
