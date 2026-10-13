# Current Research Mainline

## Mainline Judgment

Current evidence supports treating `IG-DS-SSM / PFIM` as the active research mainline, not the compression-literature local-scale/GDN proposal.

This conclusion is based on code and experiment evidence, not on the missing `docs/` documents:

- `source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py` implements `PFIM`, `DeSpeckleSSM`, and `DS_VSSBlock`.
- Current PFIM experiments occupy the top of `database/experiment_ranking_recomputed.csv`; rank 1 is `PFIM...ex27` with mAP50-95 `0.74022`.
- The latest DDP reproduction run `codex3` completed and is archived, but ranks lower at mAP50-95 `0.73028`.

## DS-SSM Role

DS-SSM is the SAR-aware state-space replacement inside Mamba-YOLO blocks. In current code it performs:

- input de-biasing / despeckle residual adjustment: `x_tilde = x - lambda_s * residual`;
- energy-aware damping through `gamma`;
- selective scan with damped input;
- residual output via `D * x_tilde` outside the CUDA scan path.

## PFIM Role

PFIM estimates token-level importance `I_t`. Current `mamba_yolo.py` uses it to modulate:

- `gamma`: memory retention / damping strength;
- `write_gate`: state write strength;
- `delta`: time-step / transition sensitivity.

PFIM also uses local standard deviation when `hw_shape` is available. This is implemented as local statistics inside PFIM, not the same as adopting the separate local-scale-prior/GDN candidate design as the mainline.

## gamma / write_gate / delta

- `gamma`: controls how much previous state is retained after energy-aware and importance-aware damping. Current code applies `min_retain + (1 - min_retain) * sigmoid(...)`.
- `write_gate`: multiplies `u` before scan, changing how strongly current token information is written into the state.
- `delta`: modulates the state transition step by adding PFIM importance into `delta_logits`.

## local scale prior / GDN-style gain control

Current status: `candidate_extension`.

Reason:

- Requested candidate document is not available in workspace.
- Code search did not establish a distinct GDN-style gain normalization module in the active `work_train_single` / `work_train_ddp` main module.
- PFIM already computes local std, but that should not be equated with the full candidate local-scale/GDN proposal without document review and experiments.

## Normal and DDP Repositories

Normal and DDP repositories should share the same model structure for comparable experiments. They may differ in trainer/launcher behavior:

- `work_train_ddp` contains DDP-specific trainer changes: `find_unused_parameters=True` and `_dsssm_freeze_warmup`.
- Model files currently differ by hash between `work_train_single` and `work_train_ddp`, so they should be audited before claiming identical model structure.

## Best Current Baseline

For model-design continuation, the strongest code/experiment baseline is:

- Code lineage: `source_code/server_3090_b` / PFIM ex27 line.
- Experiment: `ssdd/PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27-gemini-20260603:17:45`.
- Metric: mAP50-95 `0.74022`.

For DDP training operations, use:

- `source_code/work_train_ddp`, but align its model module with the ex27 mainline before new publication-grade experiments.

Git CLI is unavailable locally; both work repos show `.git/HEAD` on `main` and initial clone commit `b26cbda230dfa217f96faee8dc7020db3962f3df` from `.git/logs/HEAD`.
