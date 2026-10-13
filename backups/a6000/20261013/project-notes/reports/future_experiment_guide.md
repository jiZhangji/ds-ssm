# Future Experiment Guide

## 1. Recommended Main Code Location

Recommended working baseline for new experiments:

`source_code/server_1080ti`

Reason:

- Current Top 1 experiment comes from `server_1080ti`.
- Top 1 score is `0.71534`.
- Top 1 experiment is unique in the ranking, not a three-server duplicate group.
- It uses the ex23/DDP SSDD line, which is also represented in recovered code conflicts and experiment names.

Important caveat:

Do not treat `source_code/server_1080ti` as the only valuable historical version. It is the recommended starting point for reproduction of the current best result, not a final merged codebase.

## 2. Best Experiment Configuration

Best experiment:

`ssdd/ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602:15:40`

Source server:

`server_1080ti`

Recovered local config:

`experiments/server_1080ti__ssdd__ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602_15_40/args.yaml`

Key training settings:

- Model config: `/home/zhangji/Mamba-YOLO/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml`
- Dataset: `datasets/SSDD_YOLO/ssdd.yaml`
- Epochs: `300`
- Batch: `16`
- Image size: `640`
- Devices: `0,1`
- Optimizer: `SGD`
- Seed: `0`
- AMP: `true`

Recovered weights:

- `weights/server_1080ti__ssdd__ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602_15_40__best.pt`
- `weights/server_1080ti__ssdd__ddp_mambayolo_s_ssdd_backbone_ds-ssm_300e_16bsx2-ex11-ex23-gemini-20260602_15_40__last.pt`

## 3. Best Experiment Code Version

Recommended code version for reproducing Top 1:

`source_code/server_1080ti`

Known Git commit:

`b26cbda230dfa217f96faee8dc7020db3962f3df`

Uncommitted runtime mapping:

`UNKNOWN`

Interpretation:

The experiment name strongly suggests the `ex23` line, but the exact runtime file state cannot be proven from current evidence. Before publishing or continuing from this result, manually verify:

- `ultralytics/nn/modules/mamba_yolo_ex23.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/engine/trainer.py`
- `mbyolo_train.py`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml`

## 4. Recommended Version for Future Experiments

Use this practical rule:

- For reproducing the current best SSDD result: start from `source_code/server_1080ti`.
- For continuing PFIM/ex24-ex29 experiments: start from `source_code/server_3090_b`.
- For SAR-AIRcraft follow-up experiments: start from `source_code/server_3090_a`, then compare against `server_3090_b`.

Do not merge versions automatically. First create a clean branch or copy from the chosen source snapshot, then record:

- source server snapshot
- Git commit
- modified files used
- config file
- dataset path
- command
- output directory

## 5. Historical Versions to Keep

Keep all three source snapshots:

- `source_code/server_1080ti`: best-ranked SSDD experiment, unique trainer/selective-scan differences, PFIM backups.
- `source_code/server_3090_a`: important 3090 lineage, SAR-AIRcraft experiment line, shared ex-series development.
- `source_code/server_3090_b`: important later 3090 lineage, PFIM/ex27/ex29 experiments, unique `ultralytics/utils/metrics.py` change.

Also keep:

- `reports/code_conflict_report.md`
- `reports/code_diff_matrix.md`
- `reports/experiment_code_mapping.md`
- `downloads/experiment_recovery/recovery_manifest.json`

These files are provenance records and should be treated as part of the research archive.

## 6. Experiments With Low Future Value

Low-value experiments should not be deleted, but they should not drive future research decisions:

- Experiments marked `unknown` or interrupted in `reports/experiment_quality_report.md`.
- Experiment-like `weights/` subdirectories that were indexed separately.
- Runs without `results.csv`.
- Runs without both `best.pt` and `last.pt`.
- Repeated duplicate groups where another server has the same config and equal or better score.
- Debug runs under `output_dir/debug` unless needed for environment verification.

Recommended action:

Archive their metadata, but avoid downloading more artifacts unless a later audit shows they contain unique evidence.

## 7. Experiment Management Rules

Before every new experiment:

- Commit or snapshot the code.
- Record the exact code version in a run manifest.
- Copy the final config into the experiment directory.
- Save the training command.
- Use a unique experiment name.
- Never reuse an old output directory.
- Never edit files inside a completed experiment directory.

Each experiment directory should contain:

- `args.yaml`
- `run_command.txt`
- `code_version.txt`
- `source_snapshot.txt`
- `results.csv`
- `train.log`
- `weights/best.pt`
- `weights/last.pt`

## 8. Experiment Naming Convention

Recommended format:

`{dataset}__{model}__{module}__{change}__{epochs}e__{batch}bs__{imgsz}img__{date}_{time}__{short_commit}`

Example:

`ssdd__mbyolo_s__ds_ssm_ex23__ddp__300e__16bsx2__640img__20260605_2100__b26cbda`

Rules:

- Use only letters, numbers, `_`, and `-`.
- Avoid `:` in new names.
- Include dataset.
- Include model scale.
- Include key module or ablation name.
- Include batch and image size.
- Include date/time.
- Include Git short commit or snapshot ID.

## 9. Result Recording Convention

Maintain one row per experiment in `database/experiment_summary.csv` or a future database table.

Required fields:

- experiment_name
- date_time
- source_code_version
- git_commit
- dirty_files_hash
- dataset
- model_config
- train_command
- epochs
- batch
- image_size
- optimizer
- seed
- precision
- recall
- mAP50
- mAP50_95
- best_pt_path
- last_pt_path
- results_csv_path
- notes

After each experiment:

- Add the row immediately.
- Save `results.csv`.
- Save `args.yaml`.
- Save `train.log`.
- Record whether the run completed normally.
- Record whether the result is selected, duplicate, failed, or archived.

## Immediate Next Step

Before launching new experiments, manually confirm the Top 1 reproduction code path:

1. Compare `source_code/server_1080ti/ultralytics/nn/modules/mamba_yolo_ex23.py` with the Top 1 config and `tasks.py`.
2. Confirm that `Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml` resolves to the intended module.
3. Run a small dry validation or short training smoke test in a new output directory.
4. Only then create the first new experiment branch/snapshot.
