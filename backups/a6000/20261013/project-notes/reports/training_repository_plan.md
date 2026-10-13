# Training Repository Plan

## Recommended Local Names and Paths

Normal training version:

- Name: `work_train_single`
- Path: `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\work_train_single`
- Initialized from: `source_code/server_3090_a`
- Intended future Git branch: `train-single`

Distributed training version:

- Name: `work_train_ddp`
- Path: `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\work_train_ddp`
- Initialized from: `source_code/server_1080ti`
- Intended future Git branch: `train-ddp`

Historical code snapshots:

- `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\server_1080ti`
- `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\server_3090_a`
- `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\server_3090_b`

These historical snapshots must remain unchanged unless explicitly auditing history.

## Why This Layout

This follows the combined plan:

- Plan A: keep two long-lived training lines, `train-single` and `train-ddp`.
- Plan B: keep explicit normal and DDP training entry points inside each line.

Because Git is not currently available in this local shell, the branch plan is represented as two working directories for now. Once Git is available, these should be converted into real branches or separate local repositories.

## How To Use Later

For normal training requests, use:

`source_code/work_train_single`

For DDP/distributed training requests, use:

`source_code/work_train_ddp`

When you ask for a new experiment, specify:

- training line: `single` or `ddp`
- dataset
- model YAML
- epochs
- batch size
- image size
- experiment name
- whether code changes are allowed

Example:

`基于 work_train_ddp，使用 SSDD，配置 Mamba-YOLO-S-DS-SSM_Backbone-SSDD-ex7.yaml，epochs=300，batch=16，imgsz=640，实验名 ssdd_ex23_ddp_rerun_001，开始训练并记录结果。`

## Update Rules

Before future training:

- Work only in `work_train_single/` or `work_train_ddp/`.
- Do not edit `server_1080ti/`, `server_3090_a/`, or `server_3090_b/`.
- Save each run under a unique experiment directory.
- Record source path, code version, command, config, weights, and results.
- Update `database/experiment_ranking.csv` or future experiment summary files after training.

## Current Caveats

- The exact Top 1 runtime uncommitted code mapping is still not fully proven.
- DDP work copy is initialized from the Top 1 source server, but `mamba_yolo_ex23.py`, `tasks.py`, `trainer.py`, and the Top 1 YAML should be manually checked before publication-grade reproduction.
- `work_train_single` is initialized from 3090-A because it is a clean 3090 lineage for continued non-DDP/SAR-related work.
