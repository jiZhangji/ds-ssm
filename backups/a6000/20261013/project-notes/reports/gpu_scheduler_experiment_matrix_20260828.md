# IG-DS-SSM/PFIM P3 Automatic Experiment Queue

## Fixed protocol

- Baseline: original Mamba-YOLO/VSS backbone.
- Ours: IG-DS-SSM/PFIM inserted at backbone P3, `state_dim=16`.
- Initialization: scratch/random for both methods.
- Seed: 0.
- Optimizer: AdamW, learning rate `2e-4`.
- Module warmup: 0.
- A result is complete only when `eval_test.json` or `result.json` exists.
- The scheduler samples every 60 seconds and launches work only after a GPU has
  remained below 10% utilization and 2048 MiB usage for 900 continuous seconds.
- Existing processes are never stopped. Existing checkpoints are resumed by the
  task runner, and completed result files are never overwritten.

## Main matrix

| # | Task | Dataset | Variant | Image | Batch | Epochs | Current state at deployment |
|---:|---|---|---|---:|---:|---:|---|
| 1 | Detection | SSDD | Baseline | 640 | 16 | 150 | Complete |
| 2 | Detection | SSDD | P3 d16 | 640 | 16 | 150 | Complete |
| 3 | Detection | SAR-AIRcraft1.0 | Baseline | 640 | 32 | 150 | Blocked: prepared data absent on A-D |
| 4 | Detection | SAR-AIRcraft1.0 | P3 d16 | 640 | 32 | 150 | Blocked: prepared data absent on A-D |
| 5 | Detection | SADD | Baseline | 640 | 32 | 200 | Complete |
| 6 | Detection | SADD | P3 d16 | 640 | 32 | 200 | Complete |
| 7 | Detection | AIR-SARShip-2.0 | Baseline | 640 | 32 | 200 | Running on D |
| 8 | Detection | AIR-SARShip-2.0 | P3 d16 | 640 | 32 | 200 | Queued after baseline on D |
| 9 | Detection | SRSDD-v1.0 | Baseline | 640 | 32 | 200 | Blocked: complete prepared data absent on A-D |
| 10 | Detection | SRSDD-v1.0 | P3 d16 | 640 | 32 | 200 | Blocked: complete prepared data absent on A-D |
| 11 | Classification | SAR-ACD | Baseline | 224 | 64 | 200 | Running in existing C queue |
| 12 | Classification | SAR-ACD | P3 d16 | 224 | 64 | 200 | Existing d16 waiter on C |
| 13 | Classification | SAMPLE | Baseline | 224 | 64 | 200 | Existing C queue |
| 14 | Classification | SAMPLE | P3 d16 | 224 | 64 | 200 | Existing d16 waiter on C |
| 15 | Segmentation | FUSAR-Map | Baseline | 512 | 4 | 160 | Running on B |
| 16 | Segmentation | FUSAR-Map | P3 d16 | 512 | 4 | 160 | Running on B |
| 17 | Change detection | Ottawa | Baseline | 192 | 16 | 150 | Complete |
| 18 | Change detection | Ottawa | P3 d16 | 192 | 16 | 150 | Complete |
| 19 | Change detection | Yellow River | Baseline | 192 | 16 | 150 | Complete |
| 20 | Change detection | Yellow River | P3 d16 | 192 | 16 | 150 | Running on A |
| 21 | Change detection | Farmland | Baseline | 192 | 16 | 150 | Complete |
| 22 | Change detection | Farmland | P3 d16 | 192 | 16 | 150 | Existing A queue |
| 23 | Change detection | Sulzberger | Baseline | 192 | 16 | 150 | Existing A queue |
| 24 | Change detection | Sulzberger | P3 d16 | 192 | 16 | 150 | Existing A queue |

## Deployment

- Controller: A (`powerleader-PT6620W`).
- Project root: `/home/powerleader/ig-ds-ssm`.
- Scheduler: `/home/powerleader/ig-ds-ssm/work_train_single/tools/gpu_idle_scheduler.py`.
- Queue: `/home/powerleader/ig-ds-ssm/work_train_single/tools/ig_ds_ssm_experiment_queue.json`.
- Runtime state: `/home/powerleader/ig-ds-ssm/scheduler/state.json`.
- Compact status: `/home/powerleader/ig-ds-ssm/scheduler/status.json`.
- Scheduler log: `/home/powerleader/ig-ds-ssm/scheduler/scheduler.log`.
- Per-job logs: `/home/powerleader/ig-ds-ssm/scheduler/logs/`.
- B/C/D are accessed from A using the dedicated key
  `~/.ssh/id_ed25519_ig_scheduler`.
- E is explicitly disabled until SSH access is verified.

The scheduler currently manages only the authoritative 24-run d16 main matrix.
Existing d32 experiments remain preserved as auxiliary state-dimension evidence
and are not automatically repeated.
