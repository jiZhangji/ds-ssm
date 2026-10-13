# Remote GPU Validation and Launch Record (2026-08-27)

## Safety Rules Applied

- No running experiment was terminated, restarted, or overwritten.
- New shared-task runs use `corrected_shared_v2_sd32_20260827`.
- CUDA dependencies were not installed on Windows.
- GPU smoke tests ran on the existing 1080Ti environment and project.

## GPU Smoke Validation

Host: `172.26.193.188` (`yf`, 2 x GTX 1080 Ti)

Project: `/home/zhangji/ds_ssm_sarmix_1080/code_cda7901/work_train_single`

Environment: `/home/zhangji/miniconda3/envs/mamba-yolo`

Validated for both `baseline` and `ours` (`P3`, `state_dim=32`):

- classification
- semantic segmentation
- change detection
- CUDA forward and backward
- AMP
- validation and test evaluation
- `last.pt` checkpoint resume from epoch 1 to epoch 2
- `best.pt` and `result.json` generation

Persistent smoke logs:

- `/home/zhangji/ds_ssm_sarmix_1080/logs/shared_v2_smoke_baseline.log`
- `/home/zhangji/ds_ssm_sarmix_1080/logs/shared_v2_smoke_ours.log`

## Preserved Running Experiments

### A6000-D

- Experiment: `corrected_v2_sadd_sd32_b32`
- Current stage at the audit: SADD baseline, epoch 128/200
- Queue: baseline then P3
- Configuration: 640 px, batch 32, scratch, no module warmup, `state_dim=32` for P3
- Root: `/home/powerleader/ig-ds-ssm`

### RTX 3090-4

- Experiment: `ig_dsssm_p3_phase1_aircraft_sd32_v1`
- Current stage at the audit: SAR-AIRcraft P3, epoch 143/150
- Configuration: 640 px, batch 16, scratch, no module warmup, `state_dim=32`
- Result root: `/mnt/disk1/zhangji/ig_dsssm_cross_task/experiments/detection`

## Newly Launched Corrected-v2 Queues

All queues compare baseline and P3 under the same per-dataset settings.

| Host | PID | Datasets | Image size | Batch | Epochs | Queue log |
|---|---:|---|---:|---:|---:|---|
| A6000-A | 168428 | Ottawa, Yellow River, Farmland, Sulzberger | 192 | 16 | 150 | `/home/powerleader/ig-ds-ssm/logs/corrected_shared_v2_sd32_20260827_change.queue.log` |
| A6000-B | 325966 | FUSAR-Map | 512 | 4 | 160 | `/home/powerleader/ig-ds-ssm/logs/corrected_shared_v2_sd32_20260827_fusar.queue.log` |
| A6000-C | 248712 | SAR-ACD, SAMPLE | 224 | 64 | 200 | `/home/powerleader/ig-ds-ssm/logs/corrected_shared_v2_sd32_20260827_classification.queue.log` |

Formal result root on each A6000 host:

`/home/powerleader/ig-ds-ssm/experiments/corrected_shared_v2_sd32_20260827`

## Batch-size Comparability

Batch size is held constant between baseline and P3 within each dataset. Different task families use different batches because 512-pixel dense segmentation, 224-pixel classification, and Siamese change detection have substantially different activation memory. Cross-task metrics are not ranked directly against one another, so forcing one physical batch across all tasks would reduce throughput without improving fairness. Any same-dataset comparison must keep the batch and all other training settings identical.

## State-dimension 16 Follow-up (2026-08-28)

Historical controlled SSDD evidence at batch 16 and seed 0 favors `state_dim=16`:

| State dim | Recall | AP75 | AP50:95 |
|---:|---:|---:|---:|
| 16 | 83.516% | 33.756% | 42.991% |
| 32 | 80.220% | 28.646% | 40.596% |

New matched follow-up runs use the same corrected-v2 manifests, seed, batch,
image size, epoch limit, and optimization settings as the corresponding d32
runs. Existing baselines are reused rather than retrained.

- A6000-D: `corrected_v2_sadd_sd16_b32_seed0`, P3 only, running immediately.
- A6000-A: change-detection d16 P3 queue started; the existing d32 queue remains running.
- A6000-B: FUSAR-Map d16 P3 queue started; the existing d32 queue remains running.
- A6000-C: the first d16 classification attempt overlapped the active batch-64
  d32 baseline and exited with CUDA OOM. The d32 task was not interrupted. A
  replacement waiter (PID 257300) will start the same batch-64 d16 P3 queue only
  after the d32 wrapper PID 248712 exits.

The old SADD d16 seed-2 result is not a strict comparator for the new d32 seed-0
run because its seed and prepared-data artifact differ.

## A6000-F SSDD d16 / Batch-32 Follow-up (2026-08-28)

Host `powerleader-a6000-f` has two RTX A6000 GPUs. The prescribed project root
was initially absent, so code, the existing environment, and the frozen SSDD
source were copied over the `10.6.2.0/24` LAN from the other A6000 hosts to
`/home/powerleader/ig-ds-ssm`.

The first concurrent launch (`ssdd_scratch_p3_sd16_b32_v1`) exited before the
first optimizer step because both processes were not isolated at process start
and each observed insufficient free CUDA memory. No checkpoint was produced.
The preserved retry uses explicit per-process `CUDA_VISIBLE_DEVICES` isolation:

| Physical GPU | Variant | Experiment ID | State dim | Batch | Image size | Epochs |
|---:|---|---|---:|---:|---:|---:|
| 0 | Mamba-YOLO baseline | `ssdd_scratch_p3_sd16_b32_v2` | 0 | 32 | 640 | 150 |
| 1 | IG-DS-SSM/PFIM P3 | `ssdd_scratch_p3_sd16_b32_v2` | 16 | 32 | 640 | 150 |

Both runs use AdamW at `2e-4`, seed 0, `weights=none`, no module warmup, and the
same deterministic 742/186/232 train/validation/test split. At launch, baseline
used about 33.6 GiB and P3 about 32.0 GiB; both GPUs were at 100% utilization.
