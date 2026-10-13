# Experiment Execution Log

## 2026-08-25

- Completed the read-only dataset, code, checkpoint-index, GPU, and environment scan.
- Confirmed the authoritative model topology as `full_p3`: DS-SSM plus PFIM at P3/8 only.
- Confirmed the first reproduction protocol as state dimension 16, seed 0, random initialization, and module warmup 0.
- Selected `powerleader-a6000-d` as the primary Phase 1 host because its RTX A6000 was idle and it had about 1.7 TB free disk space.
- Created the isolated environment prefix `/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env` by cloning the existing `pt` environment. Existing environments were not modified.
- Uploaded `source_code/work_train_single` to `/home/powerleader/ig-ds-ssm/work_train_single`.
- Direct pip installation failed because the server DNS could not resolve package indexes. No training was started.
- Found a complete local dependency environment at `/home/powerleader/project/envs/scaf-grpo`, but it lacks selective-scan and uses PyTorch CUDA 12.4 while the system compiler is CUDA 11.7. It will be used only as an offline dependency source, not modified.
- Added the missing `sm_86` build target to `selective_scan/setup.py` for RTX 3090/A6000 compatibility.
- Prepared `tools/queue_sarmix_p3_when_idle.sh` for the first reproducible detection phase. It runs only `baseline` and the locked `full_p3` model, preserves the recovered protocol, and waits for GPU 0 on the 3090 host to become idle before starting.
