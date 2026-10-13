# Integrated Context From Thread 019f2406

Source thread: `codex://threads/019f2406-c105-7d63-bd7d-a3b993b2a169`

This file integrates the useful working context from the former main thread into the current workspace. It is meant to preserve experimental decisions, paper strategy, and caveats without requiring future agents to reopen the old thread.

## 1. Project State

- Workspace: `D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm`.
- Mainline: `IG-DS-SSM / PFIM` for lightweight SAR target detection.
- DG-DS-SSM remains exploratory; do not make it the current paper's main contribution.
- Active code should be edited in `source_code/work_train_single/` or `source_code/work_train_ddp/`.
- Historical server snapshots under `source_code/server_*` are evidence archives and should not be overwritten.

## 2. Current Paper Design

Treat the work as a method paper, not a parameter-search report.

Recommended core question:

> Can SAR speckle suppression and target-importance guidance be injected directly into SSM state updates to improve localization-sensitive detection metrics while preserving a lightweight detector?

Recommended contribution structure:

1. `DS-SSM`: feature-space de-speckling, local residual attenuation, and energy-aware state damping.
2. `PFIM`: token-level importance estimation that modulates `gamma`, `write_gate`, and `delta`.
3. Shallow deployment and lightweight analysis: show why high-resolution shallow stages are preferred for SAR small targets.

Important wording caveat:

- In the executed CUDA path, `gamma` and `write_gate` are two factors on the same scan drive.
- `delta` is the separate state-transition control.
- Do not overstate `gamma/write/delta` as three fully independent recurrent update locations.

## 3. Evidence Tiers

### Strongest Controlled Evidence

OGSOD-1.0 single-run matched comparison:

| Model | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| Mamba-YOLO-T | 0.771733 | 0.402109 | 0.423018 |
| IG-DS-SSM/PFIM-T | 0.772630 | 0.409711 | 0.436021 |

Interpretation:

- AP75 gain: `+0.760` percentage points.
- AP50:95 gain: `+1.300` percentage points.
- This supports the localization-quality story, but remains single-seed evidence.

### Best Recovered SSDD Archive

Historical PFIM ex27:

| Experiment | Source | P | R | AP50 | AP50:95 | Params | GFLOPs |
|---|---|---:|---:|---:|---:|---:|---:|
| `PFIM...ex27` | `server_3090_b` | 0.97378 | 0.95219 | 0.98704 | 0.74022 | 5.9648M | 13.2079 |

Interpretation:

- Use as best recovered archive evidence.
- Do not use as the main causal comparison because the historical baseline and ex27 differ in optimizer, batch size, learning rate, warmup, and initialization path.

### H100 SSDD Development Sweep

Development split:

- `742 train`
- `186 val`
- `232 official test`

Current sweep setup remembered from the former thread:

- Single H100.
- `imgsz=640`.
- Global batch `32`.
- AdamW.
- `lr=2e-4`.
- 300 epochs.
- Improved model used 20-epoch module warmup.
- Baseline did not use DS module warmup.
- Last known progress: `9/16` completed.
- TTA was worse than standard inference in completed runs.

Partial single-seed validation result:

| Configuration | VAL AP50:95 | Delta vs baseline |
|---|---:|---:|
| Baseline | 0.548384 | - |
| P2, state_dim=16 | 0.566725 | +0.018341 |
| P3, state_dim=8 | 0.561184 | +0.012800 |
| P3, state_dim=32 | 0.560158 | +0.011774 |
| P4, state_dim=8 | 0.558172 | +0.009788 |

P2-dim16 changed metrics relative to baseline:

| Metric | Baseline | P2-dim16 | Delta |
|---|---:|---:|---:|
| AP50 | 0.931291 | 0.915125 | -1.62 pp |
| AP75 | 0.588958 | 0.668575 | +7.96 pp |
| AP50:95 | 0.548384 | 0.566725 | +1.83 pp |

Interpretation:

- The gain is mainly localization-sensitive, especially AP75.
- P2-dim16 is the temporary best, not a final conclusion.
- Finish remaining 7 configurations before locking placement/state_dim.

Remaining H100 SSDD sweep jobs:

| Placement | Remaining state_dim | Count |
|---|---|---:|
| P4 | 32 | 1 |
| P5 | 8, 16, 32 | 3 |
| P2+P3 | 8, 16, 32 | 3 |
| Total |  | 7 |

## 4. Initialization And Pretraining Conclusions

The former thread corrected several conflicting interpretations. Use the latest interpretation below.

### Current H100 Sweep

- Baseline and improved models use `--weights none`.
- They are random-initialized/scratch experiments.
- The 20-epoch module warmup is not pretraining.

### Historical SSDD Runs

- Historical PFIM ex27 did execute `model.load("yolov8n.pt")`.
- Rechecking the same load path matched only about `6/519` items.
- Therefore ex27 is technically partial-transfer initialized, but practically most parameters remained random.
- Do not relabel old experiments as clean scratch.

### Fair Paper Protocols

Use two evidence layers:

1. Scratch controlled experiments:
   - Baseline and ours both `weights=none`.
   - Same split, seed, optimizer, batch, augmentation, and epoch count.
   - Prefer `module_warmup=0` for the strictest scratch comparison.

2. Pretrained final performance, if compute allows:
   - Prefer real Mamba-YOLO COCO detection weights or a properly matched pretraining route.
   - Baseline and ours must receive the same pretraining opportunity.
   - New DS-SSM/PFIM parameters should be initialized fairly and reported.

Original Mamba-YOLO note:

- The official Mamba-YOLO paper/repo reports COCO training from scratch, not ImageNet backbone pretraining.
- If ImageNet pretraining is introduced, both baseline and ours should be pretrained under the same ImageNet protocol.

## 5. Experiment Plan For A Strong Paper

Minimum paper-strength plan:

1. Finish SSDD placement/state_dim seed-0 sweep.
2. Select the final structure using validation only.
3. Run core models with seeds `0,1,2`:
   - Mamba-YOLO-T baseline.
   - DS-SSM only / no-PFIM.
   - Parameter-matched global importance control.
   - Full IG-DS-SSM/PFIM.
4. Run PFIM path diagnostics:
   - gamma only.
   - write only.
   - delta only.
   - gamma + write.
   - full PFIM.
5. Add conventional de-speckle preprocessing control, such as Lee or fixed filtering, to prove the method is not replaceable by simple preprocessing.
6. Report Params, GFLOPs, FPS/latency, and memory with a corrected profiler.
7. Use official test only after structure and training protocol are fixed.

Recommended final SSDD flow:

1. Development: `742 train / 186 val` for structure selection.
2. Final: merge train+val into official `928 train`.
3. Evaluate once on `232 official test`.
4. Do not use TTA in the main table unless new evidence reverses the current negative trend.

## 6. Final External Comparisons

Suggested controlled main table:

| Method | Role |
|---|---|
| YOLOv8n | Common lightweight CNN detector |
| YOLOv10s or closest-size YOLOv10 variant | Newer real-time CNN detector, closer parameter scale |
| RT-DETR-R18 | Transformer detector route |
| Mamba-YOLO-T | Most important direct baseline |
| MaDiNet | Closest SAR + Mamba strong reference |
| IG-DS-SSM/PFIM-T | Ours |

Report:

- P
- R
- AP50
- AP75
- AP50:95
- Params
- GFLOPs
- FPS or latency

Literature table:

- Faster R-CNN.
- YOLOv5n/YOLOv5s.
- YOLO-SRBD.
- DiffDet4SAR.
- CPN-YOLO.
- SSD-YOLO.
- Other recent SSDD methods.

Comparison caveat:

- SSDD literature uses different splits, preprocessing, pretrained weights, and evaluators.
- Only re-rank methods actually reproduced under the same protocol.
- Self-reimplemented methods must be labeled `Our reimplementation`, not official reproduction.
- Methods without reliable official code should be cited as reported literature values.

## 7. Multi-Dataset Proxy Experiment Strategy

The proposed low-cost structure-search idea is reasonable if treated as multi-fidelity search.

Principle:

- Do not randomly sample each dataset naively.
- Use stratified, fixed proxy subsets to avoid misleading rankings.

Suggested proxy sizes:

| Dataset | Proxy training subset |
|---|---:|
| SSDD | Use full development train; it is small |
| OGSOD | Stratified 20%-25% |
| SAR-ShipDataset | Stratified 10%-15% |
| SAR-Aircraft | About 20%, sampled by original large-image ID |

Proxy experiment candidates:

- Mamba-YOLO-T baseline.
- The top three SSDD candidate structures.

Proxy training:

- 80-120 epochs.
- Fixed seed.
- Fixed validation set.
- Scale module warmup with total epochs, e.g. about 7 warmup epochs for a 100-epoch proxy run.
- Do not keep 20 warmup epochs in a 50-epoch proxy run.

Final selection:

- Choose one final structure that is stable across datasets.
- Do not choose a different architecture for each dataset.
- If candidate ranking changes across proxy subsets, enlarge proxy data before concluding.

## 8. Server And Backup Notes

Known server notes from the former thread:

- A full `ds-ssm` directory snapshot had been uploaded to ModelScope around 2026-08-17, about `25.2 GB`.
- GitHub branch mentioned: `codex/ssdd-h100-sweep`, last known commit `746c823`.
- Server-side source/data should not be deleted until verifying:
  - post-snapshot changes;
  - datasets, especially SSDD;
  - conda environment or dependency export;
  - server-local patches such as `ssdd_server_local_fix.patch`;
  - training outputs and best checkpoints.

Previous GPU status snapshot from the thread:

- `1080Ti-2`: mostly idle; suitable for smoke tests and small proxy runs.
- `3090-5`: about 3.6 GB used by two processes.
- `3090-1`: about 11 GB used.
- `3090-3`: two GPUs with about 24 GB and 9 GB used.
- `3090-4`: about 18.8 GB used and high utilization.
- A6000 machines were busy.

This status is temporal and must be rechecked before scheduling new runs.

## 9. Current Practical Priority

Do not add new modules first. Prioritize:

1. Finish and verify the current SSDD/H100 sweep state.
2. Complete OGSOD three-seed matrix and factor diagnostics.
3. Establish strict scratch and, if possible, pretrained protocols.
4. Recompute complexity/speed with a reliable profiler.
5. Update the paper using only completed controlled evidence.

The paper does not need to claim absolute SOTA. A safer claim is:

> IG-DS-SSM/PFIM improves localization-sensitive SAR detection metrics under matched protocols while preserving a compact Mamba-YOLO-scale detector.
