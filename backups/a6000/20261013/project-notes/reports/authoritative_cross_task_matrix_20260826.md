# IG-DS-SSM/PFIM Cross-Task Experiment Matrix

This file records the authoritative matrix from the user's Phase 0-4 protocol.

## Fixed Method Protocol

- Model: Mamba-YOLO baseline versus Mamba-YOLO + IG-DS-SSM/PFIM at backbone P3.
- P3 state dimension: 16 for the current screening run.
- Initialization: random (`weights=none`).
- Module warmup: 0 epochs.
- Seed: 0.
- Detection screening: 640 input, batch 16, AdamW, 150 epochs, patience 30.
- Model selection: validation split; final metric: held-out test split.

## Authoritative Datasets

| Phase | Task | Datasets | Baseline | Ours |
|---|---|---|---|---|
| Phase 1 | Detection | SSDD; SAR-AIRcraft1.0 | Mamba-YOLO | Mamba-YOLO + IG-DS-SSM/PFIM(P3) |
| Phase 2 | Detection | AIR-SARShip-2.0; SRSDD-v1.0; SADD | Mamba-YOLO | Mamba-YOLO + IG-DS-SSM/PFIM(P3) |
| Phase 3 | Classification | SAR-ACD; SAMPLE | VMamba | VMamba + IG-DS-SSM/PFIM(P3) |
| Phase 3 | Segmentation | FUSAR-Map | VMamba-Seg | VMamba-Seg + IG-DS-SSM/PFIM(P3) |
| Phase 4 | Change detection | Ottawa; Yellow River; Farmland; Sulzberger | Mamba-CD, or UNet-CD fallback | Baseline + IG-DS-SSM/PFIM(P3) |

## Scope Boundary

OGSOD and SAR-Ship-Dataset are not part of this authoritative matrix. Their previous runs remain historical evidence only and must not be included in the current main comparison table.

## Current Execution

- SSDD: A6000-D, baseline/P3 serial queue, resumed from the checkpoint after a non-experimental Ray Tune callback failure.
- SAR-AIRcraft1.0: 3090-4, baseline/P3 serial queue, scratch initialization.
- Logs and checkpoints are stored on the respective server experiment roots.
