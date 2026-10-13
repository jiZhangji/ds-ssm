# IG-DS-SSM Environment Scan

Scan date: 2026-08-25  
Workspace: `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`  
Branch: `codex/ssdd-h100-sweep`  
HEAD: `2d13bc01b79bd0560460a472504e4973535f78e1`

## Scope

This is an inventory-only scan. No training was started, no conda environment was created, and no training code was modified. The detailed machine-readable inventory is in `dataset_inventory.json`.

## Dataset Findings

Detection data that is already close to the current YOLO pipeline:

- SSDD: 928 train and 232 official test images, one class, COCO annotations.
- SAR-AIRcraft-1.0: 13,976 train, 1,923 val, and 2,989 test images, seven classes, COCO annotations.
- SAR-Ship-Dataset: 39,729 image/TXT pairs, one class, YOLO annotations.
- SADD candidate: 2,966 BMP/TXT pairs, but the archive is named `SAR-aircraft.zip`; it must not be reported as SAR-AIRcraft-1.0.

Detection data that exists but needs conversion:

- AIR-SARShip-2.0: 600 TIFF entries and 300 XML annotations; XML parsing found 2,040 ship instances.
- SRSDD-V1.0: 532 train and 134 test PNG/TXT pairs; labels contain polygon/rotated-box information and cannot be silently treated as horizontal YOLO boxes.

Other task data:

- SAR-ACD: 2,522 JPG images in five aircraft class directories.
- MSTAR-SOC: 5,172 images, 2,761 train and 2,439 test, ten-class SOC protocol.
- SAMPLE: 5,380 PNG images split across decibel/qpm and real/synth; class and split policy still need to be frozen.
- FUSAR-Map: 610 SAR TIFF and 610 label TIFF files at 1024x1024; masks are color-coded and need an explicit class map.
- Ottawa, Yellow River, and Farmland: one image pair plus binary ground truth each; patch extraction and a change-detection trainer are required.
- Sulzberger: not found in either scanned local data root.

## Code Audit

The active implementation is:

- Module file: `source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py`.
- `PFIM` estimates token-level importance from local trend and local variation.
- `DeSpeckleSSM` uses the importance to modulate gamma, write, and delta when enabled.
- `DS_VSSBlock` defaults to PFIM enabled with `state_dim=16`.

The following topology candidates were found and must remain distinct:

| Candidate | Location | Actual placement | Notes |
|---|---|---:|---|
| `Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml` | active single-GPU config | P2 | Replaces only the P2 backbone stage with `DS_VSSBlock`; the file comment identifies it as the historical IG-DS-SSM/PFIM backbone. |
| `Mamba-YOLO-T-DS-SSM_Backbone-SSDD.yaml` | active single-GPU config | P2+P3 | Replaces both P2 and P3 with `DS_VSSBlock`. |
| `run_sarmix_proxy.py` generated `full_p3` | active experiment runner | P3 | Uses `DS_VSSBlockSweep` with state dimension supplied by the command; this is the clean dynamic P3 candidate. |
| `run_sar4_full_suite.py` generated `full_p3` | active experiment runner | P3 | Generates a baseline/full pair for SSDD, OGSOD, SAR-Ship, and SAR-AIRcraft; defaults to state dimension 32 and module warmup 0. |
| `Mamba-YOLO-T-DS-SSM-OGSOD-pfim-full.yaml` | OGSOD ablation config | P2 | Despite the experiment context, the only `DS_VSSBlock` is at P2. It is not a P3 config. |
| `Mamba-YOLO-T-DS-SSM_head-SSDD.yaml` | historical config | P2+P3 plus DSCA head blocks | Contains extra head changes and must not be used as the clean DS-SSM/PFIM comparison. |

Additional OGSOD ablation modules exist: no-PFIM, PFIM gamma-only, write-only, delta-only, gamma+write, global importance, and no-2D-std. These are useful for a controlled ablation matrix, but they are not interchangeable P3 definitions.

## Existing Results and Weights

Most useful controlled result recovered from the local records:

| Experiment | Placement | AP50 | AP75 | AP50:95 | Evidence |
|---|---:|---:|---:|---:|---|
| Mamba-YOLO-T baseline on OGSOD | baseline | 0.771733 | 0.402109 | 0.423018 | controlled single run |
| IG-DS-SSM/PFIM-T on OGSOD | P3 | 0.772630 | 0.409711 | 0.436021 | controlled single run |

The P3 run improves AP75 by 0.760 percentage points and AP50:95 by 1.300 percentage points in that controlled record. It is still a single seed and is not multi-seed evidence.

The historical SSDD archive contains a stronger but less controlled PFIM record: AP50 `0.98704`, AP50:95 `0.74022`, approximately 5.9648M parameters and 13.2079 GFLOPs. Its training settings and code lineage are not fully matched to the controlled baseline, so it must not be used as the main causal ablation claim.

The local `output/` directory contains many evaluation JSON files for `sarmix_scratch_d16_v1`, including baseline, DS-only, PFIM-only, global, and full placements. Their JSON points to remote checkpoint paths such as `/mnt/disk1/zhangji/...`; the corresponding checkpoint files are not present in the local workspace. The overall sarmix metrics are unusually low and should be treated as historical exploratory records pending protocol and dataset verification, not as final paper results.

Local checkpoint files are mainly archived SSDD/SAR-Aircraft weights under `weights/`, including Mamba-YOLO-T baseline and historical DS-SSM/PFIM runs. No local checkpoint was found that proves all four sarmix datasets can be directly evaluated from this workstation.

## Server Scan

The server scan used read-only `nvidia-smi` and non-interactive SSH checks. No process was terminated.

| Host | GPU status | Environment status |
|---|---|---|
| `powerleader-a6000-c` | RTX A6000, 49,140 MiB total, 977 MiB used, 0% utilization | `/home/powerleader/miniconda3` exists but `python` was not on the non-interactive PATH |
| `powerleader-a6000-d` | RTX A6000, 49,140 MiB total, 672 MiB used, 0% utilization | `python` and `conda` not found on the non-interactive PATH |
| `powerleader-a6000-b` | RTX A6000, 47,945 MiB used, 99%; Ray process uses about 47 GiB | busy; do not use |
| `172.26.150.72` | RTX 3090, 12,449 MiB used, 96%; Python process active | busy |
| `172.26.133.16` | RTX 3090, 19,283 MiB used, 100%; Python process active | busy |
| `172.26.151.143` | two RTX 3090 cards, both heavily used | busy |
| `172.26.150.36` | SSH connection timed out | unknown; do not assume free |
| `172.26.193.188` | two GTX 1080 Ti cards, 167 MiB and 2 MiB used, 0% | system Python 3.6.9 has no torch; `/home/zhangji/miniconda3` exists but was not activated |

The only clearly idle candidates from this scan are A6000 C, A6000 D, and the two 1080 Ti cards. Before any training, their conda activation and PyTorch/CUDA environment must be checked interactively or by absolute environment path.

## Readiness Assessment

1. The authoritative placement is confirmed as the dynamic `full_p3` topology generated by `run_sarmix_proxy.py`/`run_sar4_full_suite.py`: complete DS-SSM plus PFIM at the P3/8 backbone stage, with all other backbone stages left as original VSSBlock stages. The first reproduction target uses the recovered protocol `state_dim=16`, seed 0, random initialization, and no module warmup.
2. SSDD and SAR-AIRcraft-1.0 are the safest first datasets because their splits and formats are already close to the current detector pipeline.
3. SAR-ACD, MSTAR, and SAMPLE are classification datasets; the current active project is primarily a detector and does not yet contain a completed VMamba/IG-DS-SSM classification experiment path.
4. FUSAR-Map and the change-detection datasets require separate task adapters. They should not be put into `run_sar4_full_suite.py` as if they were object detection datasets.
5. No experiment should be launched from this scan alone. The next decision is the authoritative P3 candidate and the first detection-only reproduction set.

## Current Git Hygiene

The worktree was already dirty before this scan, with user and historical changes across reports, outputs, configs, and tools. No existing changes were reverted or cleaned. The only files added by this scan are `dataset_inventory.json` and this report.
