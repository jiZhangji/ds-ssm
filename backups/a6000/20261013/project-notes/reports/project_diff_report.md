# Project Diff Report

## Baseline

- Official comparison baseline: shared Git HEAD `b26cbda230dfa217f96faee8dc7020db3962f3df`.
- Remote on all servers: `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.
- No server was selected as final/master. This is a recovery analysis only.
- No weights or experiment result files were downloaded or inspected by content.

## Summary

- Files changed relative to official HEAD across any server: `43`.
- Cross-server code/config/doc paths with different hashes or presence: `16`.
- Deleted code files detected relative to HEAD: `0`.
- New untracked code/config/doc files detected: `34`.

## By Server

| Server | Modified tracked files | Untracked code/config/doc files | HEAD |
|---|---:|---:|---|
| `server_1080ti` | `8` | `27` | `b26cbda230df` |
| `server_3090_a` | `6` | `30` | `b26cbda230df` |
| `server_3090_b` | `7` | `30` | `b26cbda230df` |

## Impact Assessment

- Model-structure impact is concentrated in `ultralytics/nn/**`, `ultralytics/nn/tasks.py`, and Mamba-YOLO YAML/model variants.
- Training-flow impact is concentrated in `mbyolo_train*.py`, `ultralytics/engine/trainer.py`, `ultralytics/cfg/default.yaml`, dataset preparation tools, and selective-scan setup changes.
- Inference/validation/result impact is possible where `ultralytics/utils/metrics.py`, validation modules, task parsing, or model definitions differ.
- Most research changes are not committed to Git; recovery must preserve server provenance per file.

## Important Observations

- `server_1080ti` uniquely modifies tracked `selective_scan/setup.py` and `ultralytics/engine/trainer.py`, and contains PFIM-specific scripts/backups.
- `server_3090_b` uniquely modifies tracked `ultralytics/utils/metrics.py`, which can affect reported metrics and final result interpretation.
- `server_3090_a` and `server_3090_b` share many DS-SSM/ex-series files and are closer to each other than to 1080Ti for several tracked files.
- Untracked model variants such as `mamba_yolo_ex*`, `mamba_yolo_PFIM.py`, and DS-SSM YAMLs are likely central to experiment reconstruction.

## Detailed Lists

- Full modified-file inventory: `reports/all_modified_files.md`.
- Matrix by server and cross-server hashes: `reports/code_diff_matrix.md`.
