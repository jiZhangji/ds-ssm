# Source Code Recovery

## Scope

- Source snapshots were downloaded into separate local directories; no merge was performed.
- Servers were accessed read-only by SFTP.
- Excluded from download: `output_dir/`, `datasets/`, `weights/`, `runs/`, common weight/media/log suffixes, and non-Git files larger than 5MB.
- Experiment recovery was not performed.

## Local Repository Structure

```text
source_code/
  server_1080ti/
  server_3090_a/
  server_3090_b/
reports/
inventory/
database/
downloads/
```

## Download Statistics

| Server | Host | Files downloaded | Bytes downloaded | Skipped files | Skipped dirs | Error |
|---|---|---:|---:|---:|---:|---|
| `server_1080ti` | `172.26.193.188` | `316` | `12638913` | `6` | `29` | `` |
| `server_3090_a` | `172.26.150.36` | `310` | `12456811` | `5` | `27` | `` |
| `server_3090_b` | `172.26.133.16` | `313` | `12607822` | `5` | `27` | `` |

## Version Notes

- `server_1080ti`: preserved as an exploratory/PFIM-heavy code version with unique trainer/selective-scan modifications and PFIM files.
- `server_3090_a`: preserved as an important 3090 lineage with SAR-aircraft/ex-series files.
- `server_3090_b`: preserved as an important 3090 lineage with additional metrics modification and later ex-series files.

## Difference Against Official Mamba-YOLO

- See `reports/project_diff_report.md`, `reports/code_diff_matrix.md`, and `reports/all_modified_files.md` from phase 2A.
- Baseline remains shared commit `b26cbda230dfa217f96faee8dc7020db3962f3df` from `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.
- The recovered local snapshots include `.git` metadata for each server copy.

## Multiple Versions

- Paths present with different content on at least two servers: `8`.
- Paths with either different content or missing on one or more servers: `29`.
- No same-name files were overwritten; all versions remain under their source server directory.

## Files Needing Manual Confirmation

High-priority candidates are same-path conflicts that can affect model structure, training, inference/validation, or final reported results.

| File | 1080Ti | 3090-A | 3090-B | Impact |
|---|---|---|---|---|
| `mbyolo_train.py` | `43a95c335b55` | `3427e9cdd3b9` | `3427e9cdd3b9` | `train, result` |
| `selective_scan/setup.py` | `36019920897a` | `cb67db4a7828` | `cb67db4a7828` | `train, result` |
| `ultralytics/engine/trainer.py` | `33b19dade2e4` | `024587c50417` | `024587c50417` | `train, result` |
| `ultralytics/nn/modules/mamba_yolo_ex23.py` | `bedbdf70e121` | `391bd24be3c6` | `391bd24be3c6` | `model, result` |
| `ultralytics/nn/modules/mamba_yolo_ex24-26.py` | `851cf25916d8` | `<missing>` | `d2c14e7773e6` | `model, result` |
| `ultralytics/nn/tasks.py` | `e1c96ad68a32` | `c22332f2942c` | `c22332f2942c` | `model, infer/val, result` |
| `ultralytics/utils/metrics.py` | `015cd5b295d7` | `015cd5b295d7` | `89480260a543` | `infer/val, result` |
