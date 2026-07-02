# Server Relationship

## Confirmed Facts

- All three servers are reachable and contain `/home/zhangji/Mamba-YOLO`.
- All three are Git repositories with remote `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.
- All three have the same HEAD commit `b26cbda230dfa217f96faee8dc7020db3962f3df` and commit count `23`.
- The latest committed history is official/upstream-like and ends at `2024-12-14 22:05:35 +0800`; the research work appears mostly uncommitted/untracked on servers.

## Relationship Assessment

- They are very likely three working copies derived from the same official Mamba-YOLO project state.
- They are not code-identical at the working-tree level. Tracked files differ and each server has a different set of untracked experiment modules/backups.
- 3090-A and 3090-B are closest by branch state (`main`), counts, and many identical modified tracked file hashes.
- 1080Ti has no branch name reported, more Git status lines, `.pytest_cache`, PFIM-specific files/backups, and a smaller output experiment directory count.

## Main Development Environment Likelihood

- Most likely active development candidates: `server_3090_b` and `server_1080ti`, but no final/master server is selected.
- `server_3090_b` has the largest weight count and an extra modified `ultralytics/utils/metrics.py`, suggesting later metric/result experimentation.
- `server_1080ti` has the largest Git status count and many timestamped backup files around DS-SSM/PFIM changes, suggesting heavy code-edit activity.
- `server_3090_a` looks like a close sibling/training host with one extra `sar_aircraft_test` output directory and fewer dirty files.

## Possible Roles

- `server_1080ti`: older or exploratory development host, with PFIM and numerous local backup snapshots.
- `server_3090_a`: training/evaluation host for SAR aircraft and SSDD variants.
- `server_3090_b`: active training/evaluation host with additional metrics modification and most weight artifacts.

No server is treated as final. This report is only an evidence-based relationship inventory.
