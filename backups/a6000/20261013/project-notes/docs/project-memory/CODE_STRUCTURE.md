# Code Structure

## Repository and working copies

- **Repository root**: `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`
- **Active single-process copy**: `source_code/work_train_single/`
- **Active DDP copy**: `source_code/work_train_ddp/`
- **Historical evidence copies**: `source_code/server_1080ti/`, `source_code/server_3090_a/`, `source_code/server_3090_b/`
- **Observed branch**: `codex/ssdd-h100-sweep`
- **Worktree status**: migration前已有大量 modified/untracked files；这些改动不是本次迁移产生的，未清理。

## Main model implementation

### `source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py`

- Exports `PFIM`, `DeSpeckleSSM`, `DS_VSSBlock` and ablation variants.
- `PFIM` computes token-level importance from local trend and, when `hw_shape` is available, 2-D local variation.
- `DeSpeckleSSM` performs residual correction, energy-aware damping, selective scan and residual output.
- `gamma` is the retention/damping factor; `write_gate` scales the current scan input; `delta` changes transition step sensitivity.
- CUDA path currently uses gamma/write as factors on the same scan drive; `delta` remains separate.
- Ablation classes include no-PFIM, PFIM-only, gamma, write, delta, gamma+write, global importance and no-2D-std controls.

### `source_code/work_train_single/ultralytics/nn/modules/common_utils_mbyolo.py`

- Contains selective-scan wrappers/reference implementation and CUDA path integration.
- Historical fix addressed fused CUDA selective-scan detection and non-finite checks; current migration did not rerun or modify it.

## Training and evaluation entrypoints

- `source_code/work_train_single/tools/run_shared_sar_tasks_v2.py`: shared cross-task runner; includes detection, classification, segmentation and change-detection options, warmup/selection/loss controls.
- `source_code/work_train_single/tools/run_ogsod_ablation_matrix.sh`: OGSOD primary/diagnostic queue; skips completed stable runs and records protocol/data fingerprints.
- `source_code/work_train_single/tools/evaluate_ogsod_ablation.py`: fixed-checkpoint TEST evaluation.
- `source_code/work_train_single/tools/summarize_ogsod_ablation_matrix.py`: raw and aggregate summaries.
- `source_code/work_train_single/tools/evaluate_ogsod_strata.py`: density/area/nearest-neighbor diagnostic.
- `source_code/work_train_single/tools/visualize_pfim_importance.py`: importance arrays and qualitative panels.
- `source_code/work_train_single/tools/recheck_change_bs_matrix_20260905.sh`: change-detection BS matrix.
- `source_code/work_train_single/tools/recheck_sar_acd_bs64_fixed_20260905.sh`: corrected SAR-ACD batch.
- `source_code/work_train_single/tools/run_openearthmap_bs64_queue_20260905.sh`: OpenEarthMap effective BS64 queue.

## Config and data flow

1. Dataset manifests/YAML or task-specific manifest are converted/prepared under `datasets/`, server data roots or report-specified paths.
2. Model YAML selects original VSS blocks or DS-SSM/PFIM block and placement.
3. Runner trains, selects checkpoint on validation metric, then evaluation writes `result.json` and task-specific metrics.
4. Reports and database CSVs are evidence indexes; they are not guaranteed to describe the latest state unless their date/provenance says so.

## Tests and verification history

- Historical handoff records local cross-task tests as passing (`15 passed`, later corrected batch `19 passed`) and selective-scan numerical checks as successful.
- This migration did not run tests, build code, train, or evaluate; those claims remain historical evidence only.

## Remote environments and hosting (来自 2026-09-09 迁移批次，均为时间性事实)

- **GitHub**：`jiZhangji/ds-ssm`（远端用 SSH；HTTPS 443 常失败；push 受阻时曾用 Git Data API）。Release：`cross-task-small-data-v1`、`ssdd-historical-diagnostic-20260828`。ModelScope：`shimian123/ds-ssm-workspace-backup`、`shimian123/sar-pretrain`；2026-09-18 新建私有 A6000 备份 `shimian123/ig-ds-ssm-a6000-backup-20260918`。
- **H200/H100 服务器**：`/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/ds-ssm`；conda env `sar-pretrain`（Python 3.10.20、torch 2.1.2、CUDA 12.8）；数据 `.../SAR-Generation/dataset`（HF `shimiandeshu/sar-m` 四包）。
- **A6000 集群**：A=`powerleader-ts`、B/C/D=`powerleader-a6000-b/c/d`，E 无路由，F 默认不分配；项目根 `/home/powerleader/ig-ds-ssm`，env `envs/ig-ds-ssm-env`；拓扑 Windows→D→A/B/C（D 为 LAN 跳板）。C/D 曾报 PyTorch 1.13.1+cu117。
- **3090**：`172.26.133.16`（3090-4）、`172.26.150.36`（3090-1，经 133.16 跳板）；**1080Ti**：`172.26.193.188`（env `mamba-yolo`，torch 2.6.0+cu118，/home 空间紧张）。
- 3090 与 A6000 不能直连，数据须经 Windows 中转；ModelScope 服务器端匿名 SDK 下载失败，需 Windows 下载后传。

- **A6000 备份快照（2026-09-18）**：A6000-B 主包、A6000-A/C 补充包和实验记录已上传至上述私有 Dataset；A6000-D 本次 SSH 超时，未纳入该快照。备份中不含 `envs/`、`.secrets/`、Qwen 服务目录、缓存和临时分片，环境依赖以 `metadata/ig-ds-ssm-env-pip-freeze.txt` 记录。

## Additional tooling（同批次补充）

- `tools/prepare_flat_yolo_detection.py`、`tools/prepare_hf_sar_det_datasets.py`：平铺 YOLO 与 HF 数据集整理。
- `tools/prepare_cross_task_v2.py`、`tools/run_detection_v2_queue.sh`、`tools/run_shared_v2_queue.sh`：v2 校正协议（SADD 阳性 795/44/44、AIR tiling、SRSDD 裁剪 HBB、FUSAR city-disjoint、CD 宏块）。
- `tools/run_paper_aligned_detection.py`、`tools/run_paper_aligned_shared.py`、`tools/audit_paper_aligned_detection.py`、`tools/download_modelscope_file.py`：paper-aligned 轨道。
- `tools/snapshot_yolo_dataset.py`、`tools/snapshot_ogsod_model.py`、`tools/compare_ogsod_per_image.py`、`tools/check_dsssm_cpu_fallback.py`、`tools/check_ogsod_ablation_configs.py`：指纹/快照/定性/回退校验。
- `tools/generate_paper_figures.py`：论文图生成。
- 数据 YAML：`ship_dataset_v0.yaml`、`OGSOD.yaml`、`SSDD_HF.yaml`、`SAR_AIRcraft_HF.yaml`（single 与 ddp 副本均有）。

## Safe code-work rules

- Make new implementation changes in `work_train_single` or `work_train_ddp` after reading the project memory and conflicts.
- Treat `server_*` sources, `experiments/`, `weights/`, `downloads/` and old reports as evidence; do not overwrite them casually.
- Before new publication-grade runs, freeze code/config/data fingerprint and distinguish scratch, partial-transfer and pretrained initialization.
