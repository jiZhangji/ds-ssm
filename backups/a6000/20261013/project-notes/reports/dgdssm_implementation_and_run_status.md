# DG-DS-SSM Implementation And Run Status

更新时间：2026-06-08

## 1. 本地实现状态

已在以下两个本地工作仓库中实现 DG-DS-SSM 实验代码：

- `source_code/work_train_single`
- `source_code/work_train_ddp`

新增文件：

- `ultralytics/nn/modules/mamba_yolo_ex30_dg_dsssm.py`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex30-baseline.yaml`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex31-erase.yaml`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex32-write.yaml`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex33-erase-write.yaml`
- `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml`

修改文件：

- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`

本地语法检查：

- `mamba_yolo_ex30_dg_dsssm.py`: passed
- `tasks.py` registration: passed by static grep

## 2. 代码设计

新增模块：

- `DGDeSpeckleSSM`
- `DG_DS_VSSBlock`

核心思想：

- 保留当前 DS-SSM/PFIM 主线；
- 将 `gamma` 对应的 retain/erase gate 与 `write_gate` 解耦；
- 使用 `mode` 控制消融：
  - `baseline`
  - `erase`
  - `write`
  - `erase_write`
  - `erase_write_delta`

## 3. 服务器检查结果

| 服务器 | IP | 状态 | 处理 |
|---|---|---|---|
| 3090-5 | `172.26.150.72` | SSH 认证失败 | 未上传，未启动 |
| 3090-4 | `172.26.133.16` | `/home` 磁盘 100%，仅 4.7G 可用 | 未启动 |
| 3090-3 | `172.26.151.143` | 双 3090 空闲，项目在 `/mnt/data5/zhangji/Mamba-YOLO` | 已上传，已启动 ex33/ex31 |
| 3090-1 | `172.26.150.36` | GPU 有其他进程占用 | 未启动 |
| 1080Ti-2 | `172.26.193.188` | 空闲，但显存较小 | 暂未启动完整 bs32 |

## 4. 服务器上传状态

已上传到：

`172.26.151.143:/mnt/data5/zhangji/Mamba-YOLO`

服务器备份目录：

`/mnt/data5/zhangji/Mamba-YOLO/codex_backups/dgdssm_before_20260608_120347`

上传内容：

- 新 DG-DS-SSM 模块；
- ex30-ex34 YAML；
- `__init__.py` 注册；
- `tasks.py` 解析注册。

远程语法检查：

- passed

## 5. 已启动实验

### ex33 erase-write

服务器：

`172.26.151.143`

GPU：

物理 GPU0

命令核心：

```bash
/home/zhangji/miniconda3/envs/mamba-yolo/bin/python mbyolo_train.py \
  --task train \
  --data datasets/SSDD_YOLO/ssdd.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex33-erase-write.yaml \
  --batch_size 32 \
  --imgsz 640 \
  --epochs 300 \
  --device 0 \
  --workers 4 \
  --amp \
  --project output_dir/ssdd \
  --name ssdd_DG-DSSSM_T_300e_32bs_ex33_erase-write_20260608
```

日志：

`/mnt/data5/zhangji/Mamba-YOLO/codex_reports/ssdd_DG-DSSSM_T_300e_32bs_ex33_erase-write_20260608.log`

输出目录：

`/mnt/data5/zhangji/Mamba-YOLO/output_dir/ssdd/ssdd_DG-DSSSM_T_300e_32bs_ex33_erase-write_20260608`

启动状态：

- 正常启动；
- 已通过模型构建；
- warmup 冻结日志出现；
- 前几个 epoch 未见 NaN/OOM。

### ex31 erase-only

第一次启动：

- 使用 `--device 1`；
- 实际仍碰到 GPU0 显存；
- 发生 OOM；
- 该目录不作为正式结果。

正式重启：

服务器：

`172.26.151.143`

GPU：

通过 `CUDA_VISIBLE_DEVICES=1` 绑定物理 GPU1。

命令核心：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangji/miniconda3/envs/mamba-yolo/bin/python mbyolo_train.py \
  --task train \
  --data datasets/SSDD_YOLO/ssdd.yaml \
  --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex31-erase.yaml \
  --batch_size 32 \
  --imgsz 640 \
  --epochs 300 \
  --device 0 \
  --workers 4 \
  --amp \
  --project output_dir/ssdd \
  --name ssdd_DG-DSSSM_T_300e_32bs_ex31_erase_20260608_gpu1
```

日志：

`/mnt/data5/zhangji/Mamba-YOLO/codex_reports/ssdd_DG-DSSSM_T_300e_32bs_ex31_erase_20260608_gpu1.log`

输出目录：

`/mnt/data5/zhangji/Mamba-YOLO/output_dir/ssdd/ssdd_DG-DSSSM_T_300e_32bs_ex31_erase_20260608_gpu1`

启动状态：

- 正常启动；
- 已通过模型构建；
- warmup 冻结日志出现；
- GPU1 显存占用正常。

## 6. 当前 GPU 占用

最近检查：

- GPU0: ex33, about 15.5G used
- GPU1: ex31, about 16.1G used

## 7. 暂未启动实验

| 实验 | 原因 |
|---|---|
| ex30 baseline rerun | 暂无空闲 3090；可等 ex31/ex33 完成后跑 |
| ex32 write-only | 1080Ti 空闲但 bs32 风险高；可后续在 3090 上跑 |
| ex34 erase-write-delta | 3090-1 忙，3090-4 磁盘满；暂缓 |

## 8. 后续监控与回收

需要监控：

- `codex_reports/*.log`
- `output_dir/ssdd/*/results.csv`
- GPU 显存和进程
- 是否 NaN/OOM
- 第 20 epoch 是否出现 `[IG-DS-SSM Warmup Done]`

训练完成后回收：

- `args.yaml`
- `results.csv`
- `results.png`
- `best.pt`
- `last.pt`
- 训练日志

本地应更新：

- `database/dg_dsssm_ssdd_results.csv`
- `database/dg_dsssm_ssdd_results.xlsx`
- `reports/dg_dsssm_ssdd_experiment_results.md`
- 总排行榜

## 9. 注意事项

- 本轮正式运行的结果目录以 `ex31_erase_20260608_gpu1` 和 `ex33_erase-write_20260608` 为准。
- `ex31_erase_20260608` 第一次 OOM 目录应记录为失败尝试，不参与性能比较。
- 当前不要删除服务器上的任何失败目录或日志。
