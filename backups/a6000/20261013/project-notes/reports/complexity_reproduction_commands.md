# 复杂度实验代码与运行指令归档

生成时间：2026-06-14

## 本地保存的代码

| 文件 | 作用 |
|---|---|
| `complexity_benchmark/model_complexity_benchmark.py` | 统计本项目 Mamba-YOLO / DS-SSM / DG-DS-SSM 完整检测器的 Params、GFLOPs、Latency、FPS |
| `complexity_benchmark/madinet_official_complexity.py` | 统计 MaDiNet 官方 `MambaSAR.py` 中 res4+res5 核心模块 |
| `tools/run_complexity_benchmark_server.py` | 首次上传复杂度工具包和 MaDiNet 官方源码到服务器，并运行统计 |
| `tools/run_complexity_benchmark_resume.py` | 修正后续跑本项目检测器复杂度统计 |
| `tools/run_madinet_component_resume.py` | 单独续跑 MaDiNet 官方 MambaSAR 组件统计 |
| `tools/create_complexity_reports_and_docx.py` | 汇总 CSV、Markdown 报告和 Word 文档 |

## MaDiNet 官方源码

本地位置：

`downloads/madinet_official/MaDiNet-master`

服务器位置：

`/mnt/data5/zhangji/Mamba-YOLO/external_repos/MaDiNet_official`

官方仓库：

`https://github.com/JoyeZLearning/MaDiNet`

## 服务器环境

服务器：

`172.26.151.143`

项目目录：

`/mnt/data5/zhangji/Mamba-YOLO`

Python：

`/home/zhangji/miniconda3/envs/mamba-yolo/bin/python`

环境检查结果：

- torch: `2.3.0+cu121`
- thop: 可用
- timm: 可用
- einops: 可用
- mamba_ssm: 可用
- detectron2: 不可用

## 实际运行过的命令

### 本项目完整检测器

```bash
cd /mnt/data5/zhangji/Mamba-YOLO

/home/zhangji/miniconda3/envs/mamba-yolo/bin/python \
  complexity_benchmark/model_complexity_benchmark.py \
  --configs \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml \
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --warmup 10 \
  --runs 30 \
  --out /mnt/data5/zhangji/Mamba-YOLO/complexity_benchmark/results/server_run_20260614_171404/ours_complexity.csv
```

### MaDiNet 官方 MambaSAR 核心模块

```bash
cd /mnt/data5/zhangji/Mamba-YOLO

/home/zhangji/miniconda3/envs/mamba-yolo/bin/python \
  complexity_benchmark/madinet_official_complexity.py \
  --madinet-root /mnt/data5/zhangji/Mamba-YOLO/external_repos/MaDiNet_official \
  --imgsz 640 \
  --device 0 \
  --warmup 5 \
  --runs 10 \
  --out /mnt/data5/zhangji/Mamba-YOLO/complexity_benchmark/results/madinet_component_20260614_171712/madinet_official_complexity.csv
```

## 原始结果位置

本地：

- `downloads/complexity_results_172_26_151_143/server_run_20260614_171404/ours_complexity.csv`
- `downloads/complexity_results_172_26_151_143/madinet_component_20260614_171712/madinet_official_complexity.csv`

服务器：

- `/mnt/data5/zhangji/Mamba-YOLO/complexity_benchmark/results/server_run_20260614_171404/ours_complexity.csv`
- `/mnt/data5/zhangji/Mamba-YOLO/complexity_benchmark/results/madinet_component_20260614_171712/madinet_official_complexity.csv`

## 当前限制

MaDiNet 完整检测器还没有 end-to-end 统计，原因：

1. 当前服务器环境缺少 `detectron2`。
2. MaDiNet 官方仓库没有明显提供 SSDD 专用完整 config 入口。
3. 因此当前统计的是官方核心 `MambaSAR.py` 中 res4+res5 模块，而不是完整检测器。

这部分在论文中必须如实标注。
