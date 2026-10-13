# MaDiNet 与本方法参数量/FLOPs 对比实验计划

生成时间：2026-06-14

## 背景

MaDiNet 在 SSDD 上具有很强的检测性能，直接在单一精度指标上超越它可能较难。但本项目的 DS-SSM / PFIM / DG-DS-SSM 路线具有明显轻量化潜力：模型参数量更小、FLOPs 更低、推理链路更接近实时检测模型。这可以成为论文中的重要优势，而不是把论文叙事限定为“必须超过所有 SOTA 的 mAP”。

## 对比目标

1. 统一输入尺寸 `640x640`，统计参数量和 GFLOPs。
2. 在相同服务器、相同 GPU、相同 batch 设置下记录推理延迟和 FPS。
3. 对 MaDiNet 采用两种记录方式：
   - 若能获得官方代码：使用同一脚本或等价 profiler 重新测量；
   - 若无法运行官方代码：引用论文报告值，并在表格中明确标注 `paper_reported`。
4. 形成“精度-复杂度”联合对比表，而不是只比较 mAP。

## 本地已整理文件

| 文件 | 作用 |
|---|---|
| `complexity_benchmark/model_complexity_benchmark.py` | 参数量、FLOPs、推理延迟统计脚本 |
| `complexity_benchmark/run_complexity_benchmark.sh` | 服务器一键运行脚本 |
| `complexity_benchmark/madinet_record_template.csv` | MaDiNet 与本方法统一记录模板 |
| `reports/madinet_complexity_experiment_plan.md` | 当前计划说明 |

## 建议服务器运行命令

```bash
cd /mnt/data5/zhangji/Mamba-YOLO

PYTHON_BIN=/home/zhangji/miniconda3/envs/mamba-yolo/bin/python \
DEVICE=0 \
IMGSZ=640 \
BATCH=1 \
bash complexity_benchmark/run_complexity_benchmark.sh
```

如需 batch 速度：

```bash
PYTHON_BIN=/home/zhangji/miniconda3/envs/mamba-yolo/bin/python \
DEVICE=0 \
IMGSZ=640 \
BATCH=16 \
bash complexity_benchmark/run_complexity_benchmark.sh
```

## 推荐论文表格

| Method | Params(M) | GFLOPs | FPS | P | R | mAP50 | mAP50-95 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MaDiNet | 待补 | 待补 | 待补 | 论文值 | 论文值 | 论文值 | 论文值 | 强性能参考 |
| Mamba-YOLO-T | 待测 | 待测 | 待测 | 已有实验 | 已有实验 | 已有实验 | 已有实验 | 基线 |
| DS-SSM | 待测 | 待测 | 待测 | 已有实验 | 已有实验 | 已有实验 | 已有实验 | 主创新 |
| PFIM / IG-DS-SSM | 待测 | 待测 | 待测 | 0.97378 | 0.95219 | 0.98704 | 0.74022 | 当前最优主线 |
| DG-DS-SSM | 待测 | 待测 | 待测 | 0.963 | 0.962 | 0.987 | 0.724 | 延长线，召回较强 |

## 论文叙事建议

如果 MaDiNet 的 SSDD mAP50 更高，可以把本方法定位为“高效 SAR 目标检测结构”，强调：

1. 在 SSDD 这种小目标密集场景中，本方法达到接近 MaDiNet 的 mAP50 和较高召回；
2. 相比复杂的扩散式或重型 Mamba 结构，本方法通过 DS-SSM 在状态空间更新中引入去斑选择性建模；
3. PFIM 进一步用感知重要性引导 gamma、write gate 或 delta，使模型把计算和状态更新集中到疑似目标区域；
4. 该设计不依赖额外重型分支，参数量和 FLOPs 更友好，更适合舰船检测的实际部署。

## 下一步需要确认

1. 是否允许我把 `complexity_benchmark/` 上传到指定服务器项目目录。
2. 是否使用 `/mnt/data5/zhangji/Mamba-YOLO` 作为主测试位置。
3. MaDiNet 是否有官方代码或你希望我只先引用论文中的参数/FLOPs。
4. 是否将 batch=1 和 batch=16 都作为论文速度指标。
