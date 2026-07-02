# MaDiNet 与本方法复杂度对比实验包

本目录用于整理 SSDD 论文对比中需要的参数量、FLOPs、推理速度和显存统计。

## 目标

1. 计算本项目不同模型配置的参数量和 FLOPs。
2. 为 MaDiNet 或论文中提到的对比方法预留统一记录格式。
3. 在同一服务器、同一输入尺寸、同一 batch 设置下记录推理速度，避免跨环境比较带来的误差。
4. 将“精度接近但复杂度显著更低”的优势整理成论文可用证据。

## 当前建议对比项

| 方法 | 配置/来源 | 状态 |
|---|---|---|
| Mamba-YOLO-T baseline | `ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml` | 可直接统计 |
| DS-SSM | `Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml` | 当前主线 |
| PFIM / IG-DS-SSM | `Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml` + 对应代码版本/权重 | 当前最优主线 |
| DG-DS-SSM | `Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml` | 延长线 |
| MaDiNet | 论文/官方代码 | 待补充官方实现或论文数值 |

## 推荐流程

1. 将本目录复制到服务器项目根目录 `/mnt/data5/zhangji/Mamba-YOLO` 或 `/home/zhangji/Mamba-YOLO`。
2. 在服务器项目根目录执行 `bash complexity_benchmark/run_complexity_benchmark.sh`。
3. 将输出的 `complexity_benchmark/results/model_complexity.csv` 和日志回传到本地。
4. 如果获得 MaDiNet 官方代码，将其参数量/FLOPs 通过同一脚本或同一输入尺寸重新统计；如果无法运行官方代码，则仅引用论文报告值，并在表格中标注 `paper_reported`。

## 注意

- FLOPs 必须统一输入尺寸，建议 SSDD 使用 `640x640`。
- 参数量和 FLOPs 不依赖训练结果，但推理速度依赖 GPU、CUDA、batch size、精度模式和实现。
- MaDiNet 若采用扩散式/多步迭代推理，除了模型静态 FLOPs，还应记录实际推理延迟，因为多步计算可能使实际代价明显增加。
