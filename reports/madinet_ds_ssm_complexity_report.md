# MaDiNet 与 DS-SSM 复杂度实测报告

生成日期：2026-06-14

## 服务器与环境

- 服务器：172.26.151.143
- 项目目录：`/mnt/data5/zhangji/Mamba-YOLO`
- GPU：NVIDIA GeForce RTX 3090
- PyTorch：2.3.0+cu121
- 输入尺寸：640
- batch：1
- 精度：fp32

## 实测结果

| 方法 | 范围 | Params(M) | GFLOPs | latency(ms/img) | FPS | 说明 |
|---|---|---:|---:|---:|---:|---|
| Mamba-YOLO-T-SSDD | full Mamba-YOLO detector | 5.9855 | 13.6116 | 26.3284 | 37.9817 | 640 input, batch=1, fp32 |
| Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7 | full Mamba-YOLO detector | 5.9648 | 13.2079 | 28.7546 | 34.777 | 640 input, batch=1, fp32 |
| Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta | full Mamba-YOLO detector | 5.9632 | 13.1554 | 29.0699 | 34.3998 | 640 input, batch=1, fp32 |
| MaDiNet official MambaSAR res4+res5 | official core module only; not full detector | 223.2648 | 345.283 | 27.2921 |  | Official MambaSAR.py res4/res5 block; full detector requires Detectron2 config/environment. |

## 重要说明

1. 本项目三种 Mamba-YOLO 检测器均已在服务器上实际构建并统计，结果可作为论文复杂度表的直接依据。
2. MaDiNet 官方仓库已经下载并上传到服务器，统计的是官方 `MambaSAR.py` 中 res4+res5 核心模块，不是完整 Detectron2 检测器。
3. 完整 MaDiNet 检测器暂未 end-to-end 统计，原因是服务器当前环境缺少 `detectron2`，且官方仓库未提供清晰的 SSDD 专用配置文件入口。
4. 仅 MaDiNet 核心 MambaSAR res4+res5 模块已经达到 223.26M 参数和 345.28 GFLOPs，显著大于本文完整检测器约 5.96M 参数和 13.16-13.21 GFLOPs。
5. 因 MaDiNet 使用自定义 selective scan/CUDA 路径，THOP 可能低估或不完全覆盖部分算子，论文中应将该数值标注为基于官方模块的 profiler 结果。

## 论文结论建议

本文方法不必只围绕是否超过 MaDiNet 的 SSDD 最高 mAP 展开，而应强调在接近强参考方法精度的同时，模型规模和计算量显著降低。DS-SSM / PFIM 的优势是以轻量结构实现 SAR 去斑选择性建模，适合实际部署和后续扩展。