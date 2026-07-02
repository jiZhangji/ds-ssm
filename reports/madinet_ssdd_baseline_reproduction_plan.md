# MaDiNet SSDD 表格开源方法复现计划

生成时间：2026-06-14

## 来源

方法名单来自 MaDiNet arXiv 源码 `main_MaDiNet.tex` 中 `Comparisons of detection performance on SSDD` 表格。

## SSDD 表格方法清单

| 方法 | 开源 | 类型 | R | P | AP50 | F1 | 复现路径 | 优先级 |
|---|---|---|---:|---:|---:|---:|---|---|
| SAR-ShipNet | no | SAR anchor-free | 76.3 | 95.1 | 89.1 | 85.0 | quote only | low |
| DCTC | no | SAR anchor-free |  |  | 93.9 |  | quote only | low |
| GEFCSI-Net | no | SAR-specific | 96.9 | 93.1 | 97.1 | 95.0 | quote only | medium |
| SARNas | no | SAR-specific NAS | 94.8 | 97.4 | 98.5 | 96.1 | quote only | medium |
| MGA-Net | no | SAR-specific | 99.0 | 94.7 | 98.6 | 96.8 | quote only | medium |
| Faster R-CNN | yes | two-stage | 90.4 | 87.1 | 89.7 | 88.7 | mmdetection | medium |
| Cascade R-CNN | yes | two-stage | 90.8 | 84.1 | 90.5 | 92.4 | mmdetection | medium |
| RetinaNet | yes | one-stage anchor-based | 93.1 | 94.0 | 95.0 | 93.6 | mmdetection | medium |
| CenterNet | yes | anchor-free | 93.6 | 94.3 | 95.1 | 93.9 | mmdetection | medium |
| FCOS | yes | anchor-free | 93.9 | 94.6 | 95.3 | 94.2 | mmdetection | medium |
| ConsistencyDet | yes | generative/consistency | 94.0 | 94.5 | 95.6 | 94.2 | official repo | medium |
| YOLOv3 | yes | YOLO | 94.8 | 95.5 | 96.2 | 95.2 | yolo | high |
| YOLOX | yes | YOLO | 93.8 | 95.0 | 96.2 | 94.4 | official repo or MMYOLO | high |
| DiffusionDet | yes | generative/diffusion | 93.9 | 95.1 | 96.3 | 94.5 | official repo | medium |
| YOLOv5n | yes | YOLO lightweight | 95.8 | 97.0 | 98.0 | 96.4 | yolo | high |
| YOLOv8n | yes | YOLO lightweight | 96.4 | 97.3 | 98.1 | 96.9 | yolo | high |
| YOLOv5s | yes | YOLO lightweight | 96.4 | 97.2 | 98.3 | 97.0 | yolo | high |
| DiffDet4SAR | yes | SAR diffusion | 95.2 | 95.3 | 96.9 | 95.2 | official repo | medium |
| MaDiNet | yes | SAR diffusion + Mamba | 97.0 | 98.0 | 99.0 | 97.5 | official repo | reference |
| Ours: DS-SSM/PFIM | yes | lightweight Mamba-YOLO | 95.219 | 97.378 | 98.704 |  | already measured | main |

## 推荐优先复现顺序

1. YOLOv8n、YOLOv5n、YOLOv5s、YOLOX：轻量、开源、与本文轻量化叙事最相关。
2. Faster R-CNN、Cascade R-CNN、RetinaNet、FCOS、CenterNet：使用 MMDetection 统一复现，作为传统检测器对比。
3. DiffusionDet、ConsistencyDet、DiffDet4SAR、MaDiNet：需要 Detectron2/DiffusionDet 环境，作为生成式检测器对比。
4. 非开源 SAR 专用方法：只引用 MaDiNet 表格或原论文数值，除非后续找到官方代码。

## 当前已完成

- 已下载 MaDiNet arXiv PDF 与源码。
- 已解析 SSDD 对比表格方法名单。
- 已在服务器上测得本文完整检测器复杂度。
- 已在服务器上测得 MaDiNet 官方 MambaSAR 核心模块和若干官方组件复杂度。

## 下一步前置工作

1. 固定 SSDD 数据划分，并确认是否与 MaDiNet 使用的 Official-SSDD 一致。
2. 生成 YOLO 格式和 COCO 格式两套标注。
3. 为 YOLO 系列准备统一训练脚本。
4. 单独建立 MMDetection/Detectron2 环境，不污染当前 Mamba-YOLO 环境。
5. 每个 baseline 训练完成后，统一测试并记录 P/R/AP50/F1、Params、GFLOPs、Latency。