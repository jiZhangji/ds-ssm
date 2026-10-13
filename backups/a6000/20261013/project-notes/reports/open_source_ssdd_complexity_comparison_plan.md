# SSDD 开源方法精度-复杂度对比计划

生成时间：2026-06-14

本表只包含 MaDiNet SSDD 对比表中标注为开源的方法，以及本文方法。非开源方法不进入主复现实验表，只作为论文背景引用。

| 方法 | AP50 | P | R | F1 | Params(M) | GFLOPs | Latency(ms) | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Faster R-CNN | 89.7 | 87.1 | 90.4 | 88.7 | 41.295061 | 150.343158 | NOT_MEASURED | NOT_MEASURED | Detectron2 R50-FPN，640/FP32 服务器实测 |
| Cascade R-CNN | 90.5 | 84.1 | 90.8 | 92.4 | TO_MEASURE | TO_MEASURE | TO_MEASURE | TO_MEASURE | open-source; needs COCO-format SSDD and MMDetection/Detectron2 env |
| RetinaNet | 95.0 | 94.0 | 93.1 | 93.6 | 36.276717 | 159.341683 | NOT_MEASURED | NOT_MEASURED | Detectron2 R50-FPN，640/FP32 服务器实测 |
| CenterNet | 95.1 | 94.3 | 93.6 | 93.9 | TO_MEASURE | TO_MEASURE | TO_MEASURE | TO_MEASURE | open-source; needs COCO-format SSDD and MMDetection env |
| FCOS | 95.3 | 94.6 | 93.9 | 94.2 | TO_MEASURE | TO_MEASURE | TO_MEASURE | TO_MEASURE | open-source; needs COCO-format SSDD and MMDetection env |
| ConsistencyDet | 95.6 | 94.5 | 94.0 | 94.2 | 110.551518 | 3743.693988 | NOT_MEASURED | NOT_MEASURED | 官方当前 forward，1 类、500 proposals、40 次 head |
| YOLOv3 | 96.2 | 95.5 | 94.8 | 95.2 | 103.7541 | 283.3202 | 17.4627 | 57.2649 | measured on server 172.26.151.143 using Ultralytics config, input 640, batch=1, fp32 |
| YOLOX | 96.2 | 95.0 | 93.8 | 94.4 | TO_MEASURE | TO_MEASURE | TO_MEASURE | TO_MEASURE | open-source; high priority |
| DiffusionDet | 96.3 | 95.1 | 93.9 | 94.5 | 110.551518 | 204.262993 | NOT_MEASURED | NOT_MEASURED | 官方 R50-FPN、1 类、500 proposals，640/FP32 实测 |
| YOLOv5n | 98.0 | 97.0 | 95.8 | 96.4 | 2.6548 | 7.8401 | 9.0104 | 110.983 | measured on server 172.26.151.143 using Ultralytics config, input 640, batch=1, fp32 |
| YOLOv8n | 98.1 | 97.3 | 96.4 | 96.9 | 3.1572 | 8.8575 | 7.9916 | 125.1318 | measured on server 172.26.151.143 using Ultralytics config, input 640, batch=1, fp32 |
| YOLOv5s | 98.3 | 97.2 | 96.4 | 97.0 | 9.1532 | 24.2129 | 9.0093 | 110.9966 | measured on server 172.26.151.143 using Ultralytics config, input 640, batch=1, fp32 |
| DiffDet4SAR | 96.9 | 95.3 | 95.2 | 95.2 | 111.141342 | 204.787281 | NOT_MEASURED | NOT_MEASURED | 官方 R50-FPN、1 类、500 proposals；按原配置显式字段修复 YAML 后实测 |
| MaDiNet | 99.0 | 98.0 | 97.0 | 97.5 | 223.2648* | 345.283* | 27.2921* | TO_MEASURE_FULL_DETECTOR | open-source; *measured official MambaSAR res4+res5 core module only; full detector still needs Detectron2 |
| MaDiNet official MambaSAR res4+res5 | 99.0 reference | 98.0 reference | 97.0 reference | 97.5 reference | 223.2648 | 345.283 | 27.2921 | N/A | measured on server; core res4+res5 module only, not full detector |
| Ours: DS-SSM/PFIM | 98.704 | 97.378 | 95.219 | 96.286 | 5.9648 | 13.2079 | 28.7546 | 34.777 | measured on server 172.26.151.143 |
| Ours: DG-DS-SSM | 98.7 | 96.3 | 96.2 | 96.25 | 5.9632 | 13.1554 | 29.0699 | 34.3998 | measured on server 172.26.151.143 |

## 已实测

- YOLOv3、YOLOv5n、YOLOv5s、YOLOv8n 已在服务器 `172.26.151.143` 上完成复杂度统计。
- DS-SSM/PFIM、DG-DS-SSM 已完成复杂度统计。
- MaDiNet 官方 MambaSAR res4+res5 核心模块已完成复杂度统计，但完整 detector 仍待 Detectron2 环境。
- Faster R-CNN、RetinaNet、DiffusionDet、ConsistencyDet、DiffDet4SAR 已于 2026-07-24 在服务器 `172.26.133.16` 完成统一 640、batch 1、FP32 统计。

## 当前无法直接测量的原因

当前论文实际使用的开源对比行已完成统计。计划表中未进入论文三张大表的 Cascade R-CNN、CenterNet、FCOS 和 YOLOX 仍保留为后续扩展项。MaDiNet 官方仓库只发布 MambaSAR 核心，没有完整 detector 集成与配置，因此不能诚实地给出全模型实测值。
