# 前述 SAR 目标检测数据集与方法结果总表（2026-08-25）

## 1. 阅读规则

- 数值均为百分数。
- `AP` 或 `AP50:95` 表示 COCO 式 IoU=0.50:0.95；`AP50` 和 `AP75` 分别表示固定 IoU 阈值。
- HBB 与 OBB、SAR-only 与 Optical+SAR、原图与切片、scratch 与预训练结果不能直接排名。
- `本地`表示本项目实测；其余均为对应论文作者报告值。
- 本表覆盖本次讨论中明确提出的 11 个数据集，并非穷尽世界上所有使用过这些数据集的论文。
- 按用户决定，不把 SAI-Net 本身作为最终对比方法。

## 2. 数据集规模与类别

| 数据集 | 原始图像数 | 实例数 | 类别数 | 类别 | 框/任务 | 可用划分或本地划分 |
|---|---:|---:|---:|---|---|---|
| SSDD | 1,160 | 2,456（不同版本约 2,587） | 1 | ship | HBB/OBB 检测 | 常用 928 train / 232 test；本地开发 742/186/232 |
| OGSOD-1.0 | 18,331 SAR 图（另有配对光学） | 48,589（标准化统计） | 3 | bridge, harbor, tank | HBB/OBB；SAR-only 或多模态 | 本地 SAR-only 14,664/1,834/1,833 |
| SAR-ShipDataset | 本地修订版 39,729；早期论文版 43,819 | 本地 50,885；早期论文约 59,535 | 1 | ship | HBB 检测 | 本地 31,783/3,972/3,974 |
| SAR-AIRcraft-1.0 | 4,368 原图 | 16,463 | 7 | A220, A320/321, A330, ARJ21, Boeing737, Boeing787, other | HBB 检测/识别 | 本地切片版 13,976/1,923/2,989，共 38,475 框 |
| 双极化 DSSDD | 1,236 | 3,540 | 1 | ship | HBB/OBB；VV/VH/伪彩色 | 856 train / 380 test |
| SADD | 2,966 原始切片；标准化正样本 883 | 7,835 | 1 | aircraft | HBB 检测 | SARDet 标准化 795/44/44 |
| SIVED | 1,044 | 12,013 | 1 | vehicle | OBB（可转外接 HBB） | 837/104/103 |
| AIR-SARShip-1.0 | 31 幅大图 | 461 | 1 | ship | HBB 检测 | 原始 21 train / 10 test；切片数依论文而异 |
| AIR-SARShip-2.0 | 300 | 2,040 | 1 | ship | HBB 检测 | 常见 210 train+val / 90 test；SARDet 为 270/15/15 |
| SRSDD-v1.0 | 666 | 2,884 | 6 | ore-oil, bulk cargo, fishing, law enforcement, dredger, container | OBB 细粒度检测 | 常见 532/134（不同论文需核对） |
| HRSID | 5,604 | 16,951（标准化版约 16,969） | 1 | ship | HBB 检测、语义/实例分割 | 标准化 3,642/981/981 |

规模来源：[Official SSDD](https://github.com/TianwenZhang0825/Official-SSDD)、[SARDet-100K](https://github.com/zcablii/sardet_100k)、[SAR-Aircraft 官方页](https://www.radars.ac.cn/web/data/getData?newsColumnId=81053ef7-8327-4f09-88c0-ea061f64f9ba&pageType=en)、[DSSDD](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709314/)、[SIVED](https://github.com/CAESAR-Radi/SIVED)、[SRSDD](https://www.mdpi.com/2072-4292/13/24/5104)、[HRSID](https://github.com/chaozhong2010/HRSID)。

## 3. SSDD

### 3.1 同一篇近期论文中的 scratch 统一重跑

| 方法 | AP50 | AP75 | AP50:95 | Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 96.4 | 86.1 | 71.3 | 41.35 | 208.0 |
| Cascade R-CNN | 96.4 | 87.4 | 72.7 | 69.15 | 236.0 |
| YOLOv8n | 97.1 | 91.4 | 74.0 | 3.01 | 8.7 |
| YOLOv9-tiny | 97.3 | 88.9 | 72.2 | 2.62 | 10.6 |
| YOLOv10n | 97.3 | 88.5 | 72.8 | 2.27 | 6.7 |
| YOLO11n | 97.5 | 90.2 | 73.2 | 2.58 | 6.3 |
| YOLO12n | 97.1 | 87.2 | 71.6 | 2.55 | 6.0 |
| YOLOv8s | 97.6 | 91.7 | 75.4 | 11.13 | 28.6 |
| Dynamic multi-scale method（作者方法） | 98.2 | 91.9 | 76.8 | 2.95 | 10.4 |

来源：[A Dynamic Multi-Scale Feature Fusion Network](https://www.mdpi.com/3458476)。

### 3.2 本地结果

| 方法/协议 | AP50 | AP75 | AP50:95 | 状态 |
|---|---:|---:|---:|---|
| 历史 PFIM ex27 | 98.704 | - | 74.022 | 历史档案，初始化/训练协议不完全可确认 |
| H100 Mamba-YOLO-T baseline | 93.129 | 58.896 | 54.838 | 742/186 开发 VAL，scratch |
| H100 DS-SSM/PFIM P2-d16 | 91.513 | 66.858 | 56.673 | 同协议 VAL，AP50:95 +1.834 pp |

## 4. OGSOD-1.0

### 4.1 本地 SAR-only HBB（当前最可靠成对结果）

| 方法 | P | R | AP50 | AP75 | AP50:95 | Params(M) | GFLOPs@640 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO-T | 80.143 | 71.155 | 77.173 | 40.211 | 42.302 | 5.985 | 13.6 |
| IG-DS-SSM/PFIM-T | 79.714 | 71.692 | 77.263 | 40.971 | 43.602 | 5.964 | 13.2 |
| 变化 | -0.429 | +0.538 | +0.090 | +0.760 | +1.300 | -0.021 | -0.4 |

### 4.2 SAR-only 作者报告基线（不同划分/640 输入，仅背景参考）

| 方法 | AP50 | AP75 | AP50:95 | Size/Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN | 67.4 | 27.9 | 33.0 | 41.3 | 134.0 |
| Cascade R-CNN | 73.3 | 37.6 | 38.9 | 69.3 | 162.0 |
| ATSS | 76.8 | 40.5 | 42.7 | 38.9 | 68.3 |
| RetinaNet | 67.8 | 38.5 | 38.4 | 36.5 | 130.0 |
| YOLOv8n | 80.3 | 48.8 | 49.3 | 5.4 | 6.8 |
| YOLOv10n | 81.0 | 49.5 | 49.8 | 5.5 | 6.5 |
| YOLO11n | 79.5 | 49.2 | 49.3 | 5.3 | 6.3 |
| YOLO12n | 79.5 | 48.5 | 48.6 | 5.2 | 5.8 |
| YOLO13n | 79.2 | 48.1 | 47.9 | 5.2 | 6.1 |
| RT-DETR | 88.5 | 48.5 | 49.4 | 77.0 | 56.9 |

该表来自 SAI-Net 论文的基线重跑；按当前决定，不采用 SAI-Net 方法行，也不把该表与本地结果直接排名。

### 4.3 Optical+SAR HBB（任务不同）

| 方法 | 输入 | AP50 | AP50:95 |
|---|---|---:|---:|
| SuperYOLO | Optical+SAR | 94.7 | 53.8 |
| ICAFusion | Optical+SAR | 94.8 | 64.1 |
| CFT | Optical+SAR | 95.7 | 62.9 |
| YOLO-CMFM | Optical+SAR | 96.2 | 75.1 |
| MAIENet | Optical+SAR | 90.8 | 61.0 |

## 5. SAR-ShipDataset

| 方法 | AP50 | AP50:95 | 结果类型/说明 |
|---|---:|---:|---|
| YOLOv4 | 77.9 | - | MaDiNet 作者报告表 |
| YOLOv7 | 86.1 | - | 同上 |
| YOLOv5n | 89.3 | - | 同上 |
| CenterNet | 92.1 | - | 同上 |
| Faster R-CNN | 93.1 | - | 同上 |
| RetinaNet | 93.4 | - | 同上 |
| FCOS | 94.7 | - | LRTransDet 论文重跑 |
| YOLOv5s | 94.8 | - | LRTransDet 论文重跑 |
| DiffDet4SAR | 95.1 | - | 作者报告 |
| LRTransDet | 95.1 | - | 作者报告，3.07M |
| YOLOSAR-Lite | 96.06 | - | 作者报告，2.05M |
| DAFDet | 96.5 | - | 作者报告 |
| MaDiNet | 97.5 | - | 作者报告，未公开逐图划分 |
| 本地 Mamba-YOLO-T | 94.616 | 49.568 | baseline 训练中途改变 batch |
| 本地 IG-DS-SSM/PFIM-T | 94.844 | 57.818 | 本地 test，非严格因果对照 |
| 本地 IG-DS-SSM/PFIM-S | 95.609 | 53.177 | 大尺度模型，不是轻量主模型 |

## 6. SAR-AIRcraft-1.0

### 6.1 常见作者报告 AP50

| 方法 | AP50 | 开源/协议备注 |
|---|---:|---|
| FCOS | 54.8 | 通用实现可用 |
| CenterNet | 70.5 | 通用实现可用 |
| Faster R-CNN | 75.9 | 通用实现可用 |
| YOLOX | 81.0 | 官方实现可用 |
| YOLOv3 | 83.9 | 通用实现可用 |
| YOLOv5n | 88.2 | 通用实现可用 |
| YOLOv8n | 88.2 | 通用实现可用 |
| DiffDet4SAR | 88.4 | 官方代码可用 |
| SFS-CNet | 89.7 | 官方代码可用 |
| PGD-YOLOv8 | 90.7 | 官方代码与权重可用 |
| MaDiNet | 90.5 | 作者报告，复现链不完整 |

### 6.2 自监督预训练路线（不可与 scratch 混排）

| 方法 | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| ViTDet | 84.5 | 62.6 | 55.0 |
| ViTDet4SAR | 86.6 | 64.5 | 56.5 |
| SARATR-X | 86.1 | 64.7 | 58.7 |
| SARDet-MIM | 87.0 | 70.7 | 60.5 |

来源：[SARDet-MIM](https://www.mdpi.com/2072-4292/18/4/580)。

### 6.3 本地 patch 协议

| 方法/划分 | AP50 | AP50:95 | 状态 |
|---|---:|---:|---|
| Mamba-YOLO-T reference VAL | 87.751 | 69.904 | 约 137 epoch、SGD |
| IG-DS-SSM/PFIM-T VAL | 88.825 | 70.513 | 300 epoch、AdamW |
| Mamba-YOLO-T reference TEST | 70.402 | 54.513 | 非匹配 baseline |
| IG-DS-SSM/PFIM-T TEST | 70.822 | 55.552 | 300 epoch、AdamW |
| IG-DS-SSM/PFIM-S VAL | 89.304 | 72.025 | 21.74M 参数，不是轻量主模型 |

## 7. 双极化 DSSDD

| 检测器/输入 | Backbone | P | R | AP50 | AP50:95 |
|---|---|---:|---:|---:|---:|
| R3Det / VV | R50-FPN | 93.6 | 89.3 | 88.8 | 30.4 |
| R3Det / VH | R50-FPN | 94.2 | 87.7 | 88.7 | 33.4 |
| R3Det / 伪彩色 | R50-FPN | 95.7 | 92.1 | 90.2 | 40.5 |
| R3Det / VV | R101-FPN | 94.3 | 91.3 | 89.9 | 44.0 |
| R3Det / VH | R101-FPN | 94.6 | 90.3 | 89.6 | 44.6 |
| R3Det / 伪彩色 | R101-FPN | 96.2 | 91.5 | 90.2 | 47.5 |
| YOLOv4 / VV | CSPDarknet53 | 94.4 | 92.3 | 92.4 | 57.9 |
| YOLOv4 / VH | CSPDarknet53 | 94.8 | 92.2 | 92.2 | 55.1 |
| YOLOv4 / 伪彩色 | CSPDarknet53 | 95.8 | 93.3 | 93.8 | 58.5 |
| YOLOv13 baseline（近期论文） | - | 94.7 | 95.1 | 98.5 | 70.1 |

来源：[DSSDD 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8709314/)。最后一行为不同近期协议，只作参考。

## 8. SADD

### 8.1 同一篇 YOLO-SAD 论文统一重跑

| 方法 | AP50 | AP50:95 | P | R | Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| ResNet50-CSP | 91.5 | 55.9 | 92.3 | 83.5 | 65.1 | 64.4 |
| ResNeXt50-CSP | 91.3 | 55.4 | 88.9 | 85.3 | 64.3 | 59.7 |
| YOLOv4 | 87.8 | 54.0 | 93.8 | 77.5 | 105.5 | 119.7 |
| YOLOr-p6 | 89.2 | 53.5 | 93.6 | 80.5 | 74.3 | 80.6 |
| YOLOv7 | 89.8 | 55.2 | 92.6 | 82.7 | 74.8 | 105.1 |
| YOLO-SAD | 91.9 | 57.1 | 94.0 | 81.2 | 69.9 | 97.2 |

来源：[YOLO-SAD](https://www.mdpi.com/2076-3417/14/7/3025)。

### 8.2 自监督预训练路线

| 方法 | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| ViTDet | 93.0 | 51.0 | 50.2 |
| ViTDet4SAR | 94.6 | 47.5 | 50.3 |
| SARATR-X | 94.3 | 50.7 | 50.4 |
| SARDet-MIM | 95.9 | 59.6 | 55.3 |

## 9. SIVED（OBB）

| 方法 | P | R | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| RoI Transformer | 84.40 | 95.61 | 93.47 | 16.91 | 37.45 |
| Rotated FCOS | 96.47 | 88.86 | 95.60 | 48.13 | 50.40 |
| S2A-Net | 90.90 | 97.48 | 97.72 | 57.32 | 55.49 |
| Rotated RetinaNet | 92.73 | 97.48 | 97.76 | 50.93 | 53.11 |
| KLD | 93.34 | 98.05 | 97.92 | 64.48 | 57.49 |
| Rotated Faster R-CNN | 95.55 | 97.80 | 98.09 | 59.00 | 56.03 |
| Gliding Vertex | 95.72 | 98.13 | 98.33 | 59.59 | 56.06 |
| Oriented RepPoints | 95.11 | 98.05 | 99.13 | 70.69 | 60.15 |

来源：[SIVED 原论文](https://www.mdpi.com/2072-4292/15/11/2825)。若转成 HBB，必须重新训练所有基线，不能引用这些 OBB 数字。

## 10. AIR-SARShip-1.0

### 10.1 原始数据集论文基准

| 方法 | AP | FPS |
|---|---:|---:|
| YOLOv1 | 64.7 | 160 |
| SSD-300 | 72.4 | 151 |
| SSD-512 | 74.3 | 64 |
| Faster R-CNN | 79.3 | 30 |
| Faster R-CNN-DR | 84.2 | 29 |
| DCENN | 88.1 | 24 |

来源：[AIR-SARShip-1.0 原论文](https://radars.ac.cn/article/doi/10.12000/JR19097?viewType=HTML)。

### 10.2 LPDNet 论文统一比较

| 方法 | AP | P | R | Params(M) |
|---|---:|---:|---:|---:|
| SSD | 86.11 | 82.20 | 76.73 | 23.61 |
| CenterNet | 90.00 | 94.02 | 67.07 | 32.66 |
| YOLOv7 | 81.61 | 84.06 | 68.64 | 37.19 |
| YOLOv4-tiny | 72.98 | 77.70 | 61.77 | 5.87 |
| YOLOX-tiny | 89.96 | 83.83 | 86.15 | 5.03 |
| LPDNet | 92.19 | 84.88 | 88.64 | 5.15 |

来源：[LPDNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10347162/)。不同裁剪方式会显著改变结果。

## 11. AIR-SARShip-2.0

| 方法/预训练 | P | R | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| ImageNet-SSD | 87.11 | 79.60 | 83.00 | 46.22 | 46.49 |
| OSD-SSD | 90.64 | 81.72 | 84.70 | 56.09 | 50.96 |
| OSM-SSD | 90.91 | 81.50 | 84.49 | 55.09 | 51.14 |
| WBF-DM | 85.84 | 84.97 | 85.43 | 61.56 | 54.78 |

来源：[Boosting Ship Detection with Complementary Pretraining](https://www.alphaxiv.org/abs/2103.08251v1)。这些是预训练/融合结果，不是 scratch。

## 12. SRSDD-v1.0

### 12.1 同一篇 HBB 化 YOLO 比较

| 方法 | P | R | AP50 | AP50:95 | Params(M) |
|---|---:|---:|---:|---:|---:|
| SSD | - | - | 29.9 | - | 24.41 |
| RetinaNet | - | - | 55.6 | - | 36.43 |
| YOLOv3 | - | - | 70.5 | - | 61.55 |
| YOLOv5s | 74.2 | 81.3 | 80.2 | 42.4 | 6.71 |
| GDB-YOLOv5s | 75.2 | 83.3 | 81.7 | 47.6 | 21.38 |

来源：[GDB-YOLOv5s](https://doi.org/10.1049/ipr2.13140)。原始 SRSDD 是六类 OBB；HBB 化结果不能与旋转检测器直接混排。

### 12.2 旋转检测代表结果

| 方法 | mAP（论文口径） | GFLOPs | FPS |
|---|---:|---:|---:|
| KFIoU | 74.2 | 335.74 | 24.4 |
| S2A-Net | 84.0 | 196.21 | 24.3 |
| R3Det | 83.7 | 328.7 | 22.0 |
| RoI Transformer | 84.2 | 225.29 | 18.2 |
| LPST-Det | 84.3 | 312.36 | 23.6 |
| S4Det | 84.7 | 298.32 | 31.1 |
| PRDet | 85.9 | 226.71 | 29.5 |

来源：[PDE-Guided Diverse Feature Learning](https://www.mdpi.com/2072-4292/17/17/2998)。

## 13. HRSID

### 13.1 统一 scratch HBB 比较

| 方法 | P | R | AP50 | AP75 | AP50:95 | Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 89.9 | 75.5 | 85.4 | 75.5 | 65.4 | 41.35 | 208.0 |
| Cascade R-CNN | 90.8 | 78.9 | 87.8 | 78.7 | 68.4 | 69.15 | 236.0 |
| YOLOX-tiny | 89.4 | 84.5 | 91.3 | 74.5 | 66.1 | 5.03 | 7.58 |
| YOLOv8n | 91.7 | 79.7 | 90.2 | 74.4 | 65.4 | 3.01 | 8.7 |
| YOLOv9-tiny | 90.2 | 81.3 | 90.4 | 73.2 | 64.9 | 2.62 | 10.6 |
| YOLOv10n | 90.3 | 83.2 | 92.1 | 77.9 | 68.1 | 2.27 | 6.7 |
| YOLO11n | 91.5 | 80.1 | 89.8 | 74.3 | 65.2 | 2.58 | 6.3 |
| YOLO12n | 90.0 | 79.5 | 89.2 | 72.9 | 64.4 | 2.55 | 6.0 |
| Dynamic multi-scale method（作者方法） | 92.4 | 85.6 | 93.1 | 78.5 | 69.8 | 2.95 | 10.4 |

来源：[A Dynamic Multi-Scale Feature Fusion Network](https://www.mdpi.com/3458476)。

### 13.2 另一套常见作者报告 AP50

| 方法 | AP/AP50:95 | AP50 | 说明 |
|---|---:|---:|---|
| Faster R-CNN | - | 78.2 | LPDNet 论文协议 |
| Cascade R-CNN | - | 79.2 | 同上 |
| SSD | - | 88.8 | 同上 |
| CenterNet | - | 86.3 | 同上 |
| FCOS | - | 86.6 | 同上 |
| YOLOX-tiny | - | 91.08 | 同上 |
| LPDNet | - | 93.09 | 同上 |
| YOLOv8s | 61.6 | 91.8 | DFENet 论文协议 |
| YOLOX | 62.4 | 93.3 | DFENet 论文协议 |
| DFENet | 62.3 | 92.1 | DFENet 论文协议 |
| LH-YOLO | - | 96.6 | 只报 AP50 的另一协议 |

这些数字差异本身说明 HRSID 的划分和评估器并不统一，最终主表必须自行统一重跑。

## 14. 本项目四源混合代理实验

该结果不是分别训练四个数据集，而是 1,000 train / 400 val、11 类联合训练。

| 来源/整体 | Mamba-YOLO-T AP50:95 | 完整 DS-SSM/PFIM P3 AP50:95 | 变化 |
|---|---:|---:|---:|
| 整体 | 7.097 | 7.959 | +0.862 |
| SSDD 子集 | 32.988 | 38.696 | +5.708 |
| SAR-Ship 子集 | 27.408 | 29.918 | +2.511 |
| SAR-Aircraft 子集 | 5.664 | 6.419 | +0.756 |
| OGSOD 子集 | 3.051 | 2.873 | -0.178 |

## 15. 对当前论文最有用的筛选结论

1. 已有严格成对证据以 OGSOD 为主：AP75 +0.760 pp，AP50:95 +1.300 pp。
2. SSDD、SAR-Ship 和 SAR-Aircraft 均有本地结果，但旧对照存在协议问题；需要统一重跑 baseline 与 ours。
3. 新增小数据集优先级：DSSDD（HBB、双极化）> SADD（飞机 HBB）> SIVED（车辆 OBB/HBB 转换）。
4. AIR-SARShip-1.0/2.0 样本过少且切片差异大，适合作为补充或低数据实验，不宜单独承担主结论。
5. SRSDD 与 SIVED 的官方主任务是 OBB；若本文保持 HBB，文献 OBB 数字只能作为背景。
6. HRSID 规模中等、认可度较高，若算力允许，是比 AIR-SARShip 更稳的舰船泛化数据集。
