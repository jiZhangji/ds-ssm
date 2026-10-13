# 四数据集方法、指标与开源状态审计（2026-08-23）

## 1. 范围与状态定义

本文面向四个数据集：SSDD、OGSOD-1.0、SAR-ShipDataset、SAR-AIRcraft-1.0。

状态定义：

- **A：完整官方开源**：作者仓库包含方法代码，并有可用训练入口；若还有精确配置、日志和权重则特别注明。
- **B：官方仓库不完整**：有作者仓库，但缺少该数据集的精确配置、权重或完整复现流程。
- **C：基础框架开源**：通用检测器有官方实现，但文献中的该数据集结果不是官方 checkpoint 可直接复现。
- **D：未找到官方代码**：截至本次检索，没有找到作者公开的完整方法代码。

所有表中的数值均为论文/仓库作者报告值，除非明确标记为“本地”。不同论文的数据划分、输入尺寸、预训练、HBB/OBB、评估器和置信度可能不同，不能直接组成统一排行榜。

主要综合来源：

- MaDiNet 原论文三数据集表：https://arxiv.org/html/2411.07500
- SAR ATR 综述方法表：https://arxiv.org/abs/2509.22159
- SARDet-100K 数据与工具：https://github.com/zcablii/sardet_100k
- Official SSDD：https://github.com/TianwenZhang0825/Official-SSDD

## 2. 数据集协议

| 数据集 | 任务 | 规模/本地划分 | 主要协议风险 |
|---|---|---|---|
| SSDD | 单类舰船 HBB | 官方 1160；通常 928 train / 232 test；本地开发为 742/186/232 | 很多论文仍使用不同旧版、8:2 随机划分或只报 AP50 |
| OGSOD-1.0 | bridge/harbor/tank；原始数据为 optical-SAR 配对 | 本地 SAR-only 14664/1834/1833 | SAR-only 与 optical+SAR 不公平；另有 HBB/OBB 两种任务 |
| SAR-ShipDataset | 单类舰船 HBB | 39729 图；本地 31783/3972/3974 | 原始论文划分和本地固定划分未必逐图一致 |
| SAR-AIRcraft-1.0 | 7 类细粒度飞机检测 | 4368 原图/16463实例；本地使用 512 patch 派生版 | 原图协议、patch 协议及 VOC07/VOC12/COCO 指标并存 |

## 3. SSDD

### 3.1 MaDiNet 论文中的统一 AP50 表

该表来自 MaDiNet 作者的复现实验。`C` 表示方法/基础框架有公开代码；它不代表我们已得到对应 run 的配置和权重。

| 方法 | AP50 (%) | 开源状态 | 说明 |
|---|---:|---|---|
| SAR-ShipNet | 89.1 | D | 专用方法未开源 |
| DCTC | 93.9 | D | 未开源 |
| GEFCSI-Net | 97.1 | D | 未开源 |
| SARNas | 98.5 | D | 未开源 |
| MGA-Net | 98.6 | D | 未开源 |
| Faster R-CNN | 89.7 | C | MMDetection/Detectron2 可复现 |
| Cascade R-CNN | 90.5 | C | MMDetection 可复现 |
| RetinaNet | 95.0 | C | MMDetection 可复现 |
| CenterNet | 95.1 | C | 通用实现可复现 |
| FCOS | 95.3 | C | MMDetection 可复现 |
| ConsistencyDet | 95.6 | C | 官方框架开源 |
| YOLOv3 | 96.2 | C | 通用实现可复现 |
| YOLOX | 96.2 | C | 官方实现可复现 |
| DiffusionDet | 96.3 | C | 官方框架开源 |
| YOLOv5n | 98.0 | C | Ultralytics 开源 |
| YOLOv8n | 98.1 | C | Ultralytics 开源 |
| YOLOv5s | 98.3 | C | Ultralytics 开源 |
| DiffDet4SAR | 96.9 | A/B | 官方代码有；SSDD 精确配置不完整 |
| MaDiNet | 99.0 | B | 作者仓库有，但 SSDD 完整配置/权重链不清晰 |

来源：https://arxiv.org/html/2411.07500

### 3.2 近年轻量与 SAR 专用方法

| 方法 | AP50 (%) | AP50:95 (%) | Params (M) | GFLOPs | 开源状态 | 备注 |
|---|---:|---:|---:|---:|---|---|
| AMANet | 98.5 | 74.2 | - | - | D | 标准划分描述较完整 |
| CCAI-YOLO | 98.8 | 74.9 | 4.754 | 10.1 | D | 2026；近本文规模 |
| SSD-YOLO | 99.0 | 74.3 | - | - | D | YOLOv8n 系 SAR 抗噪方法 |
| YOLO-SRBD | 98.3 | 74.5 | 7.72 | 21.5 | D | YOLOv8 系方法 |
| RSNet | - | 72.5 | 1.49 | - | D | 论文称将开源，当前未找到 |
| SFS-CNet | 99.6 | - | - | - | A | CVPR 2024 官方代码 |
| LRTransDet | 97.8 | - | 3.07 | 3.85 | D | 轻量 ViT |
| YOLOSAR-Lite | 95.32 | - | 2.05 | 4.48 | D | 文中 mAP 主要为 AP50 口径 |
| ARGO-Net | 99.5 | 79.3 | 2.30 | 6.9 | D | 2026；协议与官方划分需复核 |
| YOLOv8n（ARGO-Net文中重跑） | 98.4 | 75.2 | 3.01 | 8.1 | C | 同一论文内统一基线 |
| RT-DETR-R18（ARGO-Net文中重跑） | 98.9 | 77.1 | 20.0 | 60.0 | C | 非轻量同规模 |

来源：

- AMANet：https://arxiv.org/html/2401.13214
- CCAI-YOLO：https://www.mdpi.com/2072-4292/18/1/145/xml
- SSD-YOLO：https://ship-research.com/article/doi/10.19693/j.issn.1673-3185.04729?viewType=HTML
- YOLO-SRBD：https://www.sciencedirect.com/science/article/pii/S2405959524000201
- RSNet：https://arxiv.org/abs/2410.23073
- SFS-CNet 代码：https://github.com/like413/SFS-Conv
- LRTransDet：https://www.mdpi.com/2072-4292/15/22/5309
- YOLOSAR-Lite：https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2405525
- ARGO-Net：https://www.mdpi.com/2077-1312/14/14/1331

### 3.3 本地当前结果

| 结果 | AP50 (%) | AP50:95 (%) | 状态 |
|---|---:|---:|---|
| PFIM ex27 历史档案 | 98.704 | 74.022 | 历史最佳；初始化/协议未完全对齐 |
| H100 baseline | - | 54.8384 | 742/186 开发集 VAL，scratch |
| H100 P2-d16 | - | 56.6725 | 742/186 开发集 VAL，scratch；当前 9/16 暂定最佳 |

这些本地开发结果不能与 official test 文献值直接比较。

## 4. SAR-ShipDataset

### 4.1 MaDiNet 论文 AP50 表

| 方法 | AP50 (%) | 开源状态 | 说明 |
|---|---:|---|---|
| OE-YOLO | 86.0 | D | 未开源 |
| SAR-ShipNet | 90.2 | D | 未开源 |
| ShipDetector | 91.2 | D | 未开源 |
| Improved FCOS | 94.1 | D | 专用改进未开源 |
| GEFCSI-Net | 94.3 | D | 未开源 |
| CMFT | 94.5 | D | 未开源 |
| YOLOv4 | 77.9 | C | 通用实现开源 |
| YOLOv7 | 86.1 | C | 通用实现开源 |
| YOLOv3 | 88.8 | C | 通用实现开源 |
| YOLOv5n | 89.3 | C | Ultralytics 开源 |
| CenterNet | 92.1 | C | 通用实现开源 |
| FCOS | 92.8 | C | MMDetection 开源 |
| Cascade R-CNN | 92.7 | C | MMDetection 开源 |
| Faster R-CNN | 93.1 | C | MMDetection 开源 |
| RetinaNet | 93.4 | C | MMDetection 开源 |
| ConsistencyDet | 94.2 | C | 官方框架开源 |
| DiffusionDet | 95.0 | C | 官方框架开源 |
| DiffDet4SAR | 95.1 | A/B | 作者代码有；本数据集精确 config 不完整 |
| MaDiNet | 97.5 | B | 作者仓库有；精确复现链不完整 |

来源：https://arxiv.org/html/2411.07500

### 4.2 其他可核对结果

| 方法 | AP50 (%) | AP50:95 (%) | Params (M) | GFLOPs | 开源状态 |
|---|---:|---:|---:|---:|---|
| Faster R-CNN（LRTransDet文中） | 92.7 | - | 41.3 | 26.2 | C |
| RetinaNet（LRTransDet文中） | 87.2 | - | 36.3 | 13.1 | C |
| FCOS（LRTransDet文中） | 94.7 | - | 32.1 | 12.57 | C |
| YOLOX（LRTransDet文中） | 89.5 | - | 8.94 | 2.13 | C |
| YOLOv5s（LRTransDet文中） | 94.8 | - | 7.03 | 2.56 | C |
| LRTransDet | 95.1 | - | 3.07 | 0.96 | D |
| SRDet | 95.1 | 65.9 | 35.2 | - | D |
| YOLOSAR-Lite | 96.06 | - | 2.05 | 4.48 | D |
| DAFDet | 96.5 | - | - | - | D |

来源：

- LRTransDet：https://www.mdpi.com/2072-4292/15/22/5309
- SRDet：https://www.mdpi.com/2072-4292/15/1/103
- YOLOSAR-Lite：https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2405525
- 综合方法表：https://arxiv.org/abs/2509.22159

### 4.3 本地当前结果

| 方法 | AP50 (%) | AP50:95 (%) | 状态 |
|---|---:|---:|---|
| Mamba-YOLO-T 档案 | 94.6158 | 49.5684 | batch 中途改变，不能作严格因果基线 |
| IG-DS-SSM/PFIM-T | 94.8440 | 57.8182 | 已完成本地 test |
| IG-DS-SSM/PFIM-S | 95.6093 | 53.1767 | 已完成本地 test；不是轻量主模型 |

本地固定随机划分未确认与 MaDiNet 的未公开逐图划分完全一致。

## 5. SAR-AIRcraft-1.0

### 5.1 MaDiNet 论文 AP50 表

| 方法 | AP50 (%) | 开源状态 | 说明 |
|---|---:|---|---|
| SKG-Net | 70.5 | D | 未开源 |
| SA-Net | 77.7 | D | 未开源 |
| MLSDNet | 82.7 | D | 未开源 |
| FCOS | 54.8 | C | MMDetection 可复现 |
| CenterNet | 70.5 | C | 通用实现可复现 |
| RetinaNet | 72.3 | C | MMDetection 可复现 |
| RepPoints | 72.6 | C | MMDetection 可复现 |
| Cascade R-CNN | 75.4 | C | MMDetection 可复现 |
| Faster R-CNN | 75.9 | C | MMDetection 可复现 |
| YOLOX | 81.0 | C | 官方实现开源 |
| YOLOv3 | 83.9 | C | 通用实现开源 |
| ConsistencyDet | 85.1 | C | 官方框架开源 |
| DiffusionDet | 86.6 | C | 官方框架开源 |
| YOLOv5n | 88.2 | C | Ultralytics 开源 |
| YOLOv8n | 88.2 | C | Ultralytics 开源 |
| DiffDet4SAR | 88.4 | A | 官方代码；有该数据集说明 |
| MaDiNet | 90.5 | B | 作者代码有；完整 config/checkpoint 不清晰 |

来源：https://arxiv.org/html/2411.07500

### 5.2 近年方法与不同指标体系

| 方法 | AP50 (%) | AP75 (%) | AP50:95/其他 AP (%) | 开源状态 | 备注 |
|---|---:|---:|---:|---|---|
| SADRN | 95.0 | 71.5 | - | D | 原始集成检测识别论文，协议与 MaDiNet 表不同 |
| SFS-CNet | 89.7 | - | - | A | CVPR 2024 官方代码 |
| PGD-YOLOv8 | 90.7 | - | - | A | 官方代码与预训练权重 |
| MSAD-B | 90.3 | - | - | D | VSSM 航空检测方法 |
| SARDet-MIM | 87.0 | 70.7 | COCO AP 60.5 | D | 自监督预训练，不能与 scratch 混排 |
| MSPaDet | - | - | mAP 68.65 / AP12 69.78 | D | MMRotate；指标不是 AP50 |
| DenoDet | - | - | VOC07 68.60 / VOC12 69.56 | A | 完整 config/log/checkpoint |
| DenoDet V2 | - | - | VOC07 69.93 / VOC12 70.73 | A | 完整 config/log/checkpoint |

来源：

- SAR-AIRcraft 集成检测识别：https://www.mdpi.com/2072-4292/16/13/2471
- SFS-CNet：https://github.com/like413/SFS-Conv
- PGD：https://github.com/XAI4SAR/PGD
- MSAD：https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70032
- SARDet-MIM：https://www.mdpi.com/2072-4292/18/4/580
- MSPaDet：https://doi.org/10.3390/rs18101513
- DenoDet/DenoDet V2：https://github.com/GrokCV/GrokSAR
- DiffDet4SAR：https://github.com/JoyeZLearning/DiffDet4SAR

### 5.3 本地 patch 协议结果

| 方法/划分 | AP50 (%) | AP50:95 (%) | 状态 |
|---|---:|---:|---|
| Mamba-YOLO-T reference VAL | 87.7509 | 69.9035 | 约137 epoch、SGD；协议不匹配 |
| PFIM-T VAL | 88.8252 | 70.5129 | 300 epoch、AdamW |
| Mamba-YOLO-T reference TEST | 70.4018 | 54.5131 | 协议不匹配 |
| PFIM-T TEST | 70.8220 | 55.5516 | 300 epoch、AdamW |
| PFIM-S VAL | 89.3042 | 72.0249 | 最终 FP32 验证；21.74M 参数、48.0G，不是轻量主模型 |
| 历史 PFIM-T 档案 VAL | 89.0660 | 72.5910 | 3090 历史运行；仅作档案证据，训练协议未严格受控 |

这里的本地数据来自 SARDet-100K 标准化流程：4368 张原图先按原图 ID 划分，再裁成 512x512、重叠 200 像素的 patch，得到 13976/1923/2989 个 train/val/test patch。它与 MaDiNet 使用的 4368 张原图协议以及 PGD 等论文的具体评估实现不完全一致，不能直接排名。

必须同时区分 split 和指标：本地 PFIM-T 在 VAL 上是 88.8252 AP50，而在分布更困难的 TEST 上是 70.8220 AP50；DenoDet V2 报告的 69.93 是 COCO-style AP50:95，不是 AP50。因而“约 89 AP50”“约 70 test AP50”和“69.93 COCO AP”描述的是不同口径，并不互相矛盾。

**论文主比较口径：** 当前 SAR-AIRcraft-1.0 文献对照表按验证结果比较，因此本文主表使用 PFIM-T VAL 的 88.8252 AP50 / 70.5129 AP50:95；若报告容量版本，则使用 PFIM-S VAL 的 89.3042 / 72.0249。PFIM-T TEST 的 70.8220 / 55.5516 只进入补充材料的跨域泛化分析，不与 MaDiNet、PGD、DiffDet4SAR 的验证结果混排。

## 6. OGSOD-1.0

### 6.1 SAR-only HBB 结果

以下来自 SAI-Net 论文的同文统一训练表。参数列沿用论文 `Size` 数值。

| 方法 | AP50:95 (%) | AP50 (%) | AP75 (%) | Params/Size (M) | GFLOPs | 开源状态 |
|---|---:|---:|---:|---:|---:|---|
| Faster R-CNN | 33.0 | 67.4 | 27.9 | 41.3 | 134.0 | C |
| Cascade R-CNN | 38.9 | 73.3 | 37.6 | 69.3 | 162.0 | C |
| ATSS | 42.7 | 76.8 | 40.5 | 38.9 | 68.3 | C |
| RetinaNet | 38.4 | 67.8 | 38.5 | 36.5 | 130.0 | C |
| YOLOv8n | 49.3 | 80.3 | 48.8 | 5.4 | 6.8 | C |
| YOLOv10n | 49.8 | 81.0 | 49.5 | 5.5 | 6.5 | C |
| YOLO11n | 49.3 | 79.5 | 49.2 | 5.3 | 6.3 | C |
| YOLO12n | 48.6 | 79.5 | 48.5 | 5.2 | 5.8 | C |
| YOLO13n | 47.9 | 79.2 | 48.1 | 5.2 | 6.1 | C |
| RT-DETR | 49.4 | 88.5 | 48.5 | 77.0 | 56.9 | C |
| SAI-Net | 51.8 | 90.2 | 50.7 | 73.0 | 68.6 | D | 

来源：https://www.nature.com/articles/s41598-026-48542-y_reference.pdf

### 6.2 本地 SAR-only 结果

| 方法 | AP50:95 (%) | AP50 (%) | AP75 (%) | Params (M) | GFLOPs@640 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Mamba-YOLO-T | 42.3018 | 77.1733 | 40.2109 | 5.9846 | 13.6 | matched single run |
| IG-DS-SSM/PFIM-T | 43.6021 | 77.2630 | 40.9711 | 5.9638 | 13.2 | matched single run |

本地使用 14664/1834/1833、1024 输入；SAI-Net 论文使用 14665 train / 3666 test、640 输入。即使都是 SAR-only，也不能直接认定同协议。

### 6.3 Optical+SAR 多模态 HBB 结果

| 方法 | 模态 | AP50 (%) | AP50:95 (%) | Params (M) | GFLOPs | 开源状态 |
|---|---|---:|---:|---:|---:|---|
| YOLOv11n | SAR | 85.5 | 55.0 | 2.6 | 6.4 | C |
| YOLOv11s | SAR | 92.5 | 63.3 | 9.4 | 21.3 | C |
| YOLOv11m | SAR | 95.6 | 68.7 | 20.0 | 67.7 | C |
| SuperYOLO | Optical+SAR | 94.7 | 53.8 | 4.8 | 56.2 | A/C |
| ICAFusion | Optical+SAR | 94.8 | 64.1 | 20.2 | 29.8 | A |
| CFT | Optical+SAR | 95.7 | 62.9 | 44.8 | 35.8 | A |
| YOLO-CMFM | Optical+SAR | 96.2 | 75.1 | - | - | D |
| MAIENet | Optical+SAR | 90.8 | 61.0 | 34.0 | 27.6 | D |

来源：

- YOLO-CMFM：https://www.mdpi.com/2072-4292/18/1/136
- ICAFusion 代码：https://github.com/chanchanchan97/ICAFusion
- CFT 代码：https://github.com/DocF/multispectral-object-detection
- MAIENet：https://www.mdpi.com/2072-4292/17/23/3866

### 6.4 Optical+SAR OBB 结果

M4-SAR 论文对 OGSOD-1.0 OBB 的统一表如下。该任务同时改变了输入模态与框类型，只能作为背景材料。

| 方法 | AP50 (%) | AP75 (%) | AP50:95 (%) | 开源状态 |
|---|---:|---:|---:|---|
| CFT | 92.9 | 72.1 | 65.3 | A |
| CLANet | 93.5 | 72.9 | 66.3 | A |
| CSSA | 92.1 | 69.3 | 63.3 | B（第三方重实现） |
| CMADet | 91.3 | 67.6 | 62.9 | A/B |
| ICAFusion | 93.5 | 72.7 | 66.1 | A |
| MMIDet | 93.5 | 73.5 | 66.7 | A |
| E2E-OSDet | 93.6 | 74.3 | 67.3 | A（代码和权重） |

来源与统一实现：https://github.com/wchao0601/M4-SAR

## 7. 哪些方法实际值得我们复现

### 四数据集统一轻量主表

| 方法 | 四数据集可运行性 | 推荐程度 |
|---|---|---|
| YOLOv5n | 高 | 必做；旧但稳定 |
| YOLOv8n | 高 | 必做；最通用轻量基线 |
| YOLOv10n 或 YOLOv10-S | 高 | 必做；新型端到端 YOLO |
| Mamba-YOLO-T | 高 | 核心必做；最直接结构基线 |
| IG-DS-SSM/PFIM-T | 高 | 本文方法 |

### 数据集专用补充

| 方法 | 数据集 | 建议 |
|---|---|---|
| SFS-CNet | SSDD、SAR-AIRcraft | 官方代码有，但仓库较精简；可作为一个 SAR 专用 CNN 尝试 |
| PGD-YOLOv8 | SAR-AIRcraft | 代码和权重较完整；适合作为 aircraft 专用补充 |
| DenoDet V2 | SAR-AIRcraft | 代码/config/log/checkpoint 最完整之一，但模型大且指标为 VOC AP |
| DiffDet4SAR | SSDD、ShipDataset、Aircraft | 代码有，复现成本高；只在 Aircraft 尝试最合理 |
| MaDiNet | 三数据集 | 概念最相关，但当前精确复现链不足；优先联系作者 |
| OGSOD 多模态方法 | OGSOD | 不进入 SAR-only 主表；除非论文增加 optical+SAR 分支 |

## 8. 论文报告原则

1. 统一主表只使用我们自己在固定 split、相同 evaluator 下得到的结果。
2. 主表可全部从随机初始化开始，标题明确写 `Unified training from scratch`。
3. Mamba-YOLO-T 与本文方法必须完全同协议并做三个随机种子。
4. 外部方法的官方论文值放在 `Author-reported` 表；即使同名 YOLO 的数值与我们不同，也同时保留而不覆盖。
5. 每行记录 split hash、输入尺寸、预训练、epoch、batch、优化器、代码 commit 和 evaluator。
6. OGSOD 必须分 SAR-only HBB、Optical+SAR HBB、Optical+SAR OBB 三组。
7. SAR-AIRcraft 必须分原图与 patch 协议，并区分 AP50、COCO AP、VOC07 AP、VOC12 AP。
8. 四数据集中的 AP50:95 才是主指标；只报告 AP50 的文献不能用来证明比本文更强或更弱。

## 9. 检索结论

- SSDD 文献最多，但协议最混乱；高于 74 的 AP50:95 并不自动意味着严格优于本文历史 74.022。
- OGSOD 的高分主要来自多模态输入或不同 split；本文 SAR-only 结果应与 SAR-only 统一复现模型比较。
- SAR-ShipDataset 可复现的专用模型很少，通用 YOLO/Mamba 的统一重跑价值更高。
- SAR-AIRcraft 的 PGD、SFS-CNet、DiffDet4SAR、DenoDet V2 具备代码，其中 PGD 与本文任务最直接，DenoDet V2 的工程证据最完整。
- 真正能在四个数据集上公平统一运行的仍是通用轻量模型与 Mamba-YOLO-T。论文的可信度应来自统一协议、多数据集和多随机种子，而不是拼接不兼容的作者报告数字。
