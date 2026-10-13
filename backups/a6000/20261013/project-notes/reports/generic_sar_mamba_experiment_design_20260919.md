---
title: IG-DS-SSM/PFIM 通用 SAR-Mamba 实验设计
date: 2026-09-19
status: design-only
---

# 1. 论文定位

本文不把 IG-DS-SSM/PFIM 定义为检测头或检测器专用模块，而定义为：

> 面向 SAR 视觉任务的通用 Mamba 状态更新方法。

DS-SSM 负责在状态更新前进行 SAR-aware 的残差修正和能量抑制，PFIM 根据局部趋势与二维变化估计 token/区域重要性，并调制状态保留、写入和步长。模块通过统一的特征输入/输出接口嵌入不同视觉 Mamba 主干；检测头、分类头和变化检测头属于任务适配层，不作为本文主要创新。

# 2. 论文主张边界

可以主张：

- IG-DS-SSM/PFIM 能把 SAR 噪声先验和局部结构重要性引入 Mamba 状态更新；
- 该状态更新思想可以迁移到检测、分类和变化检测；
- 模块在较小参数和计算开销增量下改善部分 SAR 任务的有效指标。

不能主张：

- 在所有 SAR 任务或所有数据集上都超过 baseline；
- 不同任务使用不同主干、不同 head、不同协议时仍可直接比较；
- 语义分割当前负结果已经证明模块具有通用优势。

# 3. 实验任务分层

## 3.1 主验证任务：目标检测

数据集：SSDD、SAR-ShipDataset、SAR-AIRcraft1.0。

目标：提供最完整的定量证据、P3 位置消融、PFIM 消融和复杂度对比。

核心比较：

1. 原始 Mamba 主干 + 原任务头；
2. 相同 Mamba 主干 + DS-SSM；
3. 相同 Mamba 主干 + DS-SSM/PFIM。

YOLOv8n/YOLOv5s 等 CNN 方法只作为轻量外部参照，不替代 Mamba-YOLO-T baseline。

## 3.2 跨任务验证：图像分类

数据集：优先 SAR-ACD；MSTAR-SOC 和 SAMPLE 在闭集、指标一致、协议有效后加入。

比较：原始 VMamba/视觉 Mamba 主干与加入 IG-DS-SSM/PFIM 的同一主干，使用相同分类头。

指标：Accuracy、Macro-F1、Params、GFLOPs、FPS。分类任务不使用 mAP。

## 3.3 跨任务验证：变化检测

数据集：Ottawa、Farmland、Yellow River。

比较：相同 Siamese Mamba 主干和变化检测头，仅替换状态更新模块。

指标：Precision、Recall、F1、IoU、OA、Params、GFLOPs、FPS。

三套数据集的结果完整保留。Farmland 的明显提升、Ottawa 的小幅提升和 Yellow River 的下降共同用于说明场景适用性与边界。

## 3.4 探索性任务：语义分割

数据集：FUSAR-Map、OpenEarthMap-SAR。

当前只用于分析：大面积水体/植被与细长道路、复杂边界对状态抑制机制的不同影响。除非后续统一 head、协议和训练策略后取得稳定结果，否则不把分割作为通用性主证据。

# 4. 统一实验矩阵

## 4.1 每个任务的最小主矩阵

| 组别 | 模型 | 作用 |
|---|---|---|
| B0 | 原始 Mamba 主干 + 原任务头 | 直接 baseline |
| B1 | B0 + DS-SSM | 验证 SAR-aware 状态更新 |
| B2 | B0 + DS-SSM + PFIM | 完整方法 |
| B3 | B2 的参数匹配 global importance | 排除参数量/容量因素 |

B1/B2/B3 在 SSDD 上做完整消融；分类和变化检测至少完成 B0/B2，资源允许时补 B1。

## 4.2 结构和效率分析

- 检测任务：P2/P3/P4 插入位置；P3 为当前候选最优位置。
- PFIM：no-PFIM、global importance、去除二维变化分支、完整 PFIM。
- 统一 profiler：Params、GFLOPs、batch=1 latency、FPS。
- 可视化：原始特征、DS-SSM 后特征、PFIM importance、成功/失败样例和类别级指标。

# 5. 数据和协议原则

- 每个数据集内 baseline 与 ours 使用完全相同划分、输入尺寸、epoch、batch、优化器、初始化和 evaluator。
- 训练 checkpoint 只由验证集选择；测试集只评估一次。
- 文献报告值、历史恢复值和本地同协议实测值分栏保存。
- 不使用 OGSOD 作为当前主实验；其历史记录保留，不重新启动旧队列。
- 不因为结果好坏删除同协议负结果；负结果用于定义适用边界。

# 6. 论文章节建议

1. Introduction：SAR 散斑/杂波进入 Mamba 状态递推的问题。
2. Method：通用 DS-SSM、PFIM、状态更新接口和复杂度。
3. Experimental Setup：三个任务、数据集、协议和评价指标。
4. Main Results：检测主结果、分类跨任务结果、变化检测跨任务结果。
5. Ablation and Analysis：DS-SSM/PFIM、插入阶段、重要性引导、效率、可视化。
6. Limitations：分割中细长结构受抑制、Yellow River 场景下降、任务依赖性。
7. Conclusion：通用状态更新思想有效，但收益受任务和场景结构影响。

# 7. 当前执行状态

本文件只记录实验设计重构，不表示已经启动新训练，也不表示所有待列任务已经完成。后续正式执行前需冻结每个任务的 baseline/ours 配置并生成对应的结果索引。
