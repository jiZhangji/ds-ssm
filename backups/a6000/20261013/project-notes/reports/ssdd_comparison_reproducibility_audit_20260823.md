# SSDD 对比方法与复现可行性审计（2026-08-23）

## 1. 论文贡献边界

本文只保留三项贡献：

1. SAR-aware DS-SSM 状态更新。
2. PFIM 对状态保留、写入和步长的引导。
3. 在约 6M 参数、13G FLOPs 下取得较好的精度与效率平衡。

因此，对比实验必须分别回答三个问题：

- 与原始 Mamba-YOLO-T 相比，SAR-aware 状态更新是否有效？
- PFIM 是否优于无 PFIM、全局重要性和单一门控调制？
- 与 CNN、Transformer 和其他 SAR 专用轻量检测器相比，精度、参数量、计算量和速度是否形成有竞争力的 Pareto 点？

## 2. 推荐的统一协议主表

主表只放由我们在同一代码快照、同一 SSDD 划分和同一评估器下实际训练的方法。

| 方法 | 路线 | 官方实现/权重 | SSDD 适配难度 | 建议 | 作用 |
|---|---|---:|---:|---|---|
| YOLOv8n | 轻量 CNN | 有/有 | 低 | 必做 | 经典轻量 CNN 参照 |
| YOLOv10-S | 端到端 CNN | 有/有 | 低-中 | 必做 | 近期端到端 YOLO；参数量较接近 |
| RT-DETR-R18 或 RT-DETRv2-R18 | Transformer | 有/有 | 中 | 必做 | 跨范式比较；注明其规模更大 |
| Mamba-YOLO-T | SSM | 有/有 | 很低 | 核心必做 | 最直接的结构基线，隔离本文模块贡献 |
| IG-DS-SSM/PFIM-T | 本文 | 本项目 | 已完成 | 核心必做 | 完整方法 |
| Faster R-CNN R50-FPN | 两阶段 CNN | 有/有 | 中 | 可放附录或扩展主表 | 经典两阶段参照 |

统一协议建议：

- 开发阶段使用 742 train / 186 val 选定结构和超参数。
- 最终结构固定后，将 train+val 合并为官方 928 train，在 232 official test 上评估。
- Mamba-YOLO-T 与本文方法必须做 seed 0/1/2，并报告均值和标准差。
- 其他外部基线至少做一个固定 seed；若论文预算允许，YOLOv8n、YOLOv10-S、RT-DETR-R18 也做三种子。
- 主结果统一使用 640 输入、standard inference、同一 COCO evaluator；不使用 TTA 选模。
- 报告 P、R、AP50、AP75、AP50:95、Params、GFLOPs、FPS/latency。
- Params/FLOPs/FPS 必须由同一 profiler、同一硬件、同一精度和 batch=1 重新测量。

## 3. 官方代码完整、适合复现的方法

### Mamba-YOLO-T：最高优先级

- 官方仓库：https://github.com/HZAI-ZJNU/Mamba-YOLO
- 官方模型表给出 T 版本约 5.8M 参数、13.2G FLOPs，并提供 COCO 权重。
- 与本文共享主体结构，是唯一能够直接证明“提升来自 DS-SSM/PFIM，而不是换了检测器”的外部基线。
- 正式实验应包含两套公平口径：严格 scratch 对照；可选的相同 Mamba-YOLO COCO 权重初始化对照。本文新增参数保持随机初始化。

### YOLOv8n：低风险轻量 CNN 基线

- 官方文档：https://docs.ultralytics.com/models/yolov8/
- 约 3.2M 参数、8.7G FLOPs，训练和数据接口成熟。
- 规模小于本文方法，不是严格参数匹配，但能证明本文并非只与较弱旧模型比较。

### YOLOv10-S：近期端到端 YOLO 基线

- 官方仓库：https://github.com/THU-MIG/yolov10
- 官方表给出 S 版本约 7.2M 参数、21.6G FLOPs；N 版本约 2.3M、6.7G。
- 主表建议用 S：参数量更接近约 6M；附录可补 N 以观察低计算量端点。
- 建议使用独立环境，避免其 Ultralytics 版本与本项目 fork 相互污染。

### RT-DETR-R18 / RT-DETRv2-R18：Transformer 路线

- 官方仓库：https://github.com/lyuwenyu/RT-DETR
- 官方配置与预训练权重包含 R18 版本，支持自定义 COCO 数据。
- 不是参数或 FLOPs 匹配模型，其意义是跨范式比较；表中必须同时列复杂度，避免只比精度。
- 需要把 SSDD YOLO 标签转换为 COCO JSON，环境和评估接口适配属于中等工作量。

### Faster R-CNN：经典两阶段补充

- 可用 MMDetection 官方实现；社区 SSDD 配置参考：https://github.com/ALLARDLE/ShipSARDetect_mmdetection
- 该社区仓库提供 Faster/Cascade R-CNN 的 SSDD 配置，但不能称为某篇 SAR 方法的官方复现。
- 若主表篇幅或算力有限，可移到附录；它不是本文最关键对手。

## 4. 有代码但复现风险较高的方法

### MaDiNet：概念上最接近，工程上不完整

- 官方仓库：https://github.com/JoyeZLearning/MaDiNet
- 论文：https://arxiv.org/abs/2411.07500
- 它是 SAR + Mamba 的重要相关工作，必须在相关工作和文献结果表中出现。
- 仓库公开了代码框架，但没有清晰、完整的 SSDD detector 配置、权重和一键复现流程；README 仍大量依赖 DiffusionDet。
- 本地已测的 223.2648M 参数、345.283G 仅是其 MambaSAR res4+res5 核心组件，不是完整 detector，不能作为 MaDiNet 完整复杂度。
- 建议先联系作者索要 SSDD config/checkpoint；拿不到时只引用作者报告结果，不投入大量时间猜配置。

### PPA-Net：与 SAR 噪声贡献高度相关

- 论文：https://www.mdpi.com/2072-4292/15/11/2855
- 作者代码：https://github.com/mrzhao158/ship-detection
- 模块直接针对复杂背景和多尺度舰船，适合检验本文的 SAR-aware 动机。
- 仓库缺少完整 README、发布权重和成熟训练入口，复现风险为中高。
- 推荐顺序：先做代码可运行性审计；若两天内不能得到可信 baseline，则降级为文献结果，并使用“固定去斑 + Mamba-YOLO-T”作为更可控的机制对照。

### R-Sparse R-CNN：代码完整但任务不一致

- 官方仓库：https://github.com/ka-mirul/R-Sparse-R-CNN
- 有配置、训练说明和权重，但目标是旋转框检测。
- 本文当前是水平框检测，不能把其 OBB 指标直接放入主表；只有扩展 OBB 实验时才适合放附录。

## 5. 只能作为作者报告结果的方法

以下方法在本次检索中没有找到足以进行官方同协议复现的完整代码、配置与权重。应单独建立“Author-reported results”表，逐行注明划分、输入尺寸和指标口径。

| 方法 | 作者报告的 SSDD 结果 | 代码状态 | 主要限制 |
|---|---|---|---|
| AMANet | AP50:95 74.20，AP50 98.50 | 未找到官方代码 | 500 epochs；不能冒充本地复现 |
| CCAI-YOLO | AP50:95 74.9，AP50 98.8，4.754M，10.1G | 未找到官方代码 | 2026 方法；精确文件划分与本文未必一致 |
| SSD-YOLO | AP50:95 74.3，AP50 99.0 | 未找到官方代码 | 与本文 SAR 抗噪动机相关，但无法统一复现 |
| YOLO-SRBD | AP50:95 74.5，AP50 98.3，7.72M，21.5G | 未找到官方代码 | 训练细节可见，工程实现不可得 |
| RSNet | AP50:95 72.5，1.49M | 论文称将发布，当前未找到 | 很强的轻量文献参照 |
| MCEM | AP50:95 70.2，AP50 96.9，AP75 84.7 | 未找到官方代码 | 512 输入、150 epochs、batch 4，协议不同 |
| YOLOSAR-Lite | 报告 mAP 95.32，2.05M，4.48G | 未找到官方代码 | “mAP”更接近 AP50，不能与 AP50:95 混排 |
| MaDiNet | AP50 99.0，未报告可比 AP50:95 | 代码不完整 | 无法形成严格统一复现 |

主要来源：

- SSDD 官方仓库：https://github.com/TianwenZhang0825/Official-SSDD
- SSDD 数据集论文：https://www.mdpi.com/2072-4292/13/18/3690
- AMANet：https://arxiv.org/html/2401.13214
- CCAI-YOLO：https://www.mdpi.com/2072-4292/18/1/145/xml
- SSD-YOLO：https://ship-research.com/article/doi/10.19693/j.issn.1673-3185.04729?viewType=HTML
- YOLO-SRBD：https://www.sciencedirect.com/science/article/pii/S2405959524000201
- RSNet：https://arxiv.org/abs/2410.23073
- MCEM：https://pmc.ncbi.nlm.nih.gov/articles/PMC12473879/
- YOLOSAR-Lite：https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2405525

## 6. 不进入水平框主表的方法

- DiffDet4SAR：官方仓库主要面向 SAR-AIRcraft1.0，不是 SSDD 舰船统一协议。仓库：https://github.com/JoyeZLearning/DiffDet4SAR
- R-Sparse R-CNN、MSDFF-Net：旋转框任务，与当前 HBB 评估不同。
- SLA-Net：使用额外海陆分割掩码和 CFAR 先验，输入信息不公平。论文：https://www.mdpi.com/2072-4292/17/21/3576
- SGDANet：少样本域适应协议，不是常规 SSDD 全监督检测。
- LS-SSDD / MIT-DETR 等：数据集或任务定义不同。

这些方法可在相关工作或专门附录中讨论，但不能参与本文“统一评估重排 SOTA”。

## 7. 与三项贡献直接对应的实验

### 贡献 1：SAR-aware DS-SSM 状态更新

- Mamba-YOLO-T。
- Mamba-YOLO-T + 固定 Lee/常规去斑预处理。
- DS-SSM without PFIM。
- 完整 IG-DS-SSM/PFIM。

固定去斑对照非常重要：它回答收益是否只是普通预处理造成，而不是状态更新设计带来的。

### 贡献 2：PFIM 引导 retention/write/step size

- no-PFIM。
- 参数量匹配的 global importance。
- full token PFIM。
- gamma-only、write-only、delta-only、gamma+write、full。
- PFIM 局部趋势/二维变化输入消融。

注意当前 CUDA 路径中 gamma 和 write 都因子化调制 scan drive，论文中应描述其功能分工，不应宣称它们是完全独立的动态。

### 贡献 3：约 6M/13G 的精度效率平衡

- 与 YOLOv8n、YOLOv10-S、RT-DETR-R18、Mamba-YOLO-T 画 AP50:95-Params、AP50:95-GFLOPs 和 AP50:95-latency 散点图。
- 所有模型同机测量；文献报告的 FPS 不进入统一速度图。
- 结论应写成“形成有竞争力的 Pareto trade-off”，除非严格同协议结果确实支持“最佳”。

## 8. 最省算力且审稿风险最低的执行顺序

1. 完成当前 16 组 SSDD 开发 sweep，只用 val 选定 placement/state_dim。
2. 完成内部核心消融，先用 seed 0 筛掉无效变体，再对最终关键四组补 seed 1/2。
3. 固定最终训练方案，合并 928 train，先跑 Mamba-YOLO-T 与本文方法三种子。
4. 跑 YOLOv8n、YOLOv10-S、RT-DETR-R18 的统一协议结果。
5. 加固定去斑基线；若 PPA-Net 官方代码能可靠启动，再补 PPA-Net。
6. 最后统一测复杂度和速度，再生成作者报告文献表。

这个顺序不需要在每个候选数据集上反复做全量结构搜索。结构选择以 SSDD 为主，OGSOD 用于跨数据集验证；其他数据集只验证固定最终结构，不再重新调 placement/state_dim。

## 9. 最终表格组织

- 表 1：统一协议 SSDD 主结果（我们实际复现）。
- 表 2：DS-SSM/PFIM 核心消融（三种子）。
- 表 3：效率与真实延迟（同机实测）。
- 表 4：SSDD 作者报告结果（明确协议差异，不做统一排名声明）。
- 表 5：OGSOD 跨数据集泛化结果。

顶会审稿的关键不是把所有公开或未公开方法都重写一遍，而是保证直接基线强、消融闭环、统一评估可信、跨数据集结论稳定，并把不可复现文献与本地实测严格分开。
