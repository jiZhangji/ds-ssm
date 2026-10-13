# IG-DS-SSM / PFIM 论文框架（研究提纲）

整理日期：2026-10-02  
状态：文献结构抽样 + 论文提纲；已据此完成中文内部 LaTeX 工作稿 `paper_draft/ig_ds_ssm_cn/main.tex`（24 页 A4、11 张 PDF 图）；本轮 XeLaTeX 双遍编译成功，未发现未解析引用。稿件仍不是投稿稿件。下方编号按渲染后的论文顺序核对。  
范围：围绕 SAR 视觉状态空间模型、SAR 目标检测及遥感 Mamba 论文，抽样阅读代表性期刊论文。这里的“顶刊”按本项目相关的 IEEE TGRS、IEEE TIP、IEEE TCSVT、ISPRS Journal of Photogrammetry and Remote Sensing 等高水平期刊理解；不声称穷尽所有相关论文。

## 1. 先定论文要回答的问题

### 暂定研究问题

论文的起点应是：**通用 Mamba 直接用于 SAR 图像时，状态更新机制没有显式利用 SAR 的成像与噪声特点。** SAR 散斑、海杂波、离散强散射和弱小目标会同时进入特征；selective scan 根据输入特征产生更新参数，但它本身不保证区分“目标结构”和“噪声/杂波”。因此，一个需要检验的风险是：背景扰动也参与状态更新，并沿扫描过程影响后续表征；另一方面，若简单平滑，又可能损伤弱散射目标。

本文据此研究：能否在状态更新过程中先修正局部残差，再以空间重要性调节进入 scan 的特征驱动和更新步长；这些设计分别有什么效果、组合时是否互补，以及收益能否迁移到其他 SAR 视觉任务。

这里的“潜在问题”是本文的机制假设和研究动机，不是已经被本项目实验直接证明的事实。现有 SSDD 结果显示两个组件的单独版本均值高于 baseline，但简单组合没有超过 DS-only；当前证据支持分析组件效果和交互，不支持“完整模型全面最优”，也还不能单独证明噪声确实沿 Mamba 状态递推传播。

### 暂定标题

**中文：** 面向 SAR 视觉 Mamba 的状态更新：局部残差修正与重要性调制  
**英文：** *State Updates for SAR Visual Mamba: Local Residual Correction and Importance Modulation*

标题先作为工作名。若后续受控实验表明 PFIM 不能稳定增加价值，应把论文收窄到局部残差状态更新，并将 PFIM 作为负结果或简化设计分析；若统一跨任务矩阵验证出互补性，再保留组合式标题。

### 贡献陈述草案（需由后续结果决定是否保留）

1. 提出一种 SAR 感知的视觉 SSM 状态更新设计：利用局部残差修正特征，并研究它对 scan 驱动及更新步长的影响。
2. 将 DS-SSM 与 PFIM 分开评估，并报告它们的单独效果、组合交互，以及 token-wise 与 image-wise 重要性调节的差异。
3. 用统一协议、多随机种子和跨任务实验界定方法的有效范围；在有效矩阵完成前，这条贡献只能写成研究计划，不能写成已完成结论。

不使用“首次将 Mamba 用于 SAR”“全任务 SOTA”“证明 speckle 一定沿状态传播”等主张。相关领域已有 SAR-Mamba 工作，机制性说法必须有直接实验支持。

## 2. 代表性期刊论文的结构观察

| 论文 | 期刊 / 年份 | 可借鉴的组织方式 | 对本文的启发 |
|---|---|---|---|
| RS-Mamba for Large Remote Sensing Image Dense Prediction | IEEE TGRS, 2024 | 采用引言、相关工作、方法、实验设置与结果、结论的技术论文主线；实验覆盖语义分割和变化检测，并安排消融分析。 | 方法型论文应先定义 SSM 模块，再用任务实验回答有效性；跨任务要有统一基线和分任务结果。 |
| Learning Cross-Task Features With Mamba for Remote Sensing Image Multitask Prediction | IEEE TGRS, 2025 | 以遥感多任务的共享编码器、跨任务特征模块和任务解码器为方法主线，再在多个数据集上验证。 | 本文若保留“通用 SAR 视觉”定位，必须让分类、检测、变化检测使用清楚一致的改动接口，而不能把不同 backbone 插入方式混称为同一方法。 |
| MaDiNet: Mamba Diffusion Network for SAR Target Detection | IEEE TCSVT, 2025 | 直接针对 SAR 目标检测，将 SAR 统计建模、Mamba 模块和检测任务组成完整方法。 | 相关工作必须正面讨论 SAR-Mamba；本文的差异应落在状态更新控制机制和可检验的组件交互，不可写成“该领域尚无 SAR Mamba”。 |
| SAR-NanoShipNet: A scale-adaptive network for robust small ship detection in SAR imagery | ISPRS JPRS, 2026 | 出版页面显示相关工作、网络概览、实验环境、数据集说明、讨论与局限、结论等部分；同时覆盖多个数据集和小目标分析。 | 主实验之外，应专门报告数据/训练细节、目标尺度分层、局限和失败案例。 |
| Speckle-Free SAR Image Ship Detection | IEEE TIP, 2021 | 研究滤波/斑点处理对检测结果的影响，并比较不同处理选择及多种 SAR 数据。 | 需要加入传统固定去斑预处理对照，才能判断端到端状态控制是否提供了简单滤波之外的价值。 |

### 抽样得出的写作规律

- 正文一般按“问题与缺口 → 相关方法 → 精确定义方法 → 可复现实验 → 结果和分析 → 局限与结论”展开。
- 引言通常只收敛到少数可检验的问题，并把贡献写成可由后文表格验证的陈述。
- 方法部分先给总览图，再依次定义模块、变量、计算路径和复杂度；新名字不能替代精确数学定义。
- 实验不只给一张总榜：数据协议、训练细节、基线、组件消融、复杂度和失败分析分别提供证据。
- 多任务主张需要逐任务完整报数。跨任务结果有正有负时，讨论适用范围比隐藏负结果更可信。

**范围说明：** 这是一份用于起草本文的相关论文结构抽样，不是系统综述。后续做正式 Related Work 时，还需要按投稿年份、任务和术语检索并去重；引用和新颖性结论需逐篇核验原文。

## 3. 建议的正文目录

### Abstract

按四个信息块写成一段：

1. **问题：** SAR 散斑、杂波、弱目标与现有视觉 SSM 状态更新之间的具体矛盾。
2. **方法：** 局部残差修正、PFIM 的输入线索和实际调制位置；概述插入到什么视觉 SSM 结构。
3. **证据：** 只写最终同协议实验；至少交代数据集、seed 数和主要指标，避免只报最优 seed。
4. **结论边界：** 说明收益在哪些任务/设置成立，以及组合或跨任务中的限制。

中文工作稿已按上述四块写入摘要，概括 SSDD 三种子结果并明确证据边界；后续投稿稿只能在新完成同协议实验后扩展跨数据集或外部性能结论。

### 1. Introduction

1. **SAR 与通用特征建模的矛盾：** 散斑/杂波与弱散射目标并存，固定平滑和普通特征提取都可能面临取舍。
2. **直接使用通用 Mamba 的潜在机制缺口：** selective scan 虽具输入依赖性，但没有显式 SAR 先验来区分目标、杂波和局部噪声；说明这是潜在风险，而非已有实验证实的必然失败。
3. **本文思路：** 在状态更新前作局部残差修正，再研究空间重要性如何调节 scan 驱动和更新步长；提出可检验而非预设正确的设计。
4. **研究问题：** RQ1：局部残差状态更新是否改善检测？RQ2：PFIM 单独及其与 DS-SSM 的交互是什么？RQ3：效果能否转移到分类和变化检测，边界在哪里？
5. **贡献列表：** 与最终完成的实验一一对应；未完成的跨数据集验证不列为已完成贡献。

### 2. Related Work

#### 2.1 SAR 图像去斑与目标检测

覆盖固定滤波、深度去斑、噪声感知特征与 SAR 船舶/飞机检测。总结固定前处理的优点和可能损失弱目标结构的风险，最后引出本文的可学习特征状态更新。

#### 2.2 视觉 SSM / Mamba 在遥感中的应用

按密集预测、检测、多任务/跨任务应用组织。必须纳入 RS-Mamba、TGRS 多任务 Mamba 及 MaDiNet 等直接相关工作；说明本文与已有 SAR-Mamba 的机制差别，不重复宣传“首次应用”。

#### 2.3 输入依赖状态更新与重要性调节

介绍 selective scan 的输入依赖参数、门控/注意力式重标定和局部统计。比较调节发生在特征输入、scan 驱动还是状态步长；只对已有论文中能核实的操作作归类。

#### 2.4 本文定位

以一张“方法位置对照表”总结输入线索、调节位置、任务、是否显式去斑以及证据协议，避免用泛泛的“现有方法不足”代替具体比较。

### 3. Method

#### 3.1 Problem Setup and Design Overview

定义输入特征、状态空间模块和目标任务；给出总览图，明确模块所在 stage / stride、原模块被替换还是输出特征适配、后续特征路径是否受影响。

#### 3.2 Local Residual Correction

定义局部趋势估计和残差修正。解释其设计动机，说明它是可学习的特征变换，不把它等同于物理散斑估计器或保证保边的滤波器。

#### 3.3 PFIM and State Modulation

定义用于估计空间重要性的特征、局部统计、归一化/MLP 和输出范围；再分别给出 importance 如何影响 scan 驱动和更新步长。明确这是对“哪些局部输入值得参与状态更新”的建模尝试。区分 gamma、write/input scaling 与 delta 的代码语义，避免把某个因子误写成直接调节历史隐状态保留率。

#### 3.4 Integration into Visual SSM

给出与原 block 的共同部分和改变部分：输入投影、scan 方向、状态维数、归一化、旁路、MLP、残差及插入位置。若实验版本改变了整块结构，就在此披露，不把全部收益归因于单个新机制。

#### 3.5 Complexity

给出符号复杂度及参数量，并报告完整模型同一硬件、同一输入尺寸和同一 profiler 下的实测延迟/FPS。理论线性复杂度不能代替实际速度测量。

### 4. Experimental Setup

#### 4.1 Datasets and Splits

按任务介绍 SSDD、SAR-ShipDataset、SAR-AIRcraft1.0；分类与变化检测数据集另列。每个数据集报告来源、官方/固定划分、图像或 patch 数、类别、输入尺寸和预处理。保留每个 seed 的 split manifest；明确测试集只用于最终评估。

#### 4.2 Compared Methods and Controlled Variants

- 同主干因果对照：baseline、DS-only、PFIM-only、full、image-wise/global importance。
- 预处理对照：原图 baseline 与固定 Lee 等去斑预处理，协议和预算一致。
- 外部方法：YOLO 轻量模型或其他文献基线，只有同划分、同初始化政策、同训练预算的重实现才能进入“统一协议实测”；文献原报数另表并注明不可直接排名。
- 对照表清楚标示 scratch/pretrained、模型版本、代码与权重来源。

#### 4.3 Training Protocol and Reproducibility

统一 epochs、batch、输入、优化器、增强、早停、选模和硬件。主结果至少 3 seeds，报告 mean±std 和逐 seed 数；记录训练/验证 manifest、代码版本、完整命令、checkpoint hash、评估脚本和设备。若跨 seed 的数据划分不同，应说明标准差同时包含划分变化和随机初始化变化。

#### 4.4 Metrics and Statistical Reporting

检测报告 AP50、AP75、AP50:95、precision/recall；分类报告 accuracy 及类别指标；变化检测报告 F1/IoU 等任务标准指标；效率给参数量、FLOPs、延迟/FPS。说明阈值、输入尺寸、batch 和计时是否包含预处理。小样本 seed 结果以描述性均值和方差为主，不夸大显著性。

### 5. Results

#### 5.1 Main Detection Results

在 SSDD、SAR-ShipDataset、SAR-AIRcraft1.0 分别比较同主干 baseline 和最终方法。数据集协议一致后再形成跨集结论。表格同时给每 seed 与 mean±std；外部方法文献原报数单独标注。

#### 5.2 Controlled Ablation on SSDD

报告五变体的主指标、逐 seed 配对差、参数/FLOPs/延迟。重点回答 DS-only、PFIM-only 的独立作用、full 相对单模块的交互，以及 image-wise 与 token-wise 差别。负向或不稳定结果保留。

#### 5.3 Cross-Task Results

分类和变化检测逐任务报告 baseline / DS-only / PFIM 或最终选定版本。分割单独放在探索性结果或适用边界，不将未改善任务移出报告。只有完成同接口、同预算对照后，才讨论跨任务迁移。

#### 5.4 Efficiency and Practical Cost

在同一硬件、相同输入、batch 和 profiler 下给出完整模型指标；解释精度、参数、计算量和速度之间的取舍。不能用单组件或不同环境数据拼成效率结论。

### 6. Analysis and Discussion

#### 6.1 Component Interaction and Seed Variance

展示 seed 配对变化，而不只排序均值。分析 full 组合表现低于 DS-only 的事实；讨论这是否来自调制冗余或优化差异，并把机制解释标为假设，除非有直接诊断证据。

#### 6.2 Target Scale, Clutter, and Class Analysis

按小/中/大目标、类别、背景复杂度或信噪条件分组。分组规则需预先定义，不能根据 test 结果挑切点后再声称机制发现。

#### 6.3 Visualization and Failure Cases

展示固定样例的检测结果、误检/漏检和 PFIM 图。PFIM 热图不能直接叫作“目标掩膜”或“注意力解释”；通过定量对齐、扰动/消融或相关性分析支撑解释。

#### 6.4 Generalization and Limitations

说明哪些任务改善，哪些不改善；讨论 SSDD 小规模与 split 方差、完整模型组件交互、替换 block 引入的结构差异、数据集域偏移，以及跨任务证据尚不足之处。

### 7. Conclusion

用简短段落回到研究问题，概括已被实验支持的结果和局限。避免新增结果、未验证机理和“全面适用于 SAR”的结论。

### Appendix / Supplementary Material

- 各数据集划分、manifest 哈希与预处理细节。
- 完整训练超参数、代码版本、硬件、checkpoint 哈希和评估命令。
- 所有 seed 的逐项结果、更多消融、类别/尺度结果及错误案例。
- PFIM 数学定义、参数初始化、完整模块配置与复杂度测量口径。

## 4. 图表规划

| 编号 | 内容 | 作用 | 当前状态 |
|---|---|---|---|
| Fig. 1 | SAR 特征传播问题示意：局部杂波/弱目标与状态更新 | 引出研究动机；不能画成已证实的噪声传播因果图 | 已有 `fig1_motivation.pdf` 概念图；不是实测现象图，图注保留假设边界 |
| Fig. 2 | 整体网络和模块插入位置 / 信息流 | 说明模块信息流与插入层级 | 已有 `fig2_method.pdf` 模块图；不是完整 backbone/任务头图，P3/8 与具体 stage 仍须绑定冻结代码 |
| Fig. 3 | SSDD 主实验协议流程 | 展示逐 seed 划分、五变体配对、验证选模和封存测试集 | `fig11_experiment_protocol.pdf`；概括已记录协议，不代表额外实验结果 |
| Fig. 4 | SSDD 主结果 | 展示五变体均值和误差 | `fig3_ssdd_main.pdf`，对应表格报告 mean±std |
| Fig. 5 | 逐 seed AP50:95 配对差 | 显示随机波动与模块交互，不隐藏负 seed | `fig4_ssdd_paired_delta.pdf`，五变体逐 seed 配对差可见 |
| Fig. 6 | 复杂度与精度关系 | 呈现参数量、GFLOPs 和 AP 记录 | `fig5_efficiency.pdf`；未绘制延迟，因计时元数据不完整 |
| Fig. 7--9 | 分类、变化检测、分割结果 | 展示跨任务正负方向及边界 | 对应 `fig7`、`fig6`、`fig8` PDF；均标成单 seed/探索性，不形成跨任务统一排名 |
| Fig. 10 | PFIM 可视化正反例 | 讨论解释可靠性 | `fig9_pfim_visualization_limits.pdf`；历史检查点训练协议不匹配，仅作定性审计 |
| Fig. 11 | 历史检测记录配对差 | 保留历史正负结果及协议边界 | `fig10_historical_detection.pdf`；仅单 seed/协议不统一，不能并入主表排名 |
| Table I--III | 相关工作方法位置、变体语义、数据集/证据范围 | 明确研究定位与实验边界 | 正文已包含；跨数据集公平矩阵仍未完成 |
| Table IV--VI | SSDD 三种子主结果、paired delta、复杂度 | 当前最完整的直接证据 | 已有全指标主表、逐 seed AP/P/R 原始表和配对差；checkpoint 哈希在补充 CSV；速度复测元数据仍有限制 |
| Table VII--XI | 分类、变化检测、分割和历史检测记录 | 保留跨任务正负方向与不兼容历史结果 | 主文均已单列；多数为单 seed/协议不一，不能作统一排名 |
| Table XII--XVI | SSDD 15-run 全指标、历史/失败账本、文献数值背景、完成度矩阵 | 便于审计原始记录、文献位置与主张边界 | 文献数值明确标注为非配对背景；逐 seed 表含 AP/P/R；完整权重哈希在补充 CSV |
| 后续主比较表 | 多数据集检测结果及同协议外部基线 | 支持跨数据集检测结论 | 尚未完成 SAR-ShipDataset/AIRcraft 公平三种子及同协议轻量基线 |

## 5. 当前能写与还不能写

| 主题 | 当前依据 | 框架中的写法 |
|---|---|---|
| SSDD 3-seed 五变体 | 已有 mean±std、逐 seed 结果与权重 hash；结果文件报告 baseline AP50:95 0.4203±0.0347、DS-only 0.4398±0.0144、PFIM-only 0.4312±0.0170、full 0.4291±0.0113、global 0.4377±0.0165。 | 可用于当前消融章节；措辞为“在该受控设置下的观察”，注明 n=3 和方差。 |
| 组件是否互补 | full 比 DS-only 低 1.07 个百分点；global 比 token-wise full 高 0.86 个百分点。 | 可作为重要的负交互发现；暂不宣称两个模块联合最优或 token-wise 更有效。 |
| 跨数据集检测 | 计划包含 SAR-ShipDataset、SAR-AIRcraft1.0，但尚未形成当前冻结协议下完整 3-seed 表。 | 列为主实验待完成，不能把历史非公平数搬进统一表。 |
| 分类/变化检测通用性 | 项目存在历史结果，但任务接口和训练协议并非全部统一；目前不能合并成一张可直接比较的跨任务主表。 | 只在完成同接口、同预算复核后支撑主张；否则写为探索性证据/局限。 |
| 与外部方法比较 | 有作者报告数和历史重实现记录，但不是当前同协议完整复现。 | 文献原报数独立列示，禁止伪装成公平实测排行榜。 |
| 机制解释 | 有设计动机和历史可视化；importance 不等于目标掩膜，替换 block 还影响归一化、scan 与 MLP 结构。 | 作为待检验解释，必须有额外分析；披露架构混杂因素。 |

正文起草时优先使用上述已验证部分，不复制旧 AAAI 草稿中的 ex27 验证集高分或 300-epoch OGSOD 主线作为当前结果。

## 6. 近期写作顺序

1. **投稿前待完成：**冻结代码版本，逐项把真实插入位置、PFIM 各因子的语义和共同结构绑定到公式、图和配置。
2. **工作稿已完成一轮：**Related Work 已加入代表性 SAR-Mamba、去斑检测和遥感 Mamba 工作；投稿前继续查漏并核对全部参考文献，不将抽样综述称为穷尽检索。
3. **工作稿已完成一轮：**SSDD 消融、跨任务记录、负交互、协议不匹配历史结果和失败运行已写入中文正文/附录；后续只在原始证据核验或新实验完成后更新。
4. **待实验完成：**SAR-ShipDataset / SAR-AIRcraft 公平矩阵、同协议外部轻量检测器和固定 Lee 去斑对照完成后，再决定多数据集检测主表与摘要主张。
5. **待预注册分析/复验：**补目标尺度/背景复杂度分层与跨任务统一矩阵，再判断是否保留“通用 SAR 视觉”定位；现阶段不以异质单种子记录作迁移结论。

这份框架本身不启动训练、不改变已冻结协议。主文已纳入可追溯的已完成结果、历史证据和失败/负结果；Lee 固定去斑对照、预设尺度/背景分层、跨数据集公平矩阵和同协议外部基线仍未完成，不得写成实验结果。项目当前状态与本轮稿件版本另见项目记忆记录。

## 7. 参考论文（本轮核验入口）

1. Zhao et al., “RS-Mamba for Large Remote Sensing Image Dense Prediction,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024. [DOI](https://doi.org/10.1109/TGRS.2024.3425540) · [IEEE Xplore](https://ieeexplore.ieee.org/document/10589665/)
2. Shen et al., “Learning Cross-Task Features With Mamba for Remote Sensing Image Multitask Prediction,” *IEEE Transactions on Geoscience and Remote Sensing*, 2025. [DOI](https://doi.org/10.1109/TGRS.2025.3540573) · [IEEE Xplore](https://ieeexplore.ieee.org/document/10879310/)
3. Zhou et al., “MaDiNet: Mamba Diffusion Network for SAR Target Detection,” *IEEE Transactions on Circuits and Systems for Video Technology*, 2025. [DOI](https://doi.org/10.1109/TCSVT.2025.3574657) · [IEEE Xplore](https://ieeexplore.ieee.org/document/11016924/)
4. Zhang et al., “SAR-NanoShipNet: A scale-adaptive network for robust small ship detection in SAR imagery,” *ISPRS Journal of Photogrammetry and Remote Sensing*, 2026. [DOI](https://doi.org/10.1016/j.isprsjprs.2025.12.006) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0924271625004903)
5. Chen et al., “Speckle-Free SAR Image Ship Detection,” *IEEE Transactions on Image Processing*, 2021. [DOI](https://doi.org/10.1109/TIP.2021.3089936) · [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34166190/)
6. Gao et al., “Mamba-CNN hybrid Multi-scale ship detection Network driven by a Dual-perception feature of Doppler and Scattering,” *ISPRS Journal of Photogrammetry and Remote Sensing*, 2026. [DOI](https://doi.org/10.1016/j.isprsjprs.2026.01.004) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0924271626000043)
7. Huang et al., “RSS-Net: A Mamba-Based Network for SAR Image Denoizing,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2026. [DOI](https://doi.org/10.1109/JSTARS.2025.3647971)
8. Chinchu A. et al., “MambaDIP: Mamba-Based Deep Image Prior Framework for Despeckling and Enhancement of SAR Images,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2026. [IEEE Xplore](https://ieeexplore.ieee.org/document/11557128/)
9. Wu et al., “A Novel Star Feature Decoupling Network for Multiscale SAR Image Ship Detection,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2026. [DOI](https://doi.org/10.1109/JSTARS.2026.3688771) · [IEEE Xplore](https://ieeexplore.ieee.org/document/11498663/)
