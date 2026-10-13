# Part 1 [The Review Report]

## Summary

本文提出 IG-DS-SSM：在轻量 Mamba-YOLO 的一个浅层 selective-scan 阶段中，以局部残差抑制和 PFIM 重要性场共同参数化扫描输入与动态步长；当前最干净的证据是 OGSOD 本地三划分下的一次同协议比较，显示高 IoU 定位指标改善而 AP50 基本不变。

## Strengths

1. **论点、实现和证据层级现在高度一致。** CUDA 快路径中 `gamma` 与 `write` 被准确描述为同一扫描驱动的两个因子，`delta` 是另一实际执行位置；`best.pt` 由验证集 `0.1 AP50 + 0.9 AP50:95` fitness 选择，再仅一次地在 OGSOD TEST 上进行 FP32 评估。论文不再将 TEST 用于模型选择，也不再把单次运行写成稳定增益。

2. **轻量化定位值得研究。** 在每种架构各一次的 OGSOD 同协议运行中，PFIM-T 相对 Mamba-YOLO-T 的 AP75/AP50:95 为 `+0.760/+1.300` 个百分点，AP50 仅 `+0.090`，且模型为 5.964M 参数、13.208 GFLOPs。这个信号与“改善高 IoU 定位而非普遍抬高置信度”的研究问题相符。

3. **反例和协议边界没有被掩盖。** 论文逐类报告 OGSOD tank 的 AP50/AP75 下降，并承认密度、尺度、重复外观和 NMS 困难在该类中混杂；SSDD、SAR-ShipDataset 和 SAR-AIRcraft 也被降级为非因果档案背景，而不是跨数据集稳定性的证明。

## Weaknesses (Critical)

1. **唯一直接架构比较仍然是每种模型一次运行。** 目前尚无 OGSOD baseline、no-PFIM、global-PFIM、full 在 seeds 0/1/2 上的均值、标准差和配对差值。`+1.300` AP50:95 对轻量检测器而言处于可能受随机初始化、数据顺序和 CUDA 非确定性影响的量级，因此论文只能声称一次观察，不能声称稳定或显著提升。

2. **PFIM 的必要性尚未被隔离。** `Mamba-YOLO-T -> residual-suppressed DS-SSM without PFIM -> parameter-matched image-level/global PFIM -> token-wise full PFIM` 的四阶段结果尚未完成；gamma-only、write-only、delta-only、gamma+write 与 no-2D-std 也没有实际数值。没有这些对照，收益可能来自局部 DWConv 残差抑制、额外 MLP 参数、普通门控或 transition 调制，而不一定来自 token-wise 二维重要性场。

3. **对 dense tank 的机制解释还没有证据。** 现稿已正确把“共享重要性场不利于实例分离”从结论降为假设，但仍缺少以固定 checkpoint 做的每图实例数、框面积、最近邻距离分层 AP，以及由预注册规则选择的成功/失败案例和 PFIM importance 图。该问题不是写作问题，而是需要离线诊断和受控对照才能区分检测头拥挤、NMS、平滑与共享调制。

4. **外部有效性仍未形成第二个受控数据集。** SSDD、SAR-ShipDataset 和 SAR-AIRcraft 的记录具有应用价值，却分别受多代码状态、训练中 batch 变化及优化器/切片协议不一致影响。它们不能替代 OGSOD 的核心因果实验，也不能用于回应“方法是否跨数据集稳定有效”的质疑。

## Rating

**6/10（边缘，弱拒绝）。** 论文已从“指标展示型初稿”提升为一个诚实、代码对齐且可审计的研究稿；但 AAAI 级模块论文仍需用多种子、等参数对照和结构消融证明其约 1 AP 点的定位收益不是偶然或通用门控效应。

# Part 2 [Strategic Advice]

## 问题根源

当前主要问题不是写作夸张，而是证据尚未形成最小因果闭环。历史档案是在不同服务器和训练条件下积累的，它们可以说明模型在多类 SAR 场景可运行，却不能回答 PFIM 的 token-wise 二维重要性是否是 OGSOD 提升的原因。当前代码和实验矩阵已经把这个问题压缩为一个可执行的 12-run 主实验和一个单种子诊断矩阵，缺少的是服务器产出的结果。

gamma 与 write 的问题是结构边界，而非措辞瑕疵：两者在 CUDA 中共同乘到 `u_scan`。因此，论文不应把两者包装成独立状态更新路径；它们是否比单一门控更有价值必须由 `global`、`gamma/write` 单因子和组合实验回答。现稿对此已准确，但尚无结果来建立新颖性。

dense tank 的问题也不应被过早归因。1,205 个 tank 集中在 71 张正样本图像是强烈的密度线索，但密度与目标面积、视觉重复性及后处理难度共变。需要分层评估和固定规则的可视化，才可把它写成可信的失败模式分析。

## 可救性判断

主矩阵、分支消融、全局重要性对照和分层分析都是可在修订期内完成的工程/实验工作，不要求重新设计方法。若 full 在三个种子上稳定优于 baseline、no-PFIM 和参数量完全匹配的 global 对照，论文可以合理提升到 7 分区间；此时系统会证明的不是“任何 SAR 数据都提升”，而是 token-wise PFIM 对该冻结 OGSOD 协议的定位质量有增益。

若 full 与 global 相当、与 no-PFIM 差异很小，PFIM 的核心地位不可用文字修复，应将主贡献改为 residual-suppressed DS-SSM，并把 PFIM 写成可选参数化。若 dense tank 在多种分层中都退化，则该限制是结构性边界，必须正面报告而不是继续扩大“适合小目标”的说法。历史三个档案的协议问题不能仅靠整理变为受控证据；至少需要额外重跑一个冻结协议的数据集，才可以支持跨数据集因果结论。

## 行动指南

1. 在同一提交、同一数据快照下完成 OGSOD `baseline/no_pfim/global/full`、seeds `0/1/2` 的 12 次运行。只报告 validation-selected `best.pt` 的一次 FP32 TEST，给出每种子原始值、均值、标准差和相对 baseline 的配对 AP50/AP75/AP50:95 差值。

2. 以 seed 0 完成 `gamma/write/delta/gamma_write/no_2dstd`。正文中将该表命名为“因子化扫描驱动与步长诊断”，并始终说明 gamma 与 write 是同一 CUDA 驱动位置的不同参数化因子。

3. 在四个 seed-0 主模型完成后，对固定 TEST checkpoint 按实例密度、归一化框面积和最近邻中心距离分层。候选案例由固定 IoU=0.75、confidence=0.25、NMS IoU=0.7 的每图 TP/FP/FN 统计选择，再展示 bridge/harbor 的成功案例与 tank 的退化案例及 PFIM importance。

4. 只有当 full 的三种子均值与配对差值优于 baseline、no-PFIM、global 时，才将摘要的单次 `+1.300` 改为均值加标准差；否则保留当前谨慎措辞，并按实际消融结果重新定位贡献。
