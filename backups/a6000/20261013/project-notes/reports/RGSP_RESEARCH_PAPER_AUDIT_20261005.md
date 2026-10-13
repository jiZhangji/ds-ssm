# SAR-Mamba 研究与论文梳理报告

审计日期：2026-10-05（Asia/Shanghai）  
对象：当前 IG-DS-SSM / PFIM 实现、2026-10-02 中文论文、SSDD 三种子记录，以及 RGSP 新定位建议。  
性质：本次请求的独立研究报告，不是新的正式项目状态、实验结果或已实施的代码修改。

本轮读取正式记忆、用户粘贴请求、论文定位讨论、ZIP 目录及关键内容、当前代码、训练/评估入口、逐次 CSV、实验报告和相关论文资料；未训练、推理、运行模型测试、启动服务、修改业务代码或覆盖论文。未连接实验服务器，未重新验证远端原始日志或 checkpoint。下文引用的证据编号见文末。

## 1. Executive Summary

1. 这个项目实际实现的是：在视觉特征上做可学习的局部残差修正，用局部趋势和邻域标准差生成空间重要性，再调节 selective scan 的输入驱动和时间步长；它不是图像去斑器，也没有训练可靠性概率预测器。
2. 最有价值的研究问题是：**局部结构统计能否为 Mamba 已有的输入选择性提供有效补充，以及这种补充应该作用于输入驱动还是状态转移。**“What to propagate”可以作为解释视角，但不是 Mamba 尚未考虑过的问题。
3. “Reliability-Guided”目前只成立为动机，尚不足以成为本文已经验证的核心机制名称；PFIM 更准确的定义是 task-trained local-structure-conditioned importance，而非 reliability、confidence 或 consistency estimator。
4. 当前最完整的实验记录是 SSDD 的五变体、三种子矩阵。逐次 CSV 可复算得到 DS-only AP50:95 为 43.98±1.44%，相对 baseline 平均 +1.95pp；这个数支持替换设计的描述性表现，不等于局部残差公式的独立因果收益。
5. Full 比 DS-only 平均低 1.07pp，token-wise full 比 global 平均低 0.86pp；不能写“两组件稳定互补”或“token-wise 更好”。这些结果同样不足以证明 PFIM 本质无效。
6. 最大的机制风险是**结构混杂**：原 VSSBlock 的四向 SS2D 被单向扫描块替换，同时改变投影、局部分支、MLP、残差缩放等。现实现不支持“不改变扫描拓扑、只改变传播内容”的字面主张。
7. 最大的实验风险不是某个分数低，而是原始运行证据链尚未在本地闭环，且选模指标、cosine 调度、真实 block 数量和跨任务接入方式存在可定位的文档冲突。
8. 最推荐的推进方式是：**先补同结构的机制对照，再补跨数据集和跨任务；先把文章写成局部结构引导的选择性状态调制，再由实验证据决定是否升级为 reliability-guided。**当前稿已经有真实研究基础，但还不是四任务机制论文的投稿定稿。

## 2. Project Evidence Map

### 2.1 项目结构与唯一版本入口

| 区域 | 本轮定位 | 用途/限制 |
|---|---|---|
| `docs/project-memory/` | 唯一正式 MemoryRoot | 当前范围、待办与历史冲突；其中旧语义总结仍需代码核验 |
| `paper_draft/ig_ds_ssm_cn/main.tex` | 当前中文内部稿 | 2026-10-02 版本；不是旧 AAAI 英文稿 |
| `paper_draft/aaai_ds_ssm_paper_en.tex` 等 | 历史论文线 | 不用它覆盖当前中文稿、SSDD 三种子结果或 P3 定位 |
| `source_code/work_train_single/` | 本轮核心代码证据 | PFIM、DS、SS2D、模型生成、训练评估和跨任务接入 |
| `dsh_experiment_results/` | 最新多种子账本 | 15 行逐次指标、汇总、协议、执行记录、汇总/复杂度脚本 |
| `reports/`、`database/` | 实验和文献证据索引 | 不因为文件名带 current/best 就自动成为最高优先级证据 |
| `output/`、`output_dir/`、`artifacts/` | manifest、旧评估与可视化 | 按具体实验检查，不能混入新三种子矩阵 |
| `downloads/madinet_paper/source/`、`downloads/madinet_official/` | 最邻近方法的原文/代码快照 | 定向读取 MaDiNet 方法段和 mixer；未遍历其第三方依赖 |
| `source_code/server_*`、`experiments/`、`weights/` | 历史证据 | 未改动、未加载权重；不从旧配置推断现时运行状态 |
| 用户 ZIP | 2026-10-04 快照 | 含 340 个条目；不是第二套 MemoryRoot |

ZIP 与工作区的 `mamba_yolo.py`、当前中文 `main.tex`、`results_individual.csv` 三个文件 SHA-256 完全一致。因此本报告分析的不是一个与附件无关的新版本。README 的“检测-only、OGSOD 最强受控证据”已落后于正式状态，不能据此恢复旧实验计划。[E1,E13]

引用聊天“记忆打包整理”只用于确认交接背景。其记录表明另一个桌面任务未完成最终审计；这不是科研结论，也未被用作新操作授权。

### 2.2 分类原则

- **A：已实现且有结果支持。**必须限定到结果真正支持的观察，例如“记录中 DS-only 的平均 AP 更高”；不能由此扩大到“可靠性估计准确”。
- **B：已实现但验证不足。**包括某公式存在、某分支确实执行，但其独立收益或机制解释未成立。
- **C：历史结果或协议链未完全锁定。**含旧跨任务结果；本轮未回收的原始训练/评估协议也明确保留缺口。
- **D：讨论、设计或未来实验。**RGSP 的强可靠性解释、保持原 SS2D 的新对照等属于此类。

另设独立的来源字段：Local measured / Author-reported / Historical archive / Planned。Local measured 指本项目运行所得，不表示“在 Windows 本机运行”，也不表示本轮重新测量。

### 2.3 研究主张—实现—实验对应

| 研究主张 | 实现/证据 | 实验支持 | 等级与可写范围 |
|---|---|---|---|
| 特征级局部残差修正已实现 | `DeSpeckleSSM.forward` 的 DWConv1d、残差和 λ | DS-only 是整个块替换，混入能量门控与结构变化 | 实现 B；不得把 +1.95pp 全归于残差公式 |
| PFIM 使用局部结构统计 | 趋势 + 修正特征二维标准差 + LN/MLP/sigmoid | token/global 与 PFIM-only 记录 | 实现 B；可准确描述输入线索 |
| 重要性实际进入递推 | γw 缩放 scan input，ηI 加到 Δ logits | 没有当前三种子 drive-only / Δ-only 结果 | 实现 B；可写调制位置，不可写每路已验证有效 |
| DS-only 平均 AP 高于 baseline | 15 行 CSV 可复算 | +1.9514pp，方向 +/+/- | A，限记录内描述；完整原始 protocol 回收仍待办 |
| PFIM-only 平均 AP/Recall 更高 | 同一 CSV | AP +1.0935pp；R +2.1469pp | A，限变体整体；不是原 VSSBlock 单加 PFIM 的净收益 |
| Full 稳定优于 DS-only | 同上 | 平均 -1.0726pp，逐 seed -/+/− | 不支持；不能写入主结论 |
| token-wise 优于 image-wise | 同上 | full−global 平均 -0.8566pp | 不支持；也未做等效性检验，不能反称两者等价 |
| PFIM 估计可靠性 | 无可靠性标签、校准或独立代理验证 | 旧热图有目标/背景反例 | D（强解释）/B（可检验代理假设） |
| 四任务验证同一种骨干机制 | 三套不同 backbone/接入代码 | 多为单 seed、协议异质、正负混合 | C；尚不是统一四任务机制证据 |
| 完整模型更快更省算 | 参数和 THOP/评估计时记录 | 缺 scan FLOPs 覆盖核验和统一 B=1 延迟 | 参数记录可引用；完整效率主张 B/C |

## 3. Research Problem

### 3.1 五个要素

**Problem。** SAR 的强散射、杂波和相干起伏会共同影响特征响应。响应幅度、任务相关性和结构稳定性不是同一个量；一个空间位置的价值也不能仅由是否落入目标框定义。

**Gap。** 普通 Mamba 已经通过输入生成 Δ、B、C，具有内容选择性。真正可研究的缺口是：在这一选择性之外，显式提供局部趋势与邻域变化作为归纳偏置，是否改善 SAR 任务中的状态更新。不能把“没有单独的可靠性分支”写成“不能学习可靠性”。

**Insight。** 扫描输入的幅值与状态转移的时间尺度是不同控制量。局部描述子可以参与二者的生成，而不只用于输出端重加权。

**Hypothesis。** 与同结构的普通 selective scan、普通空间重加权相比，局部统计条件化的 drive/step 调制能带来可重复的任务增益；若再观察到其与可操作定义的稳定性相关，才支持 reliability cue 解释。

**Proposed principle。** 将局部响应变换、结构描述和状态调制统一为可拆解的状态建模过程：**Correct → Describe → Modulate**。在当前证据阶段，“Describe”比“Assess reliability”准确。

### 3.2 Motivation 每个箭头的审查

| 推理箭头 | 判断 | 建议写法 |
|---|---|---|
| SAR 成像特性 → 局部强弱不等于语义价值 | 合理 | 作为领域动机，引用成像/检测文献 |
| 局部起伏 → 所有强响应均不可靠 | 不成立 | 稳定目标散射中心也会高亮，弱响应也未必可靠 |
| Mamba input-dependent → 无法处理 SAR | 不成立 | 改为“显式局部统计是否能补充其数据驱动选择性” |
| 既有工作研究 scan/fusion → 没人研究传播内容 | 不成立 | 原始 Mamba 已研究选择性传播/遗忘；相关 SAR 方法也注入先验 |
| 问题存在 → 必须先平滑再扫描 | 不是必然推论 | 残差修正是待比较的设计之一，需与无修正、普通卷积/滤波比较 |
| 局部统计 → PFIM 得到可靠性 | 未验证 | 先称结构条件化 importance；再设计代理和干预实验 |
| importance → scan input / Δ 改变 | 代码直接成立 | 明确两个作用位置及参数方向，不借助模糊的“传播强度”代替公式 |
| 任务性能提高 → 噪声被正确识别并阻断 | 不成立 | 终端性能与机制解释分开验证 |
| 多任务接入 → 跨任务机制普适 | 不成立 | 接入可行、收益一致、机制一致是三层不同证据 |

最终 Motivation 不需要“噪声递归累积”这类强命题。更扎实的表述是：**选择性参数决定局部输入如何写入及既有状态如何衰减，因此值得研究结构统计对这些参数的补充作用。**

## 4. Method Reconstruction

### 4.1 输入和局部修正：Formulation → Implementation

实际 block 输入为 \(F\in\mathbb R^{b\times d\times H\times W}\)。按行优先展平，并在 token 上 LayerNorm：

\[
X=\operatorname{LN}(\operatorname{flatten}_{HW}(F))\in\mathbb R^{b\times L\times d},\quad L=HW.
\]

\[
\bar X=\operatorname{DWConv1D}_{k=3}(X),\qquad R=X-\bar X,\qquad
\widetilde X=X-\lambda_s R.
\]

关键事实：[E2:237–245,274–277,405–488]

- DWConv1d 是**可学习卷积**，只在初始化时每个权重为 1/3；不能一直称其为固定均值滤波。
- 它作用于展平序列，不是二维 3×3 趋势估计；行尾与下一行行首发生序列邻接，边界用零 padding。
- λ 是每个块的可学习标量，不是每个像素各自预测的可靠性系数；没有 [0,1] 约束。
- `DeSpeckleSSM` 单独构造的 λ 默认 0.1，但实际 `DS_VSSBlock` 传入 **0.5**。论文应报告实际调用路径。
- 只有在 λ∈[0,1] 且趋势算子确实平滑的附加条件下，\(\widetilde X=(1-\lambda)X+\lambda\bar X\) 才能解释为收缩向局部趋势；当前训练没有保证这些条件。

**Effect 的边界。**数学上是可学习局部响应变换。将其解释为削弱无关局部起伏合理，但“识别噪声”“保边”“恢复干净 SAR”不是代码定理。建议子模块名 **Local Response Correction（LRC）**，不用 Local Reliability Correction。

### 4.2 PFIM：描述子与输出

对 \(\widetilde X\) 恢复二维形状，在每个通道计算：

\[
S=\sqrt{\max(\operatorname{AvgPool}_{3\times3}(\widetilde X^2)
-\operatorname{AvgPool}_{3\times3}(\widetilde X)^2,0)+10^{-5}}.
\]

\[
V=[\bar X,S],\quad
I=\sigma\left(W_2\operatorname{GELU}(W_1\operatorname{LN}(V)+b_1)+b_2\right).
\]

| 张量 | 形状 | 含义 |
|---|---|---|
| \(\bar X,S\) | b×L×d | 序列趋势、二维局部标准差 |
| \(V\) | b×L×2d | PFIM 的输入描述子 |
| PFIM 隐层 | b×L×max(⌊d/4⌋,16) | 学习的局部统计组合 |
| \(I\) | **b×L×1** | 每位置一个标量，在通道/状态维广播 |
| global 对照的 \(\bar I\) | b×1×1，再广播 | 先算每位置 sigmoid 输出，再沿 L 取平均 |

默认二维统计的边界均值排除 padding；无 `hw_shape` 或关闭 `use_2d_std` 时，改为沿序列的一维局部标准差，**不是把 std 分支删除**。最后线性层权重、偏置零初始化，初始 I=0.5。[E2:87–179,279–282]

PFIM 最准确地说是在做**由局部趋势和变化统计条件化、以任务损失训练的空间重要性估计**。它不是通道注意力，也不是有监督的可靠性概率。低 std 不必更重要：MLP 可学习相反关系；高 std 可能对应边缘、目标，也可能对应杂波。

术语需分开：importance 是模型为任务分配的相对作用；structural saliency 是结构突出程度；local consistency 是邻域或重复观测之间的一致性；reliability 是给定任务及扰动条件下可依赖程度；confidence 是某项预测的置信输出。correlation 只是两个观测量的统计关联，不能把它升级为因果或把上述概念互相替代。PFIM 当前没有显式计算一致性分数，也没有可靠性监督。

global 消融也不是“整块没有空间变化”：I 被平均，但能量 E、输入 u、Δ 的输入依赖部分仍随位置变化。

### 4.3 两个调制位置，而不是三个独立状态门

首先投影，分成内容和输出门控：

\[
[u,z]=W_{in}\widetilde X,\quad
B_t=W_Bu_t,\quad C_t=W_Cu_t,\quad A=-\exp(A_{log}).
\]

\[
E_t=\frac{\|\bar X_t\|_2}{\sqrt d},\qquad
\gamma_t=g_{min}+(1-g_{min})\sigma(-\alpha\,s_nE_t+\beta_i(I_t-0.5)),
\]

\[
w_t=\max\{0.1,1+\tanh(\rho_i)(I_t-0.5)\},\quad
q_t=\gamma_tw_t,\quad u^{scan}_t=q_tu_t,
\]

\[
\Delta_{t,c}=\operatorname{softplus}((W_\Delta u_t)_c+b_{\Delta,c}+\eta_i I_t).
\]

默认 \(g_{min}=0.5,s_n=0.1\)，α/β/ρ/η 初始化为 0.5/1/1/0.1；它们是可学习标量，\(s_n\) 是固定系数，**不是从 SAR 图像估出的噪声标准差**。[E2:186–267,297–367]

- **位置一：当前输入驱动。**γ 和 w 是同一个 \(u^{scan}\) 的两因子；均不直接乘历史状态。
- **位置二：Δ。**它同时影响旧状态的衰减与当前输入离散写入项，不能只叫“写入门”或“保留率”。
- B、C 由尚未乘 q 的 u 生成；不能把实现改写成 B/C 都由 gated input 生成。
- PFIM 没有直接预测 A、B、C；但局部修正改变了 u，因而会间接改变 B/C、z、Δ 和旁路。
- β、ρ、η、α 都没有正值约束。不能保证 I 越大，所有路径都越“强化”。
- PFIM 初始 I=0.5 时 w=1，但 ηI=0.05，Δ 分支不是零扰动；γ 也不等于 1。PFIM-only 令 λ=α=0，却没有去掉 γ 的常数缩放，初始 γ=0.75。
- 默认范围内 w 实际位于约 (0.5,1.5)，0.1 clamp 不意味着实现通常会把输入压到 0.1；`min_retain` 只是变量名，不应译成 hidden-state retention 下界。

### 4.4 精确递推与“传播”的含义

忽略 batch 下标，通道 c 和状态维 n 上的实现为：

\[
h_{t,c,n}=e^{\Delta_{t,c}A_{c,n}}h_{t-1,c,n}
+\Delta_{t,c}B_{t,n}q_tu_{t,c},
\]

\[
y_{t,c}=\sum_n C_{t,n}h_{t,c,n}+D_c\widetilde X_{t,c},\quad
o_t=W_{out}\big(y_t\odot\operatorname{SiLU}(z_t)\big).
\]

这里 \(h\in\mathbb R^{b\times d\times n_s}\)，\(n_s=16\) 是当前 sweep 状态维；B/C 在通道间共享其状态向量，Δ 为通道相关。代码向 scan 传入 `D=None,z=None,delta_softplus=False`，之后显式加 **D×修正特征**并做 SiLU(z) 门控；不是在 kernel 中加 D×gated u。softplus 只在外部执行一次。[E2:349–398,E3:160–196]

这是 Mamba 实现常用的 \(\Delta B u\) 写入形式，不应未经说明改写为精确 ZOH 的 \(A^{-1}(e^{\Delta A}-I)Bu\) 并声称等价。

在冻结其他系数的局部分析下，t 时刻的写入到后续 j 时刻会乘：

\[
\prod_{k=t+1}^{j}e^{\Delta_{k,c}A_{c,n}}.
\]

因此“state propagation”在技术上成立，但它描述的是递推结果；**state update/modulation 更直接描述本文改在哪里**。A<0、Δ>0 时单步系数在 (0,1) 内；增加 Δ 会加强旧状态衰减，同时改变当前写入。不能据此说“Δ 越大越保留历史”，也不能推出整个非线性网络的扰动稳定性或“噪声必然递归放大”。

### 4.5 与原始 block 的关系：必须公开的结构差异

| 项目 | baseline VSSBlock | 当前 DS_VSSBlock |
|---|---|---|
| 入口 | Conv1×1 + BN + SiLU | 直接 flatten + LayerNorm |
| 局部支路 | LSBlock；SS2D 内 depthwise Conv2d | 序列 DWConv1d 趋势 + 残差修正 |
| 扫描 | SS2D，K=4，CrossScan/CrossMerge | 单向、行优先一次 scan |
| SSM 内部宽度 | 通常 expansion=2 | u 维度为 d，无同样膨胀 |
| Δ 参数化 | 低秩投影、按方向参数 | Linear(d,d) + ηI |
| 输出/局部分支 | SS2D out-norm 等 | D×修正特征、SiLU(z)、Linear |
| FFN | RGBlock，配置 mlp_ratio=4 | 两层 Linear/GELU，ratio=2 |
| 残差尺度 | 原 VSS 残差/drop path | 两个可学习通道尺度初始 0.01 |

不能把 ratio 4→2 简化为“所有 MLP 参数刚好减半”：RGBlock 本身还有门控和宽度变换。[E2:746–839,898–945,1045–1125,E3]

保持的是选择性递推的大类形式、状态维 16（当前检测路径）以及 stage 外的检测框架；不是原 block 所有参数、数值或扫描拓扑不变。

## 5. Naming Review

### 5.1 RGSP 是否成立

**判断：作为待验证研究动机可以，作为当前定稿方法名不推荐。**

不要求一个 reliability-guided 方法必须输出校准概率；有清楚定义和验证的可靠性代理也可以。但当前 PFIM 的输入只是局部统计，学习目标只有任务损失，已有图和消融未建立“可靠性—I—状态作用”的证据链。只把 importance 上位解释成 reliability，会让名字承担尚未完成的科学证明。

IG-DS-SSM 中的 De-Speckle 则容易误导成图像恢复；不建议保留为顶层论文名称。已有类名和历史实验 ID 不应批量改名，以免破坏 checkpoint 和结果溯源。

### 5.2 更准确的候选

| 名称 | 优点 | 代价/适用条件 |
|---|---|---|
| **Structure-Guided Selective State Modulation（SGSM）** | 覆盖局部结构描述与递推调制；避开概率可靠性承诺 | 必须说明 structure 指局部结构统计线索，不是结构真值 |
| Local-Statistics-Guided State Modulation | 与趋势/标准差最直接对应 | 技术感强、辨识度较低；不用再强造与 SSM 冲突的简称 |
| Local-Response-Aware Selective State Modeling | 覆盖残差修正和状态更新，适合当前替换块 | “aware”较宽，需公式突出具体差别 |
| Importance-Guided Selective State Modulation | 与 PFIM 输出最一致，易解释 | 更偏 PFIM，不能掩盖 DS-only 才是当前最高均值版本 |

推荐层级：顶层研究思想为 local-structure-guided state modulation；方法简称 **SGSM**；具体实现称 **SGSM block**；PFIM 保留旧名以维持溯源，并在首次出现时定义成“局部结构条件化的重要性估计”；修正步骤叫 LRC 即可，无需再包装成一个独立网络。

PFIM 的 Perceptual 是历史命名，不代表用了感知损失。若后续追求更清楚的模块名，可用 Local Structural Importance Estimator；这只是命名建议，本轮没有改实现。

### 5.3 名称冲突检索范围

本轮对 RGSP 全称及 SGSM 全称做了 Crossref `query.title` 定向检索（各前 5 项），返回结果中未见完全同名条目。这个检查**不覆盖所有 arXiv/会议/代码仓库，不构成无重名保证，也不证明机制新颖**。缩写短、重用概率高，投稿前应继续核对完整标题和机制关键词。

## 6. Novelty Analysis

| 对照对象 | 已有工作做了什么 | 本文可建立的区别 | 不能声称什么 |
|---|---|---|---|
| 原始 Mamba | 输入依赖 Δ/B/C，选择性传播或遗忘内容 | 额外的局部统计描述与 drive/Δ 条件化 | “第一次考虑传播什么”或“普通 Mamba 等价传播所有 token” |
| Vision Mamba（Vim） | 双向视觉序列建模、位置编码 | SAR 特征局部响应处理与调制 | Vim 与 VMamba 是同一个方法 |
| VMamba | VSS + 四向 SS2D | 可在相同 SS2D 中检验局部统计的增量；当前代码先是整块替换 | 当前版本保留原 SS2D 拓扑 |
| Mamba-YOLO | ODMamba、局部 LS/RG 设计、检测框架 | 以局部统计参与 selective scan 为实验变量 | 它缺少局部建模，或仅修改扫描路径 |
| MaDiNet / MambaSAR | 检测 diffusion 框架、Mamba/attention 混合、SAR 空间建模 | 重点比较局部描述子进入 drive/Δ 的位置及必要性 | 仅按“它研究 how，我研究 what”就充分区分 |
| 普通 feature attention | 可在输入或输出端做任务相关重加权 | 固定 u 后，本文额外 I 可以直接改 Δ 和 scan drive | γ×w 本身不是重加权，或所有输入 attention 都不会改变递推 |
| SAR despeckling | 图像域恢复、相应统计或复原目标 | 本文输出中间特征，损失服务于下游任务 | 已实现图像去斑/ENL 或恢复质量提升 |
| scan-path redesign | 改变二维点进入序列的顺序/方向/融合 | 科学问题可以侧重调制，与拓扑研究互补 | 当前 baseline 对照完全控制了扫描方向 |

原始 Mamba 的摘要已经明确使用 “selectively propagate or forget information”。所以 **How to scan → What to propagate 应写成研究重心的补充，不能写成历史上尚无人提出的替代范式**。[R1–R5]

注意两个特别接近的比较：

1. 输入 attention 若在 Δ/B/C 生成之前作用，本来就会间接影响状态更新；必须用“相同 I、不同注入位置”的实验比较，才能说明直接 Δ 调制超出了普通预加权。
2. MaDiNet 的本地预印本写四向 SARScan，已下载 mixer 文件则展示标准 scan 调用；README 还涉及正式版 Gamma diffusion。版本之间不能互相代替。正式相关工作应以明确版本的论文和对应代码对齐；本轮没有据一个文件断言整个 MaDiNet 不含某种机制。[E11]

当前最强的潜在新意是**可分解的局部响应—驱动—转移协同设计及其受控证据**，不是新名称本身。若同结构对照没有收益，名字升级无法解决创新问题。

## 7. Claim-Evidence Matrix

| 准备写的核心结论 | 当前证据 | 缺口/处理 |
|---|---|---|
| 利用 SAR 局部统计调节 selective scan | 公式和调用链明确 | 可写为实现贡献；SAR-specific 收益另证 |
| 局部修正独立提升性能 | DS-only 整块结果 | 增加同壳层的 correction-only、energy-only、二者合用 |
| PFIM 独立提升性能 | PFIM-only 相对不同结构 baseline 的均值 | 增加同结构无 PFIM 基线；冻结参数相同初始化 |
| γ 控制历史保留、w 控制写入、Δ 控制步长 | γ/w 均在 u 上 | 删除三独立位置表述，改“两位置、输入位置含两因子” |
| 高 I 一定加强传播 | 参数符号自由，Δ 同时控制衰减/写入 | 改为“调节”；检查参数与有效门控再解释方向 |
| PFIM 是 reliability probability | sigmoid 有界，但无标签/校准 | 删除；importance 不等于 probability |
| Full 稳定互补 | full−DS 平均负，seed 间混合 | 报告当前条件下未获互补增益；机制原因待查 |
| token-wise 更好 | global 均值更高 | 删除；保留完整正负结果，不作等效性结论 |
| +1.95pp 是显著稳健收益 | n=3；配对差标准差 3.87pp | 改为“平均增加”；独立确认，不能去掉 seed2 |
| 不改扫描拓扑 | baseline 四向、DS 单向 | 当前版本删除；新版本保留四向并重新实验后才能写 |
| 适用于四任务的 backbone 更新机制 | 接入方式、stage 与训练预算不同 | 做统一接口控制后再提升主张 |
| 更轻、更快 | 参数近似降低 1.1%，计时/算子统计有限 | 分别报告参数、覆盖算子、B=1 延迟；不可整体捆绑 |
| 强响应就是噪声/残差就是 speckle | 无噪声标签或成像反演 | 删除；使用任务局部响应描述 |
| 超过 SOTA | 文献协议、组件复杂度和本地结果混杂 | 公开原报数独立表；实测榜必须统一协议 |

### 当前文稿和账本中应优先修改的具体位置

1. `main.tex:161`：“相同主干”“P3 的三个 VSSBlock”需更正。相同的是宏观框架，不是全部主干；`model_payload` 的 repeat=3 经 depth=0.33 后为 **1**。当前 P3 是 stride 8、64 通道、state_dim=16 的一处 block 替换。[E4,E5]
2. `main.tex:151`、`BASELINE_PROTOCOL.md §2.4`：验证 AP50:95 选模与代码不一致。当前 fitness 为 **0.1 AP50 + 0.9 AP50:95**；必须回原 args/log 核实运行版本。若实际就是该 fitness，改写论文即可，不要为迎合文字重新在 TEST 选权重。[E7]
3. `main.tex:149`、`BASELINE_PROTOCOL.md §2.3` 写 cosine；`EXPERIMENT_PLAN.md` 写默认关闭，runner 也默认 False。结论是协议有冲突，**不是本轮已经知道所有 run 实际使用哪种 scheduler**。[E4,E6]
4. `BASELINE_PROTOCOL.md §1.1` 把检测 baseline 写成 VMambaTinyBackbone 不准确：该 runner 走 Mamba-YOLO 配置生成。baseline 的 metadata `state_dim=0` 表示未指定 DS state，不代表原 VSS/SS2D 没有状态维。[E2,E4]
5. `main.tex:125` “它们都不直接乘到历史隐状态保留项”宜精确为“γ/w 不直接乘历史状态；Δ 进入 exp(ΔA)”。
6. PFIM 公式建议补线性层 bias、明确 b×L×1；global 是 sigmoid 后平均。`no-2dstd` 是改用一维 std，不是删除局部变化。
7. `RESULTS.md` 和记忆中的“seed2 outlier”“这是 split 方差、非方法失效”应撤掉归因；`R 提升 +0.8139` 是数值表述错误，0.8139 是水平，增加量是约 0.02147，即 2.15pp。
8. “负交互”正文最好先用可直接观察的“组合低于 DS-only”。若指严格 factorial interaction，须用共同壳层计算 \(M_{11}-M_{10}-M_{01}+M_{00}\)；当前原 baseline 不是同壳层的 M00。
9. 引言和结论大量“本文不主张/不能/待验证”移到统一的证据限制段或附录。保留数据边界，但不要把论文写成内部审计记录。
10. 当前参考文献列了 Vim，却没有独立列出最直接对应 SS2D 的 **VMamba: Visual State Space Model（2401.10166）**。应补 VMamba，并把含混的 `liu2024visionmamba` key 与 Zhu 等人的 Vim 区分。[R2,R3]

以上是建议变更清单；本轮没有修改这些原件。

## 8. Experiment Audit

### 8.1 SSDD 数值复核

由 `results_individual.csv` 逐行复算，std 使用样本标准差（n−1）；未运行训练或评估器。[E6]

| 变体 | AP50:95（%，mean±std） | 相对 baseline 平均 Δ（pp） | 配对 Δ：seed 0 / 1 / 2（pp） |
|---|---:|---:|---|
| Baseline | 42.03±3.47 | — | — |
| DS-only | **43.98±1.44** | **+1.95** | +2.900 / +5.256 / −2.302 |
| PFIM-only | 43.12±1.70 | +1.09 | +2.127 / +3.156 / −2.003 |
| Full | 42.91±1.13 | +0.88 | −0.354 / +5.935 / −2.944 |
| Global | 43.77±1.65 | +1.74 | +2.288 / +6.148 / −3.230 |

Full−DS-only：−3.254 / +0.679 / −0.642pp，平均 −1.073pp。  
Full−global：−2.642 / −0.214 / +0.286pp，平均 −0.857pp。

准确解读：DS-only 的描述性均值最高；Full 并未呈现稳定优势；global 在此矩阵中均值更高。不要由三次试验得出“global 普遍更好”或“PFIM 普遍有害”。

#### 统计上的两处重要纠正

- baseline 的三个数为 42.6943、38.2746、45.1215%。seed2 仅高于均值约 **0.89 个样本标准差**，没有可据以判为异常值的依据；更不能因所有方法在该 seed 都输而排除它。
- seed 同时控制 train/val 划分和训练随机性。方差来源无法拆成“都是 split”或“都是初始化”。未来应分离 `split_seed` 和 `train_seed`。

DS-only 配对差 std=3.867pp。若仅作小样本不确定性的演示，n=3 的配对 t 区间约为 −7.65 到 +11.56pp；其假设在三点数据下也难检验。这不是推荐以一个不可靠 p 值决定项目生死，而是说明不能把 +1.95pp 叫已确证稳健效应。未来需事先确定训练重复和确认集，不做“跑到显著就停”。

### 8.2 controlled comparison 实际满足到哪一层

| 因素 | 核查结果 |
|---|---|
| 数据与 seed 内配对 | runner 设计为 928→742/186，test=232；五变体 seed 内共享。CSV 与记录支持该运行描述，原始 split 文件尚未在本轮回收核对 |
| 初始化政策 | 记录与 runner 均为 weights=none；这不等于不同结构共享张量逐项初始化一致 |
| task head/loss/输入/增强 | 同一 runner 生成同检测框架与统一参数，具备配对基础 |
| epochs | 150 是上限，patience=30；已有不同实际早停长度，不可写“完全相同训练步数” |
| scheduler | protocol 文档与 plan/default 冲突；回收 protocol.json + args.yaml + LR 曲线再定 |
| checkpoint 选择 | 本地代码 fitness=0.1AP50+0.9AP50:95，与论文纯 AP50:95 有冲突 |
| 结构 | 不是 baseline+一个开关；四向→单向、壳层变化是真混杂因素 |
| evaluator | 项目 Ultralytics 路径，FP32、conf=0.001、NMS IoU=0.6；应明确是该实现的 COCO-style AP，非自动等同 pycocotools 官方评估 |
| P/R | 当前评估会在平滑 F1 曲线最大点取 P/R；不同模型阈值可不同。+2.15pp Recall 不是已证实的固定阈值召回收益 [E7] |
| 代码冻结 | 核心 model 文件与 ZIP、MD5 清单一致；当前 runner MD5 与两份历史清单均不同，须取每个 run 的冻结源码核查 |
| TEST 使用 | 记录称 best 后测试；本轮不能从汇总 CSV 证明完整访问历史。已有 TEST 结果已参与后续方法选择，未来新方法应增加未用于设计的确认数据 |

**原始证据缺口。**在本轮检查的最新账本、ZIP 和项目结果目录中，没有找到本批 15 次运行的完整 `eval_test.json + protocol.json + train/val lists + args.yaml + 原始训练曲线` 链。现有 CSV 和权重 hash 是可复算记录，不是训练协议的密码学证明。原始记录被指向项目服务器的 `ig-ds-ssm/experiments/ssdd_paper_v1/`；本轮未访问，因此不宣称已完成端到端独立复验。

### 8.3 跨任务：接入机制本身并不统一

| 路径 | 实际位置与传播关系 | 审计结论 |
|---|---|---|
| 检测 model_payload | Tiny 的 P3/8，64 通道；替换块后继续影响 P4/P5 | 骨干内部替换，但内部结构不纯匹配 |
| shared `full_t` | P3/8 替换后再计算 P4/P5 | 与检测的骨干接入逻辑较一致 |
| shared `legacy` | 名为 p3 的改动位于第二次 merge 后，实际 stride 16 | 不能仅凭 p3 字符串与 P3/8 实验合并 |
| `vmamba_tiny` | 先完整 encoder(image)，再修改返回列表 features[1] | **输出特征适配器**，不改变已算好的 P4/P5 或原 encoder 内递推 |
| 当前分类头 | GAP(features[1]) 后 LN/Dropout/Linear | 可让新增模块参与分类，但不是通用最终层分类头；高层仍计算却不进入该分类输出 |
| 当前变化检测 | 两时相共享 backbone，逐层 abs-diff，再 decoder | 是明确的任务框架，需保持两边相同接口和预算 |

另外，VMamba 编码器配置的 `ssm_d_state=1` 与新增 DS 的 16 不同；warmup 只给 ours，冻结/恢复原 encoder 后再完整训练，通常总 epoch 为 warmup+正式 epoch。warmup20/50 结果不能与没有这段预算的 baseline 当作纯模块对照。[E8]

### 8.4 历史跨任务与检测结果的正确用法

以下都是已有记录，不是本轮重新实测；主要用于识别边界和选择下一轮验证，不能充当统一 leaderboard。[E1,E9]

| 记录 | baseline → 方法 | 来源/协议边界 |
|---|---|---|
| OGSOD TEST AP50:95 | 42.3018→43.6021% | Historical archive，单次匹配记录；非当前主集 |
| SSDD ex27 | 历史 VAL 74.022%；复核 TEST 71.4454% | VAL≠TEST；partial transfer≠clean scratch；不与当前 42–44% 直接归因比较 |
| SAR-Ship AP50:95 | 49.568→57.818% | 非严格协议配对；不能取 +8.25pp 作主结果 |
| SAR-AIRcraft patch TEST AP50:95 | 54.513→55.552% | 单次历史协议，原图/patch与初始化需回收确认 |
| SAR-ACD scratch accuracy | 94.882→89.370% | 单种子、负结果 |
| SAR-ACD VMamba/no warmup | 89.764→91.339% | 不与 scratch 轨道混排 |
| SAR-ACD VMamba/warmup50 | 89.764→96.850% | ours 专属训练日程，非同预算净效应 |
| SAMPLE scratch accuracy | 98.519→97.037% | real-only 选样协议，负结果 |
| Ottawa d16/BS16 F1 | 92.091→92.249% | 小幅正差；d32 记录则为负，不能当多 seed |
| Yellow River d16/BS16 F1 | 88.368→86.891% | 负结果 |
| Farmland d16/BS16 F1 | 23.564→64.428% | 大正差但 baseline 召回仅13.887%；需先查欠拟合、阈值、类别不平衡和协议 |
| FUSAR-Map 四类 mIoU | 44.2973→43.5607% | 部分可复核历史记录；road IoU约4.78→1.09%；不支持全任务增益 |
| OpenEarthMap-SAR mIoU | 36.551→35.898%（no warmup） | SAR-only、标签策略和预算需锁定；warmup版本也未改善主指标 |

检测的 HBB/OBB、航空原图/patch、分类 MSTAR SOC/EOC/SAMPLE 域划分、分割 ignore/黑色标签、变化检测相邻 patch 与场景隔离，必须作为结果主键，而不是表格脚注里的可忽略差别。

### 8.5 reliability 和效率证据

历史六样例热图中，PFIM 目标/背景比值既有 3.67 也有 0.45，均值约1.19、中位数约0.72。它说明图不是稳定目标掩膜；但背景也可能对任务可靠，故**这既不能证明可靠性，也不能单凭前景比值否定可靠性**。可靠性必须另行操作化。[E10]

复杂度账本：baseline 5,985,547 参数，DS-only 5,918,104，full 5,920,441。参数减少是实际记录的架构差异，不是“零成本机制”的证据。PFIM 差额 2,337 不应称“全是几个标量”：包含 LN/MLP，具体以模型参数表为准。

`measure_complexity.py` 的 FPS 是 **1000 / eval_test inference_ms**，不是独立 batch=1 端到端 benchmark。eval 默认 batch=16；指标不包含同一口径的预处理/NMS全流程。`get_flops` 用 THOP 且未见此调用注册 scan 的 custom counter；非零 GFLOPs 也可能漏掉自定义 kernel，而不是只在结果为0时才有问题。[E12]

所以可以保留“参数/已有计时账本”，不能据此写可靠 FPS 排名；尤其不能用 MaDiNet 的单组件复杂度对比本方法完整 detector 宣称数量级优势。

## 9. Recommended Paper Story

推荐主线如下：

**Problem：**SAR 局部响应幅度不足以确定任务价值，表征需要同时处理局部结构与长程上下文。  
**Gap：**Mamba 的输入选择性提供了控制接口，显式局部统计在这些接口中的作用及必要性尚需针对性验证。  
**Observation：**现有实现显示 drive 与 Δ 是不同作用位置；已有 SSDD 实验提示单模块、组合、粒度具有不同效果，不支持简单叠加逻辑。  
**Method：**用局部响应修正和结构描述子生成选择性状态调制；通过开关区分 correction、energy、drive 和 step。  
**Evidence：**先以同结构对照证明净收益，再以检测多集、跨任务受控迁移和干预验证用途与边界。  
**Contribution：**提出可复现设计，解释其精确作用位置，并用实验确定何时有效，而不是把“可靠性”当作先验正确的标签。

建议主文只保留：问题与相关工作、机制、真正可比的检测主表、核心消融、机制分析、紧凑跨任务表和局限。把庞大的旧失败账本、文献原报数字、不同 batch/state_dim 轨道放入可追溯附录，不让24页内部稿的证据管理结构主导投稿叙事。

不要现在根据 SSDD TEST 最高均值就宣布 DS-only 是最终赢家。它是值得优先确认的候选；Full 的有效性未确认，也不应继续默认 Full 才是“完整方法所以必然主打”。若严格复验仍只有 DS-only 有效，论文应转向局部响应条件化的状态块，把 PFIM 作为探索分支，不能硬保留“三阶段联合最优”的故事。

## 10. Recommended Title and Method Name

### 当前最推荐

**Structure-Guided Selective State Modulation in Mamba for SAR Imagery**  
**面向 SAR 图像的局部结构引导 Mamba 选择性状态调制方法**

方法名：**SGSM**。中文补“局部”以明确结构线索的尺度，不要求逐词硬译。它准确表达控制位置，不承诺可靠性真值、图像复原或所有 SSM 理论。

### 三个备选标题

| 标题 | 优点 | 缺点/采用条件 |
|---|---|---|
| Local Response Correction and Selective State Modulation in Mamba for SAR Imagery | 与现实现最贴合；不预支机制解释 | 稍长且较描述性，需要引言把两部分统一成研究问题 |
| Local-Statistics-Guided State Modeling for SAR Imagery | 词义最保守准确，覆盖 correction/energy/importance | 没有直接点明 Mamba；若不验证其他体系，正文应限定范围 |
| Reliability-Guided Selective State Propagation in Mamba for SAR Imagery | 研究问题集中、有辨识度 | 仅在可靠性代理、干预、同结构净收益和跨数据集确认后再采用；当前不推荐定稿 |

“Selective state propagation”可理解，但不像 selective scan 那样是具体算子名；“modulation”更准确表示本次控制。`in Mamba`不是过度限制，反而避免泛化到所有 SSM。`SAR Imagery`自然，也不自动表示 foundation model、VLM 或大模型；不必为了避免误解再加一长串排除性词语。

## 11. Revised Introduction

以下是可替换当前引言的中文文本，采用本报告建议的 SGSM 名称；名称尚未在源文件中实施。内容按现有实验证据书写，没有把待做的四任务统一验证当作完成。引文编号对应文末已核验入口；嵌入 LaTeX 时换成正式 bibliography keys。

---

合成孔径雷达（Synthetic Aperture Radar，SAR）具有全天时、全天候成像能力，是海面监测、地表观测和灾害分析的重要信息来源。与自然光学图像不同，SAR 图像中的局部响应由目标散射特性、成像几何及相干成像过程共同决定。目标散射中心、背景杂波与局部起伏均可形成较强响应，而弱小目标边缘和细长结构也可能表现为低幅度特征。因此，响应强度与任务相关的结构信息并不等价。有效的 SAR 表征不仅需要聚合大范围上下文，还需要根据局部结构组织信息进入后续特征更新的方式。

选择性状态空间模型为这一问题提供了有价值的建模接口。Mamba 通过输入依赖的状态参数调节信息写入与状态演化，以随序列长度线性增长的计算量建立长距离依赖[R1]。Vision Mamba 和 VMamba 分别通过双向建模与二维多方向扫描扩展了其视觉表征能力，Mamba-YOLO 则进一步结合局部特征建模与层级检测结构[R2–R4]。这些进展说明，视觉状态空间模型能够在局部表示与长程关联之间形成不同于全局自注意力的计算路径。对于 SAR 数据，一个进一步的问题是：局部趋势及邻域变化能否作为显式线索，补充 Mamba 从输入中学习到的选择性，并改善任务相关信息的状态更新？

已有 SAR-Mamba 研究从扫描方向、局部与全局分支、特征融合及任务框架等方面增强表征。例如，MaDiNet 将 MambaSAR 的空间建模与检测框架结合，以刻画目标及复杂背景之间的关系[R5]。在这些设计之外，本文关注局部结构线索进入选择性递推的具体位置：一方面，扫描输入决定当前响应以何种内容和幅度写入状态；另一方面，动态步长同时参与历史状态衰减和当前输入写入。这两个作用并不等同于在输出端赋予一个空间权重，也不意味着更细粒度或更多调制分支必然带来更好的结果。明确区分这些作用，有助于将局部特征处理与状态建模之间的关系转化为可以直接检验的设计问题。

为此，本文提出局部结构引导的选择性状态调制方法（Structure-Guided Selective State Modulation，SGSM）。该方法首先在中间特征上建立可学习的局部趋势，利用特征相对趋势的残差修正送入投影分支的响应；随后，由趋势特征与二维邻域变化构造结构描述子，通过感知特征重要性模块（Perceptual Feature Importance Module，PFIM）生成空间重要性。所得重要性用于调节扫描输入驱动和动态步长，使局部结构线索直接参与状态递推。这里的重要性由下游任务损失学习，其作用是为状态调制提供条件，而不是输出图像域的去斑结果。局部修正、重要性估计和状态调制共同构成一个能够逐项拆解和比较的建模过程。

本文以 SAR 目标检测作为主要验证场景，在 SSDD 上比较原始状态空间块及四种替换变体，报告三个随机种子的完整结果。现有实验中，DS-only 变体的 AP50:95 平均值相对基线增加1.95个百分点，PFIM-only 变体的召回均值较高；完整组合没有超过 DS-only，token-wise 重要性也没有显示出相对 image-wise 对照的优势。这些观察表明，局部响应处理、状态调制位置与空间粒度之间的关系需要分别评估，而不能由模块叠加直接推断其互补性。我们同时将分类、变化检测与语义分割的已有记录作为适用范围分析，区分各任务的接入方式、训练条件与正负结果。

本文的主要贡献如下：

1. 提出一种面向 SAR 特征的局部结构引导状态调制设计，将局部响应变换、结构描述和选择性递推联系起来，研究局部统计参与状态更新的具体方式。
2. 建立扫描输入驱动与动态步长两类调制的明确数学描述，区分局部修正、输入缩放与状态转移的作用，并给出相应可拆解实现。
3. 在 SSDD 三种子五变体实验中分析单模块、组合及空间粒度的表现，报告完整配对变化，并结合跨任务记录讨论该设计的适用范围。

---

这版引言刻意不把“整理证据”“没有超越 SOTA”当作科学贡献，也不许诺四任务都提升。完成第12节的严格对照后，应把末段与第三条贡献更新为确认后的机制和跨数据集结果。若论文最终只保留 DS-only，需要相应简化第四段与方法名，而不是继续把 PFIM 写成主贡献。

## 12. Next Actions

优先顺序按“能否改变论文成立性”排序，不按数据集数量排序。本节都是后续方案，**不是本轮已执行的实验**。

### P0：先封住证据与因果缺口

#### P0-1 回收原始证据并冻结版本

1. 在正式项目记录与用户 SSH 配置范围内确认候选服务器，先查主机身份、磁盘、CPU/GPU及已有任务；新任务不能沿用旧“空闲”结论。当前记录包含 A6000 系列及其他实验机，但本轮未做在线资源检查，因此不指定现时空闲卡。
2. 新建独立任务目录，仅读取/复制本批 15 run 的 protocol.json、args.yaml、model YAML、train/val/test 清单、eval_test.json、训练曲线、train_complete、日志和冻结源码；保留原目录不覆盖。只需证据时先不要重新训练。
3. 按 CSV 的15个权重 hash逐项绑定 checkpoint；核对实际选模 fitness、cos_lr、AMP、早停 epoch、初始化加载记录和数据 hash。权重需要检查时在合适服务器做，不在本机自动加载/推理。
4. 复算数值并生成逐 run 证据清单。缺哪项就标哪项，不用“文件齐全”概括不可见产物。
5. 区分冻结版源码与当前改动版；当前 runner MD5=1801B6BEDAF57D8490B48F6A5602CCF2，历史 local/remote 清单分别为 e1f5ba4196a62e20a00adee9b17bed79 / 20c7b4e1c6ca3c39b6658c71383ed311。这个差别是版本核查任务，不自动证明实验出错。

验收：每个主表单元格均能追到“运行ID→数据划分→模型/训练配置→选模→权重→评估器→原始输出”；scheduler/fitness 冲突关闭。不要为了修复文档而悄悄改变旧实验口径。

#### P0-2 先选正确的因果对照，而不是继续铺大矩阵

**短路径：保留现有 DS 替换块，补一个 neutral-shell 对照。**保留其单向扫描、MLP、LayerNorm、state size、残差尺度、投影与旁路，只关闭待研究机制；明确令 λ=0、关闭 energy/PFIM、q=1、Δ不加ηI。这里 q=1 必须显式实现，不能只把 α=0 后留下 γ=0.75。

建议第一轮核心六变体：

| 逻辑标识（新对照，尚未实现） | 作用 |
|---|---|
| U：原 VSSBlock | 原系统参照 |
| S：neutral-shell | 隔离“换了整个壳层”的贡献 |
| S+D：局部修正+能量项 | 复验现有 DS-only 的机制增量 |
| S+I：PFIM 调制 | 在共同壳层衡量 PFIM |
| S+D+I | 完整组合 |
| S+D+I-global | 粒度对照 |

先在训练/验证数据上做服务端形状、梯度、开关与数值等价检查；主矩阵固定同一 split，训练 seed=0/1/2，六变体共18 run。因旧矩阵的 neutral 条件、初始化和协议未完全一致，**不能直接把旧15 run冒充新矩阵中的格子**。旧结果保留为历史系统级对照。

**更契合论文目标的中路径：保留原四向 SS2D 的壳层，只注入局部统计。**保持 scan方向、投影宽度、归一化、LS/RG、任务头、状态维数和初始化不变，新增 correction、drive、Δ开关；二维重要性必须随四个 scan方向同步展开/还原。该路线才能干净回答“相同 Mamba + 状态调制”。它是一个需要新实验的新版本，不可把旧结果改名继承。

建议：先用短路径判断现有优势是否主要来自壳层替换；若目标坚持是 backbone/state-modeling 机制论文，再优先做中路径。短路径 S 本身解释了大部分增益时，继续加数据集的价值低于解决机制问题。

验收：不仅比较 U vs S+D，更要比较 S vs S+D、S vs S+I；报告所有 seed，包括负向结果。门控关闭测试、参考 scan 与 fused scan 的前向/梯度一致性，都在服务器完成；不把 CPU fallback 的通过当成 CUDA 通过。

#### P0-3 冻结未来实验政策

- 固定 split_seed，独立 train_seed；另立重复划分实验分析 split 敏感性，不混成同一方差。
- 事先确定主指标、选模规则、最大预算和停止标准；若继续使用 fitness，明文记录。早停可统一规则但不是同训练步数；若要求同计算预算，应固定更新步数并另报最好验证权重。
- 同 seed 尽量复制共享参数初值；新增模块参数用独立 RNG，避免增加层后扰动其他层初始化。
- 方法/超参选择只用训练/验证；SSDD TEST 已看过，未来确认应增加未参与设计的数据集/独立确认集，并披露适应性开发过程。
- 先定义预计最小有意义效应和确认重复数；3 seeds 是起点而非显著性的保证。若波动大，可预先安排5 seeds，但不按试验结果反复追加直到“通过”。

#### P0-4 同步改论文事实，不等待新实验

按第7节修订 block数、fitness、scheduler待核验标记、PFIM含义、两调制位置和跨任务接口；把过时README改成短入口而非另一套研究状态。此次报告未实施这些修改；后续获准修改时继续使用同一论文源文件，保留原稿与结果。

### P1：建立真正的机制证据和外部有效性

#### P1-1 分离 correction、energy、drive、Δ

在 P0 共同壳层基础上，按以下顺序增加实验：

1. correction-only、energy-only、二者合用；检查 λ 与卷积核是否仍表现为收缩/平滑。若不是，按实际学到的局部变换解释。
2. PFIM drive-only、Δ-only、drive+Δ；γ-only 与 w-only 可以做参数化诊断，但不要再当三个独立状态位置。
3. 相同 I 放在普通输入预加权、scan内部驱动、Δ logits、scan输出重加权的位置对照。控制描述子、额外参数与训练条件。
4. 固定均值/可学习卷积、1D趋势/2D趋势、2D std/1D std/去掉std三种情况分别设计；不得把现 `no-2dstd` 当“无std”。

新开关应在关闭时返回明确的参考函数。若保留旧式γ参数化，还应补固定q=0.75与q=1的控制，以区分常数幅度/初始化改变和内容依赖门控；若改为零初始化、恒等中心化的门控，须登记为新版本，不能沿用旧版本的性能数字。

先用开发集筛选真正有辨别力的对照，再冻结确认矩阵；在确认结果表中保留原始 Full 和global，不因新方案改进而删去旧负结果。

#### P1-2 若仍希望叫 RGSP，增加可靠性代理与干预

先给可靠性一个与任务相关但独立于 I 的操作定义，例如对标签保持扰动的预测/特征稳定性；如果有可配准的独立观测，再评估跨观测一致性。不能直接定义“高 I 就可靠”再画 I 的热图证明自己。

建议完整流程：

1. 在开发集事先确定干净/扰动图对、扰动强度、区域与对齐规则；记录目标区域、结构背景和杂波，而不把所有背景都标为不可靠。
2. 获取 I、E、λ、有效 q、Δ、状态/输出变化；同时给出单纯能量、局部std及普通空间门控的对照。
3. 用恒定 I、空间置换 I、跨样本交换 I、反转 I 做干预；干预I时固定内容流，避免把输入破坏与重要性作用混在一起。冻结模型干预可能造成分布外输入，需补训练期对照，不能把任意干预跌分全归为可靠性机理。
4. 报告 I 与代理的关联，以及 I 真正通过 q/Δ 带来的任务变化。两个条件同时成立才有“reliability-associated cue”的证据。
5. 若对SAR强度使用Gamma乘性扰动，说明图像是否线性强度、分布/looks、归一化以及原图已含speckle；不能把对log图或任意中间特征乘Gamma说成真实SAR成像模拟。

无配对干净真值时不要声称恢复指标；不需要为了本任务强行增加 PSNR/SSIM。可靠性命名的最低门槛是定义、代理有效性、控制实验三者闭环，而不是一个漂亮热图。

#### P1-3 检测泛化与四任务矩阵

- **检测优先：**在 SSDD 确认候选后，对 SAR-ShipDataset 和 SAR-AIRcraft1.0 至少跑 U/S参照、确认方法的同协议三种子；需要解释组合时再保留关键消融。先按源场景分组再切 patch，避免航空源图泄漏；SAR-ShipDataset 不等于 AIR-SARShip-2.0。
- **分类：**先选 SAR-ACD 或有效 MSTAR-SOC；固定真实类别数、SOC角度/目标实例或样本域协议，使用相同接入位置和分类头。SAMPLE 的 real-only 不能冒充原始 synthetic-to-real 设置。
- **变化检测：**至少同时保留 Farmland 与 Yellow River/Ottawa 中一个非大幅受益数据集；固定成对图、宏块划分、边界缓冲、类权重、阈值和整图指标汇总。Farmland先检查低召回基线，不能拿失效基线作机制卖点。
- **分割：**选 FUSAR-Map 或 OpenEarthMap-SAR 作为完整受控任务。锁定4类/ignore/黑色类处理、SAR-only标签策略、原场景级划分、decoder和选模指标；除mIoU外给road/细长结构类别指标，保留负结果。
- **统一机制：**若主张 backbone 内部更新，四任务都必须在真正的 encoder内部接入并参与后级计算；输出适配器可另成扩展实验，不能与替换混写。同一任务内头/损失一致，跨任务头可以不同。
- **初始化/预算：**scratch 与共同预训练分两轨。若新增模块必须 warmup，baseline也提供合理且相同预算的对照；不要用ours多50 epoch后称纯模块收益。

当前 `run_sar4_full_suite.py` 的数据集注册表**没有 SAR-ShipDataset**，虽然有 SAR-AIRcraft1.0；现共享runner只提供baseline/ours等已有路径，也没有上述所有neutral和路径消融。扩展前应实现明确注册和冻结配置，不能把报告中的逻辑变体名直接当可执行CLI选项。

#### P1-4 效率与竞争力

1. 在同一空闲服务器GPU、同精度/同输入下，分别测完整模型 B=1 latency、B=16 throughput；固定预热、同步和重复次数，报告中位数及分位数。
2. model-only与含预处理/NMS的端到端速度分开。保存GPU/驱动/torch/CUDA/kernel、输入、batch、精度、计时范围和原始耗时。
3. Params区分总参数与可训练参数；FLOPs记录MAC约定、自定义scan计数及未覆盖算子。线性复杂度不替代真实延迟。
4. 加至少一个现代轻量检测器与一个直接Mamba/SAR参照。内部机制对照严格同配方；外部竞争力比较使用公开合理配方或相等调参预算，别用一套明显不适合外部方法的低学习率强行做弱基线。
5. 文献原报数单表，不能把MaDiNet组件profile、不同分辨率/预训练/OBB数字拼成排行榜。Lee等固定去斑前处理可作补充，但优先级低于同结构与调制位置对照。

### P2：投稿表达与交付

- 对直接相关文献补完正式版本和对应代码核验，特别区分 Mamba、Vim、VMamba、Mamba-YOLO、MaDiNet预印本/正式版。
- 以“同框架实测、机制消融、作者报告、历史探索”四类组织表格；每行固定数据版本、split、HBB/OBB、patch/原图、初始化和evaluator。
- 图示必须画出 q→输入、Δ→衰减及写入、D×修正特征旁路、扫描方向与真实P3数量。模拟图标 conceptual，实测图注明checkpoint和固定样例选择规则。
- 确认主结果后统一题名、摘要、引言、贡献与结论，再按目标期刊排版。不要通过提高文稿语气掩盖P0缺口，也不需要把每段都写成免责说明。
- 获准发生持久论文/代码变更后，按共享协议提交独立记忆增量，经核验再合并唯一MemoryRoot。本轮报告不改正式记忆，不把 SGSM 建议写成用户已经采纳的决策。

### 可执行工作流的停止点

**原始证据闭环 → 同结构开关正确 → 机制矩阵确认 → 检测多集确认 → 跨任务/干预与效率 → 定稿。**

没有合适服务器时，只继续静态核查与文稿修订，不自动回退本机CPU/GPU；不启用4050服务。一次阶段未过关就先定位原因，不并行扩大成大量数据集任务。具体服务器、安装环境、训练启动和共享资源使用，应在下一次实际实施任务中确认；本报告提供流程，不视为已经启动它们。

## 证据索引与核验边界

文中 E#:行号 是本次工作区中的一基行号/定位，区间用于阅读说明。点击链接从关键入口查看；历史文件未经本轮重测。

- E1：[正式当前状态](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/PROJECT_STATE.md)、[TODO](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/TODO.md)、[冲突](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/CONFLICTS.md)、[决策](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/DECISION_LOG.md)。
- E2：[PFIM、DeSpeckleSSM、DS/VSSBlock](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/nn/modules/mamba_yolo.py:87)。
- E3：[CrossScan 与 selective_scan_reference](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/nn/modules/common_utils_mbyolo.py:68)。
- E3补充：[本地CUDA scan写入实现](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/selective_scan/csrc/selective_scan/cus/selective_scan_fwd_kernel.cuh:119)。静态核对δ×u与B相乘的写入；本轮未运行该kernel。
- E4：[模型生成](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/run_sarmix_proxy.py:71)、[正式检测runner](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/run_sar4_full_suite.py:47)、[评估](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/evaluate_sarmix.py)。
- E5：[深度/宽度缩放与模块解析](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/nn/tasks.py:865)。
- E6：[逐run CSV](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/results_individual.csv)、[汇总CSV](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/results.csv)、[协议](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/BASELINE_PROTOCOL.md)、[计划](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/EXPERIMENT_PLAN.md)、[执行记录](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/EXPERIMENT_LOG.md)。
- E7：[fitness](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/utils/metrics.py:745)、[P/R阈值](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/utils/metrics.py:616)、[best保存](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/engine/trainer.py:527)。
- E8：[shared backbone和任务头](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/tools/run_shared_sar_tasks_v2.py:245)、[VMamba输出适配器](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/nn/modules/vmamba_backbone.py:76)。
- E9：[当前中文论文](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/paper_draft/ig_ds_ssm_cn/main.tex)、[已有审稿清单](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/paper_draft/ig_ds_ssm_cn/REVIEWER_AUDIT.md)。跨任务数字本轮主要按此稿与项目记录核对，未恢复远端原始结果。
- E10：[历史可视化审计](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/reports/dsssm_pfim_historical_visualization_audit_20260823.md)。
- E11：[MaDiNet预印本方法原文](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/downloads/madinet_paper/source/main_MaDiNet.tex:312)、[下载的官方mixer](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/downloads/madinet_official/MaDiNet-master/MambaSAR.py:162)。
- E12：[复杂度脚本](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/dsh_experiment_results/measure_complexity.py)、[THOP入口](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/source_code/work_train_single/ultralytics/utils/torch_utils.py:315)。
- E13：[用户提供的ZIP](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/PROJECT_HANDOFF/Mamba_memory_key_info_export_20261004.zip)、[题目与引言讨论总结](C:/Users/13207/Downloads/SAR_Mamba_论文定位题目与引言讨论总结.md)。讨论中的结论按待核查主张处理，不自动作为科研事实。

### 本轮在线核对的一手入口

- R1：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)。核对标题、作者和摘要中选择性传播/遗忘表述。
- R2：[Vision Mamba（Vim）](https://arxiv.org/abs/2401.09417)。核对标题、作者及双向建模。
- R3：[VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)。核对SS2D、四条扫描路线及其与Vim的区别。
- R4：[Mamba YOLO](https://arxiv.org/abs/2406.05835)。核对ODMamba、局部RG建模和检测背景。
- R5：[MaDiNet预印本](https://arxiv.org/abs/2411.07500)、[正式论文DOI](https://doi.org/10.1109/TCSVT.2025.3574657)。正式标题、作者、出版信息另与Crossref核对；未将元数据查询当作正式全文已完整核验。

其余稿中MCMN/RSS-Net/MambaDIP等文献本轮未逐篇独立全文核验，故未据其细节作排他性新颖性结论。未做截至2026-10-05的穷尽文献检索。

### 关键版本指纹（SHA-256）

| 文件 | 本轮值 |
|---|---|
| `mamba_yolo.py` | `48E6BA384C1A4FCCAA98894A664375A276605E2F14830B6B89AA92E4250567E5` |
| 当前中文 `main.tex` | `3AFA6EAA7328A4BEC9F04F39F359564D16FDF4DB8213D57E9BD0AFE22F8F243E` |
| `results_individual.csv` | `4632E57FDC9FF5E1A3ABB771471D05BEF82504D4E0C6E5A183E64C5735975036` |
| 当前检测runner | `D7A27FC4C3F725B58691A6B54A81147BA5D5B050BEC4C45AD9D84C93B2562478` |
| shared runner v2 | `E90E621C6738902F39A65E18AE4344095DE54162E67ED5E5BC5EF9B37F365AA0` |
| VMamba adapter | `5FB46B975BD7B6DF4ADDA579D84CE553BEB40C30EFFFCAC7D6E9D0C76FD8AA38` |

前三项已与用户ZIP逐字节哈希对照一致。指纹用于固定本报告对象，不代替运行真实性、数据合法性或协议完整性审计。
