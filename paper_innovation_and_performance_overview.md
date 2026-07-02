# 论文创新点与当前性能概览

## 1. 当前建议论文主线

当前最建议作为论文主线的核心创新是：

**DS-SSM（De-Speckle Selective State Space Model）：面向 SAR 图像目标检测的去斑选择性状态空间模块。**

在这个主线中，`PFIM` 或“感知重要性引导”更适合作为 DS-SSM 内部的关键创新部分，而不是单独压过 DS-SSM 成为总标题。这样论文叙事会更清楚：

- `DS-SSM` 是核心模块，负责把 SAR 去斑/抑噪思想引入状态空间建模；
- `PFIM` 是 DS-SSM 内部的感知重要性估计分支，负责判断哪些 token 或局部区域更重要；
- `PFIM` 进一步调制 `gamma`、`write_gate`、`delta`，让 DS-SSM 对重要目标区域保留更多有效信息，对噪声区域进行更稳健的抑制。

因此，推荐论文主线写法是：

**以 DS-SSM 为核心创新，以 PFIM 感知重要性引导作为 DS-SSM 的关键增强机制。**

## 2. 当前核心创新点

| 层级 | 创新点 | 当前代码状态 | 当前实验支持 | 论文价值 |
|---|---|---|---|---|
| 核心模块 | DS-SSM | 已实现 | 有实验支持 | 面向 SAR 斑点噪声的状态空间建模改造 |
| DS-SSM 内部机制 | input de-biasing / 去斑输入修正 | 已实现 | 有间接实验支持 | 用 `x_tilde = x - lambda_s * residual` 抑制局部斑点残差 |
| DS-SSM 内部机制 | energy-aware damping | 已实现 | 有间接实验支持 | 根据局部能量调节状态传播，适合 SAR 噪声特性 |
| DS-SSM 内部机制 | soft gamma 保留机制 | 已实现 | 有间接实验支持 | 避免过强抑制导致目标信息丢失 |
| 关键增强机制 | PFIM importance map | 已实现 | 有较强实验支持 | 为 DS-SSM 提供感知重要性引导 |
| 关键增强机制 | PFIM 调制 gamma | 已实现 | 有组合实验支持 | 控制记忆保留强度 |
| 关键增强机制 | PFIM 调制 write_gate | 已实现 | 有组合实验支持 | 控制当前信息写入强度 |
| 关键增强机制 | PFIM 调制 delta | 已实现 | 有组合实验支持 | 控制状态转移步长/动态响应 |
| PFIM 特征来源 | 2D local std 局部统计 | 已实现 | 有组合实验支持 | 给 PFIM 提供局部纹理/斑点强度线索 |
| 训练策略 | 20 epoch warmup | DDP 版本已实现 | 最新 DDP 实验可运行，但不是最优 | 更像训练策略，不建议作为主要理论创新 |
| 候选增强 | local scale prior / GDN-style gain control | 候选方案 | 暂无明确实验支持 | 暂不建议进入当前主线 |

## 3. 当前最佳性能

当前根据已恢复 `results.csv` 重新统计，SSDD 上最优结果是：

| 排名 | 实验 | 来源 | P | R | mAP50 | mAP50-95 |
|---:|---|---|---:|---:|---:|---:|
| 1 | `PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27` | `server_3090_b` | 0.97378 | 0.95219 | 0.98704 | **0.74022** |
| 2 | `PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex24` | `server_3090_b` | 0.95701 | 0.95238 | 0.98287 | **0.73571** |
| 3 | `mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex7` | 多服务器重复实验 | 0.96099 | 0.95421 | 0.98226 | **0.73520** |
| 10 | `ddp_PFIM...ex27-codex3` | 当前双 3090 DDP | 0.97310 | 0.93040 | 0.97976 | **0.73028** |

说明：

- 当前最强结果是 `ex27`，mAP50-95 达到 `0.74022`。
- `PFIM ex24` 和 `PFIM ex27` 都明显处于前列，说明在 DS-SSM 中加入感知重要性引导是有实验价值的。
- 早期 DS-SSM `ex7` 也排名靠前，说明 DS-SSM 本身已经具备较强效果。
- 最新 DDP 80 batch 的 `codex3` 能完整训练，但性能低于 32 batch 的 `ex27`，说明大 batch / warmup / DDP 设置还需要继续调参。

## 4. 当前主线是否足够写论文

从目前证据看，**DS-SSM 适合作为论文的核心创新模块，PFIM 适合作为 DS-SSM 内部的重要增强机制**。

理由是：

1. 有明确的 SAR 任务动机：斑点噪声会影响目标检测特征表达。
2. 有明确的结构创新：DS-SSM 把去斑思想、能量抑制和状态空间建模结合起来。
3. 有进一步增强机制：PFIM 把感知重要性注入 `gamma`、`write_gate`、`delta`。
4. 有较好的实验表现：当前最优 mAP50-95 为 `0.74022`，在已恢复实验中排名第一。
5. 有可解释模块：`gamma`、`write_gate`、`delta` 分别对应记忆保留、状态写入、动态步长。

但还需要补强：

- 需要明确 baseline 对比，例如原始 Mamba-YOLO-T、普通 DS-SSM、DS-SSM + PFIM 完整版。
- 需要做 PFIM 消融：只调制 `gamma`、只调制 `write_gate`、只调制 `delta`、三者组合。
- 需要确认 `ex27` 对应的确切代码版本和配置文件。
- 需要决定是否把 DDP 版本作为复现实验版本，还是只作为后续训练工具。

## 5. 暂不建议写进主线的内容

以下内容目前建议暂时作为候选或背景，不要直接写成已经验证的论文贡献：

- `local scale prior`
- `GDN-style gain control`
- compression hyperprior 相关机制
- SAR-specific loss / augmentation
- 20 epoch warmup 作为核心创新

这些内容可以作为“未来工作”或“候选增强”，但目前缺少明确代码实现和实验排名支持。

## 6. 当前建议论文表述

可以暂时把论文创新点概括为：

> 本文提出一种面向 SAR 图像目标检测的 De-Speckle Selective State Space Model（DS-SSM）。该模块通过局部平滑残差构造去斑输入表示，并利用能量感知的状态抑制机制缓解斑点噪声对长程特征建模的干扰。在此基础上，本文进一步设计感知特征重要性模块 PFIM，估计 token 级重要性，并将其注入 DS-SSM 的记忆保留、状态写入和动态步长控制中，从而在抑制斑点噪声的同时增强目标区域特征表达。

## 7. 下一步最重要的工作

优先级建议：

1. 固定 `ex27` 为当前最佳主线版本。
2. 对齐 `work_train_single` 和 `work_train_ddp` 的模型结构。
3. 补 DS-SSM 与 PFIM 的分层消融实验。
4. 重新整理 baseline 对比表。
5. 再决定是否引入 local scale prior / GDN-style gain control。
