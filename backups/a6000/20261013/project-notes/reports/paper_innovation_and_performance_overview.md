# 论文创新点与当前性能概览

## 1. 当前建议论文主线

当前最建议作为论文主线的创新点是：

**IG-DS-SSM / PFIM：面向 SAR 图像目标检测的感知重要性引导去斑状态空间模块。**

它不是单纯把 Mamba-YOLO 换成一个新模块，而是在 SAR 图像斑点噪声和目标结构特征之间建立一个可解释的调制机制：

- `DS-SSM` 负责把 SAR 去斑/抑噪思想引入状态空间建模；
- `PFIM` 负责估计 token 或局部区域的重要性；
- `PFIM` 进一步调制 `gamma`、`write_gate`、`delta`，让模型对重要区域保留更多信息，对噪声区域更强抑制。

## 2. 当前核心创新点

| 创新点 | 当前代码状态 | 当前实验支持 | 论文价值 |
|---|---|---|---|
| input de-biasing / 去斑输入修正 | 已实现 | 有间接实验支持 | 用 `x_tilde = x - lambda_s * residual` 抑制局部斑点残差 |
| energy-aware damping | 已实现 | 有间接实验支持 | 根据局部能量调节状态传播，适合 SAR 噪声特性 |
| soft gamma 保留机制 | 已实现 | 有间接实验支持 | 避免过强抑制导致目标信息丢失 |
| PFIM importance map | 已实现 | 有较强实验支持 | 提供感知重要性图，是当前主线核心 |
| PFIM 调制 gamma | 已实现 | 有组合实验支持 | 控制记忆保留强度 |
| PFIM 调制 write_gate | 已实现 | 有组合实验支持 | 控制当前信息写入强度 |
| PFIM 调制 delta | 已实现 | 有组合实验支持 | 控制状态转移步长/动态响应 |
| 2D local std 局部统计 | 已实现 | 有组合实验支持 | 给 PFIM 提供局部纹理/斑点强度线索 |
| 20 epoch warmup | DDP 版本已实现 | 最新 DDP 实验可运行，但不是最优 | 更像训练策略，不建议作为主要理论创新 |
| local scale prior / GDN-style gain control | 候选方案 | 暂无明确实验支持 | 暂不建议进入当前主线 |

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
- `PFIM ex24` 和 `PFIM ex27` 都明显处于前列，说明 PFIM 方向有实验价值。
- 最新 DDP 80 batch 的 `codex3` 能完整训练，但性能低于 32 batch 的 `ex27`，说明大 batch / warmup / DDP 设置还需要继续调参。

## 4. 当前主线是否足够写论文

从目前证据看，**IG-DS-SSM / PFIM 具备作为论文创新点的潜力**，理由是：

1. 有明确的 SAR 任务动机：斑点噪声会影响目标检测特征表达。
2. 有明确的结构创新：不是只加注意力，而是把重要性引导加入状态空间动态过程。
3. 有较好的实验表现：当前最优 mAP50-95 为 `0.74022`，在已恢复实验中排名第一。
4. 有可解释模块：`gamma`、`write_gate`、`delta` 分别对应记忆保留、状态写入、动态步长。

但还需要补强：

- 需要明确 baseline 对比，例如原始 Mamba-YOLO-T、普通 DS-SSM、PFIM 完整版。
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

> 本文提出一种面向 SAR 图像目标检测的 Importance-Guided De-Speckle State Space Module。该模块首先通过局部平滑残差构造去斑输入表示，再利用感知特征重要性模块 PFIM 估计 token 级重要性，并将其注入状态空间模型的记忆保留、状态写入和动态步长控制中，从而在抑制斑点噪声的同时增强目标区域特征表达。

## 7. 下一步最重要的工作

优先级建议：

1. 固定 `ex27` 为当前最佳主线版本。
2. 对齐 `work_train_single` 和 `work_train_ddp` 的模型结构。
3. 补 PFIM 三分支消融实验。
4. 重新整理 baseline 对比表。
5. 再决定是否引入 local scale prior / GDN-style gain control。
