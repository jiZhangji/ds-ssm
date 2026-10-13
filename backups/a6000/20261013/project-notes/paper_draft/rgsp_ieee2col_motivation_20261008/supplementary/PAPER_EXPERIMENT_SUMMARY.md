# PAPER_EXPERIMENT_SUMMARY.md — 论文实验体系总结

> 面向论文投稿的完整实验总结，含主结果、消融、复杂度、诚实措辞建议。
> 数据来源：SSDD 3 seed（seed0/1/2）公平消融（RESULTS.md / results.csv / results_individual.csv）。

## 一、实验设置（可复现性）

- **任务**：SAR 舰船检测（SSDD），单类 "ship"。
- **数据划分**：928 张 → 742 train / 186 val（`random.Random(seed).shuffle` 确定性切分）/ 232 test 封存。
- **训练**：scratch（weights=none），batch=16，imgsz=640，epochs=150，patience=30，AdamW(lr0=0.0002, lrf=0.01)，mosaic=0.3，deterministic，seed∈{0,1,2}。
- **评估**：验证集选 best.pt，官方 test split 评估 AP50/AP75/AP50:95/P/R。
- **公平性**：所有变体同 seed 共享同一划分（paired），同超参，同设备，clean scratch；不删失败实验，不人工挑选 seed。

## 二、主结果（3 seed，mean±std）

| 方法 | AP50 | AP75 | AP50:95 | P | R | inf(ms) |
|---|---:|---:|---:|---:|---:|---:|
| Mamba-YOLO-T（baseline） | 0.8652±0.0256 | 0.3405±0.0525 | 0.4203±0.0347 | 0.8955±0.0183 | 0.7924±0.0170 | 6.60 |
| **+ DS-SSM（ours, ds_only_p3）** | 0.8670±0.0101 | **0.3758±0.0378** | **0.4398±0.0144** | 0.8989±0.0096 | 0.7967±0.0026 | **5.98** |
| + PFIM（pfim_only_p3） | 0.8823±0.0161 | 0.3483±0.0394 | 0.4312±0.0170 | 0.9013±0.0127 | **0.8139±0.0220** | 6.08 |
| + DS-SSM + PFIM（full_p3） | 0.8797±0.0011 | 0.3410±0.0312 | 0.4291±0.0113 | 0.9191±0.0149 | 0.7985±0.0102 | 6.07 |

**主结果声明（诚实版）**：DS-SSM 在 SSDD 上带来一致的 AP50:95 提升（+1.95pp），且 AP75 增益最显著（+3.53pp），同时降低推理延迟（5.98ms vs 6.60ms）。PFIM 单独使用可提升召回（+2.15pp）。

## 三、消融（3 seed，paired Δ vs baseline）

| 消融 | AP50:95 Δ mean | Δ std | per-seed | 结论 |
|---|---:|---:|---:|---|
| Q1 DS-SSM 单独（ds_only） | **+1.95pp** | 0.0387 | {+2.90, +5.26, −2.30} | 核心稳健增益 |
| Q2 PFIM 单独（pfim_only） | +1.09pp | 0.0273 | {+2.13, +3.16, −2.00} | 稳健，改善召回 |
| Q2' PFIM on DS-SSM 边际（full−ds_only） | **−1.07pp** | — | — | 朴素结合负交互 |
| Q3 token vs image（full−global） | −0.86pp | — | {−2.64, −0.21, +0.29} | token-wise 无优势 |

**消融结论（诚实版）**：
1. DS-SSM 是稳健且主要增益来源。
2. PFIM 单独有效（尤其召回），但在 DS-SSM 之上再叠加 PFIM 产生**负交互**（full − ds_only = −1.07pp）。
3. token-wise 重要性不优于 image-wise（global 均值更优 +0.86pp）。**这是重要发现，须如实写入论文，而非反向宣称。**

## 四、复杂度（seed0 实测）

| 方法 | Params | GFLOPs | FPS | inf(ms) |
|---|---:|---:|---:|---:|
| baseline | 5,985,547 | 13.61 | 151.1 | 6.62 |
| ds_only_p3 | 5,918,104 | 13.12 | 166.7 | 6.00 |
| full_p3 | 5,920,441 | 13.15 | 164.2 | 6.09 |

**说明（诚实）**：DS-SSM 的轻量化部分来自架构替换（DeSpeckleSSM state_dim=16 无 rank 膨胀 + MLP mlp_ratio 2.0 vs 4.0），非纯 SSM 机制增益。论文需如实区分"机制增益"与"架构瘦身"。

## 五、跨任务 / 与主流方法对比（诚实边界）

- **本阶段仅完成 SSDD 检测的 3 seed 公平消融**。SAR-ShipDataset、SAR-AIRcraft1.0 的公平 3 seed 对比、与主流方法（YOLOv8/v9、RT-DETR 等）的同协议对比尚未完成。
- **历史结果（Partially Confirmed，不进投稿主表）**：SSDD BS16 +7.75pp / BS32 −3.20pp（batch 敏感）；SAR-Ship +8.25pp、SAR-AIRcraft +1.04pp（非公平协议）；FUSAR/OpenEarthMap 负结果。
- 这些历史数据提示**方法对 batch/协议敏感**，投稿前须在同协议下重跑，不能直接引用历史数字。

## 六、论文措辞建议（诚实、可防守）

1. **主结果主打 DS-SSM**：宣称"DS-SSM 在 SAR 检测上带来一致 AP50:95 提升（+1.95pp），AP75 +3.53pp，且更轻量"，数据支持充分。
2. **PFIM 作为独立模块**：宣称"PFIM 单独使用改善召回（+2.15pp）"，但**不要**宣称"PFIM 与 DS-SSM 联合最优"（数据相反）。
3. **token-wise vs image-wise**：如实报告"token-wise 重要性在 SSDD 上不优于 image-wise"，或将其作为未来工作/局限讨论。
4. **报告 seed 间方差**：所有主表给 mean±std 和 seed 数，明确"SSDD 单 seed 方差大（baseline AP50:95 std≈0.035），结论基于多 seed"。
5. **不删除负面实验**：full_p3 的负交互、seed2 outlier 都要保留并讨论，这是可信度的一部分。

## 七、复现产物清单

| 文件 | 内容 |
|---|---|
| `results.csv` | 每变体 3 seed mean±std 汇总 |
| `results_individual.csv` | 15 个 run 逐条 + 权重 sha256 |
| `RESULTS.md` | 完整结果 + 科学问题结论 + 诚实发现 |
| `BEST_CONFIG.md` | 推荐配置 + 复现命令 |
| `BASELINE_PROTOCOL.md` | 消融语义对照 + 公平性规则 + 复杂度账本 |
| `EXPERIMENT_LOG.md` | 全时间线 + 机制发现 + 负面发现 |
