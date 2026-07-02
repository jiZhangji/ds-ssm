# AAAI 投稿交接文档：IG-DS-SSM / PFIM 实验与论文上下文

更新时间：2026-06-22

## 1. 当前工作一句话总结

当前项目已经从多服务器实验恢复阶段进入 AAAI 投稿前收口阶段。论文主线建议固定为 **IG-DS-SSM / PFIM：面向 SAR 舰船检测的重要性引导去斑状态空间建模**。核心思想是把 SAR 斑点噪声抑制和目标重要性感知直接注入 Mamba/SSM 的状态更新过程，而不是只做外部去噪或简单注意力增强。

## 2. 当前最重要结论

当前最稳的主线不是 DG-DS-SSM，而是 **DS-SSM + PFIM / IG-DS-SSM**。

最佳本地实验：

| 项目 | 内容 |
|---|---|
| 实验名 | `PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_32bs-ex27` |
| 来源 | `server_3090_b` |
| 数据集 | SSDD |
| Precision | `0.97378` |
| Recall | `0.95219` |
| AP50 / mAP50 | `0.98704` |
| AP50:95 / mAP50-95 | `0.74022` |
| 参数量 | `5.9648M` |
| GFLOPs | `13.2079` |
| Latency | `28.7546 ms/img` |
| FPS | `34.777` |

DG-DS-SSM 是扩展实验，Recall 有竞争力，但目前没有超过 PFIM ex27：

| 实验 | P | R | AP50 | AP50:95 |
|---|---:|---:|---:|---:|
| DG-DS-SSM ex34 standard | `0.9700` | `0.9560` | `0.9850` | `0.7350` |

因此论文中应把 DG-DS-SSM 写成 extension / ablation / future work，不宜作为主贡献。

## 3. 论文定位

推荐论文标题方向：

> Importance-Guided De-Speckle State Space Modeling for Lightweight SAR Ship Detection

核心卖点：

1. **De-speckled input modeling**：通过局部背景估计和残差抑制构造去斑 token。
2. **Energy-aware state damping**：根据局部能量抑制不可靠背景状态传播。
3. **PFIM importance guidance**：估计 token 级目标重要性。
4. **Importance-guided SSM update**：用 PFIM 调制 `gamma / memory retention`、`write_gate / state writing`、`delta / transition step`。
5. **Lightweight SAR detector**：在接近强 SOTA 精度的同时保持约 `5.96M` 参数和 `13.21` GFLOPs。

不要主张“全面超过 MaDiNet”。更稳的表述是：

> Compared with MaDiNet, our method emphasizes a much lighter detection pathway while retaining competitive SSDD accuracy. Compared with AMANet-style COCO AP results, our AP50:95 is very close, but our design is explicitly built around SAR-aware state-space noise suppression.

## 4. 当前已生成论文文件

| 文件 | 用途 |
|---|---|
| `paper_draft/aaai_ds_ssm_paper_en.tex` | AAAI 风格英文 LaTeX 草稿 |
| `paper_draft/aaai_ds_ssm_paper_en.pdf` | 英文草稿 PDF |
| `paper_draft/aaai_ds_ssm_paper_zh.md` | 中文翻译稿 |
| `paper_draft/aaai_ds_ssm_refs.bib` | 临时参考文献 BibTeX |
| `paper_draft/aaai26.sty` | 临时本地编译 shim，正式投稿时必须替换为 AAAI 官方模板 |

注意：`aaai26.sty` 是为了本地生成 PDF 的临时文件，不是官方 AAAI 模板。

## 5. 关键实验与数据表文件

换模型或新对话继续工作时，优先阅读以下文件：

| 文件 | 为什么重要 |
|---|---|
| `reports/current_research_mainline.md` | 判断当前主线是 PFIM / IG-DS-SSM，不是旧 ex23 或 DG-DS-SSM |
| `reports/experiment_ranking_recomputed.md` | 重算实验排名，确认 PFIM ex27 是当前最佳 |
| `reports/current_codex3_experiment_report.md` | DDP codex3 复现实验结果，说明它完成但低于 ex27 |
| `reports/design_evidence_matrix.md` | 各设计点是否有代码/实验支撑，避免乱写没有证据的贡献 |
| `reports/madinet_ds_ssm_complexity_report.md` | MaDiNet core 与本方法复杂度对比 |
| `reports/open_source_ssdd_complexity_comparison_plan.md` | SSDD 开源方法复杂度/复现实验计划 |
| `reports/lightweight_detection_comparison_madinet_datasets.md` | 最新整理的轻量方法对比表 |
| `database/lightweight_detection_comparison_madinet_datasets.csv` | 上述对比表的 CSV 版本 |
| `database/complexity_comparison_madinet_ds_ssm.csv` | 本方法、DG-DS-SSM、MaDiNet core 的复杂度实测 |
| `database/open_source_ssdd_complexity_comparison_plan.csv` | SSDD 公开方法与复杂度计划表 |

## 6. MaDiNet 相关对比结论

MaDiNet 论文使用的主要实验数据集是：

1. `SSDD`
2. `SAR-ShipDataset`
3. `SAR-AIRcraft1.0`

MaDiNet 表中关键轻量方法结果：

| Method | SSDD AP50 | SAR-ShipDataset AP50 | SAR-AIRcraft1.0 AP50 |
|---|---:|---:|---:|
| YOLOv3 | `96.2` | `88.8` | `83.9` |
| YOLOX | `96.2` | `--` | `81.0` |
| YOLOv5n | `98.0` | `89.3` | `88.2` |
| YOLOv8n | `98.1` | `--` | `88.2` |
| YOLOv5s | `98.3` | `--` | `--` |
| DiffDet4SAR | `96.9` | `95.1` | `88.4` |
| MaDiNet | `99.0` | `97.5` | `90.5` |
| Ours: IG-DS-SSM / PFIM ex27 | `98.704` | `--` | `--` |

外部检索到的轻量化方法包括：

| Method | 主要数据集 | 备注 |
|---|---|---|
| YOLO-Lite | SSDD, SAR-ShipDataset | 轻量 SAR ship detector |
| LH-YOLO | SAR-ShipDataset, HRSID | 改进 YOLOv8n，轻量高精度 |
| SHIP-YOLO | SAR-ShipDataset | LH-YOLO 表中引用 |
| YOLOv10n | SAR-ShipDataset, HRSID | 新 nano YOLO 参考 |
| CPN-YOLO | SSDD, HRSID | 2026 去噪增强 YOLO |
| YOLO-SAATD | SAR-AIRcraft1.0 / SAR airport-aircraft | 轻量机场/飞机检测，报告相对提升 |

外部论文结果可能使用不同划分和指标口径，不要直接和 MaDiNet 表混为严格 leaderboard。

## 7. SSDD SOTA 背景

SSDD 没有统一官方 leaderboard，因此需要按指标口径分别对标：

| 口径 | 关键方法 | 结果/意义 |
|---|---|---|
| AP50 | MaDiNet | SSDD AP50 `99.0`，强 SOTA 参考 |
| AP50:95 | AMANet | 报告约 `74.20` AP，和 PFIM ex27 的 `74.022` 接近 |
| 2026 新方法 | CPN-YOLO | SSDD mAP/AP50 约 `98.9`，主打去噪增强 YOLO |
| 轻量化 | SURGE | 轻量 SAR ship detection via contrastive distillation |

推荐论文表述：

> On SSDD, recent strong SAR ship detectors include MaDiNet, which reports 99.0 AP50, and AMANet, which reports 74.20 AP and 98.50 AP50 under a common SSDD split. Our PFIM-enhanced IG-DS-SSM achieves 74.022 AP50:95 and 98.704 AP50 with a lightweight 5.96M-parameter detector.

## 8. 后续实验优先级

如果继续补实验，按下面顺序：

1. **固定 ex27 权重重新标准测试 SSDD**  
   输出 P / R / AP50 / AP50:95 / F1 / Params / GFLOPs / Latency。

2. **补 PFIM 分支消融**  
   最少需要：
   - baseline Mamba-YOLO-T
   - DS-SSM only
   - PFIM on gamma only
   - PFIM on write only
   - PFIM on delta only
   - full PFIM / ex27

3. **补 SAR-ShipDataset 实验**  
   这样可以和 MaDiNet 的三数据集表更完整对比。

4. **决定是否做 SAR-AIRcraft1.0**  
   如果论文主打 ship detection，可以不做；如果想对齐 MaDiNet 三数据集，则需要做。

5. **HRSID 可选**  
   用于对比 LH-YOLO、CPN-YOLO、SURGE 等轻量方法。

## 9. 如果用别的模型/新对话，应让它先读什么

建议给新模型的第一句话：

> 请先阅读 `reports/aaai_handoff_experiment_and_paper_context.md`，然后按其中列出的关键文件继续。不要重新假设论文主线；当前主线是 IG-DS-SSM / PFIM，DG-DS-SSM 只是扩展实验。

然后要求它按顺序读取：

1. `reports/aaai_handoff_experiment_and_paper_context.md`
2. `reports/current_research_mainline.md`
3. `reports/experiment_ranking_recomputed.md`
4. `reports/design_evidence_matrix.md`
5. `reports/lightweight_detection_comparison_madinet_datasets.md`
6. `paper_draft/aaai_ds_ssm_paper_en.tex`

如果任务是补实验表，还要读：

7. `database/lightweight_detection_comparison_madinet_datasets.csv`
8. `database/complexity_comparison_madinet_ds_ssm.csv`
9. `database/open_source_ssdd_complexity_comparison_plan.csv`

如果任务是继续写论文，还要读：

10. `paper_draft/aaai_ds_ssm_paper_zh.md`
11. `paper_draft/aaai_ds_ssm_refs.bib`

## 10. 明确不要做的事

1. 不要把 DG-DS-SSM 写成当前主线。
2. 不要声称完整 MaDiNet detector 复杂度已经测完；当前只测了官方 MambaSAR res4+res5 core module。
3. 不要把不同论文、不同划分、不同指标口径的结果放进同一张严格 SOTA 表却不加说明。
4. 不要声称 local scale prior / GDN-style gain control 已经是主线贡献；当前证据不足。
5. 不要在没有统一代码版本验证前，过度解释 PFIM 三个分支的单独贡献。

## 11. 当前最适合写进论文的压缩版总结

当前 SSDD SAR 舰船检测方向没有统一官方 leaderboard，不同论文在数据划分和指标口径上存在差异，因此需要按 AP50 与 AP50:95 分别对标。现有强方法中，MaDiNet 是重要 SOTA 参考，在 SSDD 上报告 P 98.0 / R 97.0 / AP50 99.0 / F1 97.5，代表高精度 SAR Mamba/扩散式检测路线；AMANet 则在常用 SSDD 划分上报告 AP50:95 74.20 / AP50 98.50，是 COCO-style AP 口径下非常关键的对手。2026 年新方法中，CPN-YOLO / Denoising-Enhanced YOLO 在 SSDD 上报告 P 97.0 / R 95.1 / mAP 98.9，主打去噪增强 YOLO；SURGE 强调轻量化 SAR 舰船检测中的对比蒸馏。相比这些方法，我们当前主线应定位为轻量化、SAR 去斑先验驱动的状态空间检测器：IG-DS-SSM / PFIM 通过去斑输入建模、能量感知状态抑制和重要性引导的 gamma/write/delta 调制，将 SAR 噪声抑制机制直接注入 Mamba 状态更新过程。当前最佳实验 PFIM ex27 在 SSDD 上达到 P 0.9738 / R 0.9522 / AP50 0.9870 / AP50:95 0.7402，模型规模约 5.96M 参数、13.21 GFLOPs。因此，论文叙事不应单纯声称全面超过 MaDiNet，而应强调：在接近强 SOTA 精度、尤其 AP50:95 接近 AMANet 的同时，本文方法保持了明显更轻量的检测结构，并提供了更可解释的 SAR 噪声感知状态空间建模机制。
