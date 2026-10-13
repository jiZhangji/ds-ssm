# Paper Status

## Current paper direction

- **主题**：面向 SAR 视觉任务的通用 Mamba 状态更新方法（`IG-DS-SSM/PFIM：面向 SAR 视觉任务的通用 Mamba 状态更新方法`，2026-09-18 按此重构；chat-reported）。
- **主方法**：IG-DS-SSM / PFIM，作为可插入视觉 Mamba 主干的 SAR-aware 状态更新模块；方法层面通用、面向 SAR-Mamba，不限定为检测专用。
- **DG-DS-SSM**：探索性延长线，不是当前主贡献。
- **目标叙事**：把 SAR 去斑先验和目标/区域重要性直接注入 Mamba 的状态更新，以目标检测为主要验证任务，分类与变化检测作跨任务验证，分割列为探索/适用边界实验，同时报告参数量、GFLOPs、延迟/FPS。当前不宣称对所有 SAR 任务或所有分割场景全面有效。
- **实验三层**（chat-reported）：(1) 检测主验证 SSDD / SAR-ShipDataset / SAR-AIRcraft1.0；(2) 分类+变化检测跨任务 SAR-ACD / MSTAR-SOC / Ottawa / Farmland / Yellow River；(3) 分割探索边界 FUSAR-Map / OpenEarthMap-SAR，不强行声称全面提升。每任务对比原始 Mamba 主干+原任务头 / +DS-SSM / +DS-SSM+PFIM，保持相同划分/输入/训练/指标。
- **OGSOD 处置**：从论文主线移出，仅保留历史结果，不作为必须完成实验（chat-reported）。

## Three contributions

1. **SAR-aware DS-SSM state update**：对输入局部残差进行去偏/去斑修正，并通过能量感知对 scan 输入/状态贡献进行抑制。
2. **PFIM guidance**：从局部趋势和二维局部变化估计 token importance，调制 `gamma`、`write_gate` 和 `delta`。
3. **Accuracy-efficiency balance**：在约 6M 参数、13G FLOPs 的 Mamba-YOLO 级别模型上形成有竞争力的点。

## Claim boundaries

安全表述：在匹配协议的 SSDD、SAR-ShipDataset 和 SAR-AIRcraft 结果中，P3/PFIM 在部分定位指标上改善；分类与变化检测提供跨任务验证。所有结论必须结合同协议 baseline、消融和复杂度说明。

不能表述：已经在所有 SAR 任务或所有外部 SOTA 上超越；不能把文献值、历史非匹配结果和本地统一实测直接重排；不能把 MaDiNet 核心模块复杂度当作完整 detector。

## Current manuscript artifacts

- Evidence-based Chinese internal draft (2026-10-02): paper_draft/ig_ds_ssm_cn/main.tex; Overleaf package: paper_draft/ig_ds_ssm_cn_overleaf.zip; compiled PDF: output/pdf/ig_ds_ssm_cn_paper.pdf; 8 standalone figures are in output/pdf/ig_ds_ssm_cn_figures/.
- This draft uses SSDD 5-variant × 3-seed results as the primary controlled evidence; non-unified detection and cross-task records are separated and labeled exploratory. It is not submission-ready: fair SAR-ShipDataset/SAR-AIRcraft multi-seed validation and same-protocol external baselines remain incomplete. Reviewer audit: paper_draft/ig_ds_ssm_cn/REVIEWER_AUDIT.md.
- The AAAI English/Chinese drafts below are retained historical working files and were not overwritten or reconciled with the new internal draft.
- English draft: `paper_draft/aaai_ds_ssm_paper_en.tex`
- Chinese draft: `paper_draft/aaai_ds_ssm_paper_zh.tex`
- PDFs: `paper_draft/aaai_ds_ssm_paper_en.pdf`、`paper_draft/aaai_ds_ssm_paper_zh.pdf`
- Current status report: `reports/aaai_paper_current_status.md`
- Local `aaai26.sty` is a temporary compilation shim and must be replaced by official author-kit style before submission.

## Review / evidence status

- Strict review records in `reports/top_conference_review_round1_20260807.md` through `round4` are around 5/10 to 6/10.
- Primary blockers remain the cross-task same-backbone matrix, no-PFIM/global/full controlled matrix, factor diagnostics, per-task analysis and unified complexity.
- Cross-task results do not support a universal backbone claim: P3 underperforms baseline on OpenEarthMap-SAR mIoU and FUSAR-Map mIoU, and underperforms on Yellow River F1; it improves Farmland strongly and Ottawa slightly.

## External comparison policy

- Must-run controlled baselines: Mamba-YOLO-T, IG-DS-SSM/PFIM-T; optionally YOLOv8n, YOLOv10-S/N and RT-DETR-R18 under one protocol.
- Literature-only rows: methods without complete code/config/weights or with incompatible tasks/protocols, such as MaDiNet, ARGO-Net, SFS-CNet, DiffDet4SAR and some SAR-specific methods.
- A reimplementation must be labeled `Our reimplementation`; it must not be called official reproduction.
- When reproduced values differ from a paper, report both paper-reported and our unified-protocol values, explain split/initialization/evaluator differences, and keep the main causal table unified.

### Author-reported reference values（作者报告，未经本地复现）

- MaDiNet（arXiv:2411.07500 核对）：SSDD P/R/AP50/F1 `98.0/97.0/99.0/97.5`；SAR-ShipDataset `96.2/95.6/97.5/95.9`；SAR-Aircraft AP50 `90.5`。
- DiffDet4SAR AP50：SSDD `96.9`、SAR-ShipDataset `95.1`、SAR-Aircraft `88.4`。
- GDB-YOLOv5s（IET Image Processing 2024，DOI `10.1049/ipr2.13140`）SRSDD HBB：YOLOv5s `80.2/42.4`，GDB `81.7/47.6`（AP50/AP50:95）。
- RSPC（电子与信息学报 2025）SAMPLE 类别增量 AIA `93.30/93.23%`（N=1/2），非闭集 Accuracy，不能排名。
- A-ConvNet（MSTAR-SOC）：官方 `99.13%`，PyTorch 复现报 `99.34%`。

### Paper wording constraints（来自线程 019f2406 细化）

- PFIM 初始化：gamma/write 在 0.5 附近中性，delta 带小初始偏置（raw importance、eta_init=0.1）；不能写“整个模块精确中性”。
- SSDD 引用 ex27 时必须区分验证集 `0.74022` 与官方 TEST `0.714454`。
- OGSOD 局限表述为“密集重复目标”（tank 1205 实例/71 张 TEST 图），不能写成“对所有小目标差”（SSDD 反证）。
- 用户要求的“剔除更强方法”对比表只能作为选择性子集并明确标注，不能当完整 SOTA 排行榜。

## Pending paper changes

1. Freeze the same-backbone cross-task matrix for detection, classification and change detection; OGSOD is no longer a publication-critical dependency.
2. Add branch-level PFIM diagnostics and parameter-matched controls.
3. Add per-class/density/scale evidence and interpretable PFIM visualizations.
4. Update efficiency table using one profiler and same hardware.
5. Keep negative cross-task outcomes visible and avoid universal generalization claims.
6. 统一最终版本表述为 P3，消除英文稿中 “only the shallow P2/4 stage is replaced” 的旧表述（CONFLICT-013，2026-09-11 窗口提出）。
7. 补固定去斑预处理（Lee/常规滤波）对照，证明收益非简单前处理可替代（2026-09-11 窗口建议，未执行）。

## Scope discussion (2026-09-11, superseded)

- 早期曾建议主论文收缩为只做 SAR 目标检测；该建议已被 2026-09-19 的新决定替代。原因是用户明确将方法定位为“面向 SAR 视觉任务的通用 Mamba 状态更新方法”，而不是检测专用模块。
- 用户确认贡献控制为三个（SAR-aware DS-SSM 状态更新、PFIM 引导、约 6M/13G 精度效率平衡），与既有 Three contributions 一致。
- 写作主线表述（2026-09-11 窗口，历史版本）：SAR 杂波/散斑可能被 selective scan 当作有效信息并沿状态递推传播；方法在浅层、信息进入多尺度传播之前对进入 SSM 的特征驱动做 SAR-aware 控制。该窗口曾以 OGSOD 作为主要证据，现已由 2026-09-19 的跨任务定位替代。
- 助手声称（2026-09-11，未验证）已将思路/动机/方法/实验设计与结果表整理到记忆根之外的笔记 `C:/Users/13207/Desktop/反代/project/IG-DS-SSM-PFIM/02-实验结果/当前实验结果.md`。

## Scope decision (2026-09-19)

- **正式定位**：IG-DS-SSM/PFIM 是面向 SAR 视觉任务的通用 Mamba 状态更新方法，不限定为目标检测器插件。
- **主验证任务**：目标检测、图像分类、变化检测。检测作为证据最完整的主场景；分类和变化检测用于证明跨任务迁移。
- **探索性任务**：语义分割暂不作为通用性主证据。FUSAR-Map 和 OpenEarthMap-SAR 当前存在低于强 baseline 的结果，保留为适用边界/后续改进方向，不将负结果隐藏。
- **主数据集**：检测使用 SSDD、SAR-ShipDataset、SAR-AIRcraft1.0；分类优先 SAR-ACD，并补 MSTAR-SOC/SAMPLE 的有效闭集协议；变化检测保留 Ottawa、Farmland、Yellow River 的完整同协议结果。
- **OGSOD**：不纳入当前主实验，保留历史记录，不重新启动 OGSOD 队列，也不把其单次结果当作核心证据。
- **方法对比原则**：每个任务使用同一 Mamba 主干和同一任务头，baseline 与 ours 只改变 DS-SSM/PFIM 状态更新；传统 CNN/Transformer 只作外部参照，不作为核心因果 baseline。
- **声明边界**：论文可声称“在多个 SAR 视觉任务中验证了通用状态更新思想”，不能声称“在所有 SAR 任务上全面领先”。

## 2026-10-08 external IEEE conversion audit (not canonical manuscript)

- Input: user-provided archive C:/Users/13207/Downloads/rgsp_cn_revision_20261006 (2).zip, SHA256 016A7D709392F13260E9A7A03C109D0B6281AB8B334DC77C8C42AE1D9AB9FD8D.
- Created an independent IEEEtran journal/XeLaTeX two-column source and 12-page preview under C:/Users/13207/Downloads/rgsp_cn_revision_20261006_ieee2col; two-pass XeLaTeX compilation succeeded. The repository's canonical paper_draft/ig_ds_ssm_cn manuscript was not changed or replaced.
- The archive is internally inconsistent: its paper.pdf has 26 pages and appendices A–F, while main.tex ends after the conclusion and references only two figures. The preview therefore covers only the available source, not the full 26-page PDF. Twenty-one bibliography items were recovered from the supplied PDF; the two citation keys already present in main.tex were added from ICLR 2025/NeurIPS 2025 proceedings records.
- The supplied fig1_motivation.pdf has partially missing/mispositioned vector labels when rendered; it was preserved unchanged. A matching full .tex source and the figure source are needed before calling the conversion complete.
- Evidence and rebuild instructions: C:/Users/13207/Downloads/rgsp_cn_revision_20261006_ieee2col/README_ieee2col.md.