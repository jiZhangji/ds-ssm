# Decision Log

## DEC-001：以 IG-DS-SSM / PFIM 作为论文主线

- **日期/阶段**：日期未知；方法收敛阶段
- **来源**：`reports/current_research_mainline.md`、`HANDOFF_FOR_NEXT_MODEL.md`
- **背景问题**：项目存在 DS-SSM、PFIM、DG-DS-SSM 等演化版本，需要确定论文主贡献。
- **备选方案**：以 DG-DS-SSM 或局部尺度/GDN 方向作为主线。
- **采用方案**：IG-DS-SSM / PFIM。
- **采用理由**：当前代码实现清晰，PFIM ex27 位于恢复实验排名前列，且与 SAR-aware state update 和轻量 Mamba-YOLO 叙事一致。
- **放弃理由**：DG-DS-SSM 证据不足；局部尺度/GDN 候选缺少完整文档与独立实现验证。
- **影响范围**：方法代码、消融矩阵、论文主线和比较方法。
- **当前状态**：有效。

## DEC-002：保留三项贡献边界

- **日期/阶段**：日期未知
- **来源**：`reports/ssdd_comparison_reproducibility_audit_20260823.md`
- **背景问题**：若同时强调多个模块和跨任务结果，创新点容易分散。
- **采用方案**：只保留 SAR-aware DS-SSM 状态更新、PFIM retention/write/step 引导、约 6M/13G 的精度效率平衡。
- **采用理由**：三项贡献能分别对应机制、引导策略和工程效率证据。
- **替代方案**：将跨任务泛化或 DG-DS-SSM作为独立主贡献。
- **被否决原因**：当前跨任务证据不稳定，DG-DS-SSM不是主线。
- **当前状态**：有效。

## DEC-003：严格区分证据层级

- **日期/阶段**：2026-08-07 至 2026-08-12 修订阶段
- **来源**：`reports/aaai_paper_current_status.md`
- **采用方案**：同协议本地实测、历史恢复证据、作者报告文献值分别标记；只有统一协议结果用于严格主表。
- **采用理由**：不同划分、初始化、输入尺寸、评估器和 batch 会改变指标，不能直接重排 SOTA。
- **限制**：目前 OGSOD 只有单次匹配结果，SSDD/SAR-Ship/AIRcraft 部分历史结果仍协议受限。
- **当前状态**：有效。

## DEC-004：初始化分为 scratch 与公平预训练两层

- **日期/阶段**：日期未知
- **来源**：`reports/thread_019f2406_integrated_context.md`
- **采用方案**：严格因果实验中 baseline 与 ours 都使用 `weights=none`；若做预训练最终性能，双方使用同一个真实可迁移的预训练机会，新 DS-SSM/PFIM 参数单独说明初始化。
- **采用理由**：避免把不匹配的 `yolov8n.pt` 部分加载误称为 Mamba-YOLO 预训练。
- **历史事实**：当前 H100 sweep 是 scratch；历史 PFIM ex27 执行过 `model.load(yolov8n.pt)`，但只匹配少量项目，不能重标为 clean scratch。
- **当前状态**：有效，历史运行需保持原标签。

## DEC-005：检测结构选择与跨任务部署分开

- **日期/阶段**：日期未知
- **来源**：`reports/thread_019f2406_integrated_context.md`、`reports/current_research_mainline.md`
- **采用方案**：检测结构搜索重点放在 SSDD/OGSOD；最终检测主线使用浅层 stage；跨任务不强制所有任务都用同一 stage。
- **理由**：P2/P3 对小目标检测与分割细长道路的需求不同，检测最优位置不能直接迁移到分割。
- **限制**：P3 在检测证据中有优势，但 FUSAR/OpenEarthMap 分割中没有超过 baseline。
- **当前状态**：有效但需在论文中准确限定适用任务。

## DEC-006：不要为追求 SOTA 删除强基线或隐藏负结果

- **日期/阶段**：跨任务实验阶段
- **来源**：当前窗口关于外部复现和结果差异的讨论、`reports/all_results_recomputed_20260831.md`
- **采用方案**：可复现的关键 baseline 自己跑；不能公平复现的方法只列作者报告，并写明协议差异。
- **理由**：复现值与论文值不一致是正常现象，关键是同协议 baseline/ours 的因果比较和透明报告。
- **当前状态**：有效。

## DEC-007：v4 记忆根目录与业务代码分离

- **日期/阶段**：2026-09-07 迁移
- **来源**：`Codex_Project_Memory_Migration_Agent_v4.0.txt`、本次预检查
- **采用方案**：正式记忆根目录为 `docs/project-memory/`；根目录 `PROJECT_MEMORY.md` 与 `AGENTS.md` 只提供短入口。
- **理由**：兼容多窗口唯一项目发现，同时避免把业务根目录知识文件与旧实验目录混淆。
- **保留事项**：历史 v3 创建的空根 `archive/` 不删除、不接管。
## DEC-008：v1 共享任务迷你 runner 结果作废，由 v2 校正协议取代

- **日期/阶段**：2026-08-25 至 2026-08-31 跨任务阶段
- **来源**：压缩线程 `019f2406` 交接摘要、`reports/authoritative_cross_task_matrix_20260826.md`
- **背景问题**：v1 迷你 runner（自定义浅层 VSS backbone、abs-diff 变化检测、逐 batch 指标、随机 tile 泄漏、SADD 含 2083 空图、AIR 整图缩到 640、SRSDD OBB 误当 HBB）产生的结果不可靠。
- **采用方案**：v1 结果仅保留为历史诊断并标记作废；v2（`prepare_cross_task_v2.py` + `run_shared_sar_tasks_v2.py`）采用 SADD 阳性 795/44/44、AIR 源图级 tiling、SRSDD 裁剪 HBB、FUSAR city-disjoint、CD 空间不相交宏块 + Siamese；另建 paper-aligned 轨道（FUSAR 530/80 四类、SRSDD 532/134、AIR 210/90、MSTAR-SOC A-ConvNet 协议）。
- **采用理由**：消除协议性假阴/假阳，避免把 v1 数值写入论文。
- **影响范围**：EXPERIMENT_LOG 证据分级、CONFLICTS-002/004/011、TODO paper-aligned 任务。
- **当前状态**：有效；paper-aligned 轨道训练完成状态待核实。

## DEC-009：历史权重只可作目标域 warm-start 诊断

- **日期/阶段**：2026-08-28
- **来源**：`reports/ssdd_historical_recheck_and_fusar_audit_20260828.md`
- **采用方案**：ex27→当前 P3-d16（450/519 张量兼容）只用于 SSDD 目标域 warm-start 诊断实验；禁止描述为外部预训练或公平论文结果；历史 ex27 引用必须区分验证集 `0.74022` 与官方 TEST `0.714454`。
- **当前状态**：有效；warm-start 配对运行状态待核实（见 TODO TASK-WARMSTART-VERIFY）。

## DEC-010：论文定位重构为通用 SAR-Mamba 状态更新

- **日期/阶段**：2026-09-19，论文实验设计重构阶段
- **来源**：用户明确决定采用“用于 SAR 视觉任务的通用 Mamba 状态更新方法”定位。
- **背景问题**：IG-DS-SSM/PFIM 的机制并不依赖检测头，但此前的实验设计过度收缩到目标检测；另一方面，当前跨任务结果并不支持对所有任务全面领先。
- **采用方案**：将 IG-DS-SSM/PFIM 定义为可插入视觉 Mamba 主干的 SAR-aware 状态更新模块。目标检测作为主验证场景，分类和变化检测作为跨任务验证，语义分割暂列为探索性边界实验。
- **实验范围**：检测主数据集为 SSDD、SAR-ShipDataset、SAR-AIRcraft1.0；分类优先 SAR-ACD，并补 MSTAR-SOC/SAMPLE 的有效闭集协议；变化检测保留 Ottawa、Farmland、Yellow River 的同协议结果；OGSOD 不纳入当前主实验。
- **公平原则**：每个任务内 baseline 与 ours 使用同一 Mamba 主干、同一任务头、同一数据划分和训练协议，核心差异只应是 DS-SSM/PFIM 状态更新。
- **采用理由**：保留方法的通用科学定位，同时利用当前检测、分类和部分变化检测的正向证据；把分割负结果作为适用边界，避免夸大 universal claim。
- **影响范围**：论文摘要、引言、方法标题、实验章节结构、任务选择、对比表和后续待办；不自动修改业务源码或启动训练。
- **当前状态**：有效；替代 DEC-005 中“检测结构与跨任务部署分开”的旧论文范围建议，但不否定其关于不同任务 stage 需单独验证的技术判断。

## DEC-011：2026-09-23 收尾判断——主线固定，阶段转为整理证据与重写论文

- **日期/阶段**：2026-09-23，收尾判断阶段（来源：批次 mamba-merge-01，发现 F004）
- **背景问题**：跨任务结果混杂正负，需要确认是否重新寻找方向，以及哪些结果可进入主表。
- **采用方案**：不再重新找方向，工作进入“整理证据、补关键对照、重写论文”阶段。论文主线固定为利用局部残差与空间重要性调节 Mamba 的状态输入与转移，以减少 SAR 不可靠信息影响并保留有用结构：DS-SSM 侧对应局部残差修正与能量相关调制，PFIM 侧对应学习空间重要性以调节扫描输入与转移步长。这里“通用”仅指模块设计与部署目标，不代表已证明所有 SAR 任务都能提升。
- **硬边界一（结构不可混用）**：模型版本不能混用——历史 P2 替换、当前 P3 替换、以及在 VMamba 输出特征上做的 P3 适配并非同一结构，比较时必须分别标注。
- **硬边界二（高分不等于可入主表）**：已有高分不都能进主表。批次记录显示 SSDD 历史权重的官方 TEST 复评 AP50:95 为 70.35→71.45，而非旧验证集的 74.02；另有一条历史配置重跑路径使用测试集选最佳权重，须单独排除其独立测试主张。
- **硬边界三（结果并非全面正向）**：分类与 Farmland 有改善，Yellow River 与两个分割任务下降，SSDD 在不同 batch 上正负变化，论文需如实呈现适用边界。
- **交付物**：`docs/research-closeout-20260923.md` 与 `docs/paper-plan.md`。
- **未执行事项**：本轮未启动训练，也未修改代码与原始结果。
- **当前状态**：有效；相关数值与排除结论的具体归档位置待核实（historical 重跑路径的测试集选权重主张仍需单独排除）。
