# Project Timeline

时间以文件名、报告日期或窗口内相对顺序为依据；没有可靠原始时间戳的事件标为“日期未知”或“迁移时观察”。迁移日期不改变原始事件的先后。

## 阶段 1：项目恢复与证据整理

- **日期未知，历史恢复阶段**：从 1080Ti、3090-A、3090-B 等历史服务器恢复源码、权重、实验目录和日志；建立 `work_train_single` 与 `work_train_ddp` 两套工作副本。来源：`README_CURRENT.md`、`HANDOFF_FOR_NEXT_MODEL.md`。
- **日期未知，代码整理阶段**：将历史服务器快照作为证据归档，活跃修改集中在 `source_code/work_train_single/` 和 `source_code/work_train_ddp/`。来源：`PROJECT_NAVIGATION.md`。

## 阶段 2：方法确定

- **日期未知**：研究主线从多个候选方向收敛为 IG-DS-SSM / PFIM；DG-DS-SSM 被保留为探索性延长线。来源：`reports/current_research_mainline.md`。
- **日期未知**：明确三项贡献为 SAR-aware DS-SSM 状态更新、PFIM 重要性引导、约 6M/13G 的精度效率平衡。来源：`reports/ssdd_comparison_reproducibility_audit_20260823.md`。
- **日期未知**：明确 `gamma` 与 `write_gate` 在已执行 CUDA 路径中共同作用于 scan drive，`delta` 是独立转移控制；不再把三者描述成三个完全独立的递归位置。来源：代码与 `reports/design_evidence_matrix.md`。

## 阶段 3：检测实验与论文证据

- **2026-06-03 附近，精确运行时间未完全确认**：历史 SSDD PFIM ex27 达到 P `0.97378`、R `0.95219`、AP50 `0.98704`、AP50:95 `0.74022`，约 5.9648M 参数、13.2079 GFLOPs。来源：`README_CURRENT.md`、`reports/thread_019f2406_integrated_context.md`。
- **2026-07-31 附近，具体运行时间未确认**：OGSOD-1.0 完成单次匹配 baseline/PFIM TEST 比较，PFIM AP75 和 AP50:95 分别提升 `0.760 pp` 和 `1.300 pp`。来源：`README_CURRENT.md`、`reports/ogsod_test_results_20260731.md`。
- **2026-08-07 至 2026-08-12**：论文从普遍 AP50 超越改为定位敏感 AP75/AP50:95 和轻量 Pareto 叙事；严格评审从约 5/10 提升到约 6/10，但三 seed 和核心消融仍缺失。来源：`reports/aaai_paper_current_status.md`。
- **日期未知**：形成 OGSOD `ogsod_t_primary_3seed_v2` 主矩阵和 branch diagnostic 的执行方案；尚不能把“脚本已准备”当作“实验已完成”。来源：`reports/ogsod_ablation_matrix_plan.md`。

## 阶段 4：跨任务扩展

- **2026-07-21 至 2026-07-23**：完成 SAR-AIRcraft patch 派生协议的 P3 记录；VAL AP50/AP50:95 为 `0.888252/0.705129`，TEST 为 `0.708220/0.555516`。确认该协议与 MaDiNet 原图协议不同。来源：`reports/sar_aircraft_experiment_20260721.md`。
- **2026-07-23**：完成 SAR-ShipDataset 轻量 P3 与历史 Mamba-YOLO 参考结果整理；确认 baseline batch 中途变化，不能作为严格因果对照。来源：`reports/sar_shipdataset_current_results.md`。
- **2026-08-25 至 2026-08-31**：跨任务实验逐步加入 FUSAR-Map、OpenEarthMap-SAR、SAR-ACD、SAMPLE、Ottawa、Yellow River、Farmland 等；形成数据、协议和异常审计报告。来源：`reports/cross_task_phase0_analysis_20260825.md`、`reports/all_results_recomputed_20260831.md`。
- **2026-08-31**：为 AIR-SARShip-2.0、SRSDD、FUSAR-Map、Farmland、Sulzberger、SAR-ACD、SAMPLE、Yellow River 等列出优化诊断计划；计划和已完成实验分开保存。来源：`reports/priority_optimization_plan_20260831.md`。
- **2026-09-05**：完成 SAR-ACD 校正 BS64 批次、OpenEarthMap-SAR 有效 BS64 批次以及 Ottawa/Yellow River/Farmland 的 BS16/BS96 批次；结果显示 P3 在 Farmland 稳定提升，在 Ottawa 小幅提升，在 Yellow River 下降。来源：当前窗口可见结果笔记和 `PROJECT_STATE.md`。
- **2026-09-06**：一次现场检查显示 A6000-A/B/F 无本项目训练进程；3090-5 和 3090-3 有其他项目进程，3090-4 无本项目训练，1080Ti 空闲。该记录仅是当时观察。来源：当前窗口工具输出摘要。

## 阶段 5：记忆迁移

- **2026-07 中下旬**：H200 服务器（`/inspire/.../SAR-Generation`）环境 `sar-pretrain` 建立；GitHub `jiZhangji/ds-ssm` 初始化并推送；HF `shimiandeshu/sar-m` 四数据集下载整理；ship T/B baseline 队列建立，后从 H200 迁移至双 H100 断点续训（BS256→128）。来源：压缩线程 `019f2406`。
- **2026-08-07 至 2026-08-12**：四轮严格评审（约 5-6/10）；明确 CUDA 语义 `u_scan = gamma*write_gate*u`；新增 OGSOD 消融矩阵基础设施（9 个 YAML、快照/分层/可视化工具）。来源：`reports/top_conference_review_round*`、压缩线程 `019f2406`。
- **2026-08-23**：仓库目录整理（README.md、README_CURRENT.md、PROJECT_NAVIGATION.md、.rgignore）；线程 `019e95f0`、`019f6040` 手动归档。来源：压缩线程 `019f1e13`。
- **2026-08-25 至 2026-08-28**：跨任务 12 数据集矩阵（`authoritative_cross_task_matrix_20260826.md`）；v1 迷你 runner 暴露系统性问题（SADD 空图、AIR 整图缩放、SRSDD OBB 误用、CD 崩溃、FUSAR 协议不符）；SSDD 历史 checkpoint 官方 TEST 复核完成（ex27 `0.714454`）；FUSAR 610 mask 全量审计完成。来源：压缩线程 `019f2406`、`reports/ssdd_historical_recheck_and_fusar_audit_20260828.md`。
- **2026-08-28 至 2026-08-31**：v2 校正协议与 paper-aligned 轨道（SADD 阳性 795/44/44、AIR 210/90、SRSDD 532/134 HBB、MSTAR-SOC、FUSAR 530/80 四类）实现并通过本地测试；四源混合代理（sarmix）部署 3090 分片；A6000 四卡 seed1/2 BS32 队列启动并完成。来源：压缩线程 `019f2406`、`019f1e13`。
- **2026-09-06**：v3 迁移流程开始但只创建了一个空的根目录 `archive/`，预检查未发现 v3 知识文件或归档文件。
- **2026-09-07**：按 v4.0 建立唯一项目记忆根目录 `docs/project-memory/`、用户目录项目注册表、接手入口、脱敏归档和迁移登记。本次不修改业务源码、论文、数据或原始实验结果。来源：本次迁移批次 `MIG-20260907-v4-01`。
- **2026-09-09**：批次 `MIG-20260909-Mamba-BATCH` 增量迁移线程 `019f1e13` 与 `019f2406` 的压缩稿，补充服务器/托管、SSDD 受控 scratch 与历史复核、SRSDD/AIR/SADD 协议、v1 作废标记、论文措辞约束等事实。
- **2026-09-11**：线程 `019f2406` 窗口讨论论文写作思路与动机（SAR 杂波沿 selective scan 传播 → SAR-aware 状态控制 → 浅层放置），重申三项贡献；发现英文论文 P3 与 “P2/4” 措辞不一致（CONFLICT-013）；提出主论文收缩为 SAR 检测 3+1 数据集的建议（未成为已执行决定）；用户提出各数据集子采样快速选结构的提议（未决）；用户要求修复 SSDD/SAR-AIRcraft/SADD/AIR-SARShip-2.0/SRSDD-v1.0/SAR-ACD/SAMPLE/FUSAR-Map/Ottawa/Farmland/Yellow River/Sulzberger 的异常并重跑（无完成证据）；用户要求数据集上传 ModelScope。助手声称将思路与结果表写入记忆根之外的笔记文件。来源：批次 `MIG-20260911-MAMBA-BATCH-003`。
- **2026-09-18**：线程 `019f2406` 窗口（仅 user 消息）用户请求整理 A6000 服务器上的数据与代码并上传到 ModelScope，并两次发送“继续任务”；无 assistant 响应或完成证据，属未验证请求。来源：批次 `MIG-20260918-MAMBA-BATCH-007`。
