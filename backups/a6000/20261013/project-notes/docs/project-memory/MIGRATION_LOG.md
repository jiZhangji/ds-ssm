# Migration Log

## MIG-20260907-v4-01

- **protocol**: Project Memory Migration Agent v4.0
- **source-id**: `SRC-CODEX-MYDS-SSM-WINDOW-01`（迁移时分配，非平台线程 ID）
- **revision**: r01
- **target project-id**: `mamba-yolo-ds-ssm--ig-ds-ssm-pfim`
- **project root**: `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`
- **memory root**: `docs/project-memory/`
- **source coverage**: 当前窗口可见历史、压缩交接摘要、当前仓库的入口/报告/代码/实验索引；无完整原始消息分页
- **source form**: mixed original-visible / summary-level / file-verified / historical tool-output summary
- **input fingerprint**: 未计算平台聊天全文指纹；平台未提供稳定可导出的完整历史。使用 source-id、工作区、早期 SSDD sweep 主题和本 revision 归档共同去重。
- **archive**: `archive/2026-09-07-migration-ig-ds-ssm-window-r01.md`
- **snapshot**: `snapshots/2026-09-07-v4-initial-state.md`
- **v3 compatibility**: v3 曾开始但只留下空根 `archive/` 目录；无 v3 知识文件可合并，目录保留。

### Preparation

- 项目识别：高置信度，单一明确项目。
- 既有正式知识库：未发现。
- 其他候选项目：未发现。
- 写入前 Git 工作区：已有大量业务修改/未跟踪文件，未由本批次清理或覆盖。
- 并发：未发现其他 v4 迁移锁或既有正式记忆文件；本次为顺序执行。未宣称跨进程原子锁。

### Written files

- Root pointers: `PROJECT_MEMORY.md`, `AGENTS.md`
- Managed memory: `PROJECT_MEMORY.md`, `PROJECT_STATE.md`, `TIMELINE.md`, `DECISION_LOG.md`, `EXPERIMENT_LOG.md`, `TODO.md`, `PAPER_STATUS.md`, `CODE_STRUCTURE.md`, `CONFLICTS.md`, `MIGRATION_LOG.md`, `MEMORY_RULES.md`
- Knowledge: `knowledge/method_definition.md`, `knowledge/experiment_protocols.md`
- Archive: `archive/2026-09-07-migration-ig-ds-ssm-window-r01.md`
- Snapshot: `snapshots/2026-09-07-v4-initial-state.md`
- Registry mirror: `registry_entry.md`; copied to the user registry after file validation.

### Fact mapping

- Core method and CUDA semantics → `PROJECT_STATE.md`, `knowledge/method_definition.md`
- Experiment parameters/results/failures → `EXPERIMENT_LOG.md`
- Decisions and rejected alternatives → `DECISION_LOG.md`
- Paper claims and blockers → `PAPER_STATUS.md`
- Current tasks and completed queue status → `TODO.md`
- Code locations and historical verification → `CODE_STRUCTURE.md`
- Protocol/version conflicts → `CONFLICTS.md`
- Detailed window narrative → archive revision r01

### Sensitive-data handling

- Passwords, tokens, cookies, private keys and authentication details were not persisted.
- Historical server/account credentials were represented only as configured/de-identified resource context where necessary.

### Status

- Preparation: complete
- Extraction/comparison: complete for accessible range
- Write: complete
- Registry update: complete; user registry entry copied and re-read
- File/link/content verification: complete
- Final result: verified complete for the accessible source range

### Verification result

- Required managed files: all present and readable.
- Root entry: `PROJECT_MEMORY.md` points to the formal memory root; `AGENTS.md` contains one bounded v4 entry block.
- Registry: `C:/Users/13207/ProjectMemory/registry/mamba-yolo-ds-ssm--ig-ds-ssm-pfim.md` exists and matches the project mirror.
- Link correction: the formal entry uses `../../AGENTS.md` for the repository-root rules file.
- Git scope check: this batch added only `PROJECT_MEMORY.md`, `AGENTS.md` and `docs/project-memory/`; pre-existing business changes remain untouched.
- Credential-shape scan of new Markdown files found no password/token/key/cookie assignments.
- Archive revision r01 and snapshot are readable; no previous archive was overwritten.

### Handoff rehearsal

Using only the new memory files, a future AI can identify the research goal and hard constraints, current IG-DS-SSM/PFIM mainline, code and paper artifacts, completed experiments and metrics, negative/failed runs, unresolved protocol conflicts, publication blockers and the exact files to read next. Remote runtime availability remains an external dependency that must be checked live.

## MIG-20260907-v4-02

- **protocol**: Project Memory Migration Agent v4.0
- **source-id**: `SRC-CODEX-MYDS-SSM-WINDOW-01`（沿用 r01，非平台线程 ID）
- **revision**: r02
- **target project-id**: `mamba-yolo-ds-ssm--ig-ds-ssm-pfim`
- **project root**: `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`
- **memory root**: `docs/project-memory/`
- **source coverage**: 当前可见上下文、既有 v4 r01 覆盖范围、项目记忆文件和用户提供的 v4 附件全文；无平台完整原始消息分页
- **source form**: mixed current-visible / prior migration summary / file-verified
- **archive**: `archive/2026-09-07-migration-ig-ds-ssm-window-r02.md`
- **snapshot**: `snapshots/2026-09-07-v4-r02-before-migration-log.md`

### Preparation and comparison

- 项目身份与 r01 相同，复用既有 project-id、正式记忆根目录、用户注册表和入口。
- 已完整读取 `Codex_Project_Memory_Migration_Agent_v4.0.txt`，共 575 行；协议内容作为执行规则，不作为项目事实。
- 已读取本批次可能受影响的入口、状态、待办、规则、决策、实验、论文、代码结构、冲突、知识文件、r01 归档和 r01 快照。
- 未发现并发迁移锁；写入前已保存本批次脱敏快照。

### Incremental result

- **新增事实**：无。
- **重复事实**：本窗口已有科研内容均已在 r01 或现有正式记忆中覆盖，本批次不重复追加。
- **排除内容**：Windows 启动项、VMware、MuMu、D 盘和硬件优化属于主机维护，不纳入科研项目记忆；附件中的示例和操作协议不纳入科研事实。
- **更新文件**：仅本登记文件追加 r02；新增 r02 归档和迁移前脱敏快照。
- **未修改文件**：`PROJECT_MEMORY.md`、`PROJECT_STATE.md`、`TODO.md`、`TIMELINE.md`、`DECISION_LOG.md`、`EXPERIMENT_LOG.md`、`PAPER_STATUS.md`、`CODE_STRUCTURE.md`、`CONFLICTS.md`、`MEMORY_RULES.md`、项目根入口、用户注册表及全部业务文件。

### Verification

- 归档和快照已写入预期路径并可重读。
- 既有 v4 r01 归档未改写；已有历史事实、实验结果、决策、冲突和人工补充保持可追溯。
- 未新增 project-id、未删除/移动/重命名业务文件，未执行训练、部署、论文修改或 Git 操作。
- 新增 Markdown 未保存密码、API Key、Token、Cookie、私钥、数据库凭据或认证 URL。
- **结果**：已验证完成（针对本次可访问范围；无新增科研事实）。

### Handoff

- 新 AI 仍应从 `docs/project-memory/PROJECT_MEMORY.md` 开始，并按其读取顺序核对状态、待办和冲突。
- 本批次没有改变接手入口或当前研究结论；远程 GPU/服务器状态仍需在新实验前现场检查。

## MIG-20260909-MAMBA-BATCH

- Protocol: Project Memory Migration Agent v4.0; batch source extraction was prepared by Codex and reviewed with DSH.
- Source IDs: `019f1e13-9661-7dd2-ab99-8ccdd57cf567`, `019f2406-c105-7d63-bd7d-a3b993b2a169`.
- Archive: `archive/2026-09-09-migration-Mamba-batch.md`.
- Snapshot: `snapshots/MIG-20260909-Mamba-BATCH/`.
- Write status: completed; existing v4 memory was compared and updated incrementally.
- Scope: source code, data, paper files, configuration and Git files were not changed.
- Sensitive-data handling: credentials and authentication material were excluded from the extracted memory.

## MIG-20260909-COVERAGE-AUDIT-Mamba

- Protocol: Project Memory Migration Agent v4.0.
- Evidence sources indexed: 2.
- Archive: `archive/2026-09-09-coverage-audit-Mamba.md`.
- Snapshot: `snapshots/MIG-20260909-Mamba-COVERAGE-AUDIT/`.
- No source code, data, configuration, Git or runtime files were changed.

## MIG-20260911-MAMBA-BATCH-003

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-669c5f49d43e46b5930f260d54967f1b-003.md`（inner source `delta-20260911-161011-001.md`，collected UTC 2026-09-11T08:10:10Z；thread `019f2406`）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260911-211105-AUTO-669c5f49d43e46b5930f260d54967f1b-2`（外部托管，未修改）。
- Archive: `archive/2026-09-11-migration-Mamba-batch-003.md`（新建，未覆盖任何旧归档）。
- Incremental updates: `CONFLICTS.md`（新增 CONFLICT-013 P3 vs P2/4 措辞）、`TIMELINE.md`（2026-09-11 窗口事件）、`TODO.md`（新增 TASK-PAPER-P3-WORDING、TASK-DESPECKLE-CONTROL、TASK-DATASET-ANOMALY-FIX、TASK-DATASET-HOSTING）、`PAPER_STATUS.md`（pending 修改 6/7、范围收缩建议节）、`PROJECT_MEMORY.md`（本批次记录）。
- Files intentionally unchanged: `PROJECT_STATE.md`、`DECISION_LOG.md`、`EXPERIMENT_LOG.md`、`CODE_STRUCTURE.md`、`MEMORY_RULES.md`（无新的已验证状态/决策/实验事实；三项贡献为用户重申 DEC-002，未新增决策条目）。
- Sensitive-data handling: 证据含 SSH 明文密码与主机账户，全部未落盘；仅以既有脱敏远程环境描述为准。
- Chat-claim handling: 助手的写作主线表述与外部笔记写入均标记为聊天声称/建议，未升级为已验证事实。
- Scope: 未修改源码、数据、论文源文件、配置、.git 或运行状态。

## MIG-20260918-MAMBA-BATCH-007

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-aea9ef23f82246c2b67dca21cd5f1c34-007.md`（inner source `delta-20260918-112745-001.md`，collected UTC 2026-09-18T03:27:45Z；thread `019f2406`，2026-09-18 窗口仅 user 消息）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260918-125911-AUTO-aea9ef23f82246c2b67dca21cd5f1c34-6`（外部托管，未修改）。
- Archive: `archive/2026-09-18-migration-Mamba-batch-007.md`（新建，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实。新增一条未验证用户请求——用户请求整理 A6000 服务器上的数据与代码并上传到 ModelScope，并两次重复“继续任务”；无完成证据。
- Overlap: 该请求与 `TODO.md` 的 `TASK-DATASET-HOSTING`（2026-09-11 ModelScope 上传）同类；本批次扩展请求日期至 2026-09-18 并明确“整理 A6000 数据/代码”对象。
- Incremental updates: `TODO.md`（`TASK-DATASET-HOSTING` 请求日期与对象扩展）、`PROJECT_MEMORY.md`（本批次记录）。未改状态/实验/决策/冲突文件（无新验证事实或新冲突）。
- Sensitive-data handling: 本窗口证据不含凭据；未落盘任何密钥/令牌/认证 URL。
- Chat-claim handling: 请求与“继续任务”均标记为聊天请求，未升级为已完成。
- Scope: 未修改源码、数据、论文源文件、配置、.git 或运行状态。

## MIG-20260918-MAMBA-BATCH-005

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-16f9e8f7600141c49e28d25fc0b027d3-005.md`（inner source `delta-20260918-192745-001.md`，collected UTC 2026-09-18T11:27:45Z；thread `019f2406`，assistant 消息 2026-09-18 10:10:09）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260918-203928-AUTO-16f9e8f7600141c49e28d25fc0b027d3-4`（外部托管，未修改）。
- Archive: `archive/2026-09-18-migration-Mamba-batch-005.md`（新建，未覆盖任何旧归档）。
- New facts: 无新增已验证事实。本窗口为 assistant 报告的 A6000 ModelScope 备份上传完成声明（B 主数据 19.61GB、A 补充 5.43GB、C 补充 116MB、实验/配置/依赖清单、SHA256；排除 `envs/`、`.secrets/`、Token/Cookie、缓存/临时；A6000-D SSH 超时未纳入）。
- Overlap: 该完成声明全部内容已在 `PROJECT_STATE.md`（A6000 ModelScope backup 一节）、`TODO.md`（`TASK-DATASET-HOSTING` 部分完成记录）、`CODE_STRUCTURE.md`（Remote environments）中完整记录，字节数与排除/限制一致；本批次未做核心文件增量修改。
- Incremental updates: 仅新增本归档与迁移日志条目；未改状态/实验/决策/冲突文件。
- Sensitive-data handling: 本窗口证据不含凭据；未落盘任何密钥/令牌/认证 URL。
- Chat-claim handling: 备份完成声明属助手聊天报告，未独立复核，未升级为已验证事实（既有记忆对此的“部分完成”表述不变）。
- Scope: 未修改源码、数据、论文源文件、配置、.git 或运行状态。

## MIG-20260919-MAMBA-BATCH-006

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-13281980333d40f88143d62db130dc8a-006.md`（inner source `delta-20260919-102750-001.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:02 起窗口）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260919-112147-AUTO-13281980333d40f88143d62db130dc8a-5`（外部托管，未修改）。
- Archive: `archive/2026-09-19-migration-Mamba-batch-006.md`（新建，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实；本窗口以用户请求/设计为主。主要新增内容——(1) 用户提出把 SSDD/OGSOD-1.0/SAR-ShipDataset/SAR-AIRcraft1.0 抽样子集整合为单一新数据集做 DS-SSM 与 PFIM 泛化 + 模块阶段消融；(2) SAI-Net 明确不进入对比；(3) H100 不可用，仅剩 3090/1080Ti/A6000，需更小 SAR 检测数据集；(4) 聊天报告 H100 工作区上传至 ModelScope `shimian123/ds-ssm-workspace-backup`（完整包约 25.2G，未独立复核）。
- Overlap: 三贡献/核心结果/初始化结论/复杂度/遗憾缺口均与既有 `PROJECT_STATE.md`、`PAPER_STATUS.md`、`TODO.md`、`DECISION_LOG.md`、`CONFLICTS.md` 内容一致，未重复改写。
- Incremental updates: `TODO.md`（新增 `TASK-INTEGRATED-QUICK-ABLATION`）、`PROJECT_MEMORY.md`（本批次记录）、本归档与迁移日志条目。未改状态/实验/决策/冲突/代码结构文件（无新验证事实或新冲突）。
- Sensitive-data handling: 证据含明文 SSH 口令/主机账户/凭据提示与 ModelScope token，全部未落盘；本批次任何记忆文件不含密钥/令牌/认证 URL。
- Chat-claim handling: 论文进展总结、`+1.098 pp` 历史 TEST 复核、H100 备份上传完成等均为聊天报告/用户请求，未独立复核，未升级为已验证事实。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260920-MAMBA-BATCH-009

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-ab9d41b7ac264278bb3dcf6a65379b3a-005.md`（inner source `delta-20260919-102750-002.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:10:33 起窗口，仅 user 消息）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260920-094017-AUTO-ab9d41b7ac264278bb3dcf6a65379b3a-4`（外部托管，未修改）。
- Archive: `archive/2026-09-20-migration-Mamba-batch-009.md`（新建；用 009 序号规避既有 `2026-09-19-migration-Mamba-batch-005.md`，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实。本批次为 IG-DS-SSM/PFIM 系统实验验证的长规划提示词与状态/硬件/实验跟随请求（均 user 级、聊天级，未完成）。唯一新增信号：用户要求为 FUSAR-Map 更换更强分割基线（提及 SegFormer 分割头）并逐个方法统计 Params/FLOPs；另要求把各数据集结果+方法指标整理进 Obsidian 工作台 `C:\Users\13207\Desktop\反代\.obsidian`。
- Overlap: 长规划（三贡献/分阶段/公平比较/不强制复现 MaDiNet 等方法/复杂度测量）、SSDD 早期 99 与 300 epoch、跨服务器调度与 ModelScope/GitHub 传输、FUSAR-Map 受控实验结果均与既有 `PROJECT_STATE.md`、`PAPER_STATUS.md`、`TODO.md`、`DECISION_LOG.md` 及 batch-006/007/008 归档一致，未重复改写。
- Incremental updates: 仅新增本归档与迁移日志条目；未改状态/实验/决策/冲突/待办文件（无新验证事实、无新冲突，新增信号均为未落地请求）。
- Sensitive-data handling: 证据含主机账户/口令/凭据提示与 ModelScope token 描述，全部未落盘；本批次记忆文件不含密钥/令牌/认证 URL。
- Chat-claim handling: 全部内容为 user 请求与规划，未升级为已验证事实；FUSAR“换更强基线”是否落地（SegFormer 等）未知，需后续现场核对。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260920-MAMBA-BATCH-014

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-a8a11946b3b14d2b8040ee1716327424-004.md`（inner source `delta-20260919-102750-003.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:10:33 起窗口）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260920-193014-AUTO-a8a11946b3b14d2b8040ee1716327424-3`（外部托管，未修改）。
- Archive: `archive/2026-09-20-migration-Mamba-batch-014.md`（新建；与 batch-008/011/012/013 同一内层 delta `delta-20260919-102750-003.md` 的又一个分片（-004），序号 014 规避既有归档，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实（全部为聊天级/未独立复核）。本分片相对既有 batch 的增量信号：(1) 论文对齐 FUSAR-Map 协议描述——四类（building/veg-other/water/road）、`WuHan_03_02` 测试、17560/4390/1280 划分、256×256 batch10、Adam LR 1e-3、LR×0.93 每两 epoch、150 epoch、FP32（AMP 约 epoch25 NaN）、scratch、Weighted CE+Dice、按 val mIoU 选择并保存 best_miou/best_fwiou；(2) 用户决定 OpenEarthMap-SAR 采用 SAR-only、P 标签协议并用双 A6000 训练（是否落地未验证）；(3) 论文定位经“仅检测”讨论后最终采纳“通用 SAR-Mamba 状态更新方法”三层重构，与 batch-008 结论一致。
- Overlap: FUSAR UPerNet 受控实验、公开 FUSAR 对比、代码修复、论文重构、OGSOD 移出主线、A→B 数据同步、road 诊断均与既有 `PROJECT_STATE.md`/`PAPER_STATUS.md`/`DECISION_LOG.md`/`TODO.md` 及 batch-008/011/012/013 归档一致，未重复改写核心文件。
- Incremental updates: 仅新增本归档、迁移日志条目及 `PROJECT_MEMORY.md` 批次记录；未改状态/实验/决策/冲突/待办/代码结构文件（无新验证事实、无新冲突，新增信号均为聊天级）。
- Sensitive-data handling: 证据含主机/凭据/ModelScope token 相关描述，全部未落盘；本批次记忆文件不含密钥/令牌/认证 URL（仅提及 A→B 临时 SSH key 存在）。
- Chat-claim handling: 所有新协议描述/决策/建议均为聊天报告或 user 请求，未独立复核，不视为已验证项目结果。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260919-MAMBA-BATCH-008

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-13281980333d40f88143d62db130dc8a-008.md`（inner source `delta-20260919-102750-003.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:10 起窗口）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260919-114152-AUTO-13281980333d40f88143d62db130dc8a-7`（外部托管，未修改）。
- Archive: `archive/2026-09-19-migration-Mamba-batch-008.md`（新建，未覆盖任何旧归档）。
- New facts（均为聊天报告，chat-reported，未独立复核为已验证事实）: (1) FUSAR-Map UPerNet 受控实验 `fusar_mambayolo_t_upernet_miou_d16_v1` 完成——baseline test mIoU 44.2973%/FWIoU 65.3817%/OA 77.1050%/3.783M/2.366 GFLOPs/183.43 FPS；ours P3-d16 43.5607%/65.0833%/76.7702%/3.718M/2.329/183.09；per-class IoU（building/veg/water/road）baseline `[14.28,71.49,86.63,4.78]`、ours `[15.23,71.07,86.85,1.09]`；val mIoU baseline epoch146 44.5149、ours epoch118 41.8919。(2) 早期 VSS-Seg FUSAR baseline/P3 对（44.293/43.336）。 (3) 公开 FUSAR 对比（FUSAR-Map Net 52.31 mIoU、MASA-SegNet 50.70、DeepLabv3+ 47.62、VGG-SegNet 44.03、U-Net 43.43、SegNet 38.52，文献值）。 (4) 论文重构为通用 SAR-Mamba，三层实验（检测主验证 SSDD/SAR-Ship/SAR-AIRcraft；分类+变化检测跨任务 SAR-ACD/MSTAR-SOC/Ottawa/Farmland/Yellow-River；分割探索边界 FUSAR-Map/OpenEarthMap-SAR），OGSOD 移出主线；新增 `reports/generic_sar_mamba_experiment_design_20260919.md`。(5) 本实验代码修复（fused CUDA selective scan ~400x、non-finite 检查、FUSAR 关 AMP、P3/8 修正、UPerNet-lite 与新 CLI）。
- Overlap: FUSAR-Map 受控 baseline/P3 mIoU 对比已在 `PROJECT_STATE.md` 简短记录（44.2973/43.5607、road 4.78→1.09%）；本批次仅在其上增量补 Params/GFLOPs/FPS、per-class、旧 VSS-Seg 对、公开对照与诊断。论文"通用 SAR-Mamba"定位与 `PROJECT_STATE.md` objective 既有表述一致，本批次补充三层划分与 OGSOD 移出主线。
- Incremental updates: `PROJECT_STATE.md`（FUSAR 节补充效率/每条类/旧对/公开对照/诊断）、`PAPER_STATUS.md`（新增通用 SAR-Mamba 试验设计、OGSOD 移出主线说明）、`PROJECT_MEMORY.md`（本批次记录）、`MIGRATION_LOG.md`（本条目）、归档 `2026-09-19-migration-Mamba-batch-008.md`。
- Sensitive-data handling: 证据不含凭据；未落盘任何密钥/令牌/认证 URL。
- Chat-claim handling: 全部新指标为聊天报告（assistant/用户），未独立复核，除既有已记录事实外均标记为 chat-reported；不视为已验证项目结果。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260920-MAMBA-BATCH-010

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-2f23d9c777834115bb4ef82adc03754f-008.md`（inner source `delta-20260919-102750-002.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:10:33 起窗口，仅 user 消息）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260920-141016-AUTO-2f23d9c777834115bb4ef82adc03754f-7`（外部托管，未修改）。
- Archive: `archive/2026-09-20-migration-Mamba-batch-010.md`（新建；与 batch-009 同一 inner delta 的另一分片，序号 010 规避既有归档，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实（全部为 chat-level 用户请求/规划，未独立复核）。本分片相对 batch-009 的增量信号：SSDD 300 epoch 重跑 AP50:95 仅 72.261%（聊天报告）用户怀疑代码错误并按“历史最高配置+P3”重训；A6000 A 上离线 GPU 空闲监测自调度脚本（30→15 分钟空闲判定）；对话末尾用户要求删除定时任务（是否删除未验证）；MSTAR-SOC/FUSAR-Map 时间估算（100 epoch 约 6–8 天 / 4–7 天）；FUSAR“放大八倍”配置异常终止并按官方论文配置重训、换更强分割基线（SegFormer 头）；数据集取舍（去 OGSOD/SRSDD-HBB/AIR-SARShip/Sulzberger，保留 Ottawa/Yellow-River 等）与既有主数据集表述一致。
- Overlap: 长规划（三贡献/分阶段/公平比较/不强制复现 MaDiNet 等）、多服务器拓扑与本机克隆、数据集矩阵、SSDD 早期 99 与 300 epoch、FUSAR 受控结果、state_dim 32 不如 16 均与既有 `PROJECT_STATE.md`/`PAPER_STATUS.md`/`TODO.md`/`DECISION_LOG.md` 及 batch-006/008/009 归档一致，未重复改写核心文件。
- Incremental updates: 仅新增本归档与迁移日志条目；未改状态/实验/决策/冲突/待办/代码结构文件（无新验证事实、无新冲突，新增信号均为未落地 user 请求）。PROJECT_MEMORY.md 追加本批次迁移记录。
- Sensitive-data handling: 证据含主机账户/SSH-Tailscale 连接与 ModelScope/GitHub 传输描述，不含明文凭据；本批次记忆文件不含密钥/令牌/认证 URL。
- Chat-claim handling: 全部内容为 user 请求/规划或聊天级状态；FUSAR“换更强基线”是否落地、定时任务是否已删除、自调度脚本是否落地均未知，需后续现场核对。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260920-MAMBA-BATCH-013

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-a8a11946b3b14d2b8040ee1716327424-003.md`（inner source `delta-20260919-102750-003.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:02/17:10:33 起窗口）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260920-192016-AUTO-a8a11946b3b14d2b8040ee1716327424-2`（外部托管，未修改）。
- Archive: `archive/2026-09-20-migration-Mamba-batch-013.md`（新建；与 batch-008/011/012 同一内层 delta `delta-20260919-102750-003.md` 的又一个分片，序号 013 规避既有归档，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实（全部为聊天级/未验证信号）。本分片相对 batch-008/011/012 的增量信号：(1) SSDD core sweep 配置 dry-run `ssdd_2xh100_core_v1`（16 runs；baseline + p2/p3/p4/p5/p2p3 × state_dim，含 sgd_ref 与多种 adamw 变体），用户要求改单 H100 串行并保留 20 epoch 预热，对应 `ssdd_1xh100_gpu1_warmup20_v1`（adamw_warmup20）；均 `--weights none`，无结果；(2) SAI-Net 被明确排除出对比（分割方法/期刊层次低）；(3) 因 H100 不可用（仅 3090/1080Ti/A6000）用户要求检索小规模 SAR 检测数据集与抽样快速消融；(4) 聊天报告 H100 工作区备份上传 ModelScope `shimian123/ds-ssm-workspace-backup`（25.2G tar，未独立复核）；(5) 用户要求归档/删除两个旧线程（bookkeeping，非科研事实）。
- Overlap: 论文进展总结、三贡献、OGSOD 匹配结果、SSDD 取舍、初始化结论（无 ImageNet 预训练/随机初始化）、DG-DS-SSM 非主线、H100 备份上传均与既有 `PROJECT_STATE.md`/`PAPER_STATUS.md`/`DECISION_LOG.md`/`TODO.md` 及 batch-006/008/011/012 归档一致，未重复改写核心文件。
- Incremental updates: 仅新增本归档、迁移日志条目及 `PROJECT_MEMORY.md` 批次记录；未改状态/实验/决策/冲突/待办文件（无新验证事实、无新冲突）。
- Sensitive-data handling: 证据含明文 SSH 口令/主机账户/ModelScope token 登录错误描述，全部未落盘；本批次记忆文件不含密钥/令牌/认证 URL（仅保留需定位的仓库 id）。
- Chat-claim handling: SSDD sweep、小规模数据集检索、SAI-Net 排除、ModelScope 备份完成均属聊天级/请求，未独立复核，不视为已验证项目结果。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## MIG-20260920-MAMBA-BATCH-012

- Protocol: Project Memory Migration Agent v4.0.
- Evidence: `auto-memory-sync/batches/Mamba/batch-23924c83b8344bb69b947f2004c3dd6a-004.md`（inner source `delta-20260919-102750-003.md`，collected UTC 2026-09-19T02:27:50Z；thread `019f2406`，2026-09-18 17:10 起窗口）。
- Pre-write snapshot: `auto-memory-sync/snapshots/Mamba/MIG-20260920-173020-AUTO-23924c83b8344bb69b947f2004c3dd6a-3`（外部托管，未修改）。
- Archive: `archive/2026-09-20-migration-Mamba-batch-012.md`（新建；与 batch-008/011 同一内层 delta `delta-20260919-102750-003.md` 的又一个分片，序号 012 规避既有归档，未覆盖任何旧归档）。
- New facts: 无新增已验证项目事实（全部为聊天级建议/运维描述或与 batch-008/011 重叠的重述）。唯一新增信号：用户两次发起记忆迁移指令（附 v3/v4 规则，为本窗口任务来源，非科研事实）；assistant 论述实验设计收缩(检测为主表)+跨任务+分割探索边界的三层方案，与 batch-008 既有重构结论一致。
- Overlap: FUSAR UPerNet 受控实验、公开 FUSAR 对比、代码修复、论文“通用 SAR-Mamba”重构、OGSOD 移出主线、A→B 数据同步、road 诊断建议均与既有 `PROJECT_STATE.md`/`PAPER_STATUS.md`/`TODO.md`/`DECISION_LOG.md` 及 batch-008/011 归档一致，未重复改写核心文件。
- Incremental updates: 仅新增本归档、迁移日志条目及 `PROJECT_MEMORY.md` 批次记录；未改状态/实验/决策/冲突/待办文件（无新验证事实、无新冲突）。
- Sensitive-data handling: 证据仅提及 A→B 临时 SSH key 存在及 ModelScope/GitHub 传输意图，无明文凭据；本批次记忆文件不含密钥/令牌/认证 URL。
- Chat-claim handling: 全部内容为聊天级建议/运维/请求或重复迁移；所有新信号未独立复核，不视为已验证项目结果。
- Scope: 未修改源码、数据、实验输出、论文源文件、配置、.git 或运行状态。

## 2026-09-21 — Mamba fast-pass sync (batch-7484b5607cac)

Source: delta-20260919-102750-003.md (chat-thread handoff summary; treated as chat-reported claims) + delta-20260921-163334-001.md (memory-agent instructions, no project facts).

**FUSAR-Map segmentation experiment `fusar_mambayolo_t_upernet_miou_d16_v1` (completed, chat-reported)**
- New controlled pair on paper-aligned FUSAR-Map protocol (4 classes, WuHan_03_02 test, 17,560/4,390/1,280 patches, 256x256 bs10, Adam 1e-3, x0.93/2ep, 150ep, FP32, scratch, weighted CE+Dice, val-mIoU selection).
- Baseline Mamba-YOLO-T + UPerNet-lite: test mIoU 44.2973%, FWIoU 65.3817%, OA 77.1050%, 3.783M params, 2.366 GFLOPs, 183.43 FPS (best ep146).
- Ours + IG-DS-SSM/PFIM true-P3 d16: test mIoU 43.5607%, FWIoU 65.0833%, OA 76.7702%, 3.718M, 2.329 GFLOPs, 183.09 FPS (best ep118).
- Ours improves building (14.28->15.23) and water (86.63->86.85) IoU but road collapses (4.78->1.09); lighter by ~65K params / 0.037 GFLOPs. Stronger decoder + corrected P3 still does not beat baseline.
- Code fixes synced to A/B: fused CUDA selective-scan detection (~400x faster), non-finite loss/grad checks, AMP disabled for FUSAR (NaN at ~ep25), new CLI flags (--backbone-layout full_t, --decoder upernet_lite, --segmentation-loss weighted_cross_entropy_plus_dice, --selection-metric miou_all); 15 local tests passed.
- Conclusions (chat-reported): main failure is road segmentation (thin low-contrast structures suppressed); P3 placement good for detection does not transfer to segmentation; do not claim P3 improves FUSAR segmentation; do not hide negative results.
- Public FUSAR baselines recorded (per-class [water, road, building, vegetation]): FUSAR-Map Net 52.31 mIoU, MASA-SegNet 50.70, DeepLabv3+ 47.62, VGG-SegNet 44.03, U-Net 43.43, SegNet 38.52.
- Recommended next steps (not executed): validation-only screening — true P2, P2+P3, identity-gated residual around DS-SSM/PFIM, rare-class sampler, road boundary/topology loss; 40-60ep screening before full runs.

**Paper direction decision (2026-09-19, user-approved, applied by prior session)**
- Paper repositioned as "IG-DS-SSM/PFIM: general Mamba state-update method for SAR vision tasks" (not detection-only). Three experiment tiers: detection (SSDD, SAR-ShipDataset, SAR-AIRcraft1.0) as main validation; classification (SAR-ACD, MSTAR-SOC) + change detection (Ottawa, Farmland, Yellow River) as cross-task validation; segmentation (FUSAR-Map, OpenEarthMap-SAR) as exploratory/boundary only, no universal-improvement claims.
- OGSOD removed from main line (results weak, protocol insufficient); SADD, Yellow River negative results kept in records, not main tables.
- Detection-side claims (chat-reported): SSDD P3-d16 BS16 AP50:95 ~+7.75pp over baseline; SAR-ShipDataset P3 ~57.82% vs 49.57% baseline (needs fixed-protocol rerun). Main model locked to P3, state_dim=16.
- Prior session reports updating PAPER_STATUS.md, PROJECT_STATE.md, TODO.md, DECISION_LOG.md and creating reports/generic_sar_mamba_experiment_design_20260919.md.

**Other durable facts**
- Servers: A6000-A `powerleader-ts`, A6000-B `powerleader-a6000-b`; server project `/home/powerleader/ig-ds-ssm/work_train_single`, env `ig-ds-ssm-env`. A pulled FUSAR data from B over LAN (~111 MB/s) after temporary A->B SSH key (credential not stored).
- BS64 runs across OpenEarthMap-SAR / Ottawa / Yellow River / Farmland / SAR-ACD were being distributed across A6000 servers incl. dual-GPU "F" server; several runs still queued/in-progress at evidence time (unresolved status).
- User standing preferences: run experiments not proposals; always report Params/GFLOPs (+latency/FPS); fair baseline treatment; no AMP for FUSAR protocol; don't claim val results as test rankings; don't interrupt valid running experiments.

**Unresolved / watch**
- Segmentation (FUSAR-Map, OpenEarthMap-SAR) and Yellow River change detection remain below baseline — road collapse root cause unfixed.
- BS64 experiment queue status and final results not captured in this evidence.


## 2026-10-07 RGSP 方法与架构图更新

- FactKey: Mamba.Paper.MethodFigure20261007; EvidenceStatus: CONFIRMED; LifecycleStatus: CURRENT（修改产物，非实验结论）。用户在 Codex 线程 01a11234-f281-7012-873b-4e2b32bb00e1 要求将已生成架构图导出 PDF、插入论文并修改介绍。
- 修改对象：paper_draft/rgsp_cn_revision_20261006/main.tex 与 figures/fig2_method.pdf。方法依次为总体网络与 RGSP 块、LRC、LSGM、结构条件化选择性状态更新；网络集成前移，删除独立复杂度分析。图为生成 PNG 的 PDF 封装，非矢量重绘；检测 P3/8 示例不证明其他任务采用同一接入位置。
- 核验：图 PDF 单页、含一个图像，Poppler 回渲染并视觉检查；源码标签/引用及环境嵌套静态检查通过。内置编译器环境错误，修改后全文编译未核验，无新论文 PDF。
- 原稿、旧图、README 与旧 PDF 备份于该目录 archive/20261007_method_figure_before/；源码包 paper_draft/rgsp_method_figure_update_20261007_overleaf.zip。不改业务代码或实验结果。


## 2026-10-07 全量论文比较表候选稿

- RunId: rgsp-all-comparison-tables-20261007-01a11234
- 用户要求将外部基线与本项目各配置并列供选择；已生成 [候选表](../../output/paper_tables_20261007/全部任务比较表.md)、独立LaTeX审阅稿、论文片段及ZIP，覆盖16个数据集、257条候选行（不等于独立实验次数）。
- 保留来源与协议分组；历史值未升级为新核验结论。源文档、主论文与业务代码未修改；未运行实验。
- 静态结构检查通过；内置LaTeX编译器环境错误，PDF未生成。证据与哈希见中央独立增量 rgsp-all-comparison-tables-20261007-01a11234。
