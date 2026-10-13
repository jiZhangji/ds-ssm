# Mamba-YOLO / IG-DS-SSM Project Memory

## Identity

- **project-id**: `mamba-yolo-ds-ssm--ig-ds-ssm-pfim`
- **正式名称**: Mamba-YOLO / IG-DS-SSM / PFIM SAR research workspace
- **别名**: DS-SSM、IG-DS-SSM、PFIM、Mamba-YOLO SAR detection
- **项目根目录**: `D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`
- **正式记忆根目录**: `docs/project-memory/`
- **当前分支观察**: `codex/ssdd-h100-sweep`
- **识别置信度**: 高

## 识别依据

1. 当前工作区是 Git 仓库，包含 `source_code/`、`experiments/`、`weights/`、`reports/`、`database/` 和 `paper_draft/`。
2. `README_CURRENT.md` 和 `HANDOFF_FOR_NEXT_MODEL.md` 明确将项目描述为 Mamba-YOLO、DS-SSM、PFIM 的 SAR 目标检测研究。
3. 当前代码包含 `PFIM`、`DeSpeckleSSM`、`DS_VSSBlock`，报告包含 SSDD、OGSOD、SAR-AIRcraft 和跨任务 SAR 实验。
4. 未发现相反证据或其他同等匹配的项目候选。

## 当前摘要

项目主线是轻量 SAR 目标检测：在 Mamba-YOLO-T 中加入 SAR-aware DS-SSM 和 PFIM。DS-SSM 对局部残差进行去斑/抑制并进行能量感知状态调节，PFIM 根据 token 的局部趋势和二维变化估计重要性，调制状态保留、写入和步长。当前最干净的受控证据是 OGSOD-1.0 的单次匹配实验；SSDD 最佳结果和其他数据集结果中仍有历史协议、初始化或划分差异，不能混合成统一排行榜。

跨任务分割、分类和变化检测实验已产生大量诊断结果，但目前并不支持“在所有 SAR 任务上统一超过外部方法”的结论。正式论文应以直接 baseline 的同协议比较、核心消融和复杂度/速度 Pareto 证据为主。

## 推荐读取顺序

1. `PROJECT_MEMORY.md`
2. `../../AGENTS.md`
3. `PROJECT_STATE.md`
4. `TODO.md`
5. `MEMORY_RULES.md`
6. `DECISION_LOG.md`
7. `TIMELINE.md`
8. `EXPERIMENT_LOG.md`
9. `PAPER_STATUS.md`
10. `CODE_STRUCTURE.md`
11. `CONFLICTS.md`
12. 与当前任务相关的 `knowledge/` 和 `archive/`

## 文件导航

| 文件 | 用途 |
|---|---|
| `PROJECT_STATE.md` | 当前有效研究、代码、实验和阻塞状态 |
| `TODO.md` | 待办、依赖和完成标准 |
| `DECISION_LOG.md` | 方法、协议和论文取舍 |
| `TIMELINE.md` | 项目演化和阶段顺序 |
| `EXPERIMENT_LOG.md` | 实验条件、结果、失败和证据路径 |
| `PAPER_STATUS.md` | 论文主线、贡献、证据边界和审稿风险 |
| `CODE_STRUCTURE.md` | 代码入口、模块和工具 |
| `CONFLICTS.md` | 未决或已澄清的版本/协议冲突 |
| `MIGRATION_LOG.md` | v4 迁移批次、覆盖范围和验收记录 |
| `MEMORY_RULES.md` | 未来 AI 的项目记忆工作规则 |
| `knowledge/` | 方法定义、协议和跨任务深层说明 |
| `archive/` | 本窗口及后续窗口的详细脱敏迁移记录 |
| `snapshots/` | 记忆文件变更前的脱敏快照 |

## 来源覆盖

- **来源形态**：当前窗口可见消息、当前窗口压缩交接摘要、当前仓库报告/代码/结果文件的混合来源。
- **覆盖结论**：已迁移当前窗口中可访问且与项目长期工作有关的主要背景、技术路线、决策、实验和论文信息。
- **缺口**：平台没有提供本窗口全部原始消息的独立分页接口；较早对话的细节依赖当前上下文摘要和仓库文件，不能声称恢复了不可见的完整聊天。外部服务器历史也只按仓库已有报告和当前窗口曾经可见的输出记录。
- **历史迁移**：v3.0 迁移曾开始创建根目录 `archive/`，但本次预检查未发现 v3 生成的知识文件或归档文件；该空目录保留，正式 v4 知识库使用本目录。

## 当前未决问题

入口中的冲突与不确定项共 9 条，见 `CONFLICTS.md`；其中外部协议差异和跨任务适用边界仍需持续核对。尤其注意：不同数据集使用了不同任务指标、划分、初始化和训练协议；旧报告中的 queued/running 状态可能早于后续已完成结果。不得仅凭旧状态启动训练。

## 最近更新

- 迁移规则版本：v4.0
- 迁移日期：2026-09-07（Asia/Shanghai）
- 最近有效实验事实：当前窗口可见的 2026-09-05 跨任务批次结果；现场 GPU 检查记录为 2026-09-06，属于时间性观察。
- 本次迁移未修改业务源码、论文、实验数据或原始结果。

## Coverage audit 2026-09-09

- All listed source windows are indexed in `2026-09-09-coverage-audit-Mamba.md`.
- High-signal provisional excerpts are in `EVIDENCE_SUMMARY.md`.
- This audit did not overwrite previous memory or archive revisions.

## Migration 2026-09-11 (batch-003)

- 增量迁移线程 `019f2406` 的 2026-09-11 窗口：论文写作思路/动机、实验范围收缩建议、P3 vs P2/4 措辞冲突（CONFLICT-013）、跨数据集异常修复与 ModelScope 上传的用户请求（均未验证完成）。
- 归档：`archive/2026-09-11-migration-Mamba-batch-003.md`；证据中的服务器明文密码已脱敏，未写入记忆。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-18 (batch 007)

- 增量迁移线程 `019f2406` 的 2026-09-18 窗口（仅 user 消息）：用户请求整理 A6000 服务器上的数据与代码并上传到 ModelScope，并两次发送“继续任务”；证据中无 assistant 响应或完成结果，仍属未验证的用户请求。
- 未新增已验证项目事实；该请求是对 `TODO.md` `TASK-DATASET-HOSTING` 的 2026-09-18 重申与对象扩展（题材不明）。无新冲突或决策。
- 归档：`archive/2026-09-18-migration-Mamba-batch-007.md`；本次证据不含凭据。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-19 (batch-008)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:10 起窗口（chat-reported）：FUSAR-Map UPerNet 受控实验 `fusar_mambayolo_t_upernet_miou_d16_v1` 完成（baseline mIoU 44.2973 vs ours P3-d16 43.5607，road IoU 4.78→1.09%，Params/GFLOPs/FPS/val mIoU 详见 `PROJECT_STATE.md`），早期 VSS-Seg 对（44.293/43.336）与公开 FUSAR 文献对比。
- 论文重构为通用 SAR-Mamba 状态更新方法：三层实验（检测主验证、分类+变化检测跨任务、分割探索边界），OGSOD 移出主线；新增 `reports/generic_sar_mamba_experiment_design_20260919.md`。
- 归档：`archive/2026-09-19-migration-Mamba-batch-008.md`；证据不含凭据，chat-reported 指标未独立复核。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-009)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:10:33 起窗口（仅 user 消息）：IG-DS-SSM/PFIM 系统验证长规划提示词与后续状态/硬件/调度/实验请求，均聊天级、未完成、未升级为已验证事实。
- 无新增已验证项目事实；唯一新增信号为用户请求——FUSAR-Map 更换更强分割基线（提及 SegFormer 分割头）、逐个方法统计 Params/FLOPs，以及把各数据集结果+方法指标整理进 Obsidian 工作台 `C:\Users\13207\Desktop\反代\.obsidian`。
- 长规划（三贡献/分阶段/公平比较/不强制复现 MaDiNet 等/复杂度测量）与 FUSAR 受控实验结果均与既有 `PROJECT_STATE.md`、`PAPER_STATUS.md`、`TODO.md`、`DECISION_LOG.md` 及 batch-006/007/008 归档重叠，未重复改写核心文件。
- 归档：`archive/2026-09-20-migration-Mamba-batch-009.md`；证据不含凭据（主机账户/口令/ModelScope token 描述未落盘）。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-010)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:10:33 起窗口（仅 user 消息），与 batch-009 为同一 inner delta（`delta-20260919-102750-002.md`）的另一分片，内容大量重叠，无新增已验证项目事实。
- 相对 batch-009 的增量信号（均为聊天级 user 请求/状态，未验证落地）：SSDD 300 epoch 重跑 AP50:95 仅 72.261%（用户怀疑代码错误，要求按“历史最高配置+P3”重训）；A6000 A 上离线 GPU 空闲监测自调度脚本（30→15 分钟空闲判定）；对话末尾要求删除定时任务；MSTAR-SOC/FUSAR-Map 时间估算；FUSAR“放大八倍”配置异常终止并按官方论文配置重训、换更强分割基线（SegFormer 头）；数据集取舍（去 OGSOD/SRSDD-HBB/AIR-SARShip/Sulzberger，保留 Ottawa/Yellow-River 等）与既有 `DECISION_LOG.md`/`PAPER_STATUS.md` 主数据集表述一致。
- 归档：`archive/2026-09-20-migration-Mamba-batch-010.md`；证据不含明文凭据。未改状态/实验/决策/冲突/待办核心文件（无新验证事实、无新冲突）。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-011)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:10:33 起窗口；**inner source 与 batch-008 相同（`delta-20260919-102750-003.md`），判定为同一 delta 的重复拆分**，FUSAR-Map UPerNet 受控实验、公开 FUSAR 对比、论文通用 SAR-Mamba 重构与代码修复均已由 batch-008 归档并写入核心文件，本批次不重复改写。
- 相对 batch-008 仅新增 chat 级增量信号：A→B FUSAR 数据经临时 SSH key 走 LAN 同步（约 111 MB/s）；recommended next diagnostic——identity-scaled residual 围绕 DS-SSM/PFIM 以避免 P3 擦除细长道路、P2/P2+P3/road loss/rare-class 采样筛查；用户偏好与"下一步 road 诊断"重述；本窗口末尾用户两次要求执行记忆迁移（附 v3/v4 规则，为本次任务发起指令，非科研事实）。
- 归档：`archive/2026-09-20-migration-Mamba-batch-011.md`；仅提及 A→B 临时 SSH key 的存在，未保存任何密钥/口令/认证 URL。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-012)

- 增量迁移线程 `019f2406` 的 2026-09-18 起窗口；**inner source 仍为 `delta-20260919-102750-003.md`，与 batch-008/011 相同 delta 的又一个分片**，FUSAR UPerNet 受控实验、代码修复、论文“通用 SAR-Mamba”重构与 A→B 数据同步均已由 batch-008/011 归档并写入核心文件，本批次不重复改写。
- 仅新增 chat 级增量信号（未独立复核）：用户两次发起记忆迁移指令（v3/v4，为本窗口任务来源，非科研事实）；assistant 论述实验设计收缩（检测为主表 SSDD/SAR-Ship/SAR-AIRcraft，剥离 OGSOD/SADD）+ 跨任务验证 + 分割探索边界的三层方案，与 batch-008 既有重构结论及 `PAPER_STATUS.md` 一致。
- 归档：`archive/2026-09-20-migration-Mamba-batch-012.md`；仅提及 A→B 临时 SSH key 的存在，未保存任何密钥/口令/认证 URL。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-013)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:02/17:10:33 起窗口；**inner source 仍为 `delta-20260919-102750-003.md`，与 batch-008/011/012 同一 delta 的分片**，无新增已验证项目事实。
- 相对既有 batch 的增量信号均为聊天级/未验证：SSDD core sweep 配置 dry-run（`ssdd_2xh100_core_v1` → 单 H100 串行 `ssdd_1xh100_gpu1_warmup20_v1`，16 runs、`--weights none`、保留 20 epoch 预热，无结果）；SAI-Net 明确排除出对比；H100 不可用时检索小规模 SAR 检测数据集与抽样快速消融的用户请求；聊天报告 H100 工作区备份上传 ModelScope `shimian123/ds-ssm-workspace-backup`（25.2G，未独立复核）；用户要求归档/删除两个旧线程（bookkeeping）。
- 论文进展、三贡献、OGSOD 匹配结果、初始化结论与 H100 备份上传均与既有记忆重叠，未重复改写核心文件。
- 归档：`archive/2026-09-20-migration-Mamba-batch-013.md`；证据含明文 SSH 口令/令牌描述的凭据全部未落盘。
- 本次未修改业务源码、论文、数据、配置或运行状态。

## Migration 2026-09-20 (batch-014)

- 增量迁移线程 `019f2406` 的 2026-09-18 17:10:33 起窗口；**inner source 仍为 `delta-20260919-102750-003.md`，与 batch-008/011/012/013 同一 delta 的又一个分片（-004）**，FUSAR UPerNet 受控实验、公开 FUSAR 对比、代码修复、论文“通用 SAR-Mamba”重构、OGSOD 移出主线与 A→B 数据同步均已由 batch-008/011/012/013 归档并写入核心文件，本批次不重复改写。
- 仅新增 chat 级增量信号（未独立复核）：论文对齐 FUSAR-Map 协议细节（四类 building/veg-other/water/road、`WuHan_03_02` 测试场景、17560/4390/1280 划分、256×256 batch10、Adam LR 1e-3、LR×0.93 每 2 epoch、150 epoch、FP32 因 AMP 约 epoch25 NaN、scratch、Weighted CE+Dice、按 val mIoU 选并保存 best_miou/best_fwiou）；用户决定 OpenEarthMap-SAR 采用 SAR-only、P 标签协议并双 A6000 训练（是否落地未知）；论文定位经“仅检测”讨论后采纳“通用 SAR-Mamba 状态更新方法”。
- 归档：`archive/2026-09-20-migration-Mamba-batch-014.md`；证据含凭据描述全部未落盘，仅提及 A→B 临时 SSH key 存在。
- 本次未修改业务源码、论文、数据、配置或运行状态。
