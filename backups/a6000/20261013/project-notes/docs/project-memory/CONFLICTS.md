# Conflicts and Uncertainties

## CONFLICT-001：旧跨任务报告的 queued 状态与后续结果不一致

- **来源 A**：`reports/current_cross_task_results_20260904.md` 记录部分 BS32/64 变化检测、SAR-ACD 和 OpenEarthMap 队列仍 queued。
- **来源 B**：当前结果笔记及后续结果摘要记录这些批次在 2026-09-05 已生成 `result.json`，并给出最终指标。
- **当前判断**：这是时间演化，不是同一时刻的矛盾；后续具体结果优先于旧队列快照，但运行状态必须以对应结果文件和最新现场检查为准。
- **影响**：不能依据旧报告再次启动已完成任务。
- **状态**：已澄清，保留旧记录。

## CONFLICT-002：FUSAR-Map 旧五类/不兼容结果与新四类 paper-aligned 结果

- **来源 A**：旧诊断 mIoU 约 21%，协议/类别和文献不兼容。
- **来源 B**：新四类 WuHan_03_02 协议 baseline 44.2973、P3 43.5607 mIoU。
- **当前判断**：两组不能排序或计算增益；新四类结果是当前可用分割对照。
- **状态**：已澄清。

## CONFLICT-003：OpenEarthMap “BS64”与物理 batch

- **来源 A**：启动记录称 physical BS32 + grad accumulation2 为 effective BS64。
- **来源 B**：部分 `result.json` 只记录 physical BS32，不能自动推断 effective BS64。
- **当前判断**：只有明确带 accum2 的结果可标记有效 global BS64；物理 BS32 结果必须单独标注。
- **状态**：已澄清。

## CONFLICT-004：SAR-ACD 旧分类头结果与校正头结果

- **来源 A**：旧版实验含 Accuracy 约 96-99% 等记录，部分预热实验因 P3 分支未连接分类输出而失败/退出。
- **来源 B**：校正后的分类头读取 `features[1]`，新 BS64 批次 baseline 89.764、P3 warmup50 96.850。
- **当前判断**：两版不能混合排序；校正批次优先用于后续分析，但迁移未重新验证代码。
- **状态**：已澄清。

## CONFLICT-005：历史 SSDD 初始化标签

- **来源 A**：当前 H100 sweep 命令明确 `--weights none`，属于 scratch。
- **来源 B**：历史 PFIM ex27 执行过 `model.load(yolov8n.pt)`，但只匹配约 6/519 项。
- **当前判断**：H100 sweep 可标 scratch；ex27 只能标 partial-transfer/大部分随机，不能重标为 clean scratch。
- **状态**：已澄清。

## CONFLICT-006：外部论文数值与本地复现值不同

- **来源 A**：公开论文使用不同 split、输入、预训练、epoch、评估器或指标口径。
- **来源 B**：本地统一协议/重实现结果与论文报告不同。
- **当前判断**：外部值进入作者报告表，本地值进入统一协议表；不可直接声称重新排名 SOTA。
- **状态**：长期未决，按方法/数据集逐项核对。

## CONFLICT-007：FUSAR-Map OA 的“超过外部最好”说法需纠正

- **来源 A**：曾以 P3 OA `76.770%` 高于 FUSAR-Map Net OA `75.80%` 作次级比较。
- **来源 B**：外部表中 MASA-SegNet OA 记录为 `79.67%`。
- **当前判断**：P3 也没有超过外部最佳 OA；主 mIoU 更低。此前说法只能解释为相对单一 FUSAR-Map Net 的比较，不能称外部最好。
- **状态**：已纠正，需避免旧文本误导。

## CONFLICT-008：P3 最优阶段因任务而异

- **来源 A**：SSDD/OGSOD 检测实验支持浅层 P2/P3 的选择讨论，P3 是当前论文版本。
- **来源 B**：分割中 P3 导致道路 IoU 明显下降，P3 并未超过 stronger baseline。
- **当前判断**：不能把 detection 的 P3 优势推广为所有任务的固定部署规则。
- **状态**：已澄清。

## CONFLICT-009：跨任务泛化主张

- **来源 A**：用户曾希望验证 IG-DS-SSM 作为通用 SAR 状态空间模块。
- **来源 B**：OpenEarthMap/FUSAR/Yellow River 结果中 P3 低于 baseline，Farmland/Ottawa/SAR-ACD 部分结果为正向。
- **当前判断**：当前证据支持“跨任务诊断/潜在泛化”，不支持“普遍提升”。论文主张应收缩到轻量 SAR 检测并透明报告跨任务边界。
- **状态**：有效限制。

## CONFLICT-010：SSDD ex27 的 0.74022 是验证集而非官方 TEST

- **来源 A**：历史排名与多处报告记录 ex27 AP50:95 `0.74022`。
- **来源 B**：2026-08-28 固定 232 图官方 TEST 复核 ex27 为 `0.714454`，历史 baseline 为 `0.703479`。
- **当前判断**：`0.74022` 是历史验证集选模口径；对外/论文引用时必须标注 split。ex27 相对历史 baseline 的官方 TEST 增益约 `+1.098 pp` AP50:95。
- **状态**：已澄清。

## CONFLICT-011：SAR-ACD 旧迷你 runner 99.2% 与后续批次口径

- **来源 A**：v1 迷你 VSS runner 记录 baseline Accuracy `99.21875%`、ours `98.4375%`。
- **来源 B**：校正批次（VMamba-Tiny 官方初始化、BS64、200 epoch）baseline `89.764`、P3 warmup50 `96.850`。
- **当前判断**：两套 backbone、初始化、划分均不同，不能排序；v1 值已标记作废/仅历史诊断。另外 SAR-ACD 真实清单为 2522 图 5 类，旧文档“六类 3032 图”有误。
- **状态**：已澄清。

## CONFLICT-012：FUSAR-Map 五类 city-disjoint 与发表四类协议

- **来源 A**：当前本地五类（含黑色未知类）、city-disjoint 408/125/77、512 crop、scratch。
- **来源 B**：发表协议 530/80、四类、256 patch、七次旋转、光学预训练编码器；官方代码黑色类处理不一致（U-Net 并入 vegetation，一条 DeepLab 路径并入 road）。
- **当前判断**：本地 mIoU 不能与发表四类 43-52% 直接比较；已新增 paper-code 对齐轨道（黑色并入 vegetation）。黑色策略必须在报告前冻结。
- **状态**：部分澄清；与 CONFLICT-002 并存，对齐轨道结果待出。

## CONFLICT-013：论文中最终版本写作 P3 还是 P2/4 不统一

- **来源 A**：项目记忆与近期实验把最终版本称为 P3（浅层 P3 stage 替换）。
- **来源 B**：2026-09-11 窗口指出当前英文论文中仍有 “only the shallow P2/4 stage is replaced” 的表述。
- **当前判断**：两种说法不能并存；正式修改论文前必须统一为实际采用的最终结构（当前主线为 P3）。
- **状态**：未决，待论文修改时统一。

## CONFLICT-014：SSDD 冻结协议、版本归属与 epoch 轨道边界

- **来源 A**：SSDD 公平协议记录（2026-10-02 复核）显示当前已跑完的对照为 150 epoch、scratch、batch 16、640 输入、742/186/232 划分；每个 seed 内五个变体共用 split 与 GPU，`ssdd_full_train=false`，训练用 val 选模、最终评估用官方 test。文件名层面未发现 train/val 与官方 test 重叠。
- **来源 B**：历史内部方法 ex27 使用旧训练口径与部分初始化，其数字属于历史内部结果，不能当作干净 baseline；其 0.74022 亦为历史验证集口径（见 CONFLICT-010）。
- **当前判断**：后续同协议主流模型对比应沿用上述冻结设置（150 epoch、scratch、batch 16、640 输入、固定 742/186/232 划分、val 选模 + 官方 test 评估）。若推进 9 月收尾建议提到的 300 epoch 轨道，必须整组模型统一用 300 epoch，并与 150 epoch 结果分表呈现，不能单独取 ex27 的 300 epoch 数字作对照。协议冻结与版本归属边界以此为准。
- **状态**：已澄清并冻结；300 epoch 轨道尚未启动（unverified，需整组同口径后再比较）。
- **来源标注**：批次 mamba-merge-01，2026-10-02。

## Uncertainty Notes（待核实项）

- sarmix 四源代理实验的最终汇总与结论未在压缩源中给出，本地 `output/sarmix_scratch_d16_v1*` 摘要需复核。
- SSDD warm-start 配对诊断（ex27→P3-d16）是否实际启动及其结果待核实（启动时 `ALREADY_RUNNING` 可能是 pgrep 误报；最后观察约 epoch 73/150、best VAL AP50:95 53.821%）。
- SSDD/SADD seed1/2 BS32 批次的 `eval_test.json` 指标尚未汇总。
- A6000-C/D 环境报 PyTorch 1.13.1+cu117，与 `sar-pretrain`（torch 2.1.2）及 1080Ti（2.6.0+cu118）不同；跨机结果需标注环境。
- ModelScope 服务器端 SDK 匿名下载曾失败（`username cannot be empty or null`），需 Windows 中转。
- SSDD 300 epoch 轨道尚未执行；若启动需保证整组模型同一 epoch 口径，并与 150 epoch 结果分表（见 CONFLICT-014）。

- 当前窗口平台没有提供全部原始聊天的可验证分页；本迁移使用可见对话、压缩交接摘要和项目文件混合证据。
- 若历史助手声称某服务器或任务完成但没有结果文件/工具输出，不能仅凭声称升级为已验证完成。
- 远程服务器状态和 GPU 空闲状态均是时间性观察，新的运行前必须现场检查。


## CONFLICT-015：旧PFIM-only冻结语义与实际权重不一致（2026-10-08）

- [CONFIRMED] 旧trainer重新启用非显式freeze参数；3个PFIM-only best.pt的lambda/alpha均非零且权重SHA匹配旧结果表。具体数值及证据见 `reports/RGSP_TRAINING_EXECUTION_20261008.md`。
- 状态：实现事实已解决；历史“严格独立PFIM收益”解释不成立，数值保留。新控制使用buffer，旧模块不静默改写。

## CONFLICT-016：文件名互斥并不等于图像内容互斥（2026-10-08）

- [CONFIRMED] 当前副本22个重复内容组，其中10组跨官方train/test；旧CONFLICT-014只核对文件名，不能排除这种内容重叠。
- 新实验：保留官方test232，从本轮开发集合排除12重叠训练副本，余下按内容组划分730/186；正式campaign `rgsp_controls_20261008_v3`。不修改原始文件，不混算旧742/186/232结果。
- 状态：字节级重叠已识别并在新轨道隔离；近重复/场景相关性与标注差异的完整审计仍OPEN。不得声称所有SSDD分发版本均如此。

## CONFLICT-017：SAR-AIRcraft原test重复内容及语义标注冲突（2026-10-09，OPEN）

- FactKey: Mamba.Data.AirTestConflict20261009；EvidenceStatus: CONFIRMED（正式状态与既有审计文档的索引一致性）；ConflictStatus: OPEN；RecordedAt: 2026-10-11 +08:00；RunId：20261011-memory-resume-01a128b2-a1。
- 仅适用于2026-10-09所审源副本：完整test内部范围为873个byte重复组、1892张test成员图、797组语义标注冲突；859/1864/792仅是成员全在test的子范围，不能混用或推广到全部SAR-AIRcraft分发版。
- 直接来源：[原审计报告](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/external_data/AIR_ARCHIVE_DUPLICATE_PROTOCOL_20261009.md)与[完整test范围补充](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/external_data/AIR_FULL_TEST_SCOPE_ADDENDUM_20261009.md)。本次核读报告并复核SHA，未重读大数据快照、图像、标签或重跑审计；文档内的源快照签名不是本次独立数据复验。
- 三个AIR seed0位置A2/B1继续BLOCKED，等待用户确认权威修正版或另行协议；不裁决标签、不移除位置、不把57缩成54、不放行seed1。跨split隔离不能修复test内部标注矛盾，也不能证明近重复/场景独立。
