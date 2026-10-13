# Mamba — Active TODO

## TASK-SARACD-EXPANSION-20261013 · P0 · DOING

- 用户确认：3032/2992两版都跑、释放B旧空等队列、BS16×4有效64、官方明确结构、优先我们的三组方法；排队后不持续监测，等用户通知完成再统一回收补表。
- 已落实：官方六类克隆和B/A完整数据绑定；两版共用240val/1792test，full1000train含冲突40，clean960train；全部原文件保留。B核心6probe通过并正式训练，首epoch5.895秒；B重排只改stage并保存旧plan/state/launch，未重跑已完成probe。
- 队列：B Original/Matched075/Full两版全部seed0后依次seed1/2，再ResNet50；A原SAMPLE结束后六通用模型两版。全20配置最多60run，实际3/2/1上限依耗时下调；不能将已排队写成已完成。B后续SSDD FasterRCNN/RetinaNet/FCOS R50-FPN三个adapter已CPU fixture/数据检查，等GPU预检通过才训练，失败不自动替代模型或重试。
- 验收待用户通知：生产者result/completion/protocol/FP32 checkpoint与连续账本，按版本/方法/共同seed补全部指标和独立论文组；保留负结果、未覆盖FLOPs和缺作者代码项。当前未回填新性能，不关闭110候选或旧57门槛；AIR三个位置未删除。
- 来源：experiments/rgsp_table_expansion_20261013_v1/README.md；evidence/HANDOFF_VERIFIED.json核104文件及C clone记录；legacy_air_release、reprioritize_ours_01、qa及queue回执。核心最多3seed估4.5–6h，全分类20配置估14–18h，不含后置检测。RunId：rgsp-acd-expansion-20261013-1053。

## TASK-UNIFIED-A6000-20261012 · P0 · DOING

- 当前授权：用户2026-10-13要求分析聊天01a1301f-cdc1-7092-ad5c-9f6a5625c9d4与01a12fc7-e68c-7a80-a48c-d30deaa53719，补全前者中断的重分配任务。统一配方按现行150轮，原30配置+6个SAMPLE通用补充基线；不代表110候选全复现。旧300轮和旧矩阵产物独立保留。
- 调度交接已核：A/a0samplex六项预检及正式队列通过，AlexNet seed0第83轮早停已验收；C/c0shipfull接管Ship Full且首轮完整ledger已出；F双卡Original/Matched075继续，F1在当前子进程完成后退出，其Full的历史PENDING位置由C承接，不重复启动。D SADD继续；该快照时B未接管；10/13已按新增授权释放，见TASK-SARACD-EXPANSION-20261013。当前入口 experiments/rgsp_rebalance_20261013_v1/README.md 与 Read-LiveStatus.ps1。
- 完成证据：68次完整运行/27配置，A28/C12/D19/F9；704小文件SHA与服务器checkpoint/result/completion/protocol通过，跨seed协议及Ship转移/SAMPLE共同协议一致。RESUME_1014_VERIFICATION为A后续+前次C/D/F分时快照，不是全任务完成；原30配置中22组计划seed已齐。旧67/25次快照保留历史。
- 后续：回收剩余SADD、三个Ship以及A六基线的全部计划seed；按实际耗时上限3/2/1顺序执行，完整同协议配对后补论文。AlexNet首项Accuracy9.4620%/Macro-F1 3.5052%、57.044810M/1.420303744G covered FLOPs，见SUPPLEMENTAL_RESULTS.md；保留跨域负结果，不用test调参。六新增通用基线不能替代原SAMPLE专用方法；其余缺源码/配置候选仍须就绪核验。
- 阻塞继续：SAR-ACD等待修正版的决定已被10/13两版都训练授权替代；MDDNet/IHMEIDA/SARCDNet和其他专用候选准确代码/协议缺口；AIR三位置与旧57seed0门槛保留。训练未全完成，不关闭本项。旧SSDD四臂300轮已独立整理，不混入新test成绩。
- 本轮仅补只读诊断与证据核验脚本、文档/记忆；没有修改训练源码/标签、安装、服务变更或本机模型执行。来源 experiments/rgsp_rebalance_20261013_v1/evidence/。RunId：rgsp-rebalance-resume-20261013-1003。

来源为既有未完成事项；列在这里不构成自动执行训练、提交、上传或服务操作的授权。历史状态未重新核实的，在恢复执行前确认。

## TASK-TABLE-COMPLETION-20261012 · P0 · DOING

- 授权与范围（2026-10-12）：优先快速实验补充当前表格；独立任务 `experiments/rgsp_table_completion_20261012_v1/`。只使用已登记服务器现有环境、独立任务目录，保留旧实验、负结果、旧57项seed0门槛及AIR三个阻塞位置，不放行旧seed1。
- 已核验回填：六张原主表共补24个原空格（12 Params+12 covered FLOPs），290→266（152性能/114复杂度）；Ship新增独立三臂seed0表21值（15性能+3 Params+3 covered FLOPs）全齐，新增45值不等于填45原空格。FLOPs为已覆盖算子估算；主稿SHA与静态验收见本轮实验日志。
- 正式首批分时快照：D于10:49核YOLOv8n 145/300epoch、4205updates；C/F于10:51核Original/Matched075/Full各3/300epoch、87updates，四项均RUNNING_VERIFIED_NOT_COMPLETE。FIRST_BATCH_STATUS.json是已保存启动/进展证据，不代表现在live；CANDIDATE_PRIORITY为4已启动/216未启动候选，不称220项已开跑。
- 新增完成核验（11:20后续）：D YOLOv8n已300轮/8700更新，最佳第168轮；P/R/AP50/AP75/AP50:95=95.0005/89.0110/95.4469/80.9772/68.1836。300行连续账本/CSV、FP32重载、最终最佳权重现场SHA均通过；见独立任务training/YOLOV8N_COMPLETION_VERIFIED.json。该结果仅属928/232 TEST-as-VAL新协议，已生成独立TeX片段，尚未并入封存主稿。其余三臂仍待完成；不关闭任务。
- 下一步：先收齐四臂300epoch/8700updates、最佳FP32重载与统一评估，再回填独立TEST-as-VAL组；SAMPLE806合成→539实测的10000update固定末权重协议正在适配准备、尚未正式训练。SADD全量/SAR-ACD六类/Ottawa无监督按就绪证据推进；旧协议结果不替代新任务。
- 验收：来源/模型/数据/代码/权重哈希、明确划分与选模口径；短测和单epoch smoke均非论文成绩，不由单seed猜mean±std。详见 EXPERIMENT_LOG 的 EXP-RGSP-TABLE-COMPLETION-20261012；RunId：`rgsp-table-completion-20261012-01a12dd6-7cb894b2`。
## TASK-GENERIC-MAMBA-MATRIX · P0 · DOING

- 当前授权（2026-10-09）：继续“astra-消融实验修改2”未完任务；全部配置先seed0，57项全部验收后才seed1，不新增seed2。独立 `rgsp_seedwise_20261009_v1` 冻结114位置/360源码；旧已完成seed1/2和负结果保留，不将旧并行计划当现行调度。
- 历史验收（2026-10-09）：21:31–21:32完整receipt分时回收32/57（A20/22、B7/27、C1/2、F0=3/4、F1=1/2），25 missing、0 invalid，不是同刻全机扫描。A恢复operator已完成15/15计划cross seed0，累计15项completion六内部产物、protocol、epoch ledger及best/test/result均递归复核`deep_verified`；A两项AIR仍阻塞。
- 较晚保存快照（2026-10-10 11:17–11:30）：45/57外层回执verified（A20/B20/C1/F0 3/F1 1），12 missing、0 invalid；五份all_requested_complete/all_global_seed0_verified_locally均false。证据同STATE的sequential_dispatch_20261010_1115；本轮仅核读保留JSON与SHA，未做新的递归深验/现时服务器检查，不放行seed1或关闭本项。记忆RunId：20261011-memory-resume-01a128b2-a1。
- Ship历史时点（2026-10-09）：C/F原始ZIP、39729图实审、独立副本、选定同一manifest与20:16原子ready发布均已核验；31735/3970/3974，原test不动。v1未知proxy拒绝保留；v2只绑定固定969个既有null名单，18单测通过。21:15 ready/source audit/manifest匹配；21:32/21:38 C Full/F0 Original/F1 Matched075仍运行，最近完整epoch行为4/3/4且有限，回执仍为1/2、3/4、1/2。
- 阻塞：AIR test有797组语义标注冲突，A2/B1三个位置待用户确认权威数据或协议；即使其他54项完成也不删位置、不缩小57门槛，见CONFLICT-017。
- 下一步：保持既有B/C/F seed0任务并回收完整生产者回执；A 15项cross已完成，A/B的AIR三位置仍阻塞。21:35四台均无barrier，五条seed1 lane无状态或receipt。在未来seed1前实现/审阅/服务器fixture验收独立原子barrier发布层及A完整19项seed1兼容入口。现有静态设计未实现，禁止直接调用旧非原子writer放行。
- 验收：每配置固定split/输入/协议与source/data/code哈希，保留所有正负结果；新seed逐轮顺序和历史已跑多seed分别登记。完整完成需57条真实seed0及随后57条seed1通过，跨seed统计待对应配对齐全；公平外部基线、效率/机制任务仍分别未完成。
- 来源：`reports/RGSP_SEEDWISE_RECOVERY_20261009.md` 与campaign完整证据，含`evidence/refresh_20261009_2130/INCREMENT_REPORT.md`；本次RunId `rgsp-seedwise-progress-20261009-01a12053`，前序恢复RunId `rgsp-seedwise-recovery-20261009-01a12053`。此前C11/12、F恢复各剩10及旧36-run状态仅为历史时点，不据此重复启动。

## TASK-SADD-RGSP-CONTROLS · P0 · RESULTS_READY

- 授权：2026-10-08用户明确要求在SADD补实验；为补充数据集，不自动替换主范围。
- 进度（2026-10-09）：12run训练/test及完成记录、权重SHA核验PASS。三seed结果见reports/SADD_RESULTS_SSDD_RESTART_20261009.md；Full AP50:95平均低于Original1.3674pp，待结合限制开展论文分析。
- 协议：fixed795/44/44，scratch/200e/patience40/640/BS16/FP32/AdamW2e-4/无模块warmup；数据与代码审计、预检完成，详见`reports/SADD_RGSP_CONTROLS_20261008.md`。
- 验收：12个train_complete、eval_test、evaluation_complete；全部seed原值、均值±样本标准差、三种配对差值、冻结常量/初始化/哈希核验；不混入旧BS32/partial-transfer，不删除负结果。
- 风险：positive-only、test44/val44，本地图像划分不等于场景独立；恢复为EMA/FP16优化器续训，非逐位等价。A94/B93历史control审计缺口保留。核验新runtime_audit_resume、逐轮/最终记录后才认定续训完成；报告见reports/RGSP_RECOVERY_QUEUE_20261008.md。

## TASK-COMPLEXITY · P1 · TODO

- 动作：统一主表复杂度测量，补全未覆盖方法。
- 验收：相同硬件、精度、batch=1、profiler；区分核心组件与完整 detector。
- 进度：SSDD 5 变体复杂度账本已测（seed0，见 BASELINE_PROTOCOL.md §4.1 / EXPERIMENT_LOG）。
- 本轮增量（2026-10-12）：9跨任务+4单类检测结构在A测得配置绑定的covered-operation FLOPs估算，独立数值审核PASS；包含特殊函数、功能式算子、Ottawa GroupNorm等覆盖缺口，未测统一FPS/延迟，不与旧THOP-only账本混为同口径，本项保持TODO。
- 阻塞：无已确认硬阻塞；执行前校对现状。

## TASK-PAPER-P3-WORDING · P1 · TODO

- 动作：核对实际结果，统一论文 P3 / P2/4 表述，并关闭 CONFLICT-013。
- **调整（2026-10-08，替代旧归因）**：撤回旧PFIM-only独立贡献解释；不以seed2为outlier、不以Full−DS为严格负交互。新36-run尚未出结果，暂不预定Full或DS必须最好；结合新证据统一P3表述。
- 最新论文交付（2026-10-09）：完整实验14页稿已统一本稿LRC/图2(c)/P3并保留负结果；文档运行验收及最终ZIP独立解压双遍编译PASS，见下方已完成TASK-PAPER-FULL-EXPERIMENTS-20261009和PROJECT_STATE现行产物。旧英文稿CONFLICT-013仍未关闭，本任务保持TODO。
- 历史进度（2026-10-02）：中文内部稿现为 24 页 A4/11 开 PDF 图，从最终 Overleaf ZIP 解压后用 XeLaTeX 双遍编译通过；SSDD 15-run 全指标、协议流程、探索性正负跨任务记录、失败/历史账本和非配对 MaDiNet 文献背景均已纳入。研究动机将通用 Mamba 未显式利用 SAR 局部线索表达为待检验风险，不写成已证明的普遍缺陷。本轮未启动训练；SAR-ShipDataset/AIRcraft 公平 3 seed、同协议外部轻量基线及跨任务统一验证仍未完成。P2/4 与 P3 的历史英文稿冲突仍开，故本 TODO 不关闭；`PAPER_STATUS.md` 的产物清单仍是旧稿，现行文件指针见 PROJECT_STATE.md。
- 新增待办（批次 round96-kimi-k3，unverified，尚未开始）：用户提出投稿前收尾请求——将中文稿改写为英文期刊投稿稿（按指定期刊模板排版，保留证据边界与全部负结果）；按冻结代码逐式核对论文方法部分，补全模块插入位置、张量维度和可复现实现细节；核实并扩展相关工作；处理审稿清单 REVIEWER_AUDIT.md 中的 P0/P1 项；最终生成投稿格式稿。其中统一 P3/P2/4 表述与上方既有动作一致，不另立项。
- 验收：方法、图、实验标签一致，并关闭 CONFLICT-013。
- 阻塞：无已确认硬阻塞；执行前校对现状。

## TASK-RESULT-CONSOLIDATION · P1 · TODO

- 动作：汇总已存在 SSDD/SADD seed1/2、sarmix、SAR-AIRcraft 结果，并核查 warm-start 与 paper-aligned 完成状态。
- 验收：逐项带运行ID、协议和产物，不把未知状态标完成。
- 阻塞：无已确认硬阻塞；执行前校对现状。

## TASK-DATASET-HOSTING · P2 · TODO

- 动作：需要补备份时先核对 A6000-D 可达性和当前文件。
- 验收：确认待补内容、包清单与哈希；不得重复上传已验收包。
- 状态更新（2026-10-08）：D已恢复可达，直连偶发超时可经已配置LAN别名；本轮仅为SADD实验读取/复制数据，备份上传未执行。

## 已完成、延后与历史候选

### TASK-PAPER-DETECTION-TABLES-20261011 · P1 · DONE（仅源码修订）

- 按用户最新检测节附件完成完整稿 `RGSP_detection_revised_20261011.tex` 与独立检测片段/书目补丁；三线表明确方法、发表年份/出处和数值转引来源，历史复评单列，不跨协议加粗排名。
- 检测37行原始成绩、其余12表保持；13新增/3更新书目，74标签/42书目及环境/括号静态检查PASS。当前源码及CHECKS哈希见PROJECT_STATE；旧完整稿、附件和main.tex未覆盖。
- 未编译/渲染、未验收表格宽度或分页；无训练/推理/测时/服务，不关闭研究矩阵、复杂度/效率或旧英文P2/4冲突TODO。RunId：`rgsp-detection-tables-20261011-01a12aa6`。

### TASK-PAPER-SELECTED-TEX-20261011 · P1 · DONE（仅源码校订）

- 按用户最新精简附件完成独立 `RGSP_revised_20261011.tex`，结构、负结果、默认Full保留；5缺失书目与5组Matched075补回，历史协议与附录纠正，不新增使用边界节。
- 15表原数字/跨任务57展示指标/引用/7资产路径静态复核通过；TeX与CHECKS哈希及入口见PROJECT_STATE。原全图main和ZIP未覆盖。
- 编译/分页验收未运行，SUMMIT Table5本轮未独立复验；不关闭实验、统一效率或旧英文P2/4冲突TODO。RunId：`rgsp-selected-tex-review-20261011-01a12aa6`。

### TASK-PAPER-ALLFIGURES-ABLATION-20261011 · P1 · DONE（仅源码整合）

- 用户指定ZIP的独立增补版已交付：64图全部引用，SSDD补齐27项，SADD9项保留，五数据集15项跨任务单seed表补回，原文非必要不改。
- 静态验收：新增144数值单元、137份源小JSON、431项ZIP及独立解压比对通过；main/ZIP SHA与证据入口见PROJECT_STATE现行源码条目。
- PDF编译与分页目视验收未运行：服务器无XeLaTeX，未自动本机回退；需可用服务器TeX或用户明确本机文档许可后再验。此DONE不表示研究57项/seed1/机制/效率完成，不重启任何实验。
- RunId：`rgsp-allfigures-ablation-20261011-01a12390`。

### TASK-PAPER-VISUALIZATION-GALLERY-20261010 · P1 · DONE

- 范围：续接原50图候选图册，完成版式修订、三格式导出、离线选图页、50页PDF及完整代码包；未补跑研究实验。
- 验收：50图逐图视觉PASS、50份单图与合册50页渲染全等、65输入和生成来源哈希绑定；259项ZIP逐成员CRC/SHA及本地独立解压/安装哈希通过。现行根 `output/pdf/rgsp_visualization_gallery_20261010/`，产物哈希见PROJECT_STATE；详细证据为包外 `package_verification.json`、`qa/local_delivery.json` 和两份QA记录。
- 边界：2026-10-10 10:07固定快照；当前单seed、历史三seed及历史ex27特征分开，负结果全保留。浏览器交互未实测；后续由用户按图号选择精修，不自动改正文或关闭研究矩阵/机制/效率TODO。
- RunId：`rgsp-visualization-gallery-20261010-01a1258d`；旧产物、代码快照和QA中间材料保留。

### TASK-PAPER-FULL-EXPERIMENTS-20261009 · P0 · DONE

- 动作与范围：完成指定ZIP的完整实验正文、消融与可编辑包交付；未补跑研究实验。
- 验收（2026-10-09）：14页/18表/2图，27run和135项逐seed指标、负结果保留；87项ZIP/86项payload与解压哈希全匹配，13项编译输入未变。用户批准本机文档回退后，当前核验器两种文本后端均PASS，最终ZIP独立解压双遍XeLaTeX、14页文本/尺寸/渲染全等。
- 证据：`output/pdf/rgsp_full_experiments_verified_20261009/verification.json`（status及roundtrip_compilation.status均PASS）；源码 `paper_draft/rgsp_cn_full_experiments_20261009/VALIDATION.md`、`evidence/BUILD_VALIDATION.json`。产物与哈希见PROJECT_STATE，旧静态包保留。
- RunId：`rgsp-full-paper-local-verified-20261009-01a12069`。此DONE不关闭SSDD36run、效率/机制/公平基线或旧英文P2/4冲突；无安装/模型运行/服务或队列操作。

### 其他历史事项

- OGSOD 与seed/分支/分层已明确延后，不留在 Active；旧历史建议、跨数据集扩展请求及 paper-aligned 轨道保留在原 TODO 供明确重启时查看。
- 2026-09-21历史整理本身未启动训练、未改论文、未上传数据；2026-10-08已授权的训练及恢复状态见上述活动任务。
- [整理前完整TODO](C:/Users/13207/ProjectMemory/_backups/20260921-live-refinement-1930/projects/Mamba/TODO.md)：包含旧完成项、重复追加和未采纳建议，保留不丢失，不作为自动执行清单。
