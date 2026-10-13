# Mamba — IG-DS-SSM / PFIM：通用 SAR-Mamba 状态更新
- [CONFIRMED; EXPERIMENTAL; 2026-10-13; FactKey=Mamba.Experiment.SAMPLEDiagnosis20261013] SAMPLE低分诊断：A/C共18份既有run混淆重算Accuracy/Macro-F1一致，539支持数与标签/目录对应通过；10关键源与生产SHA一致，未发现确定标签或指标bug。Full seed0=2/539、GoogLeNet合成val96.27%/实测2.23%，预测集中少数类。当前decibel L8 train/val均值约126、test158；MLDA原文为uint16幅度[0,1]、806训练/539测试、SGD.001/momentum.9/BS36/cosine10000更新，当前645/161/539、AdamW、BS16x4、最多1650更新不对齐。官方QPM亦L8非原幅度；固定源MAT两样例complex_img读取PASS、全量及MLDA精确缩放待核。REPORT.md及DESIGN.md位于experiments/sample_diagnosis_20261013_v1；方案保留用户BS16x4、优先三臂、另列R36，PROPOSAL未部署。因果贡献待受控重训，本轮仅低优先级CPU数据审计/文献核对，无模型运行/队列变更/自动监测；保留全部旧负结果。RunId=sample-diagnosis-20261013-v1。
整理日期：2026-09-21；本轮 3 seed 消融结果补充于 2026-10-02。这里的项目事件仍按各自原日期理解。历史核验记录不等于本轮重新验证。
- [CONFIRMED; EXPERIMENTAL; 2026-10-13; FactKey=Mamba.Experiment.UnifiedA6000V2] 统一150轮campaign原30配置+6项SAMPLE补充基线，共36配置，非110候选全部就绪。10:03起分时复核并回收，累计68次完整运行/27配置（A28/C12/D19/F9；27seed0/22seed1/19seed2），原30中22配置种子齐；704证据文件及服务器权重/协议回执核验PASS。Ship Full已从F1转至C/c0shipfull且协议SHA完全一致，首轮1164.13秒；F0/F1继续Original/Matched075，F1保留Full的PENDING槽仅历史、不可重复调度。A/a0samplex六项两阶段预检通过，AlexNet seed0第83轮早停已验收，后续顺序最多3seed；D全量SADD继续；该快照时B未接管，10/13随后授权释放见下一条。配方BS16×4/AdamW2e-4/WD5e-4/FP32/scratch/150e/patience30/实际3e预热+linear不变，旧57seed0门槛保留。当前交接与只读入口 experiments/rgsp_rebalance_20261013_v1/README.md、Read-LiveStatus.ps1；证据RESUME_1014_VERIFICATION.json。该快照取代旧67次计数，旧记录保留；训练整批未完成。RunId：rgsp-rebalance-resume-20261013-1003。
- [CONFIRMED; EXPERIMENTAL; 2026-10-13; FactKey=Mamba.Data.SARACDPair20261013; Supersedes=Mamba.Data.SARACDWait20261012] 用户改为官方3032和排除冲突2992两版都训练，允许释放B旧AIR空等队列，确认BS16×4/有效64及官方明确结构、优先Original/Matched075/Full。C已git clone官方六类commit537639c406d5d37e52ba75fa433de52db7d1d483，完整导出至B/A；3032图解码、20跨类组40图。配对full1000/240/1792与clean960/240/1792，共同val/test，冲突仅full训练，原图不删改；这是本地协议、非作者划分，16/15updates每轮差异披露。B10:27释放回执RELEASED，旧57门槛/AIR3位置保留；核心6项probe PASS，正式clean-Matched seed0首轮5.895秒，14–15分钟是150轮总时长估计。B核心后ResNet50两版，A现有SAMPLE结束后六通用分类器×两版，共20配置、最多3/2/1seed依耗时；B后置3个TorchVision R50-FPN SSDD基线，CPU适配/数据fixture PASS，GPU预检待队列，未宣称性能完成。队列与104小文件SHA交接PASS；入口experiments/rgsp_table_expansion_20261013_v1/README.md。用户要求不持续监测、等其通知后回收补表；未建监测自动化。核心seed0估1.5–2h、最多3seed估4.5–6h、全20配置估14–18h（含A旧队列，外部由SAMPLE缩放，早停可短）；不含后置检测及缺源码专用方法。SADD全2966图（883阳/2083阴）新native清单2372/296/298已核，1296框角点越界但7835归一化xywh均符合原生检查，标签原样保留；与旧阳性子集独立。D依赖经本轮授权仅装任务目录，旧失败及首次等待器失败保留。 RunId：rgsp-acd-expansion-20261013-1053。
- [CONFIRMED; CURRENT; 2026-10-13; FactKey=Mamba.Paper.PerformanceFill20261013] 用户最新45dee9dc附件的六表完整复制段落已补于 `paper_draft/rgsp_performance_data_fill_20261013/performance_comparison_filled.tex`，SHA `814b93624e5f7f1fbbc1ef4d28333b9921e5c0aada3e505d592d1c142360c2ea`；正文/旧成绩不变，45新增性能值、56原空复杂度格+24新增组复杂度格。A独立任务补测5外部CD，参数与正式运行一致、来源SHA匹配；覆盖算子FLOPs非穷尽计数。SSDD共同2seed，SADD四seed0，Original及新Ship成绩未完成保持空缺，该文稿时点六类SAR-ACD等修正版，随后新授权与训练见本页SARACDPair条。STATIC_CHECKS PASS（逆向原稿恢复/正文/环境/数学分隔符）；未整稿编译分页，不是研究验收完成。证据DATA_FILL_EVIDENCE与独立profile `experiments/rgsp_profile_fill_20261013_v1/results.remote.json`。RunId：`rgsp-performance-fill-20261013-01a1301f`。
- [CONFIRMED; EXPERIMENTAL; 2026-10-12; FactKey=Mamba.Experiment.TableCompletion20261012] 当前补表稿 `experiments/paper_table_fill_20261012_v1/input/main.tex`：填24旧空格，另有Ship新表21值；首批SSDD的YOLOv8n已核300轮/8700更新及最终结果，其余三臂在10:51快照仍运行。执行/交付入口 `experiments/rgsp_table_completion_20261012_v1/PLAN.md`；详情见本目录 TASK-TABLE-COMPLETION-20261012 与 EXP-RGSP-TABLE-COMPLETION-20261012。旧57项seed0→seed1门槛及AIR3阻塞保留；RunId：`rgsp-table-completion-20261012-01a12dd6-7cb894b2`。
## 2026-10-11 检测表修订与用户精简稿（当前复制用源码）

- [CONFIRMED; CURRENT; FactKey=Mamba.Paper.DetectionTables20261011] 以用户最新附件6481447a的检测节（SHA `5d132949a615c0441eecc82b3557f0afc4ba3a1a28b52868588191f5c217e471`）修订，现行完整稿 `paper_draft/rgsp_all_figures_ablation_20261011/RGSP_detection_revised_20261011.tex`，SHA `2e903a97c2cd776c4b43e26e2fb354abfdc6423f4c2949a1fa089bd16f43e05d`；同目录DETECTION_SECTION/REFERENCES_20261011.tex为局部替换及书目补丁。三线分组表补方法/出处/年份，原始文献与数值转引分开；SSDD历史复评独立成表，不混协议排名。
- [CONFIRMED; 检测节静态核验] 检测3→4表、37/37方法指标行保留；其他12表逐字不变，非检测正文只改一处必要历史表引用。13条新增+3条更新书目，全稿16表/74唯一标签/42引用与书目，环境/括号及7图路径通过；证据DETECTION_TABLES_20261011_CHECKS.json SHA `4c0996d2dc736a1b6c55971f21bd774a8267290bbe8af466f517e48a8ce3859f`。保留默认Full/负结果，不新增使用边界；未编译/渲染或验收分页，未训练/推理/测时，不关闭研究与效率TODO。RunId：`rgsp-detection-tables-20261011-01a12aa6`。
- [CONFIRMED; DEPRECATED; FactKey=Mamba.Paper.SelectedTexRevision20261011] 以用户最新附件1945e861的完整文本（SHA `4c22d60cee11e9fa87f918ea2e65a7fd434851b572fa2b941841c93a2cf2e256`）为唯一底稿；独立交付 `paper_draft/rgsp_all_figures_ablation_20261011/RGSP_revised_20261011.tex`，SHA `cb4873d3cc09bee63e3f11d260ca3ba9b5887efbfabccedf036e325e7e96cfe1`。本稿为本轮检测修订的非检测部分底稿；当前复制入口以上方检测新版为准。下方全图版留作历史选图资料，原main/ZIP/附件未覆盖。
- [CONFIRMED; 修订范围] 保留引言—相关工作—方法—实验—结论，不新增使用边界节；补齐5个书目与五任务Matched075共19个既有指标，修正加粗、协议/陈旧承诺、历史T/S标签及附录；原实验数字、默认Full和负结果保留。正文4图+附录3图，不等于64图全图稿。
- [CONFIRMED; 静态核验] 同目录 `RGSP_revised_20261011_CHECKS.json`（SHA `8cf45f6a620507301bdf55c0080f2439439d37ff79bb4b31d7da5db637f70d78`）及NOTES记录15表原有598数字词元保持、15份cross JSON的57展示指标核对、73标签/29书目无未定义引用、环境/括号及7资产路径通过；不是598项独立指标或重跑结果。
- [边界] 未训练/推理/测时/编译或分页验收；SUMMIT Table5本轮未独立复验，现有数值保留。其余缺失书目对应的4篇外部表已按原文核对；不宣称全稿全部外部来源已验收，不关闭研究、效率或旧英文冲突TODO。RunId：`rgsp-selected-tex-review-20261011-01a12aa6`。

## 2026-10-11 此前全图与消融独立整合稿（历史选图资料，PDF未编译）

- [CONFIRMED; DEPRECATED; FactKey=Mamba.Paper.AllFiguresAblation20261011] 按用户指定下载ZIP作唯一底稿，独立源码 `paper_draft/rgsp_all_figures_ablation_20261011/`；保留全部原文主体，仅增结果与图件、必要状态句，结论仅补“历史”限定。50张新候选图+原包14图共64图实际引用；旧50图仍固定10/10快照。SSDD新表20→27项，SADD9项保留，另补入五数据集15项跨任务seed0表；旧结果与负结果不替换、不混统计。
- [CONFIRMED; 静态与来源核验] main.tex SHA `f5224ace7cd0cc603f62fa7966f3028d09a8b493dc1b8283525e3039ae0970b1`；摘要至方法、原11图环境、参考文献及18项原包文件保持。新增144数值单元全部匹配；B的42份检测小JSON、A的95份跨任务小JSON源SHA核验一致，未重取权重或训练ledger，不冒充全链训练复验。证据见源码evidence/与MERGE_NOTES.md，不因文稿取数重跑既有实验。
- 交付 `output/pdf/rgsp_all_figures_ablation_20261011/RGSP_all_figures_ablation_20261011.zip` SHA `f228ecd04f87e716b49db58a85dfb09a89c6dca3fb11199cad5f74159798e585`；431文件经包内比对及独立解压静态复核，64引用/150图格式哈希/标签/环境通过。包外PACKAGE_VERIFICATION.json为STATIC_PASS而非PDF验收；A/B/C/D/F均可达但未找到XeLaTeX，无本机回退授权，未编译/渲染新主稿。未训练、推理、安装、服务或队列操作；现有实验/效率/机制TODO不关闭。RunId：`rgsp-allfigures-ablation-20261011-01a12390`。

## 2026-10-10 科研选图册（现行图件交付）

- [CONFIRMED; CURRENT; FactKey=Mamba.Paper.VisualizationGallery20261010] 续接聊天“1010-论文的数据表格图片生成-未完成”（01a123a3-81d9-7343-94fd-aea35b4dbc19）并完成50张候选图：C01–C16、H01–H12、Q01–Q12、M01–M06、P01–P04，每图300dpi PNG、可编辑SVG和PDF，附50页合册、离线选图页及完整10份代码。现行根 `output/pdf/rgsp_visualization_gallery_20261010/`，供用户按图号选择后精修。
- 交付ZIP `rgsp_visualization_gallery_20261010_complete.zip` SHA256 `7c1662f1e7beeaf61ccc9ab904cfd39e39df2081d2dfc2ce8ef2e461e22d1957`；合册 `gallery_booklet.pdf` SHA256 `c7d3a83e5623e8414d17ab1abce7134da204cbce7b16905563d85f73e0bc09c1`。259项包经CRC及解压字节SHA核验，本地完整回收、独立解压与最终逐文件哈希核验一致；未覆盖任何已有不同内容文件。
- [CONFIRMED; 图件验收] `qa/validation.json` 为STRUCTURAL_PASS，`qa/visual_review.json` 为VISUAL_PASS：50份单图及50页合册实际渲染、逐页RGB像素全等，50图逐图审阅；65输入、生成时7绘图脚本、全部10代码与8原始总览图绑定。包级回执 `package_verification.json`、本地回收 `qa/local_delivery.json`、详细记录 `notes/EXECUTION_AND_QA.md`。
- [CURRENT; 边界] 数据固定于2026-10-10 10:07:54–10:07:57快照（SSDD20、SADD9），不替代现时队列状态；新730/186/232与旧742/186/232分开，历史SADD12/SSDD15及全部seed和负结果保留，Q图为6个历史ex27/block0样本而非当前P3推理。示意图不是机制验证，浏览器交互未实测；图册不等于投稿定稿或研究实验完成。
- 本轮仅在B独立任务目录做低优先级CPU绘图/Poppler验证/打包，本机仅接收、解压和哈希核验；未训练/推理、安装、启动服务或改变原实验队列。只关闭图册交付项，不关闭其他实验与论文TODO；正式代码/论文/原始数据未改。RunId：`rgsp-visualization-gallery-20261010-01a1258d`。

## 2026-10-09 分轮消融续接（当前执行）

- [CONFIRMED; CURRENT; FactKey=Mamba.Experiment.Seedwise20261009] 用户续接“astra-消融实验修改2”，明确所有配置先seed0、全完成后seed1，不新增seed2。独立campaign `experiments/rgsp_seedwise_20261009_v1/`，114位置=57+57，计划canonical SHA `9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4`，360项冻结源保持；取代先前新增实验多seed同时调度，不删除/重标旧seed1/2或负结果。
- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.SeedwiseReceipts20261009] 21:31:42–21:32:32完整产地核验分时合计32/57 verified、25 missing、0 invalid：A20/22、B7/27、C1/2、F0=3/4、F1=1/2。不是同刻全机扫描或全部实验完成；五份完整JSON、缺项和SHA见`evidence/refresh_20261009_2130/INCREMENT_REPORT.md`。
- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.SeedwiseReceipts20261010] 本次核读的较晚已保存外层回执快照为2026-10-10 11:17:03–11:30:06 +08:00分时45/57 verified、12 missing、0 invalid：A20/22、B20/27、C1/2、F0=3/4、F1=1/2；五份均all_requested_complete=false且all_global_seed0_verified_locally=false。见[保存记录](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/sequential_dispatch_20261010_1115/STATUS.md)。本轮仅静态核字段/行数/SHA，不是同刻全机扫描、递归内部产物深验、现时远端状态或barrier证书；保留上述10/09深验历史及57项/AIR/seed1门槛。RunId：20261011-memory-resume-01a128b2-a1。
- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.CrossRecovery20261009] A原cross入口在首epoch前因torch1.13 UUID计数故障退出；独立兼容operator已完成15/15计划cross seed0。当前15/15均通过completion六项、protocol、连续epoch、best/test/result递归深验；operator状态complete，owner/child均已退出，A assigned GPU空闲。原失败现场与非逐位等价限制保留；结果细节见恢复报告和`evidence/refresh_20261009_2130/INCREMENT_REPORT.md`。
- [CONFIRMED; EXPERIMENTAL] 21:34:45新增Farmland与Yellow River六项cross均`deep_verified`；与此前九项合计15/15。A seed0仍有两项AIR阻塞，故A lane及全局seed0未完成；原dispatch/lane的failed状态是旧失败现场，保留。
- [CONFIRMED; CURRENT; FactKey=Mamba.Data.ShipAudit20261009] C/F完整39729图源审计一致、0 mismatch；byte/RGB各143重复组、50跨split组，原test无内部重复/语义冲突。独立副本按test>val>train排除train48/val2，保留31735/3970/3974；原test3974和剩余图/标注字节不变。v2仅接受固定源下精确969个原有null proxy，C18单测及完整数据验证/暂存/发布均通过；20:16 C/F原子no-clobber ready已发布，选定manifest SHA `2b30f37c7eaa9d5118354736959e64035841a3219ea3fd44963ae52ee6fbda78`，72项小记录逐文件回收哈希相同。21:13–21:15核验C/F ready与manifest/source audit SHA匹配；21:32/21:38现场看到C Full/F0 Original/F1 Matched075仍在assigned GPU运行，完整epoch行为4/3/4且有限，Ship回执仍分别1/2、3/4、1/2，未新增完整Ship结果。
- [CONFIRMED; CURRENT; FactKey=Mamba.Data.AirTestConflict20261009; ConflictStatus=OPEN] AIR原test完整范围873个byte重复组/1892张图/797组语义冲突，三个AIR seed0位置仍阻塞（A2/B1，见CONFLICT-017）。需用户确认权威修正版或另行协议；不裁决标签、不删位置、不缩到54项门槛。Ship同split重复、未知scene proxy及非场景独立限制保留。
- [CURRENT; TODO] 继续既有seed0队列并逐项回收深验；21:35 A/B/C/F均无seed0 barrier，五条seed1 lane均无status或receipt。全57通过且AIR阻塞解决前不发barrier/不开seed1。原全局writer存在半写JSON竞态，独立原子发布层与A完整19项seed1兼容operator仍待实现/服务器fixture验收；静态设计不等于修复已部署。A seed1须在创建compat子进程前验barrier，不能直接复用当前15项seed0恢复器。
- 续接报告 `reports/RGSP_SEEDWISE_RECOVERY_20261009.md`；10/09时点证据含`evidence/refresh_20261009_2130/`及原`refresh_20261009_2115/`、`refresh_20261009_2124/`、Ship发布和barrier静态复审目录。21:31–21:38生产端回执、A 15项深验、B/C/F现场身份及barrier/seed1缺席均已核验。本轮RunId `rgsp-seedwise-progress-20261009-01a12053`；无安装/服务改动/本机模型执行。
- 本聊天已有30分钟heartbeat `rgsp`，正常进度保持安静，重要完成/失败/需要决定才通知；需本机与Codex运行，不是服务器重启自启动。本轮RunId `rgsp-seedwise-recovery-20261009-01a12053`。并发论文节保留，不把文档验收与研究实验完成混淆。

## 2026-10-09 完整实验与消融稿（现行交付，文档运行验收完成）

- [CONFIRMED; CURRENT; FactKey=Mamba.Paper.RGSPFullExperiments20261009] 用户续接聊天“astra-论文修改2-2”，以指定上传ZIP保留完整实验正文并补消融。源码 `paper_draft/rgsp_cn_full_experiments_20261009/`，14页IEEE双栏、18表、2图；九张旧任务表及SSDD15run/SADD12run的135项逐seed指标、9组汇总、12种配对和新12配置定义均在正文，保留负结果与未验收边界。
- 现行交付根 `output/pdf/rgsp_full_experiments_verified_20261009/`：PDF `rgsp_full_experiments_verified_20261009.pdf` SHA256 `3ae0914cbd50e779cd1cd53569c16152d780fefe926eb09fec66bd08a81a00bd`；ZIP `rgsp_full_experiments_verified_20261009_editable.zip` SHA256 `00c02ad4f333d6d50638ab18f175b88ce92553df62ba2d020dd261bec1ce0f18`。
- [CONFIRMED; 文档验收] `verification.json` SHA256 `e3703fcabafd34e278f9b37b9e7ac7684aef1892dd20c9394eba6a4bfce9c651`，status和roundtrip_compilation.status均PASS：87项包/86项payload、CRC及全部解压哈希一致，原27项上传文件/9表数值序列/13项编译输入未变；最终ZIP独立解压后XeLaTeX双遍编译、14页逐页文本/尺寸/115dpi PNG全部一致，最终日志无Overfull/缺字/未解析引用；既有字体替代及参考文献Underfull保留。
- [CONFIRMED; 授权边界] 服务器失败后用户明确同意仅对文档采用本机回退；使用现有TeX Live/Python/pypdf/Poppler，当前核验器的pypdf及既有pdfinfo/pdftotext分支均实际PASS。先从旧静态包编译，再从最终包独立复编译；新旧PDF仅二进制元数据不同，全部14页渲染同哈希。依据源码 `VALIDATION.md`、`evidence/BUILD_VALIDATION.json`、两份DOCUMENT_VALIDATION及包外verification。
- [CONFIRMED; FAILED; 历史; FactKey=Mamba.Paper.DeliveryTransport20261009] 先前检查A/B/C/D/F、避开忙碌A/B后选F，SCP及后续已配置连接超时，远端文档核验未执行；未找到现成TeX且未安装。临时远端副本完整性unknown，不能作备份；本轮未重试服务器或查询/改变实验队列。
- 旧静态交付 `output/pdf/rgsp_full_experiments_20261009/` 保留：ZIP SHA256 `71ea5eb1422724637e24af487622e9bb4c18682382e1c3ffdfda471aeb801144`，其STATIC_PASS_RUNTIME_PENDING仅为回退获准前历史；PREEXISTING_BUILD与旧PDF未覆盖。
- [CURRENT; 边界] 仅TASK-PAPER-FULL-EXPERIMENTS-20261009标DONE。无训练/推理/模型测试/安装/服务改动；新SSDD36run、外部表原文核数、统一效率/机制/公平实验和旧英文P2/4冲突仍未验收。文档可编辑交付完成，不是研究实验全部完成或投稿定稿。
- 本节替代前次静态交付状态及更早7页版的现行产物入口，旧产物/历史保留；未改变研究范围或数字。RunId：`rgsp-full-paper-local-verified-20261009-01a12069`。

## 2026-10-09 论文整合修订（此前7页版，已被完整实验版替代）
 
- [CONFIRMED; DEPRECATED; FactKey=Mamba.Paper.RGSPRevision20261009] 用户授权修改并索要最新版ZIP；独立源码 `paper_draft/rgsp_ieee2col_revision_20261009/`，主文7页IEEE双栏、补充8页A4。整合并修订两张图，统一LRC/图2(c)/P3/8；更新SADD三seed负结果，纠正旧PFIM独立贡献、Full−DS交互及Yellow River Precision表述。9张历史表数值和旧15run CSV保留；未覆盖原包或旧稿。
- 交付根 `output/pdf/rgsp_revision_20261009/`：`rgsp_main_20261009.pdf` SHA256 `f630b421f787272f7a11d1d9f1e04784092f7f457cb4f18c402d00921a668f76`；`rgsp_supplement_20261009.pdf` SHA256 `04a81214a0c0c7f77ae42d5c40d35a1ad38ac9712dad2f92dff45c062e7860a5`；`rgsp_paper_revision_20261009_editable.zip` SHA256 `4a92620e44f18b28e776ccf62a8d51e51b5a4699f5acda22a42894768378e24b`。56项包、55项payload哈希、解压XeLaTeX双遍、15页渲染逐页同哈希及视觉检查通过；详见 `verification.json`、源码内 `VALIDATION.md`。
- [CONFIRMED; 边界] 正文按10月9日本地回收记录，不查询远端实时状态；冻结SSDD/SADD模块SHA均为c5248810…，与绘图来源一致。业务目录随后并发更新不属于本轮，未回滚/覆盖。外部表主文献逐行核验仍缺；SARCDNet仅核书目，不冒充核数。无训练/推理/服务/安装；新SSDD36run及其他公平/机制/效率实验不标完成。
- 本节当时替代10月5日和10月8日独立稿，现已被上方完整实验版替代，旧记录保留历史；不声称恢复旧26页PDF全部附录，不关闭旧英文P2/4冲突或实验TODO。源码修订说明与来源登记为续接依据。RunId：`rgsp-paper-revision-20261009-01a11e6d`。

## 2026-10-09 SADD结果与SSDD恢复（此前记录；调度以上方分轮节为准）

- [CONFIRMED; CURRENT] SADD12run训练及test完成，A/B/D逐项train/eval完成记录与best权重SHA核验PASS。AP50:95 Original53.3467±0.8753、Matched07552.1773±0.8791、LocalEnergy52.3676±0.4269、Full51.9793±2.5915%；Full−Original平均−1.3674pp，仅1/3正差，不能宣称稳定提升。保留test44/positive-only/恢复非逐位等价及历史审计缺口。
- [CONFIRMED; CURRENT] 用户授权暂停并禁止ComfyUI自启；C用户comfyui.service stop/reset-failed并持久mask，is-enabled=masked/is-active=inactive。未发现所查其他自启入口，不保证未来人为异名启动。
- [CONFIRMED; EXPERIMENTAL] C原队列已恢复，guidance_only测试补齐，11/12；global训练epoch8已核验。F恢复失败确因为包装器传入SSDD冻结CLI不支持参数；修复包装器，冻结业务代码/150e/30patience/BS16不变，原断点哈希通过。F新队列PID122224/122228（时点）已通过120秒空闲门控，F0从epoch32、F1从epoch34实际续训，GPU95%/100%，各剩10配置。后续查真实进程、新epoch和queue_state，不按旧PID操作。
- 证据：reports/SADD_RESULTS_SSDD_RESTART_20261009.md、experiments/rgsp_status_20261009/；RunId=sadd-results-ssdd-restart-20261009。SSDD完整36run仍未完成，SADD结果不代表论文实验全部完成。
## 2026-10-08 20:08 重启恢复后置队列（历史时点）
 
- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.RecoveryQueue20261008] A/B/D/F于15:01–15:04重启，5条原lane中断，旧status=running已失效。用户明确要求复活并排在当前任务后；已在原卡启动5条waiting_predecessor队列，独立根 `/home/powerleader/ig-ds-ssm/tasks/rgsp_recovery_20261008_v1`，不重启C。
- F0/F1恢复SSDD seed0/1（checkpoint已完成31/33轮，各剩10配置）；A/B/D恢复SADD seed0/1/2（94/93/101轮，各剩4配置）。新lane为ssdd_seed0/1、sadd_seed0/1/2；PID仅时点39856/32614/38286/41252/41253，核对各lane的queue_state/launch_receipt和真实进程。C在原v3继续，20:08完成7/12、Full-fixed-half epoch16；不重复启动。
- [CONFIRMED; CURRENT] 等对应前序整条20项队列成功完成、调度进程退出，再要求GPU无计算进程且连续120秒空闲；不在训练间隙插队。未修改前序任务/服务，无系统自启动；关闭聊天不影响服务器等待进程，但服务器再次重启仍需显式核查恢复。
- [CONFIRMED; CURRENT] 原实验66项run文件哈希未变；数据副本逐项核验、冻结代码248/251项、last/best完整性与优化器有限值、四机各3项单元测试及五个resume路径探针通过。独立包装器只改续训数据/输出路径；尚未在忙碌GPU上执行恢复训练，须以新epoch/runtime_audit_resume确认实际续训。
- [CONFIRMED; CURRENT; 边界] 旧checkpoint为EMA/FP16优化器，不保证RNG/逐位等价；只在新副本清除尾部NUL并从checkpoint.train_results恢复CSV。历史control审计A94/B93/F0-31/F1-33缺失，明确保留缺口；不混成完整无中断轨道。已完成成绩保留来源、不重跑。
- 证据与续接：[恢复报告](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/reports/RGSP_RECOVERY_QUEUE_20261008.md)，`experiments/rgsp_recovery_20261008_v1/evidence/`。本轮RunId：`rgsp-recovery-queued-20261008-01a11b5e`；后续汇总原已完成+C原轨道+新恢复lane，完整实验仍未完成。

## 2026-10-08 初始实施记录（重启后状态以上节为准）

- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.RGSPControls20261008] 用户授权补实验与 GPU 训练。正式独立轨道 `rgsp_controls_20261008_v3` 已于北京时间 12:30 启动：F GPU0 seed0、F GPU1 seed1、C GPU0 seed2，12 配置×3 seed=36 run，150 epoch/patience30。12:32:44 三个 Original 均完成 epoch5，有限损失、三卡满负载；尚无该矩阵最终结果。远端根 `/home/powerleader/ig-ds-ssm/tasks/rgsp_controls_20261008_v3`，每 lane status/manifest/runs/logs 为续接依据，不得重复启动。
- [CONFIRMED; CURRENT; FactKey=Mamba.Audit.PFIMFreeze20261008] 三个旧 PFIM-only best.pt 的 lambda/alpha 均非零，SHA 与旧表一致；旧 trainer 会重新启用 requires_grad。故撤回“严格 PFIM 独立贡献”的解释，保留原成绩。新控制将关闭参数存为 buffer，并逐轮/重载核验；原始类默认行为保留。
- [CONFIRMED; CURRENT; FactKey=Mamba.Data.SSDDContentOverlap20261008] 当前 SSDD 副本1160图仅1135个字节内容，22重复组；10组跨官方train/test，涉及12训练副本与10测试图，组内标注不完全相同。新轨道保留官方test232，仅从开发集合排除12重叠副本，剩余内容组划分730/186；源码数据不删改。与旧742/186/232分表。仅字节审计，不是近重复/场景独立性证明。
- [CONFIRMED; CURRENT] F/C 数值、梯度、12配置解析/初始化验证通过；F全部12配置两轮短训练完成，C关键配置复验通过。scratch模型统一重置stride探针造成的BN统计；旧cos_lr=false，继续线性；独立评估workers=0规避torch1.13线程退出异常，指标不变。
- [CONFIRMED; 时点操作] 用户明确同意终止C的ComfyUI；核对PID3539324身份后SIGTERM并确认释放，未删除文件或更改自启动。D后续连接超时而弃用；A/B未启动训练。未安装软件、未本机跑模型。
- [CONFIRMED; CURRENT; FactKey=Mamba.Paper.ExperimentSupplement20261008] 新增 `paper_draft/RGSP_EXPERIMENTS_SUPPLEMENT_20261008.md`，详细协议、12配置矩阵、严格交互公式、历史纠错、待填结果表与边界；执行记录 `reports/RGSP_TRAINING_EXECUTION_20261008.md` 含PID/UUID/证据和包SHA。原双栏LaTeX预览未覆盖。
- [CURRENT; TODO] 等待本轮完整训练与测试、汇总三seed；SADD补充实验现已启动（见较新节），SAR-ShipDataset/SAR-AIRcraft公平验证仍缺；公平外部基线、效率与机制诊断仍缺。不得宣称论文实验全部完成。
- 本轮RunId：`rgsp-controls-training-20261008-01a11968`；源码包与证据在 `experiments/rgsp_controls_20261008_v3/`，前期v1/v2为预检证据而非正式成绩。


## 2026-10-08 SADD 补充实验（14:18启动复核，本节为较新状态）

- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Experiment.SADDControls20261008] 用户明确要求新增SADD实验。独立 `rgsp_sadd_controls_20261008_v1` 已启动Original/Matched075/LocalEnergy/Full×3seed=12run：A0 seed0队列1959131、B0 seed1队列1917370（14:16），D0 seed2队列2217783（14:11）。14:18前后A/B完成Original epoch2、D epoch12，有限loss；尚无正式配置最终test成绩。远端根 `/home/powerleader/ig-ds-ssm/tasks/rgsp_sadd_controls_20261008_v1`。先核对status/实际命令，不重复启动或按旧PID操作。
- [CONFIRMED; CURRENT; FactKey=Mamba.Data.SADDAudit20261008] D原始副本2966图含883阳性、2083空标注；本轮沿用固定阳性795/44/44，框7215/212/408。883图字节与原副本相同，标签仅CRLF→LF；文件名/字节/解码像素无重复。此为本地恢复的单类SADD，不是七类SAR-AIRcraft1.0，也不是已证明官方/场景独立协议；test44且无空场景限制须披露。
- [CONFIRMED; CURRENT] 三训练seed共享split；200e/patience40/640/BS16/nbs64/AdamW2e-4/线性/FP32/scratch/无模块warmup，统一验证选模、独立test。不是旧BS32或partial-transfer成绩复现，不混算。A/B/D CUDA控制验证通过；D四组2epoch及A Original/B Full短训练、重载、独立val均通过，共同初始化哈希一致；251项源码哈希和数据清单一致。
- [CONFIRMED; 时点操作] 仅使用A/B/D空闲卡，C/F SSDD保持；D直连偶发超时，配置中的`powerleader-a6000-d-lan`可达且hostname/GPU UUID一致，确认训练继续。无安装/服务更改/本机模型运行/周期监控。SADD为用户新增补充集，不替换DEC-010主范围。
- 协议/结果待填：[SADD执行报告](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/reports/SADD_RGSP_CONTROLS_20261008.md)。直接证据 `experiments/rgsp_sadd_controls_20261008_v1/launch_verification.json` 及 `evidence/{a,b,d}/`；代码包SHA256 `a957bf2cfd2be3adeef4b8f692cbe74f0bfd49ea76d23b9ea7fdc789b61fe388`。完整12run结果待回收，保留任何负结果。
- RunId：`sadd-controls-training-20261008-01a11968`。

## 当前范围

以检测为主验证、分类与变化检测为跨任务验证，分割作为探索性适用边界。正式目录为 docs/project-memory；不把旧检测 only 叙事当作最新论文定位。

## 当前状态与已知证据

- [CURRENT；既有决策] DEC-010（2026-09-19）采用通用 SAR-Mamba 定位；检测主数据集是 SSDD、SAR-ShipDataset、SAR-AIRcraft1.0，OGSOD 已退出当前主实验。
- [CURRENT；既有记忆] IG-DS-SSM/PFIM 为主线，DG-DS-SSM 不是主贡献。Gamma/write_gate 是同一 scan drive 的因子，delta 控制 transition；不能写成三个完全独立位置。
- [UNVERIFIED；2026-10-02] 论文动机定位（用户确认采用，属待验证机制假设而非已证实事实）：通用 Mamba 直接用于 SAR 图像时，状态更新未显式考虑 SAR 成像噪声与目标特点，散斑、杂波与弱目标都可能进入特征，背景扰动可能参与状态更新并影响后续表征，而简单平滑又可能损伤弱散射目标；Mamba 本身有输入依赖的选择机制，但不保证能区分 SAR 杂波与目标结构。方法针对该潜在缺口：先修正局部残差，再用重要性调节 scan 输入与更新步长。当前 SSDD 消融支持继续研究这些组件，但还不能证明"噪声沿状态递推传播"，且 DS-SSM 与 PFIM 的简单组合没有优于 DS-only，因此论文应突出该动机并如实分析组合效果。（批次 round96-kimi-k3）
- [CONFIRMED；2026-10-01] 建立项目审计交接区 PROJECT_HANDOFF/（十份文档 + README 索引），整理时对照了项目记忆、代码、报告、评估 JSON 与数据 manifest；未启动训练，未修改原有代码、论文或实验结果。该包早于 2026-10-01~02 的 SSDD 三 seed 结果，其结果与状态部分已落后。（批次 round96-kimi-k3）
- [历史数值保留；归因已由2026-10-05/08审计修正] SSDD旧15-run（5变体×3seed）已产生结果：Original/DS-only/PFIM-only/Full/Global AP50:95均值分别42.03/43.98/43.12/42.91/43.77%。旧“公平/仅PFIM/最稳健/seed2 outlier/严格负交互”解释不能按原文继续采用：同壳层控制缺失、PFIM关闭参数漂移以及当前数据副本内容重叠现已核验。详见新实验补充稿与原 `dsh_experiment_results/`，不删除旧行或负结果。
- [历史实验；未在本轮重跑] OGSOD 匹配单次 AP50:95 0.423018→0.436021，可作为历史受控证据，但无三 seed 统计，也不自动恢复成当前主实验。
- [聊天报告；未独立复核] FUSAR-Map UPerNet-lite 对照 baseline/ours mIoU 44.2973/43.5607、road IoU 4.78/1.09；这是负结果，不能宣称跨任务普遍提升。
- [历史记录] A6000-B 主包与 A/C 补充包记录已上传私有 ModelScope 备份；B 当时连接超时，未纳入。现时服务器、远端文件和 GPU 状态均需重新查询。
- [CONFIRMED；2026-10-02] 产出论文结构提纲 `paper_draft/PAPER_FRAMEWORK_20261002.md`：正文主线为引言→相关工作→方法→实验设置→主结果与消融→讨论和局限→结论，并列出拟用图表、各章待答问题及哪些结论已有证据；旧中英草稿未改动。同期代表性文献抽样（非系统综述）：遗憾 Mamba 论文通常先讲方法再用多任务实验验证；SAR 检测论文经写数据集与实验环境并讨论局限；因已有 SAR-Mamba 工作，本文不能再主张"首次将 Mamba 用于 SAR"，须明确自身机制差异；另提到一类固定处理设置值得作为对照。（批次 round96-kimi-k3）

## 风险与待核实

- SSDD ex27 的 0.74022 为历史验证集口径；官方 TEST 复核为 0.714454，初始化也非 clean scratch。
- 论文 P3 与 P2/4 表述尚未统一（CONFLICT-013）；分割、Yellow River 等负结果必须保留。
- **论文核心论断需调整**：旧结果不能证明 PFIM 独立贡献、联合稳定提升或 token-wise 优势；保留全部 seed。Full−DS 不是严格因素交互；本轮同壳层和四格实验尚在运行。
- 历史结果账本（Partially Confirmed）不得进入投稿主表：SSDD BS16 +7.75pp / BS32 −1.20pp（batch 敏感）；SAR-Ship +8.25pp、SAR-AIRcraft +1.04pp（非公平协议）；FUSAR/OpenEarthMap 负结果。
- 不同 batch、split、backbone、初始化或纸面结果不得拼成统一排行榜；旧 queued/running 不代表现在仍运行。
- [时点观察；2026-10-02 复核样查] 论文与交接材料当时未跟上 SSDD 三 seed 新结果：中文草稿仍把 ex27 AP50:95 0.74022 写成当前结果（历史验证集口径，固定官方 test 复核 0.714454）；英文稿仍以单次 OGSOD 与 P2/4 结构为主；PROJECT_HANDOFF、EXPERIMENT_GOAL.md、EXPERIMENT_PLAN.md 含"待启动训练"等旧状态；PAPER_EXPERIMENT_SUMMARY.md 把 DS-only 增益写成"一致提升"，与 seed2 负值不符，措辞需修正。当时稍后的中文 LaTeX 稿已纳入新结果（见下文物产条目），其余材料的更新状态未复核。（批次 round96-kimi-k3）
- [时点观察；2026-10-02，非长期事实] A6000-F 核验时未发现 Mamba 训练进程，但两张卡被 `scaf-grpo` 的 Python 进程各占约 13.5 GB 显存；服务器双卡空闲已非现场状态，启动新任务前必须重新检查资源。工作区存在大量旧存修改与未跟踪文件（含 dsh_experiment_results、PROJECT_HANDOFF），后续操作应避免清理或覆盖。（批次 round96-kimi-k3）

## 下一步

- TASK-GENERIC-MAMBA-MATRIX [DOING] 优先收齐当前分轮57项seed0，处理AIR来源冲突；全部验收后才进入seed1。已验收SADD12run保留，旧36-run历史账本另行核验，不恢复旧并行调度或以单seed宣称稳定性。
- TASK-COMPLEXITY [TODO] SSDD 5 变体账本已测；补全未覆盖方法。
- TASK-PAPER-P3-WORDING [TODO] 按负面发现调整 P3 措辞，统一 P2/4，关闭 CONFLICT-013。
- 阶段产物：`dsh_experiment_results/` 与 RESULTS.md / results.csv / BEST_CONFIG.md / PAPER_EXPERIMENT_SUMMARY.md / NEXT_STAGE_AUTONOMOUS_PROMPT.md。
- [CONFIRMED; 2026-10-02 historical paper artifact] 按论文框架完成中文内部 LaTeX 稿：24 页 A4、11 张 PDF 图；XeLaTeX 双遍编译成功，第二遍无未解析引用，PDF 页数及 A4 尺寸已核验。稿件纳入 SSDD 15-run 全指标、受控协议图、跨任务正负结果、失败/历史账本，以及 MaDiNet 作者报告 AP50 背景表（明确非配对、不可排名）；没有把这些背景数值写成同协议性能结论。源码 `paper_draft/ig_ds_ssm_cn/main.tex`，框架 `paper_draft/PAPER_FRAMEWORK_20261002.md`，复核清单 `paper_draft/ig_ds_ssm_cn/REVIEWER_AUDIT.md`。PDF `output/pdf/ig_ds_ssm_cn_paper.pdf` SHA-256 `AD4883680BC37235EFC3127454A51AD13633693D6CD66CC48E87EC4942907AE2`；Overleaf 包 `paper_draft/ig_ds_ssm_cn_overleaf.zip` SHA-256 `77CFE6689BB74EDFEC7B1DB5D25CFC903EEA4BDDD3FA8B978D8CCB9F2D19D119`，含 main.tex、README、11 张 figures/*.pdf 和结果/协议补充材料。旧稿已存 `paper_draft/ig_ds_ssm_cn/archive/`。本轮未启动训练；SAR-ShipDataset/AIRcraft 公平 3 seed、同协议外部轻量基线等仍未完成，稿件不是投稿定稿或 SOTA 证据。`PAPER_STATUS.md` 中旧的 8 图产物索引未能通过当前 publisher 更新；本文件是现行产物指针。

## 2026-10-05 全文修订与证据溯源

- [CONFIRMED; DEPRECATED; FactKey=Mamba.Paper.RGSPRevision20261005] 已按用户所给下载目录 ZIP 和最新引言完成独立中文修订稿，保留 RGSP 为工作名称。源码 `paper_draft/rgsp_cn_revision_20261005/main.tex`；25 页 A4（页数沿用历史记录，本轮未复验）、5 幅内嵌矢量图、21 表、24 条书目；42 处单条引用沿用保存的核验记录，附录 A-F。详见同目录 REVISION_NOTES.md、REFERENCE_AUDIT.md、supplementary/PROTOCOL_CLARIFICATIONS.md。
- PDF `output/pdf/rgsp_cn_revision_20261005.pdf` SHA-256 `F3346618B5B39D76DCF2CF063F96C12CBC3EAB15593AA5B29D231B48784EA977`；ZIP `paper_draft/rgsp_cn_revision_20261005_overleaf.zip` SHA-256 `176269E212A47BF3BDBCA9DDFA1AE09A0D7DB6FCFF13A062768B7518DE298B4B`。源文件 SHA-256 `8BD5FFBFDBFD9137C43C11677F84F77D772AEBEA2BB2643E15318512459037C8`。12 项 ZIP 逐项哈希通过，18 项输入文件原样保留；解压后 XeLaTeX 双遍成功，25 页文本及尺寸与交付版一致，逐页视觉检查完成。内置编译器环境错误；使用已有本地 XeLaTeX，无新安装。
- [CONFIRMED; FactKey=Mamba.Paper.EvidenceBoundary20261005] 本次复算确认已有 15 行 SSDD 数值，但未复核全部历史 run 原始链。论文用词不再沿用上文 2026-10-02 的"seed2 outlier""最稳健核心增益""负交互"作为已证明解释：保留所有 seed；full 对基线仅 1/3 正差，full−DS=-1.0726pp 不等于严格因素交互。基线四向 DS2D 与替换单向块的壳层不同，须同壳层中性控制；当前 fitness 为 0.1AP50+0.9AP50:95，历史调参/选模版本仍需绑定。
- [DEPRECATED; 10月5日版本范围] 可靠性是设计目标，PFIM 实际为任务训练的空间控制量；扫描驱动和步长语义、P3/8 实际一个块以及跨任务输出适配器差异均已对照当前代码改写。本轮未修改业务代码、未训练/推理、未新增实验。严格三数据集/同接口跨任务、可靠性诊断和效率验证仍 planned；不关闭现有实验 TODO 或 CONFLICT-013。旧稿与历 ZIP 保持不变。
- 最近论文增量：`rgsp-paper-revision-20261005-01a109ef`；对应独立证据和回执位于中央 memory-refinement/Mamba 与 commits/Mamba。本节为历史稿索引；SupersededBy=Mamba.Paper.RGSPFullExperiments20261009（上方14页完整实验稿），不覆盖2026-10-10独立选图册入口；前述2026-10-02产物仍保留为历史。
- 2026-10-11静态纠错（RunId：20261011-memory-resume-01a128b2-a1）：同SHA的main.tex环境/书目计数与revision_checks/source_validation.json均为5图/21表/24书目；未编译、渲染或重验旧页数，未改论文与实验结论。


## 2026-10-08 动机图与图注修订（独立双栏预览）

- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Paper.Motivation20261008] 按用户“先修改动机图”请求完成真实SAR观察、内容依赖状态链、RGSP两处结构干预的新图1。仅图件与图注改动，未改引言/方法/实验数值；效果仍为设计示意，不是模型推理证据。图示不将未标注高亮区域认定为噪声，不使用未经核验的弱目标，统一原始灰度映射。
- 图件与证据：`output/pdf/rgsp_motivation_20261008/` 含PDF、可编辑SVG、PNG、LaTeX片段、provenance和verification；源脚本 `output/figures/build_rgsp_motivation_20261008.py`。图PDF SHA256 `5220b7024175a9ddcf7e33f6c0787d8ae99835e905692525ae3f1669b978dcfe`，三款字体均嵌入，180×108mm；已通过Poppler/MuPDF视觉检查。
- 独立可编辑稿 `paper_draft/rgsp_ieee2col_motivation_20261008/main.tex`，基于下载目录最新双栏稿；原稿与旧图保留。XeLaTeX双遍成功，`paper_motivation_updated.pdf` 为12页，图1在第3页，SHA256 `0d162fc0a5e0fb6ddf6cea1ef7fd2b997e7a2a8a3a6a3444d225a83b2956a4ee`。除图1路径/图注及末尾换行外正文相同；无未解析引用或Overfull，既有字体替代/Underfull警告保留。
- 这是供用户审阅的新图版本，不表示全文修改或实验完成；仍继承正文预览范围及已知旧归因/数据集不一致问题，未恢复26页旧PDF的缺失附录。无训练、推理、安装或服务器操作；本次图件审美确认及后续正文修订待用户反馈。RunId：`rgsp-motivation-20261008-01a11a4a`。

## 2026-10-08 架构图重绘（独立审阅版）

- [CONFIRMED; EXPERIMENTAL; FactKey=Mamba.Paper.Architecture20261008] 按用户给定英文/白底/淡色矢量风格重绘图2，分为检测网络、完整RGSP残差块和状态建模核心三层；对照方法正文、PFIM/DeSpeckleSSM/DS_VSSBlock及当前full配置静态核验。明确P3/8插入、多尺度颈部、两次残差/FFN、I仅接输入调制与步长、B/C来自调制前u、修正特征旁路与z门控。
- 产物 `output/pdf/rgsp_architecture_20261008/` 含原生矢量PDF、可编辑SVG、PNG、修订英文提示词、图注、provenance/verification及ZIP；生成入口 `output/figures/build_rgsp_architecture_20261008.py`。PDF为1页180×178mm、3款嵌入字体、零栅格图像，SHA256 `7d9af67c4088d9b77c4641965261544fefa755048ba9c9d2604dafb37faa5f72`；Poppler/MuPDF渲染与字体/边界静态核验通过。
- 仅独立图件，未替换或编译论文；动机图、旧架构图、正文及实验保持不变。接受后集成须将方法对旧图2(b)核心的引用改到(c)。LRC英文展开在摘要与方法仍不一致，未擅自统一全文；跨任务结构不可由检测示例外推。未运行模型、实验、服务或服务器操作。RunId：`rgsp-architecture-20261008-01a11a4a`。

## 详细入口

- [活动待办](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/TODO.md)；[来源与历史索引](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/MEMORY_HISTORY.md)。
- [决策](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/DECISION_LOG.md)；[实验](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/EXPERIMENT_LOG.md)；[冲突](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/docs/project-memory/CONFLICTS.md)。
- 旧 PROJECT_MEMORY 和迁移日志保留历史；旧"当前/进行中"段落不能覆盖本文件按事件时序整理的状态。
