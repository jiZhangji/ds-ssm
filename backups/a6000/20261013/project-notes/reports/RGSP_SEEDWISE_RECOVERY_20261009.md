# RGSP 分轮消融续接记录 — 2026-10-09

工作目录：`D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`。本报告记录已实施操作和时点证据，不是第二套正式项目记忆，也不表示全部实验完成。

## 授权、范围与冻结边界

- 本次用户要求继续聊天 `01a11b5e-55ff-76e0-8f24-bd58d9e126bc`（“astra-消融实验修改2”）未完成任务；沿用其最后明确要求：所有配置先跑第一个 seed，全部完成后才跑第二个，不再新增第三个。
- 当前独立 campaign：`experiments/rgsp_seedwise_20261009_v1/`；远端任务根统一为 `/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1`（下称 TASK）。已有 Python 为 `/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python`。
- 冻结计划共114位置：57个seed0、57个seed1。canonical SHA256 `9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4`；文件SHA256 `37e1e2254643541118e4bab29d11cb93f1483847fc73aa69be8d63be4907cbe0`；360项冻结源码。
- 不改冻结计划/业务源码/原始数据/旧结果；旧失败现场、负结果、partial文件与已有脏工作区均保留。所有模型预检、单测、训练和数据审计在服务器上执行；本机仅编辑独立操作脚本、传输与整理证据。没有安装、启停既有服务、抢占任务或启动本机4050服务。

## 服务器与生产主机回执复查

19:04–19:05已检查A/B/C/D/F。19:31–19:35再次检查并核对身份、GPU UUID、CPU负载、磁盘与已有进程：

| 目标 | 最近现场情况 | 本轮选择 |
|---|---|---|
| A / `powerleader-a6000-a` | 原队列已失败；约237GiB空闲；恢复前GPU无计算进程 | 原计划指定的15项跨任务seed0独立恢复 |
| B / `powerleader-a6000-b` | GPU100%，约622GiB空闲；本计划训练中 | 保持既有队列，不插队 |
| C / `powerleader-a6000-c` | GPU无计算进程，约788GiB空闲 | 保持Ship Full等待进程；完成数据准备后自然续行 |
| D / `powerleader-a6000-d` | GPU无计算进程，约134GiB空闲 | 冻结计划未分配，不擅自迁移/重复实验 |
| F / `powerleader-a6000-f` | 两卡无计算进程，约216GiB空闲 | 保持Ship Original/Matched075等待进程 |

网络状态会变化；已配置直连别名偶发超时，可对C/F已登记LAN目标临时使用 `-J powerleader-a6000-b`。只改变单次连接路由，未改SSH配置、密钥或服务器账户。B跳板1MiB传输仅约9.0KiB/s，未视为吞吐优化成功。

19:33五份**完整产地主机** `verify_receipts --require-complete` 输出均已回收，并逐份与产地主机SHA256相等：

| 生产主机 | 已核验 / seed0位置 | 缺失 |
|---|---:|---:|
| A | 5 / 22 | 17 |
| B | 6 / 27 | 21 |
| C | 1 / 2 | 1 |
| F0 | 3 / 4 | 1 |
| F1 | 1 / 2 | 1 |
| 合计 | **16 / 57** | **41** |

0 invalid；全部exit2是未完成的正常结果，不是全局通过。B新增 `sadd_post_gate_seed0` 完成；该时点 `ssdd_post_gate_seed0` epoch51且损失有限。五条新seed1 lane无状态/回执。尚无完成的cross回执，不能把外层两文件检查冒充内部六项验证。此表早于A恢复首轮和后续Ship准备，后续变化另据最新记录。

完整证据：`experiments/rgsp_seedwise_20261009_v1/evidence/refresh_20261009_1931/`。其中 `CAPTURE_RECEIPT.json` SHA256 `3b7f724b8f0371cf72a33ac4cd376687e3000c3d0812d7f135ba43253e0971ae`；`VERIFICATION_REPORT.md` SHA256 `6869b6ad5864d893b2ce92ef8496ee8c7fd236cfc50be57cd9765ce692300283`。

## A 的已实施恢复

原因已核实：PyTorch1.13在UUID mask下原生CUDA可见1卡，而 `torch.cuda.device_count()` 为0；原冻结cross入口未调用包内已有兼容入口，在首个epoch前失败。失败run仅有config，无checkpoint/metrics。

新增独立 `recovery_20261009_1908/operator_cross_recover.py`（SHA256 `796528f8854ab0c246bdf2e7b19edd7cee11e3e2913d5731891a6735fe01e806`），经已冻结 `cuda_uuid_compat.py` 在同进程调用原cross脚本；原计划全部参数序列保持。记录原计划/实际argv与哈希，不宣称可执行入口逐字相同或逐位等价。

- A上13项stdlib单测通过；分类/CD的Original、Matched075、Full三臂forward/backward/重载探针全部通过，仅读训练样本。
- 19:34:02 detached恢复队列启动，总15项中第1项运行、后续14项串行。19:35:41 `sar_acd_original_seed0` 真实epoch1→9，loss `1.5583444089→1.0610277963`，首步梯度有限且连接。
- 时点operator PID162471/child162560，实际 `/proc` argv与契约全等；3把原调度/队列/GPU锁持有并由child继承，指定UUID上的唯一计算进程。以后必须重新核身份，不能按旧PID操作。
- 360源码、计划、5份旧receipt、失败状态/日志/原config哈希均未变。没有AIR执行、原lane_complete、全局barrier或seed1操作。A上两个AIR位置继续保留；全局AIR共三个位置。
- 每项完成后递归验证completion内部六项产物、best/test/result一致性、连续epoch和有限指标，才发布原receipt契约。队列失败即停止，不自动重新训练任意partial。

续接以 `TASK/recovery_20261009_1908/trials_status.json` 为准；原dispatcher/lane的failed故意保留。详细报告与本地证据：`experiments/rgsp_seedwise_20261009_v1/recovery_20261009_1908/RECOVERY_STATUS_20261009.md`；`evidence_20261009_1935/first_epoch_evidence.json` SHA256 `2e861e8343951ded94194e1913e3f54ebe964f308aa279feb220d07be9d8b0b2`。34项轻量证据包SHA256 `c6a39ddcfb754dd832ca4b7992272bc56f0d26b5a7b8501277781302f5598600`。

### 20:07 A/B 增量与首项跨任务完成

- 19:58复核仍为A5/B6，无新增；20:07:46–48完整产地输出更新为A6/22、B6/27，0 invalid。新完成 `sar_acd_original_seed0` 的回执时间为20:01:09。
- 独立只读审计重新读取并哈希其六个内部产物，核对best↔test↔result、200个连续epoch及有限值、最佳epoch164；另核last.pt、operator契约/兼容记录/源码。全部通过，不仅依赖operator的深验断言。Test Accuracy94.8819%、macro-F1 94.8153%；仅单臂seed0，不作性能提升或跨seed稳定性结论。
- 20:07 A自然进入 `sar_acd_matched_scale075_seed0` epoch52/200，B `ssdd_post_gate_seed0` epoch131，损失有限；A/B seed1仍未开始。与19:33的C/F证据合并为17/57，不是全主机同刻新扫描。
- 证据 `evidence/refresh_20261009_2006/INCREMENT_REPORT.md` SHA256 `636b1443d1daaf79ede28b5a9464508ff5492844886b39adbc50c2a83a763a7a`；A独立文件审计 `producer_a/a_file_audit.json` SHA256 `dd85636d25074743a2f340a88a75186a12cf594e35de43b4f265ebf01e53a4ea`。19:58证据保留在 `evidence/refresh_20261009_1955/`。

## Ship 数据恢复与门禁

- 原始固定ZIP为426,990,175B，SHA256 `6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e`；source snapshot为5,795,637B，SHA256 `aea74cd1d3a55b750004b72249a8896c13a7751025b5398d8eb2e37098bb9b69`，C/F均已核验。
- 本机→F旧传输中断，26,195,968B partial保留；本机→C旧传输缓慢，不能把增长中的文件当作完整输入。C原606,208B旧partial也保留。
- 官方Hugging Face连接不可达；公开mirror元数据API返回403，但**公开文件地址**HEAD为200且大小正确。独立 `download_ship_mirror.py` 使用原始完整SHA256作为信任依据，不读取/输出签名跳转URL或凭据，不依据缺失的mirror元数据认定来源一致。
- F于19:41:53完整下载并校验通过，用时113.34秒；再次外部 `sha256sum` 相同。文件 `TASK/external_inputs/raw_archives/ship_dataset_v0.zip.mirror_20261009_1939_f`。C于19:44:20完成相同字节目标下载，用时98.99秒，文件尾缀 `mirror_20261009_1943_c`。两机启动审计前均重新读取完整文件核SHA；仅在canonical archive路径不存在时建立同文件系统硬链接，未覆盖partial。
- `ship_gate.py` SHA256 `1bb625fb0a8262f27ecaced01b25177b3775f4de6dc42684446bc2756a083c29` 已在C/F独立部署；原auditor/stager以pin版本在原路径不存在时新增，没有替换已有文件。C上11项stdlib测试通过，真实冻结context/接口与原子no-clobber原语核验通过；结果 `server_stdlib_c_01_result.json` SHA256 `1fffab06f70b08415072fac646ae5a5752b60b02e5d4194fc625079a0f4525f9`。独立静态复审未发现确定P1/P2。这些不是39729图完整审计通过。
- 上述v1部署时完整图像审计、reviewed report、stage、相同selected manifest及最后原子ready尚待核实；后续实际结果见下节。原始test3974保持；按固定test>val>train隔离跨split byte/RGB重复；任何test内部语义冲突停止。不得改标签、暗改split或据“文件已到”直接ready。

### 19:47–19:55 全量实审与已识别元数据边界

- F完整审计于19:47完成，报告 `TASK/external_inputs/audits/ship_recovery_20261009_1945_f/archive_audit.json`，SHA256 `9092777ec2a04caaea06420cc5df715bf703cc3f1e4ec5538cbdff65edbf763d`；C于19:49完成，路径对应 `ship_recovery_20261009_1949_c/archive_audit.json`，SHA256 `a4b42255677feac2173652ec870130a89c647c57e677eb98613551c11c4794ac`。完整报告已回收到 `recovery_ship_20261009_1910/evidence_actual_20261009_1950/{f,c}_archive_audit.json`，SHA逐份相同。
- 全79,459 ZIP成员CRC通过，39,729图像/标签来源匹配，0 mismatch；均256×256，其中14,814灰度、24,915 RGB。无重复标签行、无无效/越界框、方向均1。Pillow12.3.0/libjpeg6.2；两机counts/dimensions/label_stats/decoder/group_audit、byte/pixel重复报告及完整重复组文件SHA一致。
- byte与RGB像素均143个重复组、50跨split组；test内没有重复组或语义冲突。固定test>val>train规则排除train48/val2，预期保留31,735/3,970/3,974。训练/验证同split重复及不同标注不自动裁决，不能称完整无噪声、近重复或场景独立基准。
- v1门禁在F实际 `--validate-only` 返回2，错误为 `Unparsed source filename groups require review`；`readiness`仍不存在。完整失败调用/输出保留于 `TASK/recovery_ship_20261009_1910/evidence/v1_f_real_01`。这是保守元数据前提不符，不是绕开数据失败后发布。
- 经固定源清单SHA和全部source_group重算核对，969个未知分组行（train773/val92/test104）与实审unparsed列表有序完全一致，名称无重复；为既有newship数字族727与ship数字族242。原源/档案审计器的相同Gao/Sen正则不会解析这两族。未知scene proxy不是来源字节不匹配，也不是新划分；保留null，不能猜测场景。
- 已单独允许v2门禁只接受该固定源SHA下的精确未知名单、顺序、计数及集合SHA，其他未知仍拒绝；原v1、原auditor/stager、源/标签/split/plan保持。不因元数据缺口删除969样本。v2实测及后续发布还需独立验收，不能用静态复审替代。
- C本轮冗余慢上传SSH PID55940已在核实创建时间/精确目的路径、且独立完整ZIP二次SHA通过后终止；旧partial保留。未中断任何模型任务、其他进程或服务。

### 20:03–20:16 v2 验证、独立副本与发布

- v2门禁SHA256 `caabe5022a1d53d94747c0091f9899413b5a86b1fac21e700a343baed3be4806`，harness SHA256 `be41278d4d632d9a0623362edc667d1c19f8a2dd78c1c79c6144731fa60adeb7`。C/F无覆盖部署及语法检查通过；C 18/18 stdlib测试通过，1.810秒，完整2853B结果已回收SHA相等 `f081ae0befbde801486f2668a48592348a4f70c32b9bfed5e512c6b07c22fe8b`。独立静态复审无确定P1/P2；不把单测当完整数据验收。
- F于20:08:13、C于20:08:15 `--validate-only` 成功，完整gate输出、returncode0及空stderr已读。两机预测manifest相同：`2b30f37c7eaa9d5118354736959e64035841a3219ea3fd44963ae52ee6fbda78`；均保留31735/3970/3974，未知969 proxy原样为null，test无排除。
- 两机仅在阶段目的目录不存在时生成独立副本；F于20:10:51、C于20:12:31完成深验且当时ready仍不存在。实际manifest、descriptor、YAML、exclusions四项SHA跨机相同，原test名字及图片/标签字节不变；source_audit含产地主机/时间/源审计路径，因此两机哈希不同是预期的来源差异。
- 选定manifest为上述 `2b30f37c…fbda78`（20,455,551B）；descriptor SHA `c14a5295a56a23ccedb9edc3548e315fdafd3f89fea351274d4485131caaa3c5`，YAML SHA `d41830bc75db60353c19250a857815af787d40c9d5c8e6da7e2662c89d700386`，exclusions SHA `4177e9880749c0144941ffdf9aee7da0c407c973ef038c38521da59e581b2d8e`。
- root先读完整stage_gate_complete及stage_complete并核对引用，再单独执行发布。C reviewed stage SHA `d15725872cd253f23f00f0d143633f4b2fa35d25b94235beff24356a6fed0ebc`；F为 `a4e689cc4e5f7c18f9863d431fd4a87ce841cb51276cc140de71d3021fde1512`。两机20:16:08–10经已配置直连成功启动独立发布进程；此前B跳板banner超时发生在命令送入shell前，不是数据/训练失败，也未重复创建stage或ready。
- F于20:16:21、C于20:16:23发布成功，returncode均0；后续只读复核两机live/candidate实际samefile，均为原子no-clobber硬链接、无覆盖。F ready SHA `18fee34bcdb8e790114bfedffb79ac18619116a2f2f41e6168b65a72ce5e8c31`，publication_complete SHA `a661ea09250aa1c0650134d0f21005d5275d8819ca2b9aa89e480fdf5dcf2a67`；C ready SHA `ee7ac8c43f233355858b8e21a5a46566a272f16de5ac38d8226b39f97aa424c5`，publication_complete SHA `e0ef557df67d4076dbafdce4e51071a7f3f4900612e8da723c52fc36c8f62bb7`。既有等待队列的真实训练状态另行核验，不由PID/ready推断训练成功。
- C/F各36项完整小记录已回收到 `recovery_ship_20261009_1910/evidence_publish_20261009_2017/{c,f}/records/`，两份压缩包、生产者manifest及全部72项逐文件SHA/大小均与产地一致。C包SHA `d12760c261bdf8ea944e671da366ef8e2d20043b38c851d6cded481503aa1192`，F包SHA `30d2e54b4571d2f7b125c9e5f6cadaceb8f083e94876abee769f73cfcc05a6f3`。候选由未改冻结resolver验过完整图/标注清单、nc=1、names={0:ship}；候选验证的“not_published”是commit前时点，不与后续published完成记录冲突。
- 完整小记录位于 `TASK/recovery_ship_20261009_1910/evidence/v2_{validate,stage,publish}_{c,f}_01*`；大manifest/图像/源审计rows保留产地。旧v1拒绝、原auditor/stager、源ZIP/标签/划分及partial均保留。

### 20:24:57 三条 Ship 等待进程自然续行

两机完整启动证据与三份生产端seed0核验输出已回收，整包/清单/逐文件SHA相同，位于 `evidence/refresh_20261009_2020/producer_{c,f}/`。目录名是请求捕获时段，实际现场时间为20:24:57。该目录 `INCREMENT_REPORT.md` SHA `1ffda090f626a2e80d77b6124c8abac7c5b827bad0f3412afe35de28518c0cb8`，`CAPTURE_RECEIPT.json` SHA `541d47c9eb7a45d3b0deb95d9916b15cfe526ca29ffcb59b720167f9c4fb0828`。

| 生产lane / 配置 | 真实trainer PID / start ticks（仅时点） | 第1轮日志batch / 1984 | 日志三项loss |
|---|---|---:|---|
| C / Ship Full seed0 | 3601183 / 16647878 | 855 | 2.665 / 2.041 / 2.664 |
| F0 / Ship Original seed0 | 127524 / 7380771 | 822 | 2.653 / 2.055 / 2.615 |
| F1 / Ship Matched075 seed0 | 128688 / 7792826 | 909 | 2.669 / 2.071 / 2.724 |

- 每条恰一个精确planned argv的trainer，旧dispatcher→lane→trainer父子链与进程身份一致，均早于20:16发布已存在，说明是原等待者续行，不是重复启动。GPU compute PID和指定物理UUID一致，无其他计算进程抢占。
- runtime_audit及provenance存在；nc=1、检测头、数据/模型classes、CUDA计数和UUID、选定manifest、ready/descriptor/source-audit哈希绑定均通过。日志batch loss有限，所查尾部无错误/退出码。
- 三者均为**epoch1进行中，controls_by_epoch完整行0**；不能写已完成epoch1或已完成Ship实验。C/F完整receipt仍为C1/2、F0 3/4、F1 1/2，无新Ship完成回执；与20:07 A/B验收合计17/57，仍明确为分时快照。

## 全局门槛与未决问题

- AIR原始测试集全范围有873个byte重复组、1892张test图，其中797组语义标注冲突；不是本轮类别映射/数字格式错误。依据 `evidence/external_data/AIR_ARCHIVE_DUPLICATE_PROTOCOL_20261009.md` 与 `AIR_FULL_TEST_SCOPE_ADDENDUM_20261009.md`。未解决来源标注前，三个AIR seed0位置保持阻塞，需要用户确认权威修正版或另行决定协议，不由操作器裁决标签。
- 即使其余54项先完成，也不缩小57项门槛、不开始seed1、不新增seed2。所有结果和负结果保留，单seed不提供跨seed稳定性结论。
- `collect_global.py` 与冻结 `manifest.frozen` 的原live-path写入存在读半写JSON的竞态。最终必须使用单一指定协调者、完整产地快照与独立审核的候选验证→fsync→同FS原子no-clobber发布；当前未发布任何barrier。A seed1仍需要另审兼容入口，不能直接复启已失败原dispatcher。
- 新增独立静态复核 `recovery_barrier_20261009_static/INDEPENDENT_BARRIER_AND_SEED1_REVIEW_20261009.md`，SHA `1955268029da971a2b1a2cb34cb57dc2f2f532cd8f6ba20075380002491eecd6`。风险是reader可能失败关闭，不是半份JSON放行seed1；修复尚为设计，未实现/测试/部署。不能改plan.barrier_path来生成候选（会变更plan digest）。A完整seed1 lane共19项，15个cross子进程须分别走冻结UUID入口，且在启动兼容子进程之前验证全局barrier；现seed0-only 15项恢复器不可直接用于seed1。三个AIR为deferred command，not_ready_ids=[]不表示数据就绪或全57完成。
- 已建立本聊天30分钟自动跟进 `rgsp`（“跟进 RGSP 分轮消融”），仅在重要完成、失败或需要用户处理时通知；普通epoch进度保持安静。应用/电脑需保持运行；不是服务器重启自启动服务。

后续以正式 `docs/project-memory/PROJECT_STATE.md` 的本轮报告指针、真实服务器状态和完整回执为准；计划、日志存在或PID存活均不等于实验完成。

## 21:13–21:24 只读现场复核与回执更新

- 按配置中的A/B/C/F别名复核主机身份、负载、磁盘、GPU UUID和真实计算进程。四台均与冻结计划匹配，剩余磁盘约235/622/786/213 GiB。A独立兼容operator仍运行；21:18:51时已完成9个cross seed0，当前 `farmland_original_seed0`，PID/start ticks、PPID、计划argv与GPU进程一致。原A dispatch/lane失败状态保留，不按旧PID操作。B的seed0队列及C/F三条Ship seed0原等待队列仍运行。
- 完整生产端receipt stdout已回收并逐lane解析：A 14/22、B 7/27、C 1/2、F0 3/4、F1 1/2，分时合计26/57 verified、31 missing、0 invalid；这不是同一时刻全机扫描。A的9个cross回执分别递归重验了completion六项、best/test/result/protocol和连续epoch，9/9 `deep_verified`。详细SHA与命令回执见 `experiments/rgsp_seedwise_20261009_v1/evidence/refresh_20261009_2115/INCREMENT_REPORT.md`。
- C/F readiness仍绑定计划SHA与manifest `2b30f37c…fbda78`，计数31,735/3,970/3,974；Ship仍训练中，未产生新完整回执。21:24逐台核对四台均无global barrier，A/B/C/F0/F1均无seed1状态或receipt。
- AIR原测试冲突仍阻塞三个seed0槽；未改标签、排除槽位、缩小57项门槛或启动seed1。没有修改冻结计划/源码/数据/配置，没有安装、更改服务或本机运行模型。

## 21:31–21:38 只读现场复核与 A 恢复队列完成

- 新一轮完整产地回执为 A20/22、B7/27、C1/2、F0 3/4、F1 1/2，分时合计32/57 verified、25 missing、0 invalid。A恢复队列15个预定cross seed0全部完成；新完成的Farmland与Yellow River六项递归重验内部completion六项、protocol、连续epoch及best/test/result绑定，累计15/15 deep_verified。
- B `sadd_original_seed0` 最新完整有限epoch行为151；C Ship Full为4，F0 Original为3，F1 Matched075为4。B/C/F dispatcher、lane、planned trainer父子/argv及assigned GPU compute PID现场身份复核匹配；A operator状态complete且owner/child已退出，A卡空闲。训练进程保留运行，无干预。
- 21:35四台均无 `seed0_barrier.json`；A/B/C/F0/F1 seed1 lane均无status与receipt。AIR三位置继续因原test标注冲突阻塞；没有改标签、移除槽位、缩小57项门槛或开始seed1。
- 完整输出、SHA、六项深验明细及资源/进程/门槛证据：`experiments/rgsp_seedwise_20261009_v1/evidence/refresh_20261009_2130/INCREMENT_REPORT.md`（SHA256以该文件本身为准）。本轮只读检查，没有改冻结计划、业务源码、数据/配置、安装、服务或本机模型运行。
