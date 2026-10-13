# SAR-ACD 六类补充实验与后续检测队列

2026-10-13。正式项目状态仍在 `docs/project-memory/`；本目录保存本次代码、协议和交接证据。

## 已落实

- 官方仓库已在 C 用 git clone 下载；提交 `537639c406d5d37e52ba75fa433de52db7d1d483`。完整克隆连同文件哈希导出到 B/A，未安装软件。3032 张图片全部解码，六类；旧五类2522张副本缺A320/321的510张。
- 用户明确要求3032和2992两版都训练测试，并允许结束B的旧AIR空等队列。B释放回执、原计划和进程身份保存在 `evidence/handoff_b/legacy_air_release/`。原文件、旧57项seed0门槛及AIR三个位置保留，不因本次释放而放行。
- 两版数据准备及完整源校验通过，另有改标、重复图、篡改holdout指纹、冲突图移到test四项拒绝测试通过。
- 用户最终确认物理 **BS16 × 累积4 = 有效BS64**，模型采用官方框架明确实现；优先我们的Original/Matched075/Full。AdamW 2e-4、WD5e-4（bias/norm为0）、scratch、FP32、150轮上限、patience30、3轮预热+linear保持。
- B 的六项核心预检全部PASS，并已开始正式训练。初次完整epoch：clean2992 Matched075 seed0，960图/60microbatch/15updates，耗时5.895秒。这是运行进度，不是最终测试成绩。
- A已部署相同数据及运行代码，后置控制器已登记；B的后续三项检测基线也已登记，等待SAR-ACD队列结束再预检、通过后训练。
- 用户要求排队后不持续监测，等待其通知后统一回收、验收、补表。本任务未建立持续监测自动化。

## 数据协议

|版本|train|val|test|用途|
|---|---:|---:|---:|---|
|SAR-ACD-full3032|1000|240|1792|保留官方全部原标签，40张跨类冲突图仅进入训练|
|SAR-ACD-clean2992|960|240|1792|仅在清单中排除40张冲突图，原图不删除|

先对2992张无冲突图按固定RGB组划分，每类160张train、40张val，其余test；3032版仅加回冲突训练图。两版val/test逐项完全一致，holdout SHA `51da1dae01bcf11288d85d57514d347867581a9ee7959023a46eda94e0930400`。官方README没有指定统一训练BS或官方split，本划分是本地配对协议，不冒充作者划分。

full/clean每轮优化器更新数分别16/15，比较包含样本数量和更新次数差异；没有场景/飞机实例独立性证明。各方法共享版本内划分、输入224及配方。六类为A220、A320/321、A330、ARJ21、Boeing737、Boeing787。

## 当前调度与恢复入口

服务器共同任务根：`/home/powerleader/ig-ds-ssm/tasks/rgsp_table_expansion_20261013_v1`。

|卡/队列|内容|状态与次序|
|---|---|---|
|B / b0acd|Original、Matched075、Full × 两版；ResNet50 × 两版|核心6配置优先：全部seed0→seed1→seed2，再外部ResNet50|
|A / a0acd|AlexNet、GoogLeNet、VGG16无BN、ViT-B/16、Swin-T、Swin-B × 两版|等待原a0samplex自然完成后自动启动，共12配置|
|B / b0det|Faster R-CNN R50-FPN、RetinaNet R50-FPN、FCOS R50-FPN，SSDD|等待b0acd完成；GPU预检通过才进入正式训练|

SAR-ACD共 **20配置**，每配置按实测耗时最多3/2/1个种子（≤1h/1–5h/>5h）；并非全部已经开跑。各配置种子顺序执行。A/B各单GPU顺序队列，同卡并发未启用，亦未实测并发收益；20秒B训练采样中利用率21–82%，显存2452MiB，显存余量不等于相同比例的空闲算力。

`build_acd_plans.py`仅生成新计划，不应对已有计划重跑。B按用户“优先我们的方法”做过一次**仅改变stage**的审计迁移，当前生产计划SHA `2c1a3f1c5213ca158ff0ce0954ccf1af8b0aba94db3c83d8bcd4bb825dc436bd`。`lanes/b0acd/plan.frozen.json`保留最初计划，当前以 `plans/b0acd.production.json` 与 `reprioritize_ours_01/revision.json` 为准。所有已有预检继续使用；训练命令、代码、数据、种子未改。

启动回执见 `launch/b0acd.json`、`dispatch/a0acd.gate.launch.json`、`dispatch/b0det.gate.launch.json`。启动器保存attempt、不自动故障重试；不要因没有result就重复启动。B旧AIR任务恢复需重新核定数据及旧门槛，不能直接按历史PID恢复。

## 时长估计（非监测承诺）

- “14–15分钟”指一次完整150轮的预检外推，**不是每个epoch**。B六核心预检外推13.1–15.4分钟/次，正式首轮5.895秒，与秒级epoch一致。
- 从本轮正式训练开始算，核心三模型×两版的首轮seed0约 **1.5–2小时**；最多三个种子约 **4.5–6小时**。已计入验证、保存及队列等待余量，早停可缩短。
- 全20配置的最多计划种子暂估 **14–18小时**，包含A原SAMPLE队列余量。A外部模型尚未在这两版上预检，该估计使用既有SAMPLE实测按训练量缩放，可信度低于B核心估计；后续用户通知时可用实际结果修正。
- 上述不包括后置SSDD三基线；其GPU预检未运行，暂不给虚构完成时间。其他专用方法也不包含在20配置之内。

## 表格缺项处理

分类runner在正式训练结束后保存Accuracy、Macro-F1、Macro-Precision、Macro-Recall、逐类召回/混淆矩阵、参数量、覆盖算子FLOPs、模型延迟与吞吐。只从最终选择权重的独立test取数，单seed不猜标准差。

新增检测adapter使用TorchVision0.14.1的明确R50-FPN构造器、原生loss及冻结Ultralytics IoU匹配/AP统计，统一BS16×4；预检含真实训练更新、FP32保存/重载及训练图片推理。服务器的空标签、空预测、框坐标转换、重复预测唯一匹配、三个官方构造器和SSDD730图源检查已通过；**GPU前后向及最终性能仍待排队预检**。最终同时输出P/R/AP50/AP75/AP50:95、Params及带覆盖限制的FLOPs/延迟；ROI、NMS等未覆盖时不填假0。

这些结果将作为独立同协议组，不能补到历史文献成绩同一行去形成混合结果。VGGNet/ViT/ResNet等泛名称未确认规模时不冒认；BU-SIP、Light-ResKAN、MDB-DPL及部分SAMPLE/CD专用模型仍需准确作者代码/配置，原来源清单保留于 `external/*/source_registry.json`。本轮没有声称110候选全复现，也未修改已封存论文数字。

## 用户通知完成后的验收

1. 一次性读取各lane与真实进程，确认是否全部完成；失败attempt保留，先诊断再决定恢复。
2. 在生产端调用 `collect_evidence.py --snapshot <新的唯一名称>`，验证result/completion/protocol/checkpoint SHA，按需额外核对逐轮账本及source inventory。
3. 回收小文件；按版本、方法、相同seed汇总，保存负结果并比较3032/2992。最后补独立论文表格和正式项目记忆。

初次部署的源包在传输未结束时SHA检查失败，runtime当时已解出、source未解；随后完整包SHA通过并完成部署。所有失败现场及原归档保留。
