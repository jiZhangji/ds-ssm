# SADD：RGSP 跨数据集受控实验

日期：2026-10-08，Asia/Shanghai。用户本轮明确要求在 SADD 上补充实验。

## 1. 实验目的与范围

本轮在 SADD 上检验 SSDD 之外的关键对照，不预设完整方法必然胜出。SADD 是补充检测数据集；不因此把其他尚未执行的数据集实验标成完成，也不改写原有主数据集范围。

预先固定四组、训练种子 0/1/2，共 12 个正式 run：

| 配置 | 含义 | 主要比较 |
|---|---|---|
| original | 原始 Mamba-YOLO-T | 完整系统是否优于原系统 |
| matched_scale075 | 匹配替换块结构，关闭局部修正、能量作用及学习式引导；固定 0.75 缩放 | 区分替换结构/缩放和功能机制 |
| local_energy | 开启局部修正与能量作用，关闭学习式引导 | 局部修正与能量作用的联合贡献 |
| full | 完整局部修正、能量作用、引导机制 | 在 local_energy 基础上增加引导是否有收益 |

Full−Original 是系统比较；LocalEnergy−Matched075 和 Full−LocalEnergy 是关键机制比较。四组并不足以估计局部修正、能量、引导的全部独立作用和高阶交互，不能冒充 SSDD 的完整 12 组矩阵。

## 2. 服务器盘点与隔离

2026-10-08 14:00 左右检查当前 SSH 配置中的项目关联目标：

| 目标 | GPU 状态 | 可用磁盘 | 选择 |
|---|---|---|---|
| A | 单 A6000，0%，无计算任务 | 281 GB | seed0 |
| B | 单 A6000，0%，无计算任务 | 655 GB | seed1 |
| D | 单 A6000，0%，无计算任务；本轮恢复可达 | 176 GB | seed2；已找到 SADD 原始副本和修订数据 |
| C | A6000 约 96%，SSDD 正在训练 | 854 GB | 保持原任务 |
| F | 双 A6000 约 98%–100%，SSDD 正在训练 | 278 GB | 保持原任务 |

A/B/D 均有已安装的 Python 3.10.16、PyTorch 1.13.1+cu117、CUDA 11.7 和 selective_scan_cuda_core；没有安装新软件、终止服务或在本机训练。

GPU UUID：A=`GPU-12cf4841-46b9-94d1-2715-9932134cf964`；B=`GPU-09b018e1-7140-296a-73d4-9e010ee209a2`；D=`GPU-f1b51ae3-c020-9901-c608-cca67f6ab8f1`。

所有输出写入独立 campaign：

`/home/powerleader/ig-ds-ssm/tasks/rgsp_sadd_controls_20261008_v1`

源码、数据副本、模型、缓存和日志与既有项目及 SSDD campaign 隔离。每个种子在指定服务器顺序执行四组；每组完成训练后独立评估，再进入下一组。异常时该队列停止，不改变超参数、不替换种子、不抢占其他任务。

## 3. 数据来源、审计与限制

D 的既有数据：

- 原始展开副本：`/home/powerleader/ig-ds-ssm/data/cross_task_prepared_v2/detection/_raw_sadd/SAR-aircraft`。
- 修订副本：`/home/powerleader/ig-ds-ssm/data/cross_task_prepared_v2/detection/SADD`。
- 本地证据：[数据审计](../experiments/rgsp_sadd_controls_20261008_v1/evidence/source_audit.json)。

这是单类飞机 SADD，不是七类 SAR-AIRcraft1.0。原始副本有 2,966 张图，883 张非空标注图、2,083 张空标注图，无缺失标签文件。本轮沿用已恢复的 positive-only 修订协议：

| 划分 | 图像 | 标注框 |
|---|---:|---:|
| train | 795 | 7,215 |
| val | 44 | 212 |
| test | 44 | 408 |

直接复算确认该划分与排序后的 883 张阳性图使用随机种子 0 进行 795/44/44 划分一致。三个训练种子共享这个固定划分，种子间差异不包含重新划分数据的波动。

审计检查了全部 883 张图和标注：

- 图像与原始副本字节一致；标签的类别和归一化坐标有效。
- 修订版与原始版的 883 个标签哈希不同，逐个确认仅为 CRLF→LF，无内容变更。第一版严格字节检查因此 FAIL；保留失败证据，修正审计解释后通过，未修改数据。
- 图像文件名、图像字节哈希、解码 RGB 像素哈希均无重复，也没有跨划分完全相同内容。
- 本轮复制数据到独立目录，原始文件不删、不改；保留原标签与边缘框，不擅自修改标注。

来源清单规范哈希：`06086fe32b9f97c70ed7adb4d83ee2aed3b38560b6b0d111eb74b05e745e38c4`。

限制必须随结果一起报告：

1. 这是恢复的本地图像级划分，不宣称已确认与公开论文的官方评估协议一致。
2. 精确内容去重不等于近重复或源场景独立，尚无原场景级泛化证明。
3. positive-only 测试不覆盖无飞机场景上的虚警表现；2,083 张空图保留，不在本轮偷偷加入或删除。
4. 测试集仅 44 张、验证集仅 44 张；三训练种子不能消除有限测试样本带来的不确定性。

## 4. 冻结训练与评估协议

| 项目 | 本轮设置 |
|---|---|
| 初始化 | scratch，无预训练权重，无模块 warmup |
| 模型 | T 规模；替换块位于 P3；状态维度 16 |
| 输入 | 640；三通道输入；单类 aircraft |
| 种子 | 0、1、2，配置内配对 |
| 训练预算 | 最大 200 epoch；patience=40 |
| batch | 物理 batch=16；nbs=64；保留框架预热/梯度累积规则 |
| 优化器 | AdamW；lr0=0.0002；lrf=0.01；weight_decay=0.0005 |
| 学习率 | 线性，cos_lr=false；优化器 warmup 3 epoch |
| 数值 | FP32；amp=false；deterministic=true |
| 数据加载 | 训练 workers=4；独立评估 workers=0 |
| 增强 | 与本轮 SSDD 控制共用显式训练入口；mosaic=0.3，最后15轮关闭；其余见冻结源码 |
| 初始化审计 | 统一清除 stride 探针对 BN running statistics 的影响；保存初始张量哈希 |
| 关闭参数 | buffer，不进入优化器；训练逐轮与 checkpoint 重载后核验 |
| 选模 | 验证 fitness=0.1×AP50+0.9×AP50:95 的 best.pt |
| 最终测试 | 固定 test44；conf=0.001；NMS IoU=0.6；batch16；FP32 |
| 训练验证 | NMS IoU=0.7，不隐瞒其与最终测试的差异 |
| 失败策略 | 非有限 loss/控制漂移/子进程失败/哈希不匹配即停止；不自动改配置或续跑 |

200轮/patience40 延续 SADD 历史 scratch 预算；batch16 与本轮机制研究统一。它不是旧 batch32 单次结果的严格复现，必须独立报告，不能与旧 55.018/54.597 或 partial-transfer 72.113 混算。

源码包：[code_frozen.tar.gz](../experiments/rgsp_sadd_controls_20261008_v1/code_frozen.tar.gz)，SHA256=`a957bf2cfd2be3adeef4b8f692cbe74f0bfd49ea76d23b9ea7fdc789b61fe388`。

数据传输包 SHA256=`56225128000072ba4a9abb9e4d7a4a065d0b57fc5fb7f2b96c2b206d978b9479`。原始数据文件内容清单与每次运行的源代码/配置哈希另见各 lane 的 `manifest`。

## 5. 验证与启动状态

- A/B/D 的 Python 编译检查与完整 CUDA 控制验证已通过，包括数值/梯度、关闭参数、12配置解析与共同初始化检查。
- D 四组各两轮短训练、checkpoint 重载、独立验证集评估均完成；A 的 Original、B 的 Full 也独立通过。短训练仅用 train32/val16，不访问测试集，不进入正式成绩表。
- D 三种匹配控制的实际训练初始化共同张量哈希一致（除预先规定的 lambda/alpha 开关），差异数为0。四组的关闭参数、有限损失与重载审计通过。
- 三条正式队列均已启动，GPU约96%–100%利用率，训练进程约15.4GB显存。启动时点身份如下：

| 主机 | seed | 北京时间启动 | 队列PID | 首个Original训练PID |
|---|---:|---|---:|---:|
| A | 0 | 2026-10-08 14:16:33 | 1959131 | 1959134 |
| B | 1 | 2026-10-08 14:16:32 | 1917370 | 1917373 |
| D | 2 | 2026-10-08 14:11:36 | 2217783 | 2217786 |

PID只是时点证据。每次续接核对任务路径、实际命令和status，不按旧PID操作。初始证据存于 `experiments/rgsp_sadd_controls_20261008_v1/evidence/{a,b,d}/startup_evidence.tar.gz`；D的完整预检包为同目录 `preflight_evidence.tar.gz`。

D在14:14完成Original epoch5，loss有限。A/B初次启动快照处于第一轮，不把进程启动当作任何正式配置已完成。

后续启动复核：A/B均完成Original epoch2，D完成epoch12，loss均有限，GPU约98%–100%。三条lane的251项代码/配置来源哈希与本地源码逐项一致，数据清单SHA256均为 `23077fb38483e161a3b7675b1d0629b19cbe334482588a0bdebf7eba88372e10`。D直连偶发超时，已使用SSH配置中的 `powerleader-a6000-d-lan` 经A访问，核对hostname和GPU UUID一致；确认训练仍正常推进，未重启训练。

按D最初约30余秒/epoch的Original吞吐，四配置顺序执行可粗估整轮约6–8小时；这是初步资源估算，不是已测得其余配置耗时，也不保证早停、共享负载或故障情况下的完成时间。

## 6. 正式结果与论文解释

| 配置 | AP50 均值±标准差 | AP75 均值±标准差 | AP50:95 均值±标准差 | 状态 |
|---|---|---|---|---|
| Original | 待完成 | 待完成 | 待完成 | 不预填数值 |
| Matched075 | 待完成 | 待完成 | 待完成 | 不预填数值 |
| LocalEnergy | 待完成 | 待完成 | 待完成 | 不预填数值 |
| Full | 待完成 | 待完成 | 待完成 | 不预填数值 |

收齐12个正式测试后，应同时报告每个种子的原值、样本标准差，以及 Full−Original、LocalEnergy−Matched075、Full−LocalEnergy 的逐种子配对差值。均值±标准差不是置信区间；3个种子不支持仅凭均值差宣称统计显著。

若 Full 没有提升或没有超过 LocalEnergy，应保留结果并讨论适用边界，不删除负种子、不事后挑选最佳配置来代替预先定义的完整方法。不同数据集上的正负差异只能支持进一步的机制分析，不直接证明“散斑传播被抑制”。

## 7. 续接入口

每台主机：`seedN/status.json`、`seedN/runs/<variant>/results.csv`、`controls_by_epoch.jsonl`、`train_complete.json`、`eval_test.json`、`evaluation_complete.json` 和 `results_partial.json`。

先确认实际进程、status和证据，不重复启动队列。`smoke/` 或 `smoke_*` 的任何数值都不是正式论文结果。未建立周期监控或开机自启动；有限队列脱离SSH运行，不依赖当前聊天持续打开。
