# 新协议跨任务 seed0：只读回收与论文取数

## 范围与来源

本目录仅为 2026-10-11 论文整合补充证据，不代表新训练、重新测试、全 campaign 完成或 seed1 放行。

- 来源聊天：`01a12053-e7f7-7d02-a7b7-c1f8b7edc44b`。
- 来源主机：已配置别名 `powerleader-a6000-a`，只读确认 hostname 为 `powerleader-PT6620W`。
- 来源根：`/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1`。
- 指标读取时间：2026-10-11 11:13:51 +08；源字节清单读取时间：11:15:50 +08。
- 冻结计划 canonical SHA256：`9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4`。
- 15项均为 2026-10-09 已完成的 seed0；本轮只读回收，不把读取日期当成新的实验完成日期。
- 本轮只在获准目录保存小 JSON/CSV。未读取凭据、下载权重、重跑评估/训练、运行远端验收器、启停服务器任务或修改正式记忆。

## 文件使用

- `SOURCE_FILE_MANIFEST.json`：95份原始小文件的来源相对路径、SHA256及字节数；与上方来源根组合为完整源路径。
- 每个 `*_seed0/` 目录：原样回收 result.json、test_evaluation.json、config.json、completion.json、该 trial 的外层 receipt、deep_completion_audit.json，共 15×6=90份。
- `data_audits/`：5份原样数据审计 JSON。
- `METRICS_READ_CAPTURE.json`：15项指标、最佳轮次、参数量、split、协议与源哈希的结构化提取；数字值经语义核对，不是源 JSON 的字节副本。
- `seed0_classification.csv`、`seed0_change_detection.csv`：可直接取数的完整指标，比例值均按源文件的 [0,1] 原值保存。Kappa 是系数；不要误标成百分率。百分点差应将原值之差乘100。
- `LOCAL_VERIFICATION.json`：95/95原件 SHA256及字节数一致、15/15提取与原始 metric/test 一致，config/completion/receipt、best SHA/epoch等已提供字段绑定通过。该验收不包含重新读取权重、训练逐轮 ledger 或本轮完整远端深验。

## 协议与划分

| 数据集 | train / val / test 记录数 | 输入 | 最大epoch / patience | batch | 选模指标 |
|---|---:|---:|---:|---:|---|
| SAR-ACD | 2003 / 251 / 254 | 224 | 200 / 40 | 64 | macro-F1 |
| SAMPLE | 1075 / 135 / 135 | 224 | 200 / 40 | 64 | macro-F1 |
| Ottawa | 112 / 24 / 32 | 192 | 150 / 25 | 16 | F1 |
| Farmland | 112 / 16 / 16 | 192 | 150 / 25 | 16 | F1 |
| Yellow River | 100 / 16 / 16 | 192 | 150 / 25 | 16 | F1 |

统一 scratch、AdamW lr=0.0002、weight_decay=0.0005、余弦衰减至1%学习率、FP32（AMP/TF32关闭）、梯度L2裁剪5、无模块warmup。完整 Mamba-YOLO-T backbone，匹配控制与Full仅替换内部64通道P3/8单块（state_dim=16），P4/P5仍依赖该块。分类从P5全局均值接LayerNorm256、Dropout0.2和Linear，label smoothing=0.1。变化检测采用共享Siamese骨干、P2–P5绝对差及width96 UPerNet-lite。

分类按验证macro-F1选checkpoint；CD阈值在验证集0.10–0.90、步长0.05网格选择，再冻结到test。CD先把概率恢复到原始crop尺寸、平均重叠位置概率，再按唯一原始留出像素计数。Ottawa/Farmland/YellowRiver test分别18432/9216/9216唯一像素；32/16/16是窗口记录数，不是场景数，更不能乘192²冒充评估分母。

## 主指标与差值

下表分类为 macro-F1，CD为正类F1；前三数列为百分数，末两列为百分点（pp）。

| 数据集 | Original | Matched075 | Full | Full−Original | Full−Matched075 |
|---|---:|---:|---:|---:|---:|
| SAR-ACD | 94.8153 | 92.0511 | 92.9481 | -1.8672 | +0.8970 |
| SAMPLE | 96.8128 | 98.3546 | 97.7039 | +0.8911 | -0.6506 |
| Ottawa | 96.4222 | 96.3069 | 96.7278 | +0.3056 | +0.4209 |
| Farmland | 0.0000 | 41.3043 | 42.3326 | +42.3326 | +1.0283 |
| Yellow River | 78.9728 | 69.1929 | 81.2690 | +2.2962 | +12.0761 |

分类 Accuracy（Original / Matched075 / Full）：
SAR-ACD 94.8819 / 92.1260 / 92.9134%；SAMPLE 97.0370 / 98.5185 / 97.7778%。

## 必须保留的限制与负结果

1. 五份协议均明确 `historical_results_comparable=false`。不得与9月VMamba输出适配器等旧跨任务表合并排名、覆盖旧数值，或把本批次当成旧协议复现。
2. 只有seed0，不支持跨seed稳定性或统计显著性。Full在SAR-ACD低于Original，在SAMPLE低于Matched075。SAMPLE的Accuracy差分别仅多/少正确1个test样本。
3. Farmland Original在验证选定阈值0.90下没有阳性预测（TP=FP=0、FN=365），F1=0但OA=96.0395%。precision=0为无阳性预测的零值约定，不能解释成“所有目标预测均错”。Full仍漏掉267/365正类像素，Recall仅26.8493%；相较Matched075仅多检出3个正类像素，F1+1.0283pp。保留退化结果，不将+42.3326pp独自归因于局部结构机制。
4. Yellow River Full相对Original少637个FP、多372个FN，Precision+30.0204pp、Recall−28.0543pp；F1提升是权衡，不是所有指标改善。Ottawa Full相对Original的Recall亦下降0.1793pp。
5. CD阈值（Original / Matched075 / Full）：Ottawa 0.80/0.65/0.65；Farmland 0.90/0.85/0.90；Yellow River 0.15/0.60/0.25。它们来自各自验证集，不是统一固定阈值下的纯结构比较。
6. SAR-ACD审计从原train2017排除14条记录，保留原val/test及其中3条具有历史类别冲突的留出标注（test2、val1）；没有替用户重标注。精确解码内容隔离不证明近重复、多视角、实例或场景独立。
7. SAMPLE只使用decibel/real；相同类别的实体编号跨train/val/test，不能宣称实例独立泛化。
8. 三个CD数据集均为同一场景内空间留出，相邻区域相关、最小跨split间隔0px、无缓冲区；不是unseen-scene泛化。训练/验证/测试crop不相交不等于场景独立。
9. 兼容恢复保留冻结参数，但经进程内torch1.13 UUID适配入口执行；不声称与原始入口逐位等价。所有历史失败现场、旧seed和负结果保持。
10. result中的参数量可引用，但没有本批次同硬件延迟或scan-aware FLOPs，不能从参数量推断效率胜出。
