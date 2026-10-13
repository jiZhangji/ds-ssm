# RGSP seed0 已有结果审查 — 2026-10-10

这是本次只读结果审查的时点报告，不是第二套项目当前状态，也不替代正式记忆。来源聊天为 [astra-消融实验修改2-2-2](codex://threads/01a120c7-6ce1-7ec1-b85d-4fa82fae0529)。没有启动、停止或重跑实验，没有改计划、源码、数据、论文、服务或正式记忆。

## 1. 结论先行

截至北京时间 **2026-10-10 10:07:54–10:07:57**，五份完整产地主机回执快照合计 **44/57 verified、13 missing、0 invalid**，较10月9日21:31–21:32的32/57新增12项。此为分时回执核验，不是同刻全机扫描，也不等于57项实验已全部完成。

已回收可直接分析的检测测试数值为 **SSDD 20项 + SADD 9项**。全部是本计划seed0位置所绑定的结果，保留明确复用/历史采纳来源；不是29次全新、无中断训练。

主要发现以 test AP50:95 为准，单位为百分数，差值单位为百分点：

- **Full 低于 Original**：SSDD 42.6881 vs 43.8112（-1.1232）；SADD 49.0030 vs 54.0872（-5.0842）。
- Full 相对同壳层 Matched075 在SSDD为 **+1.3369**，在SADD为 **-3.0884**；不支持统一、稳定的增益结论。
- **Post-gate 是目前已完成配置中的最高 AP50:95**：SSDD 46.6156、SADD 54.2821。相对Original为 +2.8043 / +0.1949。这是单seed、多配置比较中的暂时排序，不能据此宣称显著提升、泛化获益或事后将其当成预先固定的最终方法。
- 正确组合级四格为 Full−LocalEnergy−GuidanceOnly+Matched075，SSDD为 **+0.2082**，不是“显著负交互”。其余因素分解有负的单seed差分，见第4节；不同四格不可混为一个结论。
- 15项跨任务结果已有完成记录及10月9日的递归深验；本次外层回执仍有效，但**14项最终测试数值尚未回收**，不能填写或推断五数据集三臂优劣表。

## 2. 检测结果与协议

SSDD固定清理后730/186/232，150 epochs / patience30；SADD固定positive-only 795/44/44，200 / patience40。均为640、batch16、scratch、FP32、AdamW 2e-4、线性学习率、nbs64、优化器warmup3轮/模块warmup0轮，val fitness选best（0.1×AP50+0.9×AP50:95），独立test conf0.001 / NMS IoU0.6。

SSDD manifest SHA256：`cd2f5314ecc8139f8e51a8c283d309f522c12abd58320003ea86bacbaac853de`。  
SADD manifest SHA256：`23077fb38483e161a3b7675b1d0629b19cbe334482588a0bdebf7eba88372e10`。

所有表格为seed0、test、百分数。行中的训练轮数来自train_complete.last_epoch_observed，不是best epoch。P/R及未四舍五入原值、trial ID、来源路径和SHA在[结构化结果](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/results_seed0.json)及四份生产主机evaluation_read中。

### SSDD：20/27项

| 配置 | AP50 | AP75 | AP50:95 | 已观察训练轮数 | 来源 |
|---|---:|---:|---:|---:|---|
| Original | 86.0842 | 38.1961 | 43.8112 | 150 | F0 · 复用 |
| Matched identity | 85.6672 | 29.5149 | 40.7174 | 150 | F0 · 复用 |
| Matched075 | 87.2861 | 29.6893 | 41.3512 | 150 | B · 本计划训练 |
| LocalEnergy（R+E） | 85.2910 | 26.4083 | 39.3382 | 150 | F0 · 历史采纳 |
| Full（P3，d16，3/3） | 87.7410 | 32.7394 | 42.6881 | 150 | B · 本计划训练 |
| Full no-LRC | 87.9631 | 38.1854 | 44.3636 | 150 | B · 本计划训练 |
| Constant guidance | 85.6757 | 32.2823 | 40.4470 | 150 | B · 本计划训练 |
| Post-gate | 88.8405 | 42.7314 | 46.6156 | 150 | B · 本计划训练 |
| LocalEnergy + I→U | 87.4954 | 39.8365 | 45.2636 | 150 | B · 本计划训练 |
| LocalEnergy + I→Δ | 87.5728 | 37.9498 | 43.4327 | 150 | B · 本计划训练 |
| Full fixed half | 86.6322 | 33.9181 | 42.3787 | 150 | B · 本计划训练 |
| Residual scale075（R） | 85.5415 | 31.9169 | 41.0612 | 150 | B · 本计划训练 |
| Energy only（E） | 87.5590 | 42.8282 | 45.4908 | 150 | B · 本计划训练 |
| Guidance only（I） | 86.6723 | 39.6177 | 44.4928 | 150 | B · 本计划训练 |
| Global | 87.4519 | 35.4908 | 43.3001 | 150 | B · 本计划训练 |
| LSGM trend only | 85.9367 | 35.7006 | 41.9561 | 150 | B · 本计划训练 |
| LSGM std only | 86.2898 | 42.6429 | 45.1484 | 150 | B · 本计划训练 |
| LSGM point only | 86.2564 | 28.3731 | 40.1865 | 150 | B · 本计划训练 |
| Full P2 | 82.8971 | 29.4908 | 38.8203 | 105 | C · 本计划训练 |
| Full P4 | 85.8254 | 35.8832 | 43.1492 | 150 | F1 · 本计划训练 |

七项尚未有完整回执：Full d8、Full d32、Full trend5、Full stats5、Full trend7、Full stats7、CNN YOLOv8n。不从部分epoch填入测试成绩。

### SADD：9/9项

| 配置 | AP50 | AP75 | AP50:95 | 已观察训练轮数 | 来源 |
|---|---:|---:|---:|---:|---|
| Original | 98.5262 | 53.7659 | 54.0872 | 200 | B · 本计划训练 |
| Matched identity | 98.6194 | 52.5264 | 52.6018 | 191 | A · 本计划训练 |
| Matched075 | 97.5739 | 49.5962 | 52.0914 | 200 | A · 复用 |
| LocalEnergy（R+E） | 97.5941 | 48.1029 | 51.9479 | 168 | A · 复用 |
| Full（P3，d16，3/3） | 97.0944 | 43.5558 | 49.0030 | 156 | A · 复用 |
| Full no-LRC | 97.9780 | 48.4441 | 52.4506 | 191 | B · 本计划训练 |
| Constant guidance | 96.8067 | 51.3325 | 52.2110 | 191 | B · 本计划训练 |
| Post-gate | 98.3196 | 58.1027 | 54.2821 | 200 | B · 本计划训练 |
| CNN YOLOv8n | 97.0339 | 50.7377 | 52.7304 | 191 | A · 本计划训练 |

SADD这9项是当前57项seed0矩阵中的9个配置。它们不是此前4配置×3seed的12-run统计表，也不将旧seed1/2混入当前单seed表。SADD只有44张test且仅正样本，不能解释为全场景或七类SAR-AIRcraft结论。

## 3. 从数据可以说什么

1. SSDD Full的AP50为87.7410，高于Original86.0842，但AP75/AP50:95较低；因此“Full低于Original”在这里特指主指标AP50:95，不能笼统说所有指标都下降，也不能仅凭这个模式证明定位退化的内部原因。
2. no-LRC相对Full的AP50:95在SSDD/SADD提高 +1.6755 / +3.4476。这一配置跳过trend卷积并改变后续能量/重要性输入，不能归结为只关闭一个残差标量带来的收益。
3. SSDD LocalEnergy+I→U为45.2636、I→Δ为43.4327，Full为42.6881；当前结果不支持“两条引导路径一起加一定更好”。
4. SSDD P2/P3/P4分别为38.8203/42.6881/43.1492；这次P4略高于P3，不能宣称P3实验证明最优。位置改变涉及通道/结构/初始化差异，同seed不代表张量完全对齐。
5. SSDD Global43.3001高于Full42.6881；LSGM std-only45.1484也高于Full，而trend-only/point-only更低。当前单seed不证明全局或某单统计量普遍优越，也不支持预先认定完整局部统计必然必要。
6. CNN YOLOv8n在SADD为52.7304，高于Full但低于Original/Post-gate；只是同数据与预算下的一种CNN参照，不代表参数匹配、充分调参、统一效率或外部SOTA比较。

## 4. 消融语义与正确四格

冻结模块源SHA256为 `05d1900840474e068aec65bc30ca81501fce55a464ca0dc3fbe60054885aade3`，已与plan核对。[控制定义](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/transfer/final_patch_v1/code/ultralytics/nn/modules/mamba_yolo.py:587)采用R=局部残差修正、E=能量调制、U=重要性作用于scan输入、Δ=重要性作用于步长：

| 配置 | R | E | U | Δ |
|---|---:|---:|---:|---:|
| Matched075 | 0 | 0 | 0 | 0 |
| LocalEnergy | 1 | 1 | 0 | 0 |
| GuidanceOnly | 0 | 0 | 1 | 1 |
| Full | 1 | 1 | 1 | 1 |

Matched075保留γ=0.75的扫描驱动锚点；Original、MatchedIdentity不能替代它计算这组因素交互。

| SSDD因素组 | 四格差分（AP50:95） | seed0结果（百分点） |
|---|---|---:|
| (R+E) × (U+Δ) | Full−LocalEnergy−GuidanceOnly+Matched075 | 0.208226 |
| R × E | LocalEnergy−ResidualScale075−EnergyOnly+Matched075 | -5.862623 |
| U × Δ（固定R+E） | Full−LocalEnergyInput−LocalEnergyDelta+LocalEnergy | -6.670049 |

这些量只描述当前数据/指标/训练方案下的非加性；**没有跨seed不确定性估计，不能标统计显著性，也不是内部物理机制的证明**。LocalEnergy来自历史恢复采纳，相关审计缺口与非逐位等价限制未消失。

[Post-gate实现](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/transfer/final_patch_v1/code/ultralytics/nn/modules/mamba_yolo.py:354)将重要性的相对倍率从scan输入移到输出、在D×x旁路之前，同时关闭importance→Δ。因此Full对Post-gate混合了位置与Δ两种改变。较接近固定其余机制的位置对照是 **LocalEnergyInput对Post-gate**，本seed SSDD差值为 +1.3519；仍不可绕开恢复/初始化及单seed限制。

[no-LRC实现](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/transfer/final_patch_v1/code/ultralytics/nn/modules/mamba_yolo.py:285)令x_bar=x_tilde=x，能量/重要性输入也随之改变，不能写成“仅令λ=0”。[初始化记录入口](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/train_trial.py:325)保留逐张量哈希；同seed不是跨架构、位置、状态维度或窗口形状的张量配对证明。

## 5. 跨任务：完成情况与数值缺口分开

A的独立UUID兼容operator已在10月9日完成15/15；当晚15项均通过completion内部六项（config、initialization、metrics、test_evaluation、result、best）及protocol/epoch/best-test-result递归检查，见[2130增量报告](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/refresh_20261009_2130/INCREMENT_REPORT.md)。

本次A外层producer verifier在10:07:54重新核验了对应result/completion等外层产物。后续独立回收跨任务正文时，已配置SSH连接返回Permission denied；没有尝试新通道、提权或更改权限。因此不将当晚深验冒称为本次六内部文件全部重新读取通过，也不从receipt存在推断数值。

| 数据集 | 指标 | Original | Matched075 | Full |
|---|---|---:|---|---|
| SAR-ACD | test macro-F1（%） | 94.8153 | 数值未回收 | 数值未回收 |
| SAMPLE | test macro-F1（%） | 数值未回收 | 数值未回收 | 数值未回收 |
| Ottawa | test unique-pixel F1（%） | 数值未回收 | 数值未回收 | 数值未回收 |
| Farmland | test unique-pixel F1（%） | 数值未回收 | 数值未回收 | 数值未回收 |
| Yellow River | test unique-pixel F1（%） | 数值未回收 | 数值未回收 | 数值未回收 |

“数值未回收”不等于未训练或失败。SAR-ACD Original的Accuracy=94.8819%、macro-Precision=94.8238%、macro-Recall=94.8427%；test254张，best epoch164/训练200轮。其唯一现有完整指标副本来自[a_file_audit.json](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/refresh_20261009_2006/producer_a/a_file_audit.json)，文件SHA256 `dd85636d25074743a2f340a88a75186a12cf594e35de43b4f265ebf01e53a4ea`。该记录绑定result SHA `78295008b867e71e47d6b221983c9b16842169e4277e2e4ac06e1b2cee329bbf`，与今早A回执一致。

跨任务使用完整backbone、内部P3单块；分类224/BS64/200e-p40，以val macro-F1选模；变化检测192/BS16/150e-p25，val选阈值后固定test，原生重叠像素聚合后每个像素只计一次。统一scratch/FP32/AdamW2e-4，不与旧VMamba输出适配器、P3-only分类头或patch-weighted CD表混算。SAR-ACD保留的heldout标注歧义、SAMPLE同实例多视角、CD单场景空间holdout限制也保留。协议及源划分见[CROSS_TASK_README](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/CROSS_TASK_README.md)。

## 6. 进度、服务器与全局门槛

| producer | 回执verified/分配数 | 缺项 |
|---|---:|---|
| A | 20/22 | AIR Original / Full |
| B | 19/27 | SSDD七项 + AIR Matched075 |
| C | 1/2 | Ship Full |
| F0 | 3/4 | Ship Original |
| F1 | 1/2 | Ship Matched075 |

按任务族：SSDD20/27、SADD9/9、cross15/15、Ship0/3、AIR0/3。完整缺失trial ID在结构化结果文件中。

10:05现场检查仅覆盖原配置A/B/C/F，hostname及GPU UUID均与计划匹配：

| 主机 | 当时磁盘余量GiB | 当时负载/已有实验 | 本轮选择 |
|---|---:|---|---|
| A | 224.61 | GPU空闲；旧dispatch失败记录保留，独立operator状态complete | 只读取已有产物，不重启 |
| B | 619.63 | GPU100%；SSDD Full-d8，已完成epoch22且损失有限 | 现有队列继续，不插队 |
| C | 786.26 | GPU96%；Ship Full，epoch46且损失有限 | 现有队列继续 |
| F | 213.34 | 两GPU100%；Ship Original/Matched075，epoch42/46且损失有限 | 现有队列继续 |

A原队列的torch1.13 UUID失败是已知历史现场，不是本次新增失败。Ship三臂已经使用审计后的独立副本在训练，不应重复传输或发布readiness。AIR原test标注冲突仍是三个位置的协议阻塞，本轮没有收到新数据协议选择；不改标签、不排除位置、不缩小57项门槛。

10:11:37–10:11:42再次只读检查：四台均无seed0 barrier，五条seed1 lane均无status或receipt。**当前campaign的seed1未启动；全57项seed0完整验收并解决数据/协议阻塞前不放行**。这不否定历史campaign已存在的seed1/2。独立原子barrier发布层及A seed1兼容wrapper的待验收限制保留，本轮未实现或部署它们。

## 7. 本轮核验范围与证据

- 五份producer回执完整JSON值已回收，57个唯一trial、44 verified/13 missing/0 invalid；不是只统计文件名。
- 29份eval均为seed0/test，数值有限且范围合法；96份捕获文件元数据的路径/SHA与生产端已核验receipt一致。
- 29项均满足eval.weights_sha256 = train.best_sha256 = receipt best SHA；28项常规evaluation_complete的complete/非smoke/eval SHA/best SHA链一致。
- SSDD LocalEnergy为command adoption：本轮回收了train/eval正文并核对best绑定；receipt核验包含adoption和completion的SHA，但这两份正文未回收，因此不能称为本轮独立语义深验。其eval权重在adopted目录、receipt在runs副本，SHA相同，不是本次新训练。
- 复用的EMA/FP16 checkpoint保存、历史恢复/RNG和控制审计缺口、以及新raw-FP32恢复不保证逐位等价的限制保留。没有本机模型执行或重做推理。
- evaluation中的speed_ms不是统一硬件、batch1、预热/同步控制下的效率实验，不换算成论文统一FPS。
- 五份receipts.capture保存的是完整JSON值重序列化副本，**不是原stdout字节副本**；original_stdout_sha256绑定原生产者stdout，本地文件另有自身SHA。这些审查快照不作为barrier候选。

冻结计划canonical SHA256：`9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4`。  
plan.json文件SHA256：`37e1e2254643541118e4bab29d11cb93f1483847fc73aa69be8d63be4907cbe0`。  
矩阵保持114位置=57 seed0+57 seed1，无新seed2。

核心证据：

- [结构化检测结果与逐行来源](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/results_seed0.json)
- [A检测原始JSON值捕获](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/producer_a_evaluation_read.json)、[B检测捕获](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/producer_b_evaluation_read.json)、[C检测捕获](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/producer_c_evaluation_read.json)、[F检测捕获](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/producer_f_evaluation_read.json)
- [A完整回执](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/producer_a_receipts.capture.json)、[B完整回执](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/producer_b_receipts.capture.json)、[C完整回执](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/producer_c_receipts.capture.json)、[F0完整回执](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/producer_f0_receipts.capture.json)、[F1完整回执](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/producer_f1_receipts.capture.json)
- [跨任务读取限制记录](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/cross/CAPTURE_ATTEMPTS.json)
- [本轮文件SHA索引](D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm/experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/CAPTURE_INDEX.json)

本轮新增内容仅为结果审查报告和独立证据副本；不修改正式MemoryRoot，不将结果审查冒充业务实施或完整实验验收。当前可以分析检测结果；要完成跨任务数值比较，仍须在既有访问恢复后回收A上对应result/test正文并递归核对completion内部六项。
