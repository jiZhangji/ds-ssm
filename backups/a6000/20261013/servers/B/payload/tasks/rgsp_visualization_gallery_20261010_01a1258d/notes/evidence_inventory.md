# 图册数据来源与科学口径核对

核对日期：2026-10-10。项目根：`D:/A-主机文件/Codex/Mamba-YOLO-ds-ssm`。除特别注明外，下列路径相对此根。本文件只是本次图册来源清单，不替代正式项目记忆，不记录当前服务器状态。原数据、原图和原生成脚本未改动；未执行模型、训练、推理或评测。

## 1. 必须隔离的协议组

| 图中建议组名 | 来源 | 可用记录 | 必须保留的边界 |
|---|---|---:|---|
| Current SSDD seed 0 | `experiments/rgsp_seedwise_20261009_v1/evidence/results_review_20261010/detection/results_seed0.json` 中 SSDD | 主绘图程序按今日原始记录筛选 | 新730/186/232内容组划分；单seed，不与旧SSDD三seed均值合并，不加重复实验误差条 |
| Current SADD seed 0 | 同文件中 SADD | 主绘图程序按今日原始记录筛选 | 原12run中的可复用seed0与新增对照可能同时出现；不把重复来源当新独立样本 |
| SADD 12-run | 下文SADD原始JSON及CSV | 4配置×3seed=12 | 固定795/44/44，positive-only，test44；三seed共享split；Original恢复非逐位等价；保留历史审计缺口 |
| Legacy SSDD 15-run | `dsh_experiment_results/results_individual.csv` | 5配置×3seed=15 | 旧742/186/232；seed同时改变split与训练随机性；内容重叠审计后已被新轨道隔离；旧PFIM-only不是严格独立贡献 |
| Historical ex27 exact-v2 diagnostic | `artifacts/dsssm_motivation_ex27_exact_v2_20260823/` | 固定6图 | 历史partial-transfer checkpoint与历史baseline非完全匹配；不能证明当前RGSP或总体可靠性 |
| Historical DDP ex27 training | 下文300e CSV | 1条300epoch轨道 | BS80、DDP、AMP、300e；不是旧SSDD15run，也没有证据把它绑定为exact-v2可视化checkpoint |

新seed0的逐行来源、完成回执与数值缺口详见 `reports/RGSP_SEED0_RESULTS_REVIEW_20261010.md`。完成回执不能替代缺失的指标原值；缺数值不等于训练失败。不得用旧结果填新格子。

## 2. SADD 12-run：精确表及核对

- 原始回收文件：`experiments/rgsp_status_20261009/sadd_seed0.json`、`sadd_seed1.json`、`sadd_seed2.json`，每份4条。
- 整洁CSV：`paper_draft/rgsp_ieee2col_revision_20261009/supplementary/sadd_results_individual.csv`，12行、8列。
- CSV列：`variant,seed,ap50,ap75,ap50_95,precision,recall,weights_sha256`。
- 原始JSON另含 `evidence` 远端评估文件路径；所有五种指标均为0–1率值，显示为百分数须乘100。差值乘100后是百分点（pp），不是相对提升百分比。
- 本次只读逐项核对：60个指标及12个权重SHA均与三份JSON完全一致，0项差异。
- 配置键：`original,matched_scale075,local_energy,full`；seed为0、1、2。
- AP50:95均值±样本标准差（%）：Original 53.3467±0.8753；Matched075 52.1773±0.8791；LocalEnergy 52.3676±0.4269；Full 51.9793±2.5915。
- Full−Original逐seed差（pp）：−4.7087、−0.7816、+1.3881；均值−1.3674，只有1/3正差。不能称稳定提升。三seed原点应与均值/SD同时显示，不做显著性星号。
- 原始精度保留到计算结束再四舍五入。说明文件：`reports/SADD_RESULTS_SSDD_RESTART_20261009.md`。

## 3. 旧SSDD 15-run与复杂度

- 逐run表：`dsh_experiment_results/results_individual.csv`，15行、9列：`variant,seed,ap50,ap75,ap50_95,precision,recall,inference_ms,weights`。
- 汇总表：`dsh_experiment_results/results.csv`，5行，`n_seeds=3`，均值/样本SD；主统计可由逐run表重算，原表本身只有6位小数。
- 五配置：`baseline,full_p3,ds_only_p3,pfim_only_p3,global_p3`。
- 五精度指标为0–1；`inference_ms`是评估记录每图平均推理毫秒，不能直接说成同步batch1延迟。`1000/inference_ms`不是已验证的batch1吞吐测试。
- 复杂度来源：`dsh_experiment_results/BASELINE_PROTOCOL.md` §4.1；原测量代码 `dsh_experiment_results/measure_complexity.py`。

| variant | Params（个） | 报告GFLOPs | seed0记录inference ms |
|---|---:|---:|---:|
| baseline | 5,985,547 | 13.61 | 6.616542 |
| full_p3 | 5,920,441 | 13.15 | 6.088878 |
| ds_only_p3 | 5,918,104 | 13.12 | 6.000570 |
| pfim_only_p3 | 5,920,441 | 13.15 | 6.105006 |
| global_p3 | 5,920,441 | 13.15 | 6.129386 |

- GFLOPs为历史thop报告值，自定义scan计数覆盖未闭合；不能当统一完整算力账本。参数/速度变化同时包含壳层与MLP改变，非纯SSM机制贡献。
- 旧 `BASELINE_PROTOCOL.md` 的PFIM冻结、cosine、纯AP50:95选模等解释不能原样沿用。后续审计确认旧PFIM-only的lambda/alpha被训练器重新启用，数值保留但撤回独立归因。
- Full−DS不是严格二因素交互；必须有匹配四格才可计算交互。禁止把seed2删成outlier。Full旧表平均比baseline高，但仅1/3配对正差。
- 解释边界来源：`docs/project-memory/CONFLICTS.md` CONFLICT-014/015/016、`paper_draft/rgsp_cn_revision_20261005/supplementary/PROTOCOL_CLARIFICATIONS.md`、较新的 `PROJECT_STATE.md`。历史文档中的调度指令不构成本次授权。

## 4. 真实热图、原图与原始生成代码

目录：`artifacts/dsssm_motivation_ex27_exact_v2_20260823/`。

- 6个 `ID_dsssm_maps.npz`、6个 `ID_dsssm_motivation.png`、`motivation_summary.csv`（6行×63列）、`manifest.json`。
- 对应原生成代码：`source_code/work_train_single/tools/visualize_dsssm_motivation.py`。本次文件SHA256与manifest完全一致：`dbb9a5f7faa40dabfa9d9bf59bf63d6e3066f0fa815ab246e55cb637682a857b`。
- 历史执行模块：`source_code/server_3090_b/ultralytics/nn/modules/mamba_yolo_ex27-E-new.py`，不是后来sweep的forward。
- checkpoint：PFIM `1cebf8735311e5f0461b079d4e50b8b5d957dd6f171532e832821857b240aa09`；baseline `c70f6a023adc51dbeaa892976fdcfba542a4b0c867a1bcb9a7ba3c726210d52e`。
- 保存推理设置：imgsz640、FP32、conf0.25、NMS IoU0.7、block_index0。这里只复用保存结果，没有重新推理。

每NPZ含14个float32数组：`raw_`和`aligned_`各有 `importance,input_energy,corrected_energy,residual,correction,drive_gate,delta_bias`。本次读取数组确认84项均有限，无NaN/Inf。

| ID | raw H×W | aligned/原图 H×W | 标注框数 | 原生token数（含padding） | importance FG/BG | drive FG/BG |
|---|---|---|---:|---:|---:|---:|
| 000099 | 144×160 | 275×320 | 1 | 23040 | 0.45385 | 0.80794 |
| 000531 | 96×160 | 297×500 | 1 | 15360 | 3.67096 | 1.10305 |
| 000929 | 120×160 | 364×500 | 1 | 19200 | 1.04041 | 1.01362 |
| 001081 | 96×160 | 281×500 | 2 | 15360 | 0.56364 | 0.92422 |
| 001131 | 104×160 | 300×464 | 7 | 16640 | 0.63152 | 0.91333 |
| 001111 | 112×160 | 321×491 | 24 | 17920 | 0.80059 | 0.98804 |

表中FG/BG来自历史summary；FG是GT框并集，不是像素级船体分割。6图中4图importance FG/BG低于1，不能只选000531。

### 原图的字节级证据

原图压缩包：`D:/BaiduNetdiskDownload/SAR-Data/SARATR-X/detection/dataset/SSDD.zip`。

- 六个 `SSDD/images/test/ID.jpg` 的SHA256逐项与历史manifest一致。
- 六图在 `SSDD/annotations/test.json` 的image_id分别为99、531、929、1081、1131、1111。官方COCO尺寸和框数与捕获记录一致。
- 老报告里的“validation”是旧文件夹名称；这些图的字节来源实际上对应官方test。图注应标官方test中的历史固定样本，不写成当前val样本。
- ZIP还含test_offshore或test_inshore的相同字节副本，不能重复计数。1081、1111属于test_inshore，其他四图属于test_offshore。
- 历史manifest没有标签SHA；新画GT框若用官方COCO，明确标“official COCO boxes”。不能把框数一致升级为旧标签逐坐标/字节相同的证明。

### 重绘时的处理规则

1. 使用 `raw_*` 原始张量，按原脚本 `align_map_to_image` 的letterbox裁边步骤取有效内容，再按需要可视化；分布图不先插值放大。raw网格本身仍含padding。
2. 裁边：`s=min(Wf/Wi,Hf/Hi)`，`cw=round(Wi*s)`、`ch=round(Hi*s)`，左右/上下居中裁至cw/ch（保留原代码边界约束）。
3. aligned使用INTER_CUBIC，已出现约−0.027的小负importance；原生importance均非负。这是插值越界，不是模型输出负“概率”。不可把aligned直方图解释为原生控制量分布。
4. Importance是任务训练空间控制量，不是经校准的可靠性概率、前景概率或固定注意力mask。统一色标便于6图比较；不要逐图分位拉伸制造共同强响应。
5. 输入/修正feature energy需要相同色标；它们是特征RMS，无物理回波强度单位。原始SAR灰度保持一致0–255映射，不能把未标注亮区称噪声真值。
6. residual和correction按定义成比例；delta_bias是importance的确定性仿射变换。可展示实现关系，但不是独立机制证据。gamma与write相乘形成同一扫描驱动，并非独立两处状态干预。
7. `motivation_summary.csv` 每个量有min/max/mean/std/fg_mean/bg_mean/fg_minus_bg/fg_bg_ratio。std是单图像素总体SD（ddof=0），不是图间SD。delta_bias_fg_bg_ratio因背景均值不大于0而为NaN，不能填0。
8. 若用新COCO框重新计算分区统计，应与原保存summary区分来源；若保留原summary，则表明来自历史框并集统计。

## 5. 现成曲线资产

精确训练CSV：

`experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3/train/results.csv`

- 300行（epoch1–300），14列：`epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2`。
- 旁边 `args.yaml`：300e，batch80，device0,1，pretrained=true，AdamW lr0=0.001，cos_lr=false，amp=true。
- 可画独立历史训练诊断：训练/验证loss，验证AP/P/R随epoch，学习率轨迹。保留原始曲线；若平滑，另标窗口并叠加原值。不能作为新seed0或SADD12run的收敛比较。
- 同run的 `train/`、`test_standard/`、`test_tta_iou0.55_conf0.001/` 各有PR/F1/P/R与confusion_matrix PNG。它们是已栅格化结果，不是曲线原始数列。TTA与standard须分开，不能由单个P/R值编造PR曲线。
- 本次本地目录检索未找到新seed0完整训练CSV；SADD旧evidence中的smoke两轮和启动时部分CSV不代表完整12run曲线，不应用来伪装完整训练过程。

## 6. 有真实来源支撑的28种图册选题

不是要求为同一数据反复换皮；按实际阅读问题选用。新seed0具体集合由今日逐行JSON决定。

| # | 要回答的问题 | 图形与来源 |
|---:|---|---|
| 1 | 当前每配置AP是多少 | 新seed0配置点图，按dataset分面 |
| 2 | 相对同协议Original方向如何 | 新seed0有符号pp差值条/点图 |
| 3 | 五指标是否一致 | 新seed0多指标热力图，不平均异质指标 |
| 4 | 匹配控制的哪个因素改变结果 | 新seed0严格四格效果图，单seed无显著性断言 |
| 5 | 结果数值与完成回执哪些缺失 | 当前覆盖矩阵，明确completed/metric-missing/blocked |
| 6 | 旧SSDD三seed分布 | 原点+均值/样本SD，不仅均值条形图 |
| 7 | 旧SSDD配对方向是否一致 | 同seed baseline→variant连线 |
| 8 | 旧SSDD配对效应大小 | 配对差值点图+均值/SD，保留所有seed |
| 9 | SADD三seed负结果结构 | 原点+均值/SD |
| 10 | SADD每seed的Full影响 | Original→Full配对坡度图 |
| 11 | 更严格IoU下表现如何 | SADD五指标/三种AP分面或热力图 |
| 12 | AP50与AP75差距 | 每配置定位阈值差值图（描述性，非定位误差分解） |
| 13 | 记录P/R怎样取舍 | P–R散点，标各模型最优F1点可能阈值不同，不称PR曲线 |
| 14 | seed改变排名多少 | 三seed排名轨迹，单协议内排序 |
| 15 | 均值与跨seed波动怎样共存 | 均值–SD散点，注明n=3，不声称稳定性概率 |
| 16 | 参数成本与AP怎样取舍 | 旧SSDD Params–AP散点 |
| 17 | 记录推理时间与AP怎样取舍 | 旧SSDD ms/image–AP散点，非batch1基准 |
| 18 | 报告算力账本覆盖如何 | Params/GFLOPs成本图，显式标扫描计数未闭合 |
| 19 | 历史训练与验证loss差异 | 300e loss小多图，独立历史协议 |
| 20 | 历史验证指标何时变化 | 300e AP/P/R曲线，不标为最终test曲线 |
| 21 | 历史学习率如何变化 | lr/pg0–2轨迹，不由协议文字虚构调度 |
| 22 | 样本包含什么真实场景 | 六图原灰度+官方COCO框的GT面板 |
| 23 | 空间importance是否总突出GT | 六图统一色标原生importance面板，正/负例全留 |
| 24 | 残差修正前后特征RMS如何变 | input_energy/corrected_energy双面板、同色标 |
| 25 | 框内/框外importance均值差 | 六图FG/BG dumbbell，明确框并集 |
| 26 | 6图FG/BG比值是否高于1 | importance与drive ratio点图、参考线1 |
| 27 | 有效token控制量如何分布 | 剔padding的raw importance ECDF/直方图，不池化假作独立图像 |
| 28 | importance与drive/energy相关形态 | 剔padding的raw二维密度/hexbin；相关非因果，确定性调制非独立证明 |

现成概念结构图来源可另外复用 `output/figures/build_rgsp_motivation_20261009.py` 和 `output/figures/build_rgsp_architecture_20261009.py`；须标示意，不与真实heatmap混称实验观察。旧 `paper_draft/ig_ds_ssm_cn/build_figures.py` 含历史绘图实现，但含早期口径，不能直接执行后当作最新结论。
