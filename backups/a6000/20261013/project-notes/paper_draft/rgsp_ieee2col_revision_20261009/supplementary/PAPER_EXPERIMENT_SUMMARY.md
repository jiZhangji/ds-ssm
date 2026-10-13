# 本版实验解读 - 2026-10-09

本文件取代当前包中的旧解释，原文归档不覆盖。指标本轮只做文件核对与描述性复算，没有训练、推理或新模型评估。

## 新 SADD

AP50:95，百分数，均值 +/- 样本标准差：

| 配置 | seed0 | seed1 | seed2 | mean +/- sd |
|---|---:|---:|---:|---:|
| Original | 53.7117 | 53.9804 | 52.3480 | 53.3467 +/- 0.8753 |
| Matched075 | 52.0914 | 53.0962 | 51.3444 | 52.1773 +/- 0.8791 |
| LocalEnergy | 51.9479 | 52.3537 | 52.8014 | 52.3676 +/- 0.4269 |
| Full | 49.0030 | 53.1989 | 53.7361 | 51.9793 +/- 2.5915 |

Full-Original配对差 -4.7087/-0.7816/+1.3881pp，平均-1.3674pp，仅1/3正。LocalEnergy-Matched075平均+0.1903pp；Full-LocalEnergy平均-0.3883pp。AP50略升不抵消更严格定位指标的下降，不能宣称Full稳定改善。
固定阳性795/44/44、小测试集、EMA/FP16恢复非逐位等价和seed0/1历史审计缺口必须保留。

## 旧 SSDD

Original/LocalEnergy/Legacy-PFIM/Full/Global AP50:95均值42.03/43.98/43.12/42.91/43.77%。原15项数值保留于CSV。
LocalEnergy-Original逐seed +2.90/+5.26/-2.30pp；Full-Original -0.35/+5.93/-2.94pp。不能删除seed2或称其为已证实outlier。
旧PFIM关闭参数漂移，故不支持“PFIM单独有效”的归因；Full-LocalEnergy=-1.07pp是边际差，不是严格交互。
新SSDD730/186/232完整36-run尚未在本版回收范围内验收，不用旧成绩替代。

## 跨任务

- SAR-ACD的96.850 vs 89.764 Accuracy含RGSP额外50轮模块预热日程，不能剥离训练策略效应。
- SAMPLE当前real-only闭集结果97.037 vs 98.519，不与文献synthetic-to-measured排名。
- Ottawa重跑批次F1略升，但早期批次下降。
- Yellow River四个BS配置F1均降；Precision仅BS16/96升，BS32/64也降，Recall均降。
- OpenEarthMap-SAR早期mIoU小升但FWIoU/OA下降；有效BS64三个指标均降。
- 旧高分检测与跨文献表仅作有条件的历史记录；详见SOURCE_REGISTRY.md。

可靠性仍为设计目标；I是由任务损失学习的结构条件信号，不是已校准可靠性。尚缺同协议外部基线、公平多数据集、效率、机制和真实预测可视化。工作稿不是投稿定稿。
