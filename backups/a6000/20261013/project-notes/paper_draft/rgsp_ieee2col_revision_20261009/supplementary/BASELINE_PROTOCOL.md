# 检测协议与审计边界 - 2026-10-09

这是论文配套说明，不是启动实验的指令。输入版原文保存在 archive/20261008/supplementary/BASELINE_PROTOCOL.md。

## 三条轨道不可混算

| 轨道 | 划分 train/val/test | seed | 状态与边界 |
|---|---|---|---|
| 历史 SSDD | 742/186/232 | 0/1/2，逐seed划分 | 15项成绩保留；PFIM关闭参数漂移、结构差异和当前副本内容重叠使纯模块归因不成立 |
| 新 SSDD | 730/186/232，内容组切分 | 0/1/2 | 12配置 x 3seed；本版所依据的10月9日回收记录尚未验收完整36run，不填写预期成绩 |
| 新 SADD | 固定阳性795/44/44 | 0/1/2，共享split | 4配置 x 3seed已回收；test44、positive-only、场景独立性未证明、恢复审计边界保留 |

SSDD当前1160张图仅1135个字节内容。10个组跨官方train/test，涉及12训练副本和10测试图；新轨道仅排除开发集合的12份重叠副本，不删除源数据或官方test。不能由字节去重推出无近重复或场景独立。

## 名称映射与机制

- baseline -> Original：原始 Mamba-YOLO-T，四向 SS2D；不是严格同壳层参考。
- ds_only_p3 -> LocalEnergy：残差修正与趋势能量调制组合，不是仅 LRC。
- pfim_only_p3 -> Legacy-PFIM：旧 trainer 重新启用了参数，三个best权重的lambda_s/alpha非零；不是严格“仅PFIM”。
- full_p3 -> Full；global_p3 -> Global。Global只对I取空间均值；内容、基础步长和趋势能量仍随位置变化。
- LRC = Local Response Correction；PFIM对应论文LSGM。DeSpeckleSSM包含整个状态建模核心；DS_VSSBlock包含归一化、残差和FFN。
- gamma与write_gate相乘调节同一扫描输入；I另一条路径进入Delta logits。B/C从调制前u生成。不能称三条完全独立干预。
- 原始块FFN倍率4，替换块倍率2，且扫描组织不同；整块对比不能全部归因于可靠性/结构引导。
- MatchedIdentity为倍率1控制；Matched075在关闭能量和引导项时gamma=0.75。FixedHalf仍有eta*I=0.05的步长偏置。
- 新12配置完整定义和两组四格交互公式见正文及 supplement.tex。关闭参数用buffer固定并核验，不仅依赖requires_grad=False。

## 训练与选模

640输入，BS16，nbs64，scratch，AdamW lr0=2e-4、lrf=0.01、weight_decay=5e-4；cos_lr=false，线性衰减。SSDD150e/patience30；SADD200e/patience40。
优化器预热3轮，无模块预热；Mosaic0.3、末15轮关闭，degrees180、translate0.1、scale0.3、flipud/fliplr各0.5、HSV(0,0,0.1)。FP32和确定性设置。
选模fitness=0.1*AP50+0.9*AP50:95，不是只按AP50:95。训练val NMS IoU=0.7；独立test conf=0.001、NMS IoU=0.6、FP32、BS16。P/R来自最优F1工作点，不是跨模型固定阈值比较。
恢复Original使用EMA/FP16优化器检查点，不保证逐位无中断等价；seed0/1历史审计缺口不补造。

## 统计与效率

报告全部种子、均值和样本标准差(ddof=1)，比例值与百分数明确区分；配对差以百分点表示。2/3同号不定义为已证明“稳定”，3seed也不等于显著性证据。Full减LocalEnergy不是严格交互。

以下为继承的2026-10-01 seed0历史复杂度账本，本轮未复测。不能作为新控制轨道统一效率结果：

| 历史key | Params | GFLOPs | FPS | inference(ms) |
|---|---:|---:|---:|---:|
| baseline | 5985547 | 13.61 | 151.1 | 6.62 |
| full_p3 | 5920441 | 13.15 | 164.2 | 6.09 |
| ds_only_p3 | 5918104 | 13.12 | 166.7 | 6.00 |
| pfim_only_p3 | 5920441 | 13.15 | 163.8 | 6.11 |
| global_p3 | 5920441 | 13.15 | 163.2 | 6.13 |

替换块更小包含FFN和扫描结构效应；不能把2337个参数差概括为三个标量。统一设备、输入、精度、batch、计时和profiler的效率验证仍待完成。
