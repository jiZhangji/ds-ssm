# 参考文献与论述逐项对照

核对日期：2026-10-05。编号与main.tex书目顺序一致。正文42处引用均使用单个文献键，具体行号和上下文见 revision_checks/citation_occurrences.json；同一文献可为不同段落提供同类依据。

## 核对口径

书目核对包括题名、作者、年份、载体及DOI/arXiv标识，不等于验证原文全部方法和实验。本表明确具体用途及可用证据层级。仅得到题名/摘要的近期工作只承担研究范围描述，不用来背书未读的公式、优先权或性能结论。本地实验事实以代码/原记录为依据，不由外部论文引用代替。

| 编号 / 键 | 文献与定位来源 | 本文中对应的具体论述 | 核对层级与限制 |
|---|---|---|---|
| 1 / ssdd2021 | Zhang等，SSDD官方发布，Remote Sensing 2021，[DOI](https://doi.org/10.3390/rs13183690) | 引言的近岸、尺度、密集停泊挑战；4.2的数据集及使用标准 | DOI书目与摘要明确列举这些标准；不证明本地742/186/232划分已完成逐文件审计 |
| 2 / saraircraft2023 | 王智睿等，SAR-AIRcraft-1.0，雷达学报2023，[出版社](https://radars.ac.cn/article/doi/10.12000/JR23043) | 引言的离散散射及机场背景干扰；飞机检测数据来源 | 出版社书目/摘要；中文6名作者、12(4):906-922以出版社为准，不用不完整DOI记录的单作者替代 |
| 3 / resnet2015 | He等，Deep Residual Learning，CVPR2016，[DOI](https://doi.org/10.1109/CVPR.2016.90) | 引言中残差学习支持深层视觉优化 | 原论文页面摘要及正式会议元数据；键中的2015仅内部标识，书目用2016 |
| 4 / vit2020 | Dosovitskiy等，ViT，ICLR2021，[arXiv](https://arxiv.org/abs/2010.11929) | 图像块自注意力与远程联系 | 原论文页面及会议注释；平方复杂度是全局成对交互的说明，不泛指所有Transformer变种 |
| 5 / mamba2023 | Gu与Dao，Mamba，2023预印本，[arXiv](https://arxiv.org/abs/2312.00752) | 输入依赖选择性、线性序列建模；2.2和3.1的SSM背景 | 页面摘要及原方法背景；本稿递推细节单独核对项目实现，不宣称是原论文精确连续离散化公式 |
| 6 / vim2024 | Zhu等，Vision Mamba，ICML2024，[arXiv](https://arxiv.org/abs/2401.09417) | 双向视觉状态空间建模 | 题名、摘要、会议注释；不与VMamba混用作者或扫描路径 |
| 7 / vmamba2024 | Liu等，VMamba，NeurIPS2024，[arXiv](https://arxiv.org/abs/2401.10166) | 四路线二维选择性扫描；原块架构背景 | 论文页面摘要及项目SS2D实现；不将当前单向替换说成扫描拓扑保持不变 |
| 8 / mambayolo2024 | Wang等，Mamba YOLO，2024预印本，[arXiv](https://arxiv.org/abs/2406.05835) | 状态空间与目标检测结合；检测实现架构背景 | 原论文页面书目/摘要及本地模型结构；不为本地改动后的性能作背书 |
| 9 / rsmamba2024 | Sijie Zhao等，RS-Mamba，TGRS2024，[DOI](https://doi.org/10.1109/TGRS.2024.3425540) | 大幅面遥感图像密集预测 | DOI题名/书目；首作者修正为S. Zhao。未据元数据补造具体扫描公式 |
| 10 / rstmtmamba2025 | Shen等，Learning Cross-Task Features With Mamba，TGRS2025，[DOI](https://doi.org/10.1109/TGRS.2025.3540573) | 遥感跨任务特征学习 | DOI题名/书目；仅支持研究方向，不宣称与本地四任务协议相同 |
| 11 / madinet2025 | Zhou等，MaDiNet，TCSVT2025，[DOI](https://doi.org/10.1109/TCSVT.2025.3574657) | SAR-Mamba相关工作及正式期刊版本定位 | 正式书目：Zhou、Y. Liu、Peng、L. Liu、Li，35(11):10787-10800；不移植v1作者/数字为正式版结果 |
| 12 / lee1980 | Jong-Sen Lee，Local Statistics，TPAMI1980，[DOI](https://doi.org/10.1109/TPAMI.1980.4766994) | 局部统计自适应估计的经典背景 | 正式题名与书目、经典方法定位；本稿不是Lee滤波器或已验证物理散斑估计 |
| 13 / specklefree2021 | Chen等，Speckle-Free SAR Image Ship Detection，TIP2021，[DOI](https://doi.org/10.1109/TIP.2021.3089936) | 散斑处理与舰船检测关系 | 题名/书目核对；没有引用未核验的模块公式或迁移其检测收益 |
| 14 / rssnet2026 | Huang等，RSS-Net，JSTARS2026，[DOI](https://doi.org/10.1109/JSTARS.2025.3647971) | Mamba用于SAR图像去噪 | 题名与书目层级，19:3539-3551；DOI含2025而正式卷年2026不矛盾，标题保留登记拼写Denoizing |
| 15 / mambadip2026 | Chinchu A等，MambaDIP，JSTARS2026，[DOI](https://doi.org/10.1109/JSTARS.2026.3702550) | Mamba结合深度图像先验用于去斑/增强 | 题名与书目层级，19:21255-21271；没有对其内部实现作逐页核验声明 |
| 16 / madinet2024v1 | Zhou等，MaDiNet v1，2024，[固定版本](https://arxiv.org/abs/2411.07500v1) | 2.3的MambaSAR/多方向/局部全局设计；附录D.3的作者报告AP50 | 版本明确的原文方法与三数据集主表；SSDD99.0、SARShip97.5、Aircraft90.5只作作者报告，非本地复现 |
| 17 / mcmn2026 | Gao等，Mamba-CNN hybrid Multi-scale ship detection Network，ISPRS2026，[DOI](https://doi.org/10.1016/j.isprsjprs.2026.01.004) | Doppler/散射线索、Mamba-CNN、多尺度舰船检测方向 | 题名/书目及既有材料的任务范围；不据此推导其具体状态调制，不与本文单通道强度输入混同 |
| 18 / cbam2018 | Woo等，CBAM，ECCV2018，[arXiv](https://arxiv.org/abs/1807.06521) | 通道/空间乘性细化；说明注入位置对照 | 原论文页面摘要与会议记录；不声称所有注意力都只在扫描输出端 |
| 19 / sarship2019 | Wang等，SAR ship dataset，Remote Sensing2019，[DOI](https://doi.org/10.3390/rs11070765) | SAR-ShipDataset来源、复杂背景与尺度 | DOI摘要/书目；公开总体规模与本地筛选子集数分开 |
| 20 / sample2019 | Lewis等，SAMPLE，SPIE2019，[DOI](https://doi.org/10.1117/12.2523460) | 实测/合成配对数据来源，4.2及附录C.1 | 正式题名/作者/会议；本地real-only选样不等于原公开协议。未将模糊元数据字段“17”擅自扩成页码范围 |
| 21 / fusar2021 | Shi等，FUSAR-Map，JSTARS2021，[DOI](https://doi.org/10.1109/JSTARS.2021.3063797) | Gaofen-3语义分割数据来源 | DOI书目与任务题名；不证明本地裁块manifest或场景隔离 |
| 22 / oemsar2025 | Xia等，OpenEarthMap-SAR，GRSM2025，[DOI](https://doi.org/10.1109/MGRS.2025.3599512) | 全球高分辨率SAR土地覆盖基准来源 | 正式书目13(4):476-487；本地SAR-only设置与发布协议分开 |
| 23 / adamw2017 | Loshchilov与Hutter，Decoupled Weight Decay，ICLR2019，[arXiv](https://arxiv.org/abs/1711.05101) | 解耦权重衰减优化器背景 | 原论文页面与会议注释；内部键不决定正式年份，不据此证明历史run实参 |
| 24 / coco2014 | Lin等，Microsoft COCO，ECCV2014，[DOI](https://doi.org/10.1007/978-3-319-10602-1_48) | 多IoU阈值检测评价的COCO背景 | 数据集/基准原论文书目与页面；当前AP由项目Ultralytics计算，不声称与官方COCO API未经比对即可逐项等价 |

## 重点纠正与未闭合项

- Vim对应Zhu等，VMamba对应Liu等；RS-Mamba首作者为Sijie Zhao。
- MaDiNet预印本v1和正式期刊版分条引用，不混作者、结构描述或成绩。本稿未保留缺乏当前版本原文支持的“Gamma先验”论断。
- ResNet、ViT、AdamW采用正式发表年份2016、2021、2019；保留旧内部引用键只为源码稳定。
- RSS-Net、MambaDIP、MCMN只承担已核对范围内的相关工作定位；后续若扩写详细机制，需重新阅读对应全文。
- SAR-ACD本地五类2522张集合与唯一首发论文/发布版本未绑定，不以同名六类数据文献替代。
- Ottawa等单场景变化检测数据、SADD、AIR-SARShip、OGSOD及SRSDD的本地发布版、划分和许可链仍需核验。这些在稿中作为历史记录而非已标准化的主基准；本次未捏造参考文献补齐来源。
- Crossref摘要/元数据、出版社页面和arXiv原文页面具有不同覆盖深度。网络核对不等于全部24篇全文逐页审计。

## 可复查材料

revision_checks内保留crossref_metadata.json与primary_pages.json（部分来源的检索快照）、citation_occurrences.json（全部42处引用位置）、source_validation.json（键/标签校验）。它们是辅助审计记录，不作为额外参考文献，也不替代DOI指向的原始出版物。少数补充DOI的在线核对没有完整响应快照，本表保留直接入口，不伪称快照覆盖全部24项。
