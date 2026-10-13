# 来源登记与核验范围 - 2026-10-09

“书目核对”“原表数值核对”“本项目复现”是不同层级。此表不把旧稿中的引用说明自动升级为本轮主文献核验。

## 项目结果

| 材料 | 当前证据 | 本轮处理与限制 |
|---|---|---|
| SADD新12run | supplementary/sadd_seed0/1/2.json；原来源experiments/rgsp_status_20261009/；reports/SADD_RESULTS_SSDD_RESTART_20261009.md | 从回收JSON复算并对照报告；不在本轮重跑模型或远端权重检查 |
| SSDD旧15run | supplementary/results_individual.csv、results.csv | 全部行保留；原key映射见BASELINE_PROTOCOL，关闭参数漂移推翻严格组件归因 |
| 新控制定义 | paper_draft/RGSP_EXPERIMENTS_SUPPLEMENT_20261008.md；冻结实现和Full配置 | 静态核对；完整SSDD36run不在本次验收范围内 |
| 恢复边界 | evidence/recovery/*_verification.json；reports/RGSP_RECOVERY_QUEUE_20261008.md | 摘要是恢复前核验，不是最终完成回执；完成状态来自10月9日报告 |
| 历史跨任务 | archive/20261008/main.tex对应分类/检测/分割/变化检测章节；项目reports/旧汇总 | 数值继承、负结果保留；没有本轮逐项重跑或重新回收全部原始日志 |

## 外部表格来源链

以下表名对应supplementary/legacy_tables.tex中的label，不依赖最终页码。

| 表label | 输入稿声称来源与协议 | 本轮核验状态 / 后续动作 |
|---|---|---|
| tab:sar_acd_comparison | Bottom-Up Table7；SUMMIT Table5；SARMAE 30%标注；MDB-DPL六类8:2、五次统计 | 表内数值保留。前三组完整书目和原页待核；MDB-DPL候选DOI 10.1016/j.sigpro.2026.110777，未本轮核数。不得混同比本文五类单种子 |
| tab:sample_comparison | MLDA，Remote Sensing 2025，Table2；synthetic-to-measured | 未定位并核验完整主文献。方法行和均值/SD只是继承记录；不补造作者、题名或DOI。与本项目real-only闭集分区 |
| tab:ssdd_comparison | MaDiNet及项目近期方法汇总，多个外部来源 | 保留整行，未拼接不同来源指标；逐行版本、表号、split、初始化待核，不可排名 |
| tab:sar_ship_comparison | MaDiNet等旧比较记录 | 同上；不以发表原方法的年份代替实际成绩来源 |
| tab:sadd_comparison | YOLO-SAD、SARATR-X、SARDet-MIM等旧记录 | 原文数据版本、预训练、空标注处理及split仍须逐行对齐 |
| tab:air_sarship_comparison | 互补预训练与FMP-Net等旧资料 | 候选原文arXiv:2103.08251；各外部行尚未逐一核验。项目640切片与416配置分开 |
| tab:openearthmap_sar_comparison | OpenEarthMap-SAR基准，SAR/P+R5 | 公开仓库cliffbb/OpenEarthMap-SAR可定位；本轮未成功取得并核对原表，35.08/36.84/37.23为继承值 |
| tab:ottawa_comparison | SARCDNet Table2 | DOI/题名/作者/卷号/文章号经Crossref核对；Table2数值未本轮成功核验 |
| tab:yellow_river_comparison | SARCDNet Table2 | 同上；本文内部Precision/Recall叙述已按保留表数值纠正 |

SARCDNet书目核验：Vibha Damodara Kevala, Vishal Mukundan, Sravya Nedungatt, Shilpa Suresh, Shyam Lal，
“SARCDNet-an enhanced deep learning network for change detection from bi-temporal SAR images”，
Scientific Reports 16, 1826，卷期年2026，在线2025-12-31，DOI 10.1038/s41598-025-31488-y。
书目核验不等于原表核验或方法复现。

## 旧资料提供的定位候选（非已核数清单）

- MaDiNet: https://arxiv.org/abs/2411.07500 （版本须明确，旧资料提及v1 Tables3-5）
- YOLO-SRBD: https://doi.org/10.1016/j.icte.2024.02.007
- ARGO-Net: https://doi.org/10.3390/jmse14141331
- YOLO-SAD: https://www.mdpi.com/2076-3417/14/7/3025
- AIR-SARShip互补预训练: https://arxiv.org/abs/2103.08251
- SARDet-MIM: https://www.mdpi.com/2072-4292/18/4/580
- OpenEarthMap-SAR: https://github.com/cliffbb/OpenEarthMap-SAR
- SARCDNet: https://doi.org/10.1038/s41598-025-31488-y

候选路径来自reports/IG_DS_SSM_各数据集方法年份出处对比表_20260830.md等输入资料；可能存在版本或日期差异。保留来源并标待核，不以这些链接的存在证明数值正确。主文其他继承书目本轮也未逐一做系统核验。
