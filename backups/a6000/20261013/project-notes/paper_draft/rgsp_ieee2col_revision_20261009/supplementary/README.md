# 补充材料索引 - 2026-10-09

本目录属于新的独立工作稿，旧解释已归档到 archive/20261008/supplementary。不要把归档里的“公平消融”“稳健增益”“PFIM 独立贡献”等旧措辞重新当作当前结论。

- results.csv / results_individual.csv：历史 SSDD 5 变体 x 3 种子的原始汇总和15行结果，未改数值或历史 key。
- sadd_seed0.json / sadd_seed1.json / sadd_seed2.json：2026-10-09 回收的 SADD 12-run 比例指标和权重 SHA256，原样复制。
- sadd_results_individual.csv：由这三个 JSON 导出的比例值；正文和补充 PDF 中乘100写成百分数。
- BASELINE_PROTOCOL.md：历史与新轨道分开、命名映射、训练/评估协议及审计边界。
- PAPER_EXPERIMENT_SUMMARY.md：与本版正文一致的解释，不删负结果。
- legacy_tables.tex：从输入稿搬入的分类、检测、分割和变化检测表；外部行并未全部完成主文献数值核验。
- ../SOURCE_REGISTRY.md：按数据集列出来源链、核验等级与缺口。
- ../evidence/：冻结实现片段、配置、恢复核验摘要和证据索引；不是完整可运行实验包。

编译根目录 supplement.tex 可生成独立补充 PDF。这些历史 CSV 不参与自动更新论文表格；修改结果时必须先核对来源并同步统计。
