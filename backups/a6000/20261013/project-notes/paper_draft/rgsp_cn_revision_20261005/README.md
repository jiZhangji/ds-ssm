# RGSP 中文论文全面修订稿

修订日期：2026-10-05。基于用户 ZIP 和最新引言方向重写，未覆盖原包。当前 PDF 为25页 A4、5幅内嵌矢量图、21张表、24条参考文献。这是完整可编辑研究稿，不是已经补齐实证证据的投稿定稿。

## 文件

- main.tex：唯一当前编译入口；正文、附录 A-F、图表与参考文献均内嵌。
- paper.pdf：本次核验的阅读版。
- REVISION_NOTES.md：论文逻辑、方法修正、实验写法及待补证据。
- REFERENCE_AUDIT.md：24条文献与具体论述、核对来源、版本边界逐项对应。
- supplementary/PROTOCOL_CLARIFICATIONS.md：对原包协议和结论的澄清，优先于旧说明。
- supplementary/results_individual.csv、results.csv：原包数据，未改动。
- supplementary/BASELINE_PROTOCOL.md、PAPER_EXPERIMENT_SUMMARY.md、README.md：原包历史说明，原样保存；其中“公平”“冻结”“outlier”等判断不自动成为修订稿结论。
- source_archive/：输入版本的源码与 README，不是当前编译入口。
- figures/：原包11幅历史图件，原样保留；当前正文不调用它们，不能将历史热图视为当前模型的可靠性验证。
- revision_checks/：引用位置索引、源码校验、数值复算及 PDF 结构核验。
- MANIFEST.json：输入 ZIP、主要来源、交付文件的 SHA-256 清单。

## Overleaf

新建项目并上传修订 ZIP，选择根目录 main.tex，将编译器设为 **XeLaTeX**；建议使用较新的 TeX Live。连续编译至交叉引用稳定。无需 BibTeX、外部绘图脚本或数据文件即可生成正文；默认 Fandol 中文字体由 ctex/TeX Live 提供。

本地等效命令为连续执行两次：

    xelatex -interaction=nonstopmode -halt-on-error main.tex

本次内置编译器因“Unable to find standard directories for platform”未能工作，已保留源码编辑页；交付 PDF 由已有 TeX Live 2026/XeLaTeX 编译并逐页渲染核验。未安装软件。当前本机缺少 hyperref 的传递依赖 hycolor，源码会退回普通 URL 排版；完整 Overleaf 环境可自动启用超链接，分页可能略有差异。未将本地编译等同于 Overleaf 云端实测。

## 投稿前检查

本轮没有训练、推理、服务器实验或新增模型成绩。SSDD 主表由原包15行记录复算，使用样本标准差并保留全部 seeds。Full 未优于 DS-only，逐位置重要性未优于图像级对照，跨任务有正负结果。没有做显著性宣称，也没有把不同协议的公开成绩拼为排行榜。

同壳层中性对照、可靠性独立诊断、SAR-ShipDataset/AIRcraft 多种子验证、跨任务接口统一及严格效率测量均在附录 E 标为 **Planned**。后续完成这些实验后，应同步修改摘要、贡献、结果和结论，而非只向表中补数。
