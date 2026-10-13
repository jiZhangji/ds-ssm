# RGSP 中文双栏工作稿 - 2026-10-09

## 当前入口

- main.tex：双栏正文。
- supplement.tex：独立补充材料，包含全部新SADD种子、旧SSDD统计及历史跨任务表。
- figures/：正文实际使用的两份PDF。
- figure_sources/：对应可编辑SVG、原生成脚本和来源记录。
- evidence/：冻结实现片段、Full配置、恢复核验摘要及来源索引。
- REVISION_NOTES.md / SOURCE_REGISTRY.md：本轮修改及仍待核验事项。
- archive/20261008/：输入稿、旧图和旧说明，仅供追溯，不能当当前结论。

## 编译

在本目录用 XeLaTeX 编译 main.tex 两次；需要补充材料时编译 supplement.tex 两次。Overleaf设置 Compiler=XeLaTeX，并分别选择对应主文件。使用IEEEtran、xeCJK/ctex和TeX Live自带Fandol字体，无需BibTeX、shell-escape或模型环境。

主文使用IEEEtran默认Letter双栏，补充材料为A4单栏。随包预编译PDF与当前源码对应。作者信息待用户提供，未自行填写。

SVG中的文本与形状可编辑；图1嵌入真实SAR图像裁块。生成脚本是本工作区的来源快照，依赖原数据归档/证据路径、ReportLab/Pillow及Windows字体，不是解压即可运行的跨平台绘图安装包。直接编辑SVG/PDF或编译论文不需要这些依赖。模型源码片段仅供核对，ZIP不是完整训练项目。

## 最新版本的范围

本机检索到的输入论文ZIP为 rgsp_cn_revision_20261006_ieee2col_body_preview_bundle.zip（文件时间2026-10-08），同时整合10月8日独立动机图和架构图并继续修订。没有覆盖原包。旧26页单栏PDF和此双栏可编辑正文并非同一覆盖范围，本版不声称恢复了缺失附录。

这是证据与图文修订工作稿，不是投稿定稿。新SSDD完整36run、同协议外部/多数据集实验、效率/机制/真实预测可视化及部分主文献逐行数值核验仍待完成。本次未启动训练、推理、服务器服务或安装软件。
