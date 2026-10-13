# RGSP 完整实验与消融修订稿

本版基于用户上传的双栏正文底稿修改，不是上一版七页主文。分类、检测、分割、变化检测九张原任务表保留在正文，新增消融、27次逐种子结果、配对分析、完整控制定义与复杂度记录也全部在同一PDF中。

## 打开与编译

- 入口：main.tex；编译器：XeLaTeX；至少编译两遍。
- Overleaf：上传整个ZIP，将主文档设为main.tex、编译器设为XeLaTeX。归档中的旧main.tex不是主文档。
- 本地已有TeX Live时：`xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex`，重复运行直至交叉引用稳定。
- ZIP根目录的paper.pdf为本轮新编译的14页预览。无需另外打开补充PDF才能查看实验。
- 项目服务器连接失败后，用户明确同意仅对文档采用本机回退。已用现有TeX Live 2026完成独立解压双遍编译，pypdf及既有pdfinfo/pdftotext核验均通过；全部14页渲染图与先前已审阅版本逐页同哈希。正文、章节、类文件和两张图的13项编译输入均未改变。
- 最终ZIP另由tools/package_delivery.py独立解压、双遍复编译和逐页渲染比较；交付目录verification.json仅在全部通过后写入PASS。旧静态交付包保留，不能用其verification_static.json替代新回执。详见VALIDATION.md。

## 内容定位

- main.tex：引言、相关工作、方法、九张历史任务结果表、结论和书目。
- sections/experimental_settings.tex：数据、划分、训练与选模、统计规则。
- sections/ablation.tex：十个消融小节，包括旧冻结审计、SSDD、SADD、结构匹配、两类四格、固定引导、复杂度及覆盖边界。
- sections/ssdd_*.tex和sections/sadd_*.tex：完整均值、逐种子和配对表。
- sections/limitations.tex：跨任务边界、真实预测证据与未验收项。
- supplementary：旧SSDD原始CSV、新SADD原始JSON及修正后的协议/解读。
- evidence/NUMERICAL_AUDIT.json：由原始比例值计算的完整统计，不用四舍五入表值反算。
- evidence/DOCUMENT_VALIDATION.json和BUILD_VALIDATION.json：本轮新PDF的文档核验与本地编译回执；历史静态记录单独保留。
- figure_sources：可编辑SVG与已有绘图代码快照；脚本可能依赖原工作区路径，SVG与PDF可直接使用。
- archive/upload_original：上传ZIP的27项原件，包含历史PDF与旧解释，不作为现行结论。

## 研究边界

已纳入SSDD历史15run及SADD新12run。新SSDD12配置×3seed的完整结果尚未在本版证据中验收，矩阵是实验定义，不是36项已完成成绩。此次本机授权仅覆盖文档编译、文本/数值/版面核验与打包；没有新增训练、推理、模型测试、软件安装或服务改动，也未查询或改变实验队列。外部论文表数值尚未全部逐行核验，不能据这些历史表宣称SOTA。详见SOURCE_REGISTRY.md和REVISION_NOTES.md。
