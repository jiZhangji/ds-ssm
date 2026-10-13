# RGSP 中文论文修订稿（2026-10-06）

基于 Overleaf 原包修订的研究工作稿；**不是投稿定稿**。
编译：XeLaTeX 连续两遍；当前 26 页 A4，0 个未定义引用、0 个错误。

## 相对原包的修订要点

1. **引言动机图**：原 `tikzpicture` 内联图因缺少 tikz 宏包无法编译；已替换为真实 SSDD 数据的三联图 `figures/fig1_motivation.pdf`（(a) 真实 SAR 裁剪中的响应歧义； (b) 通用选择性状态更新；(c) RGSP 的两处条件化）。
2. **摘要与引言**：按 RGSP 叙事重写摘要；引言补充"局部结构可靠性"的定义表述并修正若干措辞。
3. **命名统一**：正文统一为 RGSP / LRC / LSGM。对应关系：DS-only → LRC-only；PFIM-only → LSGM-only；DS-SSM 块 → RGSP 块；PFIM → LSGM（实现代码中的名称）。
4. **P3/8 替换块数量**：修正为"该阶段经深度缩放后实际执行一个块"。
5. **参考文献**：共 21 条，与正文引用一一对应；补充 resnet2015、vit2020、vmamba2024 等条目。
6. **图件更新**：fig2/3/4/5/9/11 重新生成、标签同步（fig9 文件名改为 `fig9_visualization_limits.pdf`）。

实验数值、主结果与负结果未改动；本次未新增实验。

## 文件

- `main.tex`：正文（含附录与参考文献）。上传 Overleaf 后编译器选 XeLaTeX。
- `figures/`：11 张 PDF 图件。
- `supplementary/`：SSDD 汇总 CSV、逐 run 指标与权重哈希、协议说明（原包历史材料，原样保留）。
- `paper.pdf`：本次编译产物。

## 实现代码 ↔ 论文命名对应（正文表 2 下注释）

| 论文变体名 | 实现代码 |
|---|---|
| LRC-only | ds_only_p3 |
| LSGM-only | pfim_only_p3 |
| Full | full_p3 |
| Global | global_p3 |
| LSGM | PFIM |

## 2026-10-07 方法与架构图更新

fig2_method.pdf 已替换为本聊天生成的整体网络 + RGSP 放大架构图（PNG 无损嵌入 PDF，非矢量重绘）。方法改为总体网络与 RGSP 块 → LRC → LSGM → 结构条件化选择性状态更新，网络集成前移，删除独立复杂度分析。原稿与原图备份于 archive/20261007_method_figure_before/。本次未改实验结果。旧 paper.pdf 已移入备份；新的全文 PDF 编译状态见本轮说明。
