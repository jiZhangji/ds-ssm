from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path(__file__).with_name("IG_DS_SSM_current_status_brief.docx")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(table, top=80, start=120, bottom=80, end=120):
    tbl_pr = table._tbl.tblPr
    tbl_cell_mar = tbl_pr.find(qn("w:tblCellMar"))
    if tbl_cell_mar is None:
        tbl_cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(tbl_cell_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tbl_cell_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tbl_cell_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for idx, width in enumerate(widths):
            if idx >= len(row.cells):
                continue
            cell = row.cells[idx]
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def style_table(table, widths=None):
    table.style = "Table Grid"
    set_cell_margins(table)
    if widths:
        set_table_width(table, widths)
    for row_idx, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for run in p.runs:
                    run.font.size = Pt(9)
            if row_idx == 0:
                set_cell_shading(cell, "F2F4F7")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.bold = True


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    for row_data in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
    style_table(table, widths)
    return table


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.add_run(text)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.add_run(text)
    return p


def add_callout(doc, title, body):
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F4F6F9")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.bold = True
    r.font.color.rgb = RGBColor(31, 58, 95)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.add_run(body)
    style_table(table, [6.5])
    return table


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(1)
sec.bottom_margin = Inches(1)
sec.left_margin = Inches(1)
sec.right_margin = Inches(1)
sec.header_distance = Inches(0.492)
sec.footer_distance = Inches(0.492)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Calibri"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.10

for style_name, size, color, before, after in [
    ("Heading 1", 16, "2E74B5", 16, 8),
    ("Heading 2", 13, "2E74B5", 12, 6),
    ("Heading 3", 12, "1F4D78", 8, 4),
]:
    st = styles[style_name]
    st.font.name = "Calibri"
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor.from_string(color)
    st.font.bold = True
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(3)
r = title.add_run("IG-DS-SSM / PFIM 当前研究与论文进展汇报")
r.bold = True
r.font.size = Pt(18)
r.font.color.rgb = RGBColor.from_string("0B2545")
r.font.name = "Calibri"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(12)
rr = sub.add_run("面向轻量化 SAR 舰船检测的重要性引导去斑状态空间建模 | 2026-06-23")
rr.font.size = Pt(10)
rr.font.color.rgb = RGBColor(85, 85, 85)

add_callout(
    doc,
    "一句话结论",
    "当前最稳妥的汇报主线是：我们提出 IG-DS-SSM / PFIM，将 SAR 去斑先验和 token 重要性注入状态空间更新，在 SSDD 上取得接近 MaDiNet 的 AP50，同时保持约 5.96M 参数和 13.21 GFLOPs 的轻量规模。"
)

doc.add_heading("1. 当前工作定位", level=1)
p = doc.add_paragraph()
p.add_run("当前研究主线：").bold = True
p.add_run("Importance-Guided De-Speckle State Space Model（IG-DS-SSM）/ PFIM。")
doc.add_paragraph(
    "它不是单纯在 YOLO 上叠加注意力或外部去噪模块，而是把 SAR 图像中的斑点噪声抑制、目标重要性估计和状态空间模型的状态更新过程结合起来。"
)
add_bullet(doc, "DS-SSM：构造去斑 token 输入，并通过能量感知机制抑制不可靠状态传播。")
add_bullet(doc, "PFIM：估计 token 级目标重要性，并调制 gamma、write_gate 和 delta 三条状态更新路径。")
add_bullet(doc, "DG-DS-SSM：作为解耦门控扩展与消融思路，目前不作为主模型。")

doc.add_heading("2. 目前最重要的实验结果", level=1)
doc.add_paragraph("当前可用于汇报的核心结果集中在 SSDD 数据集。PFIM ex27 是当前恢复实验中的最佳主线结果。")
add_table(
    doc,
    ["方法", "数据集", "P", "R", "AP50", "AP50:95", "Params(M)", "GFLOPs"],
    [
        ["IG-DS-SSM / PFIM ex27", "SSDD", "97.378", "95.219", "98.704", "74.022", "5.9648", "13.2079"],
        ["Mamba-YOLO-T", "SSDD", "96.29", "96.15", "98.49", "72.85", "5.9855", "13.6116"],
        ["DG-DS-SSM ex34", "SSDD", "96.3", "96.2", "98.7", "73.5", "5.9632", "13.1554"],
    ],
    [1.55, 0.62, 0.62, 0.62, 0.75, 0.8, 0.78, 0.76],
)

doc.add_paragraph(
    "解读：我们的 AP50 略低于 MaDiNet 在 SSDD 上报告的 99.0，但参数量和 GFLOPs 处于轻量级检测器范围。汇报中建议强调“近 SOTA 精度 + 显著轻量化 + SAR-aware SSM 动态机制”，不要强调全面超过 MaDiNet。"
)

doc.add_heading("3. 与 MaDiNet 三数据集结果的关系", level=1)
doc.add_paragraph("MaDiNet 论文在 SSDD、SAR-ShipDataset 和 SAR-AIRcraft1.0 三个数据集上报告结果。我们已整理统一对比表并生成 PDF。")
add_table(
    doc,
    ["方法", "SSDD AP50", "SAR-ShipDataset AP50", "SAR-AIRcraft1.0 mAP50", "复杂度口径"],
    [
        ["MaDiNet", "99.0", "97.5", "90.5", "原文精度；复杂度仅测得 MambaSAR 核心模块"],
        ["IG-DS-SSM / PFIM", "98.704", "--", "--", "完整轻量检测器，5.9648M / 13.2079 GFLOPs"],
    ],
    [1.45, 1.0, 1.35, 1.35, 1.95],
)
add_bullet(doc, "我们目前只完成了 SSDD 的可靠对比；SAR-ShipDataset 和 SAR-AIRcraft1.0 还不能填我们的方法结果。")
add_bullet(doc, "MaDiNet 复杂度数字是官方 MambaSAR res4+res5 核心模块的本地测量，不是完整 MaDiNet detector。")
add_bullet(doc, "可汇报为：我们在 SSDD 上接近 MaDiNet AP50，并以更轻量的检测链路实现 SAR-aware 状态更新。")

doc.add_heading("4. 论文目前写成什么样子", level=1)
doc.add_paragraph("当前论文已经不是空白草稿，而是形成了一个 AAAI 风格的完整初稿骨架。已有文件包括：")
add_bullet(doc, "英文 LaTeX：paper_draft/aaai_ds_ssm_paper_en.tex")
add_bullet(doc, "英文 PDF 预览：paper_draft/aaai_ds_ssm_paper_en.pdf")
add_bullet(doc, "中文同步稿：paper_draft/aaai_ds_ssm_paper_zh.md")
add_bullet(doc, "BibTeX：paper_draft/aaai_ds_ssm_refs.bib，但参考文献仍需要清理和补全。")

add_table(
    doc,
    ["章节", "当前状态", "还需要补的内容"],
    [
        ["摘要", "已写明问题、方法、PFIM 三路径调制和 SSDD 结果", "最终实验确认后微调数值和措辞"],
        ["引言", "已建立 SAR 斑点噪声、SSM 状态传播和轻量化部署动机", "需要加入更强的文献铺垫和贡献凝练"],
        ["相关工作", "已有 SAR 检测、视觉 SSM、轻量 SAR Mamba 检测器三部分", "引用需要补全，避免泛泛而谈"],
        ["方法", "已有 DS-SSM、能量抑制、PFIM、三路径调制、DG-DS-SSM 扩展", "需要配正式框架图和模块图"],
        ["实验", "已有 SSDD 主结果、复杂度表、讨论", "需要统一命令复测、补分支消融、可选跨数据集实验"],
        ["局限与结论", "已明确 MaDiNet 复杂度口径和缺失实验", "投稿版可压缩局限，增强主结论"],
    ],
    [0.95, 2.65, 2.9],
)

doc.add_heading("5. 汇报时建议怎么讲", level=1)
add_number(doc, "先讲问题：SAR 舰船目标小、稀疏，斑点噪声和海杂波会沿 SSM 状态传播，影响检测。")
add_number(doc, "再讲核心创新：不是外部去噪，而是在 SSM 状态更新内部做去斑输入、状态抑制和重要性引导。")
add_number(doc, "强调 PFIM：它调制 gamma、write_gate、delta，使模型既抑制噪声，又尽量保留真实舰船响应。")
add_number(doc, "最后讲结果：SSDD AP50 98.704、AP50:95 74.022，参数 5.9648M、13.2079 GFLOPs。")
add_number(doc, "和 MaDiNet 对比时讲轻量化和机制差异，不讲全面 SOTA。")

doc.add_heading("6. 目前不能过度声称的内容", level=1)
add_bullet(doc, "不能说全面超过 MaDiNet，MaDiNet 的 SSDD AP50 仍是 99.0。")
add_bullet(doc, "不能说完整 MaDiNet detector 的参数量和 FLOPs 已经测完，目前只测了官方 MambaSAR 核心模块。")
add_bullet(doc, "不能说 PFIM 三个分支贡献已经完全隔离，除非后续补上 gamma/write_gate/delta 分支消融。")
add_bullet(doc, "不能说我们方法已在 SAR-ShipDataset 或 SAR-AIRcraft1.0 上验证，目前这些结果为空。")

doc.add_heading("7. 下一步优先事项", level=1)
add_table(
    doc,
    ["优先级", "任务", "目的"],
    [
        ["P0", "固定命令复测 PFIM ex27", "形成投稿可追溯的主结果"],
        ["P1", "做 PFIM 分支消融", "证明 gamma、write_gate、delta 的贡献"],
        ["P1", "补 SAR-ShipDataset 实验", "支撑跨数据集泛化或至少同类船检数据集对比"],
        ["P2", "完善框架图、模块图、状态更新调制图", "让创新点更容易被审稿人理解"],
        ["P2", "清理 BibTeX 与 MaDiNet/AMANet/CPN-YOLO/SURGE 等引用", "提高论文可信度和规范性"],
    ],
    [0.75, 2.4, 3.35],
)

doc.add_heading("8. 汇报结尾可用表述", level=1)
doc.add_paragraph(
    "当前工作已经形成较清晰的论文主线：IG-DS-SSM / PFIM 通过把去斑和重要性引导注入状态空间动态，使轻量 Mamba-YOLO 在 SSDD 上达到 98.704 AP50 和 74.022 AP50:95。与 MaDiNet 相比，我们不主张 AP50 全面领先，而是强调在接近强基线精度的同时显著降低模型规模，并提供更可解释的 SAR-aware 状态更新机制。后续重点是复测主结果、补分支消融和跨数据集验证。"
)

doc.add_paragraph()
footer = doc.sections[0].footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run("IG-DS-SSM / PFIM current status brief")
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(100, 100, 100)

doc.save(OUT)
print(OUT)
