from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = r"D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\reports\SAR领域发展综述与当前项目定位报告.docx"


refs = {
    1: "Zhang, T., Zhang, X. SSDD: A SAR Ship Detection Dataset. Remote Sensing, 2017/2021 dataset analysis.",
    2: "Wei, S. et al. HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation. IEEE Access, 2020.",
    3: "Zhang, T. et al. LS-SSDD-v1.0: A Deep Learning Dataset Dedicated to Small Ship Detection from Large-Scale Sentinel-1 SAR Images. Remote Sensing, 2020.",
    4: "Sun, X. et al. AIR-SARShip-1.0: High-Resolution SAR Ship Detection Dataset. IEEE GRSS EOD / related dataset release.",
    5: "Hou, B. et al. OpenSARShip: A Dataset Dedicated to Sentinel-1 Ship Interpretation. IEEE JSTARS, 2018.",
    6: "Hou, X. et al. FUSAR-Ship: Building a High-Resolution SAR-AIS Matchup Dataset of Gaofen-3 for Open SAR Ship Recognition. Science China Information Sciences, 2020.",
    7: "Wang, C. et al. SAR-AIRcraft-1.0: High-Resolution SAR Aircraft Detection and Recognition Dataset. Journal of Radars, 2023.",
    8: "Zhang, T. et al. Deep Learning for SAR Ship Detection: Past, Present and Future. Remote Sensing, 2022.",
    9: "Recent Advances in Deep-Learning-Based SAR Image Target Detection and Recognition. IEEE Transactions on Geoscience and Remote Sensing, 2025.",
    10: "A Review of Deep-Learning-Based SAR Image Ship Interpretation. IEEE Transactions on Geoscience and Remote Sensing / related IEEE venue, 2025.",
    11: "Gu, A., Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv, 2023.",
    12: "JoyeZLearning et al. MaDiNet: Mamba Diffusion Network for SAR Object Detection. arXiv 2024; IEEE TCSVT, 2025.",
    13: "YOLO-SARSI: An Improved YOLOv8-Based SAR Ship Detection Method. Remote Sensing, 2024.",
    14: "WDFA-YOLOX: Wavelet-Driven Feature-Enhanced Attention YOLOX for SAR Ship Detection. Remote Sensing, 2024.",
    15: "Ship Detection Transformer in SAR Images Based on Key Scattering. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.",
    16: "SMEP-DETR: Detection Transformer for SAR Ship Detection. Remote Sensing, 2025.",
    17: "MSTAR Public Targets Dataset. AFRL/Sandia SAR ATR benchmark dataset.",
    18: "NUDT4MSTAR: A Large-Scale SAR ATR Dataset. arXiv, 2025.",
}


def set_font(run, name="Calibri", size=None, color=None, bold=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def style_para(p, size=11, color="000000", bold=False, italic=False, after=6, before=0, align=None):
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.10
    if align is not None:
        p.alignment = align
    for run in p.runs:
        set_font(run, size=size, color=color, bold=bold, italic=italic)


def add_p(doc, text="", size=11, color="000000", bold=False, italic=False, after=6, before=0, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, size=size, color=color, bold=bold, italic=italic)
    style_para(p, size=size, color=color, bold=bold, italic=italic, after=after, before=before, align=align)
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    r = p.add_run(text)
    if level == 1:
        set_font(r, size=16, color="2E74B5", bold=True)
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
    elif level == 2:
        set_font(r, size=13, color="2E74B5", bold=True)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
    else:
        set_font(r, size=12, color="1F4D78", bold=True)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
    p.style = f"Heading {level}"
    return p


def add_refs(doc, nums):
    p = add_p(doc, "本节参考文献：", size=10.5, color="1F4D78", bold=True, before=2, after=2)
    for n in nums:
        add_p(doc, f"[{n}] {refs[n]}", size=9.5, color="404040", after=2)


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_font(r, size=10.5, color="000000")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.10


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    hdr = t.rows[0]
    set_repeat_table_header(hdr)
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.text = h
        shade_cell(c, "F2F4F7")
        set_cell_margins(c)
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in c.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                set_font(r, size=9.5, color="000000", bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val
            set_cell_margins(cells[i])
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    set_font(r, size=9, color="000000")
    if widths:
        for row in t.rows:
            for idx, width in enumerate(widths):
                row.cells[idx].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(1)
sec.bottom_margin = Inches(1)
sec.left_margin = Inches(1)
sec.right_margin = Inches(1)
sec.header_distance = Inches(0.492)
sec.footer_distance = Inches(0.492)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
styles["Normal"].font.size = Pt(11)

add_p(doc, "SAR 领域发展综述与当前项目定位报告", size=23, bold=True, color="0B2545", after=4)
add_p(doc, "面向 SAR 目标检测、数据集、近年论文脉络与 Mamba-YOLO / DS-SSM 研究定位", size=13, color="555555", after=14)
for label, value in [
    ("项目", "Mamba-YOLO-ds-ssm"),
    ("任务", "SAR 船舶/飞机目标检测；轻量化 Mamba/SSM 检测器设计"),
    ("生成日期", "2026-06-24"),
    ("报告用途", "论文选题、相关工作、实验设计和开题/组会材料整理"),
]:
    p = add_p(doc, "", after=2)
    r1 = p.add_run(label + "：")
    set_font(r1, size=10.5, color="000000", bold=True)
    r2 = p.add_run(value)
    set_font(r2, size=10.5, color="000000")

add_p(doc, "核心结论", size=12, color="1F4D78", bold=True, before=14, after=4)
add_p(doc, "当前项目属于 SAR 目标检测中的轻量化、高精度、SAR 成像特性适配方向。它不是普通的 YOLO 模块替换，而是尝试把 Mamba/视觉状态空间模型中的选择性扫描机制改造成更适合 SAR 斑点噪声、强散射响应、小目标和复杂背景的 DS-SSM/PFIM 检测框架。", after=8)
doc.add_page_break()

add_h(doc, "1. SAR 领域概况与任务边界", 1)
add_h(doc, "1.1 SAR 成像特点", 2)
add_p(doc, "SAR（Synthetic Aperture Radar，合成孔径雷达）是一类主动微波遥感成像技术，具备全天时、全天候、一定穿透能力和远距离观测能力。相比光学图像，SAR 图像的语义外观更弱，目标通常表现为强散射点、阴影、旁瓣和局部纹理组合。")
add_p(doc, "深度学习进入 SAR 后，主要任务包括目标检测、目标识别/ATR、语义分割、变化检测、船舶解译、飞机检测、建筑/道路提取、灾害监测等。当前仓库聚焦的是 SAR 目标检测，尤其是船舶和飞机这两类高价值目标。")
add_refs(doc, [8, 9, 10])

add_h(doc, "1.2 SAR 目标检测的主要难点", 2)
for x in [
    "斑点噪声强：SAR 相干成像带来乘性 speckle noise，容易让自然图像检测器误把噪声、海杂波、岸线和建筑强散射当成目标。",
    "目标外观弱语义：船舶、飞机、车辆在 SAR 中往往不是清晰轮廓，而是由散射中心和阴影构成。",
    "小目标与大背景：SSDD、HRSID、LS-SSDD 等任务中，目标面积占比低，正负样本极不均衡。",
    "复杂背景误警：港口、近岸、机场、跑道、建筑、桥梁、岛礁等区域会产生类似目标的强散射结构。",
    "跨传感器泛化难：Sentinel-1、Gaofen-3、TerraSAR-X 等平台的频段、极化、分辨率、入射角和成像模式差异明显。",
]:
    bullet(doc, x)
add_refs(doc, [2, 3, 7, 8])

add_h(doc, "2. 当前项目在做什么", 1)
add_h(doc, "2.1 任务定位", 2)
add_p(doc, "从仓库结构、实验目录、权重命名和报告文件看，当前项目主要面向 SAR 目标检测：一条主线是 SSDD 船舶检测，另一条扩展线是 SAR-AIRcraft 飞机检测。模型基座来自 Mamba-YOLO/Ultralytics YOLO 系框架，研究重点是把视觉 Mamba/SSM 模块做 SAR-aware 改造。")
add_p(doc, "更准确地说，项目目标是：在 SAR 小目标和复杂背景检测任务上，以较低参数量和计算量获得更好的 mAP，形成可与 YOLO 系、Mamba-YOLO、MaDiNet 和其他轻量化检测器对比的 DS-SSM 检测框架。")
add_refs(doc, [1, 7, 11, 12])

add_h(doc, "2.2 方法主线：DS-SSM 与 PFIM", 2)
add_p(doc, "本地研究主线显示，当前有效模型围绕 IG-DS-SSM / PFIM 展开。DS-SSM 的作用不是简单替换 backbone，而是在 Mamba-YOLO 的状态空间块中引入 SAR 成像特性适配机制，例如输入去偏/去斑残差、能量感知阻尼、选择性扫描过程中的写入控制，以及残差输出。")
add_p(doc, "PFIM 进一步估计 token-level importance，并用该重要性调制 gamma、write_gate 和 delta。直观上，gamma 控制历史状态保留和阻尼，write_gate 控制当前 token 写入强度，delta 控制状态转移步长。这样模型可以对目标区域增强写入，对噪声背景和低价值区域抑制写入。")
add_refs(doc, [11, 12])

add_h(doc, "2.3 与普通 YOLO 改进的区别", 2)
add_p(doc, "近几年 SAR 检测论文中大量工作集中在 YOLOv5/YOLOv8 的 neck、attention、loss、小目标检测头或特征融合改造。当前项目更适合被包装为 SAR-aware state-space modeling：它把 SAR 噪声、能量响应、局部统计和长程状态建模联系起来，而不只是堆叠通用注意力模块。")
add_refs(doc, [13, 14, 11, 12])

add_h(doc, "3. SAR 目标检测发展脉络", 1)
add_h(doc, "3.1 传统检测到 CNN 检测", 2)
add_p(doc, "早期 SAR 目标检测依赖 CFAR、杂波统计建模、散射中心、纹理特征、形态学处理和模板匹配。这类方法可解释性强、对少样本友好，但对近岸、港口、机场等复杂背景鲁棒性不足。")
add_p(doc, "SSDD、HRSID 等公开数据集出现后，Faster R-CNN、SSD、RetinaNet、YOLO、FCOS 等通用检测器被迁移到 SAR 图像。此阶段的核心问题是：自然图像检测器能否适应 SAR 的 speckle、弱纹理、小目标和散射结构。")
add_refs(doc, [1, 2, 8])

add_h(doc, "3.2 小目标、复杂背景与轻量化", 2)
add_p(doc, "2021 年以后，SAR 船舶检测不再满足于在 SSDD 上刷单一 mAP，而更关注 HRSID、LS-SSDD、AIR-SARShip 等复杂场景。研究重点转向小目标增强、海陆背景误警抑制、多尺度融合、轻量部署和跨数据集泛化。")
add_p(doc, "YOLO-SARSI、WDFA-YOLOX 等工作体现了这一趋势：它们多从轻量化 backbone、注意力增强、特征融合、局部结构强化等角度提高 SAR 船舶检测精度，同时控制计算开销。")
add_refs(doc, [3, 4, 13, 14])

add_h(doc, "3.3 Transformer / DETR 路线", 2)
add_p(doc, "Transformer 和 DETR 类方法试图用全局上下文建模解决复杂背景误警和目标关系建模问题。它们的优势是长程依赖强、结构表达能力好，适合从大场景中理解目标与背景关系。")
add_p(doc, "但这一路线也存在明显限制：SAR 公开数据集普遍偏小，纯 Transformer 训练成本高、数据需求大，小目标细节也可能在 token 化过程中损失。因此，Transformer/DETR 方法常常需要结合散射先验、多尺度结构或轻量化设计。")
add_refs(doc, [15, 16, 13])

add_h(doc, "3.4 Mamba / SSM 新路线", 2)
add_p(doc, "Mamba 提供了线性复杂度的长程序列建模路径，对遥感大图和高分辨率图像尤其有吸引力。相比 Transformer 的二次复杂度注意力，SSM/Mamba 在长序列、高分辨率和轻量部署上具有潜在优势。")
add_p(doc, "MaDiNet 是 SAR 目标检测中较有代表性的 Mamba 相关工作，它把 SAR 检测建模为边界框生成过程，引入 Gamma diffusion 和 MambaSAR 模块。当前项目应重点与 MaDiNet 区分：MaDiNet 偏 diffusion/generative detection，而本项目偏轻量 YOLO 检测框架中的 SAR-aware SSM 状态更新。")
add_refs(doc, [11, 12])

add_h(doc, "4. 常用 SAR 数据集", 1)
add_p(doc, "下表整理了当前项目和近年 SAR 检测论文中最常见的数据集。写论文时，建议把 SSDD 作为基础验证，把 HRSID/LS-SSDD 作为船舶检测泛化验证，把 SAR-AIRcraft 作为复杂地面背景和多类别目标扩展验证。")
table(doc,
      ["数据集", "主要任务", "规模/来源", "特点与用途", "参考"],
      [
          ["SSDD", "SAR 船舶检测", "经典公开船舶检测数据集", "最常用基准，适合快速比较；但指标较饱和", "[1]"],
          ["HRSID", "船舶检测/分割", "5,604 图像；16,951 船舶实例", "高分辨率、COCO 风格，适合复杂海域验证", "[2]"],
          ["LS-SSDD-v1.0", "大场景小船检测", "Sentinel-1 大尺度图像", "突出大背景、小目标和纯背景误警抑制", "[3]"],
          ["AIR-SARShip-1.0", "高分辨率船舶检测", "Gaofen-3 高分辨率 SAR", "港口、岛礁、复杂海面，适合复杂背景检测", "[4]"],
          ["OpenSARShip", "船舶解译/识别", "Sentinel-1 SAR-AIS chips", "结合 AIS，偏船舶识别和属性解译", "[5]"],
          ["FUSAR-Ship", "船舶识别/分类", "Gaofen-3 SAR-AIS matchup", "船型细粒度、子类丰富，适合识别任务", "[6]"],
          ["SAR-AIRcraft-1.0", "飞机检测/识别", "4,368 图；16,463 飞机实例；7 类", "机场复杂背景、多类别飞机，是当前项目的重要扩展集", "[7]"],
          ["MSTAR", "SAR ATR", "X-band 车辆目标芯片", "经典 ATR 分类基准，检测任务相关性较弱", "[17]"],
          ["NUDT4MSTAR", "大规模 SAR ATR", "40 类、多条件", "用于缓解 MSTAR 饱和和真实泛化不足", "[18]"],
      ],
      widths=[1.15, 1.15, 1.45, 2.25, 0.5])
add_refs(doc, [1, 2, 3, 4, 5, 6, 7, 17, 18])

add_h(doc, "5. 近几年顶刊/高水平论文脉络", 1)
add_h(doc, "5.1 综述与问题定义类", 2)
add_p(doc, "综述类论文通常从传统 CFAR、CNN 检测、深度学习目标检测、船舶解译、ATR、跨域泛化和未来趋势几个方面梳理领域。它们适合放在 introduction 开头，用来说明 SAR 目标检测的重要性、挑战和研究空白。")
bullet(doc, "Remote Sensing 2022 的 SAR ship detection 综述可作为船舶检测主综述。")
bullet(doc, "IEEE TGRS 2025 左右的 SAR target detection and recognition 综述可用于支撑目标检测/识别整体脉络。")
bullet(doc, "IEEE 2025 左右的 SAR ship interpretation 综述可用于连接 ship detection、recognition 和 interpretation。")
add_refs(doc, [8, 9, 10])

add_h(doc, "5.2 轻量 YOLO / CNN 改进类", 2)
add_p(doc, "这类论文数量最多，常见做法是改 YOLO backbone、neck、检测头、注意力机制、多尺度融合或 loss。优点是工程落地强、推理快、实验容易复现；缺点是同质化严重，若只在 SSDD 上提升少量 mAP，创新性容易不足。")
bullet(doc, "YOLO-SARSI 代表轻量 YOLOv8 改进路线，强调 HRSID 和 LS-SSDD 等 SAR 数据集规模有限，轻量 CNN/YOLO 仍具有优势。")
bullet(doc, "WDFA-YOLOX 代表小波/注意力/特征增强路线，在 SSDD 和 HRSID 上验证。")
add_refs(doc, [13, 14])

add_h(doc, "5.3 Transformer / DETR 类", 2)
add_p(doc, "Transformer/DETR 路线适合强调全局上下文和目标-背景关系建模。近年的工作开始把 SAR 散射先验、关键散射点、多尺度特征和 DETR 结构结合起来，以避免纯 Transformer 对小数据和小目标不友好的问题。")
add_refs(doc, [15, 16])

add_h(doc, "5.4 Mamba / SSM 与 SAR-specific 建模类", 2)
add_p(doc, "Mamba/SSM 是近两年遥感和 SAR 检测的重要新方向。它既能建模长程依赖，又比 Transformer 更适合长序列和高分辨率场景。SAR-specific Mamba 方法的关键不应只是套用视觉 Mamba，而应显式引入 speckle、散射、能量、局部统计或传感器先验。")
add_p(doc, "MaDiNet 是最需要重点讨论的相关工作。当前项目可以把它作为直接对比对象：MaDiNet 强调 Gamma diffusion 与 MambaSAR；本项目强调轻量检测器中的 DeSpeckle SSM、能量阻尼和 PFIM 动态状态调制。")
add_refs(doc, [11, 12])

add_h(doc, "6. 当前项目的论文写作定位", 1)
add_h(doc, "6.1 推荐定位语句", 2)
add_p(doc, "现有 SAR 检测方法要么依赖 CNN 局部卷积，难以充分建模复杂背景下的长程上下文；要么使用 Transformer/DETR 获取全局依赖，但计算开销大、对小规模 SAR 数据不友好。Mamba/SSM 为线性复杂度长程建模提供了新路径，但通用 SSM 没有显式考虑 SAR 斑点噪声、强散射响应和目标-背景能量差异。为此，本文提出 SAR-aware DS-SSM/PFIM，将去斑残差、能量感知阻尼和 token 重要性调制融入 Mamba-YOLO 检测框架，在 SSDD 和 SAR-AIRcraft 等数据集上实现更好的精度-复杂度平衡。")
add_refs(doc, [8, 11, 12, 13, 15])

add_h(doc, "6.2 可写成的贡献点", 2)
bullet(doc, "提出 SAR-aware DeSpeckle State Space Module，将输入去斑/去偏残差和能量感知阻尼引入选择性扫描过程。")
bullet(doc, "提出 PFIM token importance modulation，用目标重要性动态调制 gamma、write_gate 和 delta，提高目标区域状态写入并抑制噪声背景。")
bullet(doc, "构建轻量化 Mamba-YOLO SAR 检测器，在 SSDD 船舶检测和 SAR-AIRcraft 飞机检测上验证精度、参数量、GFLOPs 和推理效率。")
bullet(doc, "通过消融、复杂度对比和可视化证明 DS-SSM/PFIM 不是通用注意力堆叠，而是针对 SAR 成像特性设计的状态空间建模。")
add_refs(doc, [11, 12, 7, 1])

add_h(doc, "6.3 实验设计建议", 2)
table(doc,
      ["实验模块", "建议内容", "目的"],
      [
          ["主结果", "SSDD + SAR-AIRcraft；若时间允许补 HRSID 或 LS-SSDD", "避免只在 SSDD 饱和基准上刷指标"],
          ["消融实验", "baseline Mamba-YOLO、+DS-SSM、+PFIM、gamma/write_gate/delta 分别消融", "证明每个模块确实贡献性能"],
          ["复杂度对比", "Params、GFLOPs、FPS、显存、mAP50、mAP50-95", "突出轻量化和部署价值"],
          ["MaDiNet 对比", "同数据集、同输入尺度、同评价指标下复现实验或引用官方结果", "回应最接近的 Mamba/SAR 检测工作"],
          ["可视化", "检测案例、误警/漏检、复杂背景、Grad-CAM 或特征响应", "说明模型为何更适合 SAR 场景"],
      ],
      widths=[1.25, 3.25, 2.0])
add_refs(doc, [1, 2, 3, 7, 12])

add_h(doc, "7. 风险与后续优先级", 1)
add_p(doc, "当前项目的优势是方向新、与 Mamba/SSM 热点契合、轻量化价值明确，并且能从 SAR 成像特性角度讲出比普通 YOLO 改模块更强的故事。主要风险是 SSDD 单数据集说服力不足、与 MaDiNet 的差异需要写清楚、消融实验必须扎实。")
bullet(doc, "优先补 HRSID 或 LS-SSDD，形成 SSDD + 复杂船舶数据集的组合。")
bullet(doc, "整理 SAR-AIRcraft 结果表，把论文从 ship-only 扩展到更广义 SAR object detection。")
bullet(doc, "把 MaDiNet 作为重点相关工作和复杂度对比对象。")
bullet(doc, "做精度-复杂度散点图和典型失败案例分析。")
add_refs(doc, [2, 3, 7, 12])

add_h(doc, "8. 完整参考文献表", 1)
for n in sorted(refs):
    add_p(doc, f"[{n}] {refs[n]}", size=9.5, after=3)

doc.save(OUT)
print(OUT)
