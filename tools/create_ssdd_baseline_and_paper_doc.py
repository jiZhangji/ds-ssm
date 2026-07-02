from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, r"C:\Users\13207\.cache\codex-runtimes\codex-primary-runtime\dependencies\python")

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path.cwd()


SSDD_METHODS = [
    # method, open, family, R, P, AP50, F1, repo_or_impl, reproduction_track, priority, note
    ("SAR-ShipNet", "no", "SAR anchor-free", "76.3", "95.1", "89.1", "85.0", "paper only", "quote only", "low", "MaDiNet table marks non-open-source."),
    ("DCTC", "no", "SAR anchor-free", "", "", "93.9", "", "paper only", "quote only", "low", "MaDiNet table marks non-open-source."),
    ("GEFCSI-Net", "no", "SAR-specific", "96.9", "93.1", "97.1", "95.0", "paper only", "quote only", "medium", "Strong non-open SAR baseline."),
    ("SARNas", "no", "SAR-specific NAS", "94.8", "97.4", "98.5", "96.1", "paper only", "quote only", "medium", "Strong non-open SAR baseline."),
    ("MGA-Net", "no", "SAR-specific", "99.0", "94.7", "98.6", "96.8", "paper only", "quote only", "medium", "Strong non-open SAR baseline."),
    ("Faster R-CNN", "yes", "two-stage", "90.4", "87.1", "89.7", "88.7", "MMDetection / Detectron2", "mmdetection", "medium", "Classic two-stage detector; useful but far below modern YOLO on SSDD."),
    ("Cascade R-CNN", "yes", "two-stage", "90.8", "84.1", "90.5", "92.4", "MMDetection / Detectron2", "mmdetection", "medium", "Classic cascade two-stage detector."),
    ("RetinaNet", "yes", "one-stage anchor-based", "93.1", "94.0", "95.0", "93.6", "MMDetection / Detectron2", "mmdetection", "medium", "Strong anchor-based baseline."),
    ("CenterNet", "yes", "anchor-free", "93.6", "94.3", "95.1", "93.9", "MMDetection", "mmdetection", "medium", "Anchor-free baseline."),
    ("FCOS", "yes", "anchor-free", "93.9", "94.6", "95.3", "94.2", "MMDetection", "mmdetection", "medium", "Anchor-free baseline."),
    ("ConsistencyDet", "yes", "generative/consistency", "94.0", "94.5", "95.6", "94.2", "https://github.com/Tankowa/ConsistencyDet", "official repo", "medium", "Open-source but requires separate Detectron2-like environment."),
    ("YOLOv3", "yes", "YOLO", "94.8", "95.5", "96.2", "95.2", "Ultralytics / MMYOLO", "yolo", "high", "Easy to reproduce, useful classic YOLO baseline."),
    ("YOLOX", "yes", "YOLO", "93.8", "95.0", "96.2", "94.4", "https://github.com/Megvii-BaseDetection/YOLOX", "official repo or MMYOLO", "high", "Good modern one-stage baseline."),
    ("DiffusionDet", "yes", "generative/diffusion", "93.9", "95.1", "96.3", "94.5", "https://github.com/ShoufaChen/DiffusionDet", "official repo", "medium", "Needs Detectron2-style environment."),
    ("YOLOv5n", "yes", "YOLO lightweight", "95.8", "97.0", "98.0", "96.4", "https://github.com/ultralytics/yolov5", "yolo", "high", "Important lightweight baseline."),
    ("YOLOv8n", "yes", "YOLO lightweight", "96.4", "97.3", "98.1", "96.9", "https://github.com/ultralytics/ultralytics", "yolo", "high", "Important lightweight baseline; closest to our code ecosystem."),
    ("YOLOv5s", "yes", "YOLO lightweight", "96.4", "97.2", "98.3", "97.0", "https://github.com/ultralytics/yolov5", "yolo", "high", "Strong lightweight YOLO baseline."),
    ("DiffDet4SAR", "yes", "SAR diffusion", "95.2", "95.3", "96.9", "95.2", "https://github.com/JoyeZLearning/DiffDet4SAR", "official repo", "medium", "Open-source SAR diffusion baseline from same group as MaDiNet."),
    ("MaDiNet", "yes", "SAR diffusion + Mamba", "97.0", "98.0", "99.0", "97.5", "https://github.com/JoyeZLearning/MaDiNet", "official repo", "reference", "SOTA reference; current server lacks Detectron2 for full detector profiling."),
    ("Ours: DS-SSM/PFIM", "yes", "lightweight Mamba-YOLO", "95.219", "97.378", "98.704", "", "local Mamba-YOLO-ds-ssm", "already measured", "main", "Current best verified row: P=0.97378, R=0.95219, mAP50=0.98704; complexity measured on server."),
]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_font(run, font="宋体"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def heading(doc, text, level=1):
    p = doc.add_heading("", level=level)
    r = p.add_run(text)
    set_font(r, "黑体")
    return p


def shade(row):
    for cell in row.cells:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "D9EAF7")
        cell._tc.get_or_add_tcPr().append(shd)


def make_registry():
    fields = [
        "method", "open_source", "family", "R_percent", "P_percent", "AP50_percent", "F1_percent",
        "repo_or_implementation", "reproduction_track", "priority", "note",
    ]
    rows = [
        {
            "method": m[0],
            "open_source": m[1],
            "family": m[2],
            "R_percent": m[3],
            "P_percent": m[4],
            "AP50_percent": m[5],
            "F1_percent": m[6],
            "repo_or_implementation": m[7],
            "reproduction_track": m[8],
            "priority": m[9],
            "note": m[10],
        }
        for m in SSDD_METHODS
    ]
    write_csv(ROOT / "database" / "madinet_ssdd_baseline_methods.csv", rows, fields)
    return rows


def make_repro_folder(rows):
    base = ROOT / "baseline_reproduction" / "ssdd_madinet_table"
    base.mkdir(parents=True, exist_ok=True)
    (base / "README.md").write_text(
        """# SSDD Baseline Reproduction Folder

This folder records the preparation work needed to reproduce open-source methods from the MaDiNet SSDD comparison table.

## Tracks

- `yolo`: YOLOv3, YOLOv5n, YOLOv5s, YOLOv8n, YOLOX. These should be reproduced first because they are lightweight, open-source, and close to the current project.
- `mmdetection`: Faster R-CNN, Cascade R-CNN, RetinaNet, FCOS, CenterNet. These can be reproduced using MMDetection/MMYOLO style configs after converting SSDD to COCO format.
- `official repo`: DiffusionDet, ConsistencyDet, DiffDet4SAR, MaDiNet. These require separate Detectron2/DiffusionDet environments.
- `quote only`: methods marked non-open-source in MaDiNet should be cited from the paper unless official code becomes available.

## Required Preparation

1. Freeze the SSDD split used by this project and MaDiNet comparison.
2. Export SSDD labels to both YOLO and COCO formats.
3. Use the same confidence threshold when comparing with MaDiNet table: confidence=0.5.
4. Record Params, GFLOPs, latency, P, R, AP50, and F1 for every reproduced method.
5. Keep each method in an isolated environment or container; do not mix Detectron2 baselines with the current Mamba-YOLO environment.
""",
        encoding="utf-8",
    )
    fields = ["method", "command_template", "status"]
    commands = [
        {
            "method": "YOLOv8n",
            "command_template": "yolo detect train model=yolov8n.yaml data=datasets/SSDD_YOLO/ssdd.yaml imgsz=640 epochs=300 batch=32 device=0 project=output_dir/ssdd_baselines name=yolov8n_ssdd_300e",
            "status": "ready after environment confirmation",
        },
        {
            "method": "YOLOv5n/YOLOv5s",
            "command_template": "python train.py --data datasets/SSDD_YOLO/ssdd.yaml --weights yolov5n.pt --img 640 --epochs 300 --batch 32 --project output_dir/ssdd_baselines --name yolov5n_ssdd_300e",
            "status": "needs YOLOv5 repo/environment",
        },
        {
            "method": "YOLOX",
            "command_template": "python tools/train.py -f exps/ssdd/yolox_s_ssdd.py -d 1 -b 32 --fp16",
            "status": "needs YOLOX repo and SSDD COCO conversion",
        },
        {
            "method": "MMDetection baselines",
            "command_template": "python tools/train.py configs/ssdd/{method}_ssdd.py --work-dir work_dirs/ssdd/{method}",
            "status": "needs MMDetection environment and SSDD COCO conversion",
        },
        {
            "method": "DiffusionDet/ConsistencyDet/DiffDet4SAR/MaDiNet",
            "command_template": "python tools/train_net.py --config-file configs/ssdd/{method}.yaml --num-gpus 1",
            "status": "needs Detectron2-style environment and official config adaptation",
        },
    ]
    write_csv(base / "command_templates.csv", commands, fields)

    md = ROOT / "reports" / "madinet_ssdd_baseline_reproduction_plan.md"
    lines = [
        "# MaDiNet SSDD 表格开源方法复现计划",
        "",
        "生成时间：2026-06-14",
        "",
        "## 来源",
        "",
        "方法名单来自 MaDiNet arXiv 源码 `main_MaDiNet.tex` 中 `Comparisons of detection performance on SSDD` 表格。",
        "",
        "## SSDD 表格方法清单",
        "",
        "| 方法 | 开源 | 类型 | R | P | AP50 | F1 | 复现路径 | 优先级 |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['open_source']} | {r['family']} | {r['R_percent']} | {r['P_percent']} | {r['AP50_percent']} | {r['F1_percent']} | {r['reproduction_track']} | {r['priority']} |"
        )
    lines += [
        "",
        "## 推荐优先复现顺序",
        "",
        "1. YOLOv8n、YOLOv5n、YOLOv5s、YOLOX：轻量、开源、与本文轻量化叙事最相关。",
        "2. Faster R-CNN、Cascade R-CNN、RetinaNet、FCOS、CenterNet：使用 MMDetection 统一复现，作为传统检测器对比。",
        "3. DiffusionDet、ConsistencyDet、DiffDet4SAR、MaDiNet：需要 Detectron2/DiffusionDet 环境，作为生成式检测器对比。",
        "4. 非开源 SAR 专用方法：只引用 MaDiNet 表格或原论文数值，除非后续找到官方代码。",
        "",
        "## 当前已完成",
        "",
        "- 已下载 MaDiNet arXiv PDF 与源码。",
        "- 已解析 SSDD 对比表格方法名单。",
        "- 已在服务器上测得本文完整检测器复杂度。",
        "- 已在服务器上测得 MaDiNet 官方 MambaSAR 核心模块和若干官方组件复杂度。",
        "",
        "## 下一步前置工作",
        "",
        "1. 固定 SSDD 数据划分，并确认是否与 MaDiNet 使用的 Official-SSDD 一致。",
        "2. 生成 YOLO 格式和 COCO 格式两套标注。",
        "3. 为 YOLO 系列准备统一训练脚本。",
        "4. 单独建立 MMDetection/Detectron2 环境，不污染当前 Mamba-YOLO 环境。",
        "5. 每个 baseline 训练完成后，统一测试并记录 P/R/AP50/F1、Params、GFLOPs、Latency。",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return base, md


def make_word(rows):
    doc = Document()
    doc.styles["Normal"].font.name = "宋体"
    doc.styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.styles["Normal"].font.size = Pt(10.5)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("面向轻量高效 SAR 舰船检测的 DS-SSM 论文写作框架")
    r.bold = True
    r.font.size = Pt(16)
    set_font(r, "黑体")

    doc.add_paragraph("生成日期：2026-06-14")
    doc.add_paragraph("说明：本文档基于 MaDiNet SSDD 对比表、本文已完成实验结果、服务器复杂度实测结果整理，用于后续论文写作。")

    heading(doc, "一、研究背景与问题动机", 1)
    doc.add_paragraph(
        "SAR 舰船检测面临斑点噪声、海杂波、近岸金属散射体干扰、小目标密集分布和目标尺度变化等挑战。"
        "在 SSDD 这类公开数据集上，MaDiNet 等扩散式 Mamba 检测方法已经取得很高的 AP50，继续单纯追求小幅精度提升的边际收益有限。"
        "因此，本文需要从更符合实际应用的角度重新定义问题：如何在保持接近强 SOTA 精度的同时，显著降低模型参数量、计算量和部署成本。"
    )
    doc.add_paragraph(
        "这一动机与 SAR 大范围海域监测高度一致。实际系统往往需要处理大量遥感图像，模型不仅要准确，还需要轻量、快速、稳定。"
        "因此，本文可以将核心贡献定位为：一种面向 SAR 斑点噪声与复杂背景的轻量高效选择性状态空间检测模型。"
    )

    heading(doc, "二、MaDiNet SSDD 对比表中的开源 baseline", 1)
    doc.add_paragraph("MaDiNet 在 SSDD 表格中同时列出了非开源 SAR 专用方法和开源通用检测器。建议论文中重点复现开源方法，并将非开源方法作为引用比较。")
    table = doc.add_table(rows=1, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["方法", "开源", "类型", "AP50(%)", "P(%)", "R(%)", "复现建议"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    shade(table.rows[0])
    for r0 in rows:
        if r0["priority"] in {"high", "medium", "reference", "main"}:
            cells = table.add_row().cells
            vals = [r0["method"], r0["open_source"], r0["family"], r0["AP50_percent"], r0["P_percent"], r0["R_percent"], r0["reproduction_track"]]
            for i, v in enumerate(vals):
                cells[i].text = str(v)

    heading(doc, "三、本文已有实验与复杂度结果", 1)
    doc.add_paragraph(
        "当前本文 PFIM / IG-DS-SSM 主线在 SSDD 上的代表性结果为：P=97.378%，R=95.219%，AP50=98.704%，mAP50-95=74.022%。"
        "DG-DS-SSM 延长线在 TTA IoU=0.55 设置下达到 P=96.3%，R=96.2%，AP50=98.7%。"
    )
    doc.add_paragraph(
        "服务器复杂度实测显示，DS-SSM 完整检测器约 5.9648M 参数、13.2079 GFLOPs；DG-DS-SSM 完整检测器约 5.9632M 参数、13.1554 GFLOPs。"
        "相比之下，MaDiNet 官方 MambaSAR res4+res5 核心模块单独达到 223.2648M 参数和 345.283 GFLOPs。"
        "虽然这不是 MaDiNet 完整检测器统计，但已经说明其核心 MambaSAR 模块属于重型设计。"
    )

    heading(doc, "四、建议的顶会级论文创新表述", 1)
    bullets = [
        "提出 DS-SSM：一种面向 SAR 舰船检测的 De-Speckle Selective State Space Model，在选择性状态更新中抑制斑点噪声和海杂波干扰。",
        "提出 PFIM 感知重要性引导：利用目标相关重要性图调制状态更新，使模型在轻量结构下更关注舰船候选区域。",
        "从精度导向扩展到精度-效率联合优化：在接近 MaDiNet SSDD AP50 的同时，保持约 6M 参数和约 13G FLOPs 的轻量规模。",
        "构建可复现实验体系：将 MaDiNet SSDD 表格中的开源 baselines 分为 YOLO、MMDetection、Diffusion/Consistency 三条复现路线，并统一记录精度与复杂度。"
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    heading(doc, "五、论文背景段落草稿", 1)
    doc.add_paragraph(
        "Recent SAR target detectors have achieved remarkable accuracy on public benchmarks such as SSDD. "
        "However, the increasing accuracy of heavy diffusion-based or Mamba-based detectors is often accompanied by substantial computational overhead. "
        "For large-scale maritime surveillance, where massive SAR images must be processed under limited computational resources, detection efficiency becomes as critical as accuracy. "
        "This motivates us to revisit SAR ship detection from an accuracy-efficiency perspective and design a compact state-space detector tailored to speckle-corrupted SAR imagery."
    )
    doc.add_paragraph(
        "中文版本：近年来，MaDiNet 等方法在 SSDD 上取得了很高的检测精度，但其扩散式检测框架和重型 MambaSAR 模块也带来了较高复杂度。"
        "对于大范围海域监测和资源受限部署场景，仅追求极限 AP50 并不足够。本文因此从精度-效率联合优化出发，提出轻量 DS-SSM，在保持接近强 SOTA 检测性能的同时显著降低参数量和 FLOPs。"
    )

    heading(doc, "六、后续复现实验路线", 1)
    steps = [
        "第一阶段：优先复现 YOLOv8n、YOLOv5n、YOLOv5s、YOLOX，建立轻量检测器对比。",
        "第二阶段：使用 MMDetection 复现 Faster R-CNN、Cascade R-CNN、RetinaNet、FCOS、CenterNet。",
        "第三阶段：单独建立 Detectron2 环境，复现 DiffusionDet、ConsistencyDet、DiffDet4SAR、MaDiNet。",
        "第四阶段：统一汇总 AP50、P、R、F1、Params、GFLOPs、Latency，形成论文主表。"
    ]
    for s in steps:
        doc.add_paragraph(s, style="List Number")

    out = ROOT / "paper_draft" / "DS_SSM_顶会论文背景动机与轻量化对比框架.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out


def main():
    rows = make_registry()
    folder, report = make_repro_folder(rows)
    docx = make_word(rows)
    summary = {"registry": str(ROOT / "database" / "madinet_ssdd_baseline_methods.csv"), "folder": str(folder), "report": str(report), "docx": str(docx)}
    (ROOT / "reports" / "ssdd_baseline_paper_outputs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
