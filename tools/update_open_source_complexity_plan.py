from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, r"C:\Users\13207\.cache\codex-runtimes\codex-primary-runtime\dependencies\python")

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path.cwd()

OPEN_SOURCE_ROWS = [
    {
        "method": "Faster R-CNN",
        "family": "two-stage",
        "AP50_percent_from_MaDiNet": "89.7",
        "P_percent_from_MaDiNet": "87.1",
        "R_percent_from_MaDiNet": "90.4",
        "F1_percent_from_MaDiNet": "88.7",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "MMDetection / Detectron2",
        "status": "open-source; needs COCO-format SSDD and MMDetection/Detectron2 env",
    },
    {
        "method": "Cascade R-CNN",
        "family": "two-stage",
        "AP50_percent_from_MaDiNet": "90.5",
        "P_percent_from_MaDiNet": "84.1",
        "R_percent_from_MaDiNet": "90.8",
        "F1_percent_from_MaDiNet": "92.4",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "MMDetection / Detectron2",
        "status": "open-source; needs COCO-format SSDD and MMDetection/Detectron2 env",
    },
    {
        "method": "RetinaNet",
        "family": "one-stage anchor-based",
        "AP50_percent_from_MaDiNet": "95.0",
        "P_percent_from_MaDiNet": "94.0",
        "R_percent_from_MaDiNet": "93.1",
        "F1_percent_from_MaDiNet": "93.6",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "MMDetection / Detectron2",
        "status": "open-source; needs COCO-format SSDD and MMDetection/Detectron2 env",
    },
    {
        "method": "CenterNet",
        "family": "anchor-free",
        "AP50_percent_from_MaDiNet": "95.1",
        "P_percent_from_MaDiNet": "94.3",
        "R_percent_from_MaDiNet": "93.6",
        "F1_percent_from_MaDiNet": "93.9",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "MMDetection",
        "status": "open-source; needs COCO-format SSDD and MMDetection env",
    },
    {
        "method": "FCOS",
        "family": "anchor-free",
        "AP50_percent_from_MaDiNet": "95.3",
        "P_percent_from_MaDiNet": "94.6",
        "R_percent_from_MaDiNet": "93.9",
        "F1_percent_from_MaDiNet": "94.2",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "MMDetection",
        "status": "open-source; needs COCO-format SSDD and MMDetection env",
    },
    {
        "method": "ConsistencyDet",
        "family": "generative/consistency",
        "AP50_percent_from_MaDiNet": "95.6",
        "P_percent_from_MaDiNet": "94.5",
        "R_percent_from_MaDiNet": "94.0",
        "F1_percent_from_MaDiNet": "94.2",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/Tankowa/ConsistencyDet",
        "status": "open-source; needs official repo + Detectron2-like env",
    },
    {
        "method": "YOLOv3",
        "family": "YOLO",
        "AP50_percent_from_MaDiNet": "96.2",
        "P_percent_from_MaDiNet": "95.5",
        "R_percent_from_MaDiNet": "94.8",
        "F1_percent_from_MaDiNet": "95.2",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "Ultralytics / MMYOLO",
        "status": "open-source; high priority",
    },
    {
        "method": "YOLOX",
        "family": "YOLO",
        "AP50_percent_from_MaDiNet": "96.2",
        "P_percent_from_MaDiNet": "95.0",
        "R_percent_from_MaDiNet": "93.8",
        "F1_percent_from_MaDiNet": "94.4",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/Megvii-BaseDetection/YOLOX",
        "status": "open-source; high priority",
    },
    {
        "method": "DiffusionDet",
        "family": "generative/diffusion",
        "AP50_percent_from_MaDiNet": "96.3",
        "P_percent_from_MaDiNet": "95.1",
        "R_percent_from_MaDiNet": "93.9",
        "F1_percent_from_MaDiNet": "94.5",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/ShoufaChen/DiffusionDet",
        "status": "open-source; needs Detectron2 env",
    },
    {
        "method": "YOLOv5n",
        "family": "YOLO lightweight",
        "AP50_percent_from_MaDiNet": "98.0",
        "P_percent_from_MaDiNet": "97.0",
        "R_percent_from_MaDiNet": "95.8",
        "F1_percent_from_MaDiNet": "96.4",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/ultralytics/yolov5",
        "status": "open-source; high priority lightweight baseline",
    },
    {
        "method": "YOLOv8n",
        "family": "YOLO lightweight",
        "AP50_percent_from_MaDiNet": "98.1",
        "P_percent_from_MaDiNet": "97.3",
        "R_percent_from_MaDiNet": "96.4",
        "F1_percent_from_MaDiNet": "96.9",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/ultralytics/ultralytics",
        "status": "open-source; high priority lightweight baseline",
    },
    {
        "method": "YOLOv5s",
        "family": "YOLO lightweight",
        "AP50_percent_from_MaDiNet": "98.3",
        "P_percent_from_MaDiNet": "97.2",
        "R_percent_from_MaDiNet": "96.4",
        "F1_percent_from_MaDiNet": "97.0",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/ultralytics/yolov5",
        "status": "open-source; high priority lightweight baseline",
    },
    {
        "method": "DiffDet4SAR",
        "family": "SAR diffusion",
        "AP50_percent_from_MaDiNet": "96.9",
        "P_percent_from_MaDiNet": "95.3",
        "R_percent_from_MaDiNet": "95.2",
        "F1_percent_from_MaDiNet": "95.2",
        "params_M": "TO_MEASURE",
        "GFLOPs": "TO_MEASURE",
        "latency_ms": "TO_MEASURE",
        "fps": "TO_MEASURE",
        "repo_or_impl": "https://github.com/JoyeZLearning/DiffDet4SAR",
        "status": "open-source; needs official repo + Detectron2 env",
    },
    {
        "method": "MaDiNet",
        "family": "SAR diffusion + Mamba",
        "AP50_percent_from_MaDiNet": "99.0",
        "P_percent_from_MaDiNet": "98.0",
        "R_percent_from_MaDiNet": "97.0",
        "F1_percent_from_MaDiNet": "97.5",
        "params_M": "223.2648*",
        "GFLOPs": "345.283*",
        "latency_ms": "27.2921*",
        "fps": "TO_MEASURE_FULL_DETECTOR",
        "repo_or_impl": "https://github.com/JoyeZLearning/MaDiNet",
        "status": "open-source; *measured official MambaSAR res4+res5 core module only; full detector still needs Detectron2",
    },
    {
        "method": "MaDiNet official MambaSAR res4+res5",
        "family": "MaDiNet core module",
        "AP50_percent_from_MaDiNet": "99.0 reference",
        "P_percent_from_MaDiNet": "98.0 reference",
        "R_percent_from_MaDiNet": "97.0 reference",
        "F1_percent_from_MaDiNet": "97.5 reference",
        "params_M": "223.2648",
        "GFLOPs": "345.283",
        "latency_ms": "27.2921",
        "fps": "N/A",
        "repo_or_impl": "official MaDiNet MambaSAR.py",
        "status": "measured on server; core res4+res5 module only, not full detector",
    },
    {
        "method": "Ours: DS-SSM/PFIM",
        "family": "lightweight Mamba-YOLO",
        "AP50_percent_from_MaDiNet": "98.704",
        "P_percent_from_MaDiNet": "97.378",
        "R_percent_from_MaDiNet": "95.219",
        "F1_percent_from_MaDiNet": "TO_COMPUTE",
        "params_M": "5.9648",
        "GFLOPs": "13.2079",
        "latency_ms": "28.7546",
        "fps": "34.777",
        "repo_or_impl": "local Mamba-YOLO-ds-ssm",
        "status": "measured on server 172.26.151.143",
    },
    {
        "method": "Ours: DG-DS-SSM",
        "family": "lightweight Mamba-YOLO extension",
        "AP50_percent_from_MaDiNet": "98.7",
        "P_percent_from_MaDiNet": "96.3",
        "R_percent_from_MaDiNet": "96.2",
        "F1_percent_from_MaDiNet": "TO_COMPUTE",
        "params_M": "5.9632",
        "GFLOPs": "13.1554",
        "latency_ms": "29.0699",
        "fps": "34.3998",
        "repo_or_impl": "local Mamba-YOLO-ds-ssm",
        "status": "measured on server 172.26.151.143",
    },
]


FIELDS = list(OPEN_SOURCE_ROWS[0].keys())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def set_font(run, font="宋体"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def heading(doc, text, level=1):
    p = doc.add_heading("", level=level)
    r = p.add_run(text)
    set_font(r, "黑体")


def shade(row):
    for cell in row.cells:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "D9EAF7")
        cell._tc.get_or_add_tcPr().append(shd)


def update_report():
    csv_path = ROOT / "database" / "open_source_ssdd_complexity_comparison_plan.csv"
    write_csv(csv_path, OPEN_SOURCE_ROWS)

    md = ROOT / "reports" / "open_source_ssdd_complexity_comparison_plan.md"
    lines = [
        "# SSDD 开源方法精度-复杂度对比计划",
        "",
        "生成时间：2026-06-14",
        "",
        "本表只包含 MaDiNet SSDD 对比表中标注为开源的方法，以及本文方法。非开源方法不进入主复现实验表，只作为论文背景引用。",
        "",
        "| 方法 | AP50 | P | R | Params(M) | GFLOPs | Latency(ms) | FPS | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in OPEN_SOURCE_ROWS:
        lines.append(
            f"| {r['method']} | {r['AP50_percent_from_MaDiNet']} | {r['P_percent_from_MaDiNet']} | {r['R_percent_from_MaDiNet']} | {r['params_M']} | {r['GFLOPs']} | {r['latency_ms']} | {r['fps']} | {r['status']} |"
        )
    lines += [
        "",
        "## 复现原则",
        "",
        "1. 非开源方法不做主表复现，不填 Params/FLOPs。",
        "2. 开源方法必须同时记录检测指标和复杂度指标：P、R、AP50、F1、Params、GFLOPs、Latency、FPS。",
        "3. YOLO 系列优先复现，因为它们与本文轻量化主线最相关。",
        "4. MMDetection 和 Detectron2 系列需要单独环境，不污染当前 Mamba-YOLO 环境。",
        "5. MaDiNet 当前只完成官方 MambaSAR 核心模块复杂度统计，完整 detector 的 Params/FLOPs 仍需 Detectron2 环境后补。",
        "6. MaDiNet 行中带 `*` 的复杂度数值表示已实测官方 `MambaSAR.py` 的 res4+res5 核心模块，不代表完整检测器。",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md


def update_docx():
    doc_path = ROOT / "paper_draft" / "DS_SSM_顶会论文背景动机与轻量化对比框架.docx"
    doc = Document(str(doc_path))
    heading(doc, "七、开源方法精度-复杂度主对比表设计", 1)
    doc.add_paragraph(
        "后续论文主对比表只纳入开源且可复现的方法。非开源 SAR 专用方法可以在相关工作或补充表中引用 MaDiNet/原论文数值，"
        "但不应进入需要 Params 和 FLOPs 的主复现实验表。"
    )
    table = doc.add_table(rows=1, cols=8)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["方法", "AP50", "P", "R", "Params(M)", "GFLOPs", "Latency", "状态"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    shade(table.rows[0])
    for r in OPEN_SOURCE_ROWS:
        cells = table.add_row().cells
        vals = [
            r["method"],
            r["AP50_percent_from_MaDiNet"],
            r["P_percent_from_MaDiNet"],
            r["R_percent_from_MaDiNet"],
            r["params_M"],
            r["GFLOPs"],
            r["latency_ms"],
            r["status"],
        ]
        for i, v in enumerate(vals):
            cells[i].text = str(v)
    doc.add_paragraph(
        "当前已经实测的复杂度结果显示，本文 DS-SSM 完整检测器为 5.9648M 参数、13.2079 GFLOPs；DG-DS-SSM 为 5.9632M 参数、13.1554 GFLOPs。"
        "MaDiNet 官方 MambaSAR res4+res5 核心模块已实测为 223.2648M 参数、345.283 GFLOPs、27.2921 ms，但这不是完整检测器结果。"
        "后续所有开源 baseline 都应补齐同样的复杂度字段，从而形成精度-效率联合对比，而不是只比较 AP50。"
    )
    doc.save(doc_path)
    return doc_path


def main():
    csv_path, md = update_report()
    doc = update_docx()
    print(json.dumps({"csv": str(csv_path), "report": str(md), "docx": str(doc)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
