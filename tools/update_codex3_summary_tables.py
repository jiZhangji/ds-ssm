import csv
import os
from pathlib import Path

from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database"
REPORTS = ROOT / "reports"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path, rows, fields, title="Sheet1"):
    wb = Workbook()
    ws = wb.active
    ws.title = title
    ws.append(fields)
    for row in rows:
        ws.append([row.get(field, "") for field in fields])
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 10), 55)
    wb.save(path)


def append_unique(rows, new_row, key_fields):
    key = tuple(new_row.get(k, "") for k in key_fields)
    out = [r for r in rows if tuple(r.get(k, "") for k in key_fields) != key]
    out.append(new_row)
    return out


def main():
    current_csv = DB / "current_codex3_performance_comparison.csv"
    current_rows = read_csv(current_csv)
    current_fields = list(current_rows[0])
    write_xlsx(DB / "current_codex3_performance_comparison.xlsx", current_rows, current_fields, "codex3")

    best = next(r for r in current_rows if r["mode"] == "train_best_val")
    standard = next(r for r in current_rows if r["mode"] == "standard_test")
    tta = next(r for r in current_rows if r["mode"].startswith("tta_"))

    summary_csv = DB / "SSDD_experiment_results_summary_recomputed.csv"
    summary_rows = read_csv(summary_csv)
    fields = list(summary_rows[0])
    new = {field: "" for field in fields}
    cn_values = {
        "实验名称": "ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3",
        "来源服务器": "server_current_3090x2",
        "最佳epoch": best["epoch"],
        "P": best["precision"],
        "R": best["recall"],
        "mAP50": best["mAP50"],
        "mAP50-95": best["mAP50-95"],
        "原rank": "NEW",
        "原score": best["mAP50-95"],
        "实际实验路径": "output_dir/ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3",
        "本地results.csv": "experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3/train/results.csv",
    }
    for key, value in cn_values.items():
        if key in new:
            new[key] = value
    summary_rows = append_unique(summary_rows, new, ["实验名称", "来源服务器"])
    if "实验序号" in fields:
        for i, row in enumerate(summary_rows, 1):
            row["实验序号"] = str(i)
    write_csv(summary_csv, summary_rows, fields)
    write_xlsx(DB / "SSDD_experiment_results_summary_recomputed.xlsx", summary_rows, fields, "SSDD")

    ranking_csv = DB / "experiment_ranking_recomputed.csv"
    ranking_rows = read_csv(ranking_csv)
    rank_fields = list(ranking_rows[0])
    rank_new = {field: "" for field in rank_fields}
    for key, value in {
        "source_server": "server_current_3090x2",
        "experiment_name": "ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3",
        "remote_dir": "output_dir/ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3",
        "training_complete": "True",
        "best_epoch": best["epoch"],
        "P": best["precision"],
        "R": best["recall"],
        "mAP50": best["mAP50"],
        "mAP50-95": best["mAP50-95"],
        "score": best["mAP50-95"],
        "epochs_recorded": "300",
        "old_rank": "NEW",
        "old_score": best["mAP50-95"],
        "results_csv": "experiments/server_current_3090x2__ssdd__ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3/train/results.csv",
    }.items():
        if key in rank_new:
            rank_new[key] = value
    ranking_rows = append_unique(ranking_rows, rank_new, ["experiment_name", "source_server"])
    def score(row):
        try:
            return float(row.get("score") or row.get("mAP50-95") or -1)
        except ValueError:
            return -1
    ranking_rows = sorted(ranking_rows, key=score, reverse=True)
    if "rank" in rank_fields:
        for i, row in enumerate(ranking_rows, 1):
            row["rank"] = str(i)
    write_csv(ranking_csv, ranking_rows, rank_fields)
    write_xlsx(DB / "experiment_ranking_recomputed.xlsx", ranking_rows, rank_fields, "ranking")

    with open(REPORTS / "current_codex3_experiment_report.md", "a", encoding="utf-8") as f:
        f.write("\n## Table Updates\n\n")
        f.write("- Updated `database/current_codex3_performance_comparison.csv` and `.xlsx`.\n")
        f.write("- Appended this run to `database/SSDD_experiment_results_summary_recomputed.csv` and `.xlsx`.\n")
        f.write("- Appended this run to `database/experiment_ranking_recomputed.csv` and `.xlsx`.\n")

    print("updated")


if __name__ == "__main__":
    main()
