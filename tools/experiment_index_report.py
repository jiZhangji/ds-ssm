import csv
import json
import os
from datetime import datetime


RAW = "inventory/raw/experiment_index_raw.json"
CSV_OUT = "database/experiment_index.csv"
XLSX_OUT = "database/experiment_index.xlsx"
REPORT_OUT = "reports/experiment_overview.md"


FIELDS = [
    "experiment_name",
    "source_server",
    "host",
    "experiment_dir",
    "created_time",
    "modified_time",
    "config_files",
    "has_best_pt",
    "has_last_pt",
    "best_pt_size_bytes",
    "log_files",
    "tensorboard_files",
    "training_complete",
]


def epoch_to_iso(value):
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return ""
    if ivalue <= 0:
        return ""
    return datetime.fromtimestamp(ivalue).isoformat(sep=" ", timespec="seconds")


def split_count(value):
    if not value:
        return 0
    return len([x for x in value.split(",") if x])


def main():
    os.makedirs("database", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open(RAW, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for server, payload in raw.items():
        for item in payload.get("rows", []):
            rows.append(
                {
                    "experiment_name": item.get("experiment_name", ""),
                    "source_server": server,
                    "host": payload.get("host", ""),
                    "experiment_dir": item.get("experiment_dir", ""),
                    "created_time": epoch_to_iso(item.get("ctime_epoch")),
                    "modified_time": epoch_to_iso(item.get("mtime_epoch")),
                    "config_files": item.get("config_files", ""),
                    "has_best_pt": "yes" if item.get("best_pt") else "no",
                    "has_last_pt": "yes" if item.get("last_pt") else "no",
                    "best_pt_size_bytes": item.get("best_pt_size", ""),
                    "log_files": item.get("log_files", ""),
                    "tensorboard_files": item.get("tensorboard_files", ""),
                    "training_complete": item.get("training_complete", "unknown"),
                }
            )

    rows.sort(key=lambda r: (r["source_server"], r["experiment_name"]))
    with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "experiment_index"
        ws.append(FIELDS)
        for row in rows:
            ws.append([row.get(field, "") for field in FIELDS])
        for col in ws.columns:
            max_len = 0
            letter = col[0].column_letter
            for cell in col:
                max_len = max(max_len, len(str(cell.value or "")))
            ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 80)
        ws.freeze_panes = "A2"
        wb.save(XLSX_OUT)
    except Exception as exc:
        with open(XLSX_OUT + ".error.txt", "w", encoding="utf-8") as f:
            f.write(repr(exc) + "\n")

    by_server = {}
    for server in raw:
        server_rows = [r for r in rows if r["source_server"] == server]
        by_server[server] = {
            "count": len(server_rows),
            "with_best": sum(1 for r in server_rows if r["has_best_pt"] == "yes"),
            "with_last": sum(1 for r in server_rows if r["has_last_pt"] == "yes"),
            "with_tb": sum(1 for r in server_rows if r["tensorboard_files"]),
            "with_logs": sum(1 for r in server_rows if r["log_files"]),
            "likely_complete": sum(1 for r in server_rows if r["training_complete"] == "likely_yes"),
        }

    report = ["# Experiment Overview", ""]
    report += [
        "## Scope",
        "",
        "- Indexed `/home/zhangji/Mamba-YOLO/output_dir` on all three servers.",
        "- Servers were read-only.",
        "- No weight files, TensorBoard files, result images, logs, or experiment artifacts were downloaded.",
        "- The index records only paths, timestamps, existence flags, and file sizes from remote metadata.",
        "",
        "## Summary",
        "",
        f"- Total indexed experiment-like directories: `{len(rows)}`.",
        "",
        "| Server | Experiments | With best.pt | With last.pt | With logs/results | With TensorBoard | Likely complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for server, s in by_server.items():
        report.append(
            f"| `{server}` | `{s['count']}` | `{s['with_best']}` | `{s['with_last']}` | "
            f"`{s['with_logs']}` | `{s['with_tb']}` | `{s['likely_complete']}` |"
        )
    report += [
        "",
        "## Indexed Files",
        "",
        f"- CSV: `database/experiment_index.csv`",
        f"- XLSX: `database/experiment_index.xlsx`",
        "",
        "## Experiment Directory List",
        "",
        "| Server | Experiment | Modified time | best.pt | last.pt | Logs/results | TensorBoard | Complete |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        report.append(
            f"| `{r['source_server']}` | `{r['experiment_name']}` | `{r['modified_time']}` | "
            f"`{r['has_best_pt']}` | `{r['has_last_pt']}` | `{split_count(r['log_files'])}` | "
            f"`{split_count(r['tensorboard_files'])}` | `{r['training_complete']}` |"
        )
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(report).rstrip() + "\n")


if __name__ == "__main__":
    main()
