import csv
import json
import os
import posixpath
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"
SERVER_ID = "server_current_3090x2"

TRAIN = "output_dir/ssdd/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3"
TEST = "output_dir/ssdd_test/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3"
TTA = "output_dir/ssdd_test/ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bsx2-ex27-codex3_v22"

TEST_METRICS = {
    "standard_test": {"precision": 0.983, "recall": 0.926, "mAP50": 0.978, "mAP50-95": 0.729},
    "tta_iou0.55_conf0.001": {"precision": 0.969, "recall": 0.952, "mAP50": 0.978, "mAP50-95": 0.715},
}


def safe_name(s):
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s.replace("/", "__"))[:180]


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def list_files(ssh, remote_dir):
    keep = (
        "-name 'best.pt' -o -name 'last.pt' -o -name 'args.yaml' -o -name 'opt.yaml' "
        "-o -name 'results.csv' -o -name 'results.png' -o -name 'PR_curve.png' "
        "-o -name 'P_curve.png' -o -name 'R_curve.png' -o -name 'F1_curve.png' "
        "-o -name 'confusion_matrix.png' -o -name 'confusion_matrix_normalized.png'"
    )
    cmd = f"find {posixpath.join(PROJECT, remote_dir)!r} -maxdepth 2 -type f \\( {keep} \\) 2>/dev/null | sort"
    _, out, _ = ssh.exec_command(cmd, timeout=60)
    return [x for x in out.read().decode("utf-8", errors="replace").splitlines() if x]


def download_group(sftp, ssh, remote_dir, local_id, group_name):
    exp_dir = os.path.join("experiments", local_id, group_name)
    os.makedirs(exp_dir, exist_ok=True)
    downloaded = []
    for rp in list_files(ssh, remote_dir):
        base = posixpath.basename(rp)
        if base in ("best.pt", "last.pt"):
            dest = os.path.join("weights", f"{local_id}__{base}")
        elif base.endswith(".log"):
            dest = os.path.join("logs", f"{local_id}__{base}")
        else:
            dest = os.path.join(exp_dir, base)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        sftp.get(rp, dest)
        downloaded.append(
            {
                "group": group_name,
                "remote_path": rp,
                "local_path": dest.replace("\\", "/"),
                "size": os.path.getsize(dest),
                "download_time": datetime.now().isoformat(timespec="seconds"),
            }
        )
    return downloaded


def parse_train_results(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clean = {k.strip(): v.strip() for k, v in r.items()}
            rows.append(clean)
    def f(row, key):
        return float(row.get(key, "nan"))
    best = max(rows, key=lambda r: f(r, "metrics/mAP50-95(B)"))
    last = rows[-1]
    return {
        "epochs": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_precision": f(best, "metrics/precision(B)"),
        "best_recall": f(best, "metrics/recall(B)"),
        "best_mAP50": f(best, "metrics/mAP50(B)"),
        "best_mAP50_95": f(best, "metrics/mAP50-95(B)"),
        "last_epoch": int(float(last["epoch"])),
        "last_precision": f(last, "metrics/precision(B)"),
        "last_recall": f(last, "metrics/recall(B)"),
        "last_mAP50": f(last, "metrics/mAP50(B)"),
        "last_mAP50_95": f(last, "metrics/mAP50-95(B)"),
    }


def write_tables(record):
    os.makedirs("database", exist_ok=True)
    csv_path = os.path.join("database", "current_codex3_performance_comparison.csv")
    fields = [
        "experiment_id",
        "source_server",
        "mode",
        "epoch",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "notes",
    ]
    rows = [
        {
            "experiment_id": record["local_id"],
            "source_server": SERVER_ID,
            "mode": "train_best_val",
            "epoch": record["train_metrics"]["best_epoch"],
            "precision": record["train_metrics"]["best_precision"],
            "recall": record["train_metrics"]["best_recall"],
            "mAP50": record["train_metrics"]["best_mAP50"],
            "mAP50-95": record["train_metrics"]["best_mAP50_95"],
            "notes": "Best row recomputed from training results.csv",
        },
        {
            "experiment_id": record["local_id"],
            "source_server": SERVER_ID,
            "mode": "train_last_val",
            "epoch": record["train_metrics"]["last_epoch"],
            "precision": record["train_metrics"]["last_precision"],
            "recall": record["train_metrics"]["last_recall"],
            "mAP50": record["train_metrics"]["last_mAP50"],
            "mAP50-95": record["train_metrics"]["last_mAP50_95"],
            "notes": "Last epoch from training results.csv",
        },
    ]
    for mode, metrics in TEST_METRICS.items():
        rows.append(
            {
                "experiment_id": record["local_id"],
                "source_server": SERVER_ID,
                "mode": mode,
                "epoch": "",
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mAP50": metrics["mAP50"],
                "mAP50-95": metrics["mAP50-95"],
                "notes": "Parsed from user pasted test output",
            }
        )
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(os.path.join("database", "current_codex3_performance_comparison.xlsx"), index=False)
    except Exception as exc:
        with open(os.path.join("database", "current_codex3_performance_comparison.xlsx.error.txt"), "w", encoding="utf-8") as f:
            f.write(repr(exc))
    return rows


def update_reports(record, rows):
    os.makedirs("reports", exist_ok=True)
    md = [
        "# Current Codex3 Experiment Report",
        "",
        f"- Source server: `{SERVER_ID}` (`{HOST}`)",
        f"- Remote train dir: `{PROJECT}/{TRAIN}`",
        f"- Local archive: `experiments/{record['local_id']}`",
        f"- Weights: `weights/{record['local_id']}__best.pt`, `weights/{record['local_id']}__last.pt`",
        f"- Downloaded files: `{len(record['downloaded'])}`",
        "",
        "## Metrics",
        "",
        "| Mode | Epoch | P | R | mAP50 | mAP50-95 | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| `{r['mode']}` | {r['epoch']} | {float(r['precision']):.5f} | {float(r['recall']):.5f} | "
            f"{float(r['mAP50']):.5f} | {float(r['mAP50-95']):.5f} | {r['notes']} |"
        )
    md += [
        "",
        "## Analysis",
        "",
        "- The standard test result is close to the final validation row, so the recovered `best.pt` is consistent with the training record.",
        "- TTA increased recall from `0.926` to `0.952`, but precision and mAP50-95 decreased. This suggests extra augmented predictions recovered more ships but introduced localization/duplicate-quality penalties under COCO-style IoU averaging.",
        "- Compared with the previous best recovered ex27 run (`mAP50-95 0.74022`), this DDP 80 batch run is slightly lower on mAP50-95. The most likely contributors are larger global batch size changing optimization dynamics, the 20-epoch freeze schedule, and DDP/find-unused behavior changing which parameters receive synchronized gradients during the frozen phase.",
        "- The cuDNN warning during test is a non-fatal plan-selection warning; inference completed normally.",
    ]
    with open(os.path.join("reports", "current_codex3_experiment_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    local_id = f"{SERVER_ID}__ssdd__{safe_name(posixpath.basename(TRAIN))}"
    for d in ("experiments", "weights", "logs", "downloads/current_codex3"):
        os.makedirs(d, exist_ok=True)
    ssh = connect()
    sftp = ssh.open_sftp()
    try:
        downloaded = []
        downloaded += download_group(sftp, ssh, TRAIN, local_id, "train")
        downloaded += download_group(sftp, ssh, TEST, local_id, "test_standard")
        downloaded += download_group(sftp, ssh, TTA, local_id, "test_tta_iou0.55_conf0.001")
    finally:
        sftp.close()
        ssh.close()
    result_csv = os.path.join("experiments", local_id, "train", "results.csv")
    record = {
        "local_id": local_id,
        "source_server": SERVER_ID,
        "host": HOST,
        "remote_train_dir": posixpath.join(PROJECT, TRAIN),
        "remote_test_dir": posixpath.join(PROJECT, TEST),
        "remote_tta_dir": posixpath.join(PROJECT, TTA),
        "downloaded": downloaded,
        "train_metrics": parse_train_results(result_csv),
        "archive_time": datetime.now().isoformat(timespec="seconds"),
    }
    exp_root = os.path.join("experiments", local_id)
    with open(os.path.join(exp_root, "SOURCE.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(os.path.join("downloads", "current_codex3", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    rows = write_tables(record)
    update_reports(record, rows)
    print(json.dumps({"local_id": local_id, "metrics": record["train_metrics"], "downloaded": len(downloaded)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
