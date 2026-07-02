import hashlib
import json
import os
from collections import defaultdict


SERVERS = ["server_1080ti", "server_3090_a", "server_3090_b"]
ROOT = "source_code"
STATS = "downloads/source_code_raw/all_download_stats.json"

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_server(server):
    base = os.path.join(ROOT, server)
    out = {}
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, base).replace("\\", "/")
            try:
                st = os.stat(full)
                out[rel] = {"sha1": sha1(full), "bytes": st.st_size}
            except OSError:
                pass
    return out


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def classify(path):
    p = path.lower()
    model = any(x in p for x in ["ultralytics/nn/", "cfg/models/", "mamba_yolo", "tasks.py"])
    train = any(x in p for x in ["train", "trainer.py", "default.yaml", "tools/", "selective_scan/setup.py"])
    infer = any(x in p for x in ["predict", "val.py", "validator.py", "metrics.py", "tasks.py"])
    result = model or train or infer or any(x in p for x in ["yaml", "metrics", "loss", "dataset"])
    return ", ".join([name for name, val in [("model", model), ("train", train), ("infer/val", infer), ("result", result)] if val]) or ""


def main():
    with open(STATS, "r", encoding="utf-8") as f:
        stats = json.load(f)

    files = {s: walk_server(s) for s in SERVERS}
    all_paths = sorted(set().union(*(set(files[s]) for s in SERVERS)))

    conflicts = []
    missing_or_diff = []
    for path in all_paths:
        vals = {s: files[s].get(path, {}).get("sha1", "<missing>") for s in SERVERS}
        present_hashes = {v for v in vals.values() if v != "<missing>"}
        if len(set(vals.values())) > 1:
            missing_or_diff.append((path, vals))
        if len(present_hashes) > 1:
            conflicts.append((path, vals))

    report = ["# Source Code Recovery", ""]
    report += [
        "## Scope",
        "",
        "- Source snapshots were downloaded into separate local directories; no merge was performed.",
        "- Servers were accessed read-only by SFTP.",
        "- Excluded from download: `output_dir/`, `datasets/`, `weights/`, `runs/`, common weight/media/log suffixes, and non-Git files larger than 5MB.",
        "- Experiment recovery was not performed.",
        "",
        "## Local Repository Structure",
        "",
        "```text",
        "source_code/",
        "  server_1080ti/",
        "  server_3090_a/",
        "  server_3090_b/",
        "reports/",
        "inventory/",
        "database/",
        "downloads/",
        "```",
        "",
        "## Download Statistics",
        "",
        "| Server | Host | Files downloaded | Bytes downloaded | Skipped files | Skipped dirs | Error |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for s in SERVERS:
        st = stats[s]
        report.append(f"| `{s}` | `{st['host']}` | `{st['files']}` | `{st['bytes']}` | `{len(st['skipped_files'])}` | `{len(st['skipped_dirs'])}` | `{st.get('error') or ''}` |")

    report += [
        "",
        "## Version Notes",
        "",
        "- `server_1080ti`: preserved as an exploratory/PFIM-heavy code version with unique trainer/selective-scan modifications and PFIM files.",
        "- `server_3090_a`: preserved as an important 3090 lineage with SAR-aircraft/ex-series files.",
        "- `server_3090_b`: preserved as an important 3090 lineage with additional metrics modification and later ex-series files.",
        "",
        "## Difference Against Official Mamba-YOLO",
        "",
        "- See `reports/project_diff_report.md`, `reports/code_diff_matrix.md`, and `reports/all_modified_files.md` from phase 2A.",
        "- Baseline remains shared commit `b26cbda230dfa217f96faee8dc7020db3962f3df` from `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.",
        "- The recovered local snapshots include `.git` metadata for each server copy.",
        "",
        "## Multiple Versions",
        "",
        f"- Paths present with different content on at least two servers: `{len(conflicts)}`.",
        f"- Paths with either different content or missing on one or more servers: `{len(missing_or_diff)}`.",
        "- No same-name files were overwritten; all versions remain under their source server directory.",
        "",
        "## Files Needing Manual Confirmation",
        "",
        "High-priority candidates are same-path conflicts that can affect model structure, training, inference/validation, or final reported results.",
        "",
        "| File | 1080Ti | 3090-A | 3090-B | Impact |",
        "|---|---|---|---|---|",
    ]
    for path, vals in conflicts:
        impact = classify(path)
        if impact:
            report.append(f"| `{path}` | `{vals['server_1080ti'][:12]}` | `{vals['server_3090_a'][:12]}` | `{vals['server_3090_b'][:12]}` | `{impact}` |")
    write("reports/source_code_recovery.md", "\n".join(report))

    conflict = ["# Code Conflict Report", ""]
    conflict += [
        "Same relative path exists with different content across at least two recovered server snapshots. These files were not merged or overwritten.",
        "",
        "| File | 1080Ti | 3090-A | 3090-B | Bytes 1080Ti | Bytes 3090-A | Bytes 3090-B | Impact |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for path, vals in conflicts:
        bytes_vals = {s: files[s].get(path, {}).get("bytes", "") for s in SERVERS}
        conflict.append(
            f"| `{path}` | `{vals['server_1080ti'][:12]}` | `{vals['server_3090_a'][:12]}` | `{vals['server_3090_b'][:12]}` | "
            f"`{bytes_vals['server_1080ti']}` | `{bytes_vals['server_3090_a']}` | `{bytes_vals['server_3090_b']}` | `{classify(path)}` |"
        )
    conflict += [
        "",
        "## Presence Differences",
        "",
        "Same path is missing from at least one server or differs by content. This section is broader than direct conflicts and helps preserve provenance.",
        "",
        "| File | 1080Ti | 3090-A | 3090-B | Impact |",
        "|---|---|---|---|---|",
    ]
    for path, vals in missing_or_diff:
        conflict.append(f"| `{path}` | `{vals['server_1080ti'][:12]}` | `{vals['server_3090_a'][:12]}` | `{vals['server_3090_b'][:12]}` | `{classify(path)}` |")
    write("reports/code_conflict_report.md", "\n".join(conflict))


if __name__ == "__main__":
    main()
