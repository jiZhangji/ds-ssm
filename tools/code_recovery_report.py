import json
import os
from collections import defaultdict


RAW = "inventory/raw/code_recovery_2a.json"
REPORTS = "reports"
SERVERS = ["server_1080ti", "server_3090_a", "server_3090_b"]


def lines(text):
    return [x for x in (text or "").splitlines() if x.strip()]


def cmd(data, server, key):
    return data[server].get("commands", {}).get(key, {}).get("stdout", "").strip()


def parse_name_status(text):
    out = {}
    for line in lines(text):
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[-1]] = parts[0]
    return out


def parse_numstat(text):
    out = {}
    for line in lines(text):
        parts = line.split("\t")
        if len(parts) >= 3:
            add, delete, path = parts[0], parts[1], parts[2]
            out[path] = {"added": add, "deleted": delete}
    return out


def parse_untracked_numstat(text):
    out = {}
    for line in lines(text):
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[2]] = {"lines": parts[0], "bytes": parts[1]}
    return out


def parse_hashes(text):
    out = {}
    for line in lines(text):
        parts = line.split("\t")
        if len(parts) >= 4:
            out[parts[3]] = {"sha1": parts[0], "lines": parts[1], "bytes": parts[2]}
    return out


def classify(path):
    p = path.lower()
    model = any(x in p for x in [
        "ultralytics/nn/", "cfg/models/", "mamba_yolo", "tasks.py", "modules/", "head.py", "block.py"
    ])
    train = any(x in p for x in [
        "train", "trainer.py", "default.yaml", "cfg/", "tools/", "scripts/", "selective_scan/setup.py"
    ])
    infer = any(x in p for x in [
        "predict", "val.py", "validator.py", "metrics.py", "engine/model.py", "tasks.py"
    ])
    result = model or train or infer or any(x in p for x in ["loss", "metrics", "dataset", "augment", "yaml"])
    return model, train, infer, result


def scale(entry):
    added = entry.get("added", "")
    deleted = entry.get("deleted", "")
    if added.isdigit() and deleted.isdigit():
        total = int(added) + int(deleted)
        return f"+{added}/-{deleted} ({total} lines)"
    if "lines" in entry:
        return f"{entry['lines']} lines, {entry['bytes']} bytes"
    return "unknown"


def yes(v):
    return "yes" if v else ""


def write(name, text):
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, name), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def main():
    with open(RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    status = {s: parse_name_status(cmd(data, s, "git_diff_name_status")) for s in SERVERS}
    numstat = {s: parse_numstat(cmd(data, s, "git_diff_numstat")) for s in SERVERS}
    untracked = {s: set(lines(cmd(data, s, "git_untracked_code"))) for s in SERVERS}
    untracked_stats = {s: parse_untracked_numstat(cmd(data, s, "git_untracked_code_numstat")) for s in SERVERS}
    hashes = {s: parse_hashes(cmd(data, s, "all_code_hashes")) for s in SERVERS}

    all_modified = sorted(set().union(*(set(status[s]) | untracked[s] for s in SERVERS)))
    all_code = sorted(set().union(*(set(hashes[s]) for s in SERVERS)))
    differing_all_code = []
    for p in all_code:
        vals = {s: hashes[s].get(p, {}).get("sha1", "<missing>") for s in SERVERS}
        if len(set(vals.values())) > 1:
            differing_all_code.append((p, vals))

    rows = []
    for p in all_modified:
        sources = []
        per_server_scale = []
        change_kinds = []
        for s in SERVERS:
            if p in status[s]:
                sources.append(s)
                change_kinds.append(f"{s}:{status[s][p]}")
                per_server_scale.append(f"{s}:{scale(numstat[s].get(p, {}))}")
            elif p in untracked[s]:
                sources.append(s)
                change_kinds.append(f"{s}:A(untracked)")
                per_server_scale.append(f"{s}:{scale(untracked_stats[s].get(p, {}))}")
        model, train, infer, result = classify(p)
        rows.append({
            "path": p,
            "sources": sources,
            "kinds": change_kinds,
            "scale": per_server_scale,
            "model": model,
            "train": train,
            "infer": infer,
            "result": result,
        })

    # all_modified_files.md
    out = ["# All Modified Files", "", "Scope: code/config/documentation files outside `output_dir/` and `datasets/`. Official baseline is the shared Git HEAD `b26cbda230dfa217f96faee8dc7020db3962f3df` from `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.", ""]
    out += ["| File | Source server(s) | Change kind | Modification scale | Model structure | Training | Inference/validation | May affect final results |", "|---|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| `{r['path']}` | `{', '.join(r['sources'])}` | `{'; '.join(r['kinds'])}` | `{'; '.join(r['scale'])}` | {yes(r['model'])} | {yes(r['train'])} | {yes(r['infer'])} | {yes(r['result'])} |")
    write("all_modified_files.md", "\n".join(out))

    # code_diff_matrix.md
    mat = ["# Code Diff Matrix", "", "## Relative to Official Shared HEAD", "", "| File | 1080Ti | 3090-A | 3090-B | Impact flags |", "|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for s in SERVERS:
            if r["path"] in status[s]:
                cells.append(f"{status[s][r['path']]} {scale(numstat[s].get(r['path'], {}))}")
            elif r["path"] in untracked[s]:
                cells.append(f"A(untracked) {scale(untracked_stats[s].get(r['path'], {}))}")
            else:
                cells.append("")
        flags = ", ".join([x for x, v in [("model", r["model"]), ("train", r["train"]), ("infer", r["infer"]), ("result", r["result"])] if v])
        mat.append(f"| `{r['path']}` | `{cells[0]}` | `{cells[1]}` | `{cells[2]}` | `{flags}` |")
    mat += ["", "## Cross-Server Code Hash Differences", "", "This section includes all collected code/config/doc paths outside experiment and dataset directories, including untracked files.", "", "| File | 1080Ti | 3090-A | 3090-B |", "|---|---|---|---|"]
    for p, vals in differing_all_code:
        mat.append(f"| `{p}` | `{vals['server_1080ti'][:12]}` | `{vals['server_3090_a'][:12]}` | `{vals['server_3090_b'][:12]}` |")
    write("code_diff_matrix.md", "\n".join(mat))

    # project_diff_report.md
    report = ["# Project Diff Report", ""]
    report += [
        "## Baseline",
        "",
        "- Official comparison baseline: shared Git HEAD `b26cbda230dfa217f96faee8dc7020db3962f3df`.",
        "- Remote on all servers: `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.",
        "- No server was selected as final/master. This is a recovery analysis only.",
        "- No weights or experiment result files were downloaded or inspected by content.",
        "",
        "## Summary",
        "",
        f"- Files changed relative to official HEAD across any server: `{len(all_modified)}`.",
        f"- Cross-server code/config/doc paths with different hashes or presence: `{len(differing_all_code)}`.",
        f"- Deleted code files detected relative to HEAD: `{sum(1 for r in rows if any(k.endswith(':D') for k in r['kinds']))}`.",
        f"- New untracked code/config/doc files detected: `{sum(1 for r in rows if any('A(untracked)' in k for k in r['kinds']))}`.",
        "",
        "## By Server",
        "",
        "| Server | Modified tracked files | Untracked code/config/doc files | HEAD |",
        "|---|---:|---:|---|",
    ]
    for s in SERVERS:
        report.append(f"| `{s}` | `{len(status[s])}` | `{len(untracked[s])}` | `{cmd(data, s, 'git_head')[:12]}` |")
    report += [
        "",
        "## Impact Assessment",
        "",
        "- Model-structure impact is concentrated in `ultralytics/nn/**`, `ultralytics/nn/tasks.py`, and Mamba-YOLO YAML/model variants.",
        "- Training-flow impact is concentrated in `mbyolo_train*.py`, `ultralytics/engine/trainer.py`, `ultralytics/cfg/default.yaml`, dataset preparation tools, and selective-scan setup changes.",
        "- Inference/validation/result impact is possible where `ultralytics/utils/metrics.py`, validation modules, task parsing, or model definitions differ.",
        "- Most research changes are not committed to Git; recovery must preserve server provenance per file.",
        "",
        "## Important Observations",
        "",
        "- `server_1080ti` uniquely modifies tracked `selective_scan/setup.py` and `ultralytics/engine/trainer.py`, and contains PFIM-specific scripts/backups.",
        "- `server_3090_b` uniquely modifies tracked `ultralytics/utils/metrics.py`, which can affect reported metrics and final result interpretation.",
        "- `server_3090_a` and `server_3090_b` share many DS-SSM/ex-series files and are closer to each other than to 1080Ti for several tracked files.",
        "- Untracked model variants such as `mamba_yolo_ex*`, `mamba_yolo_PFIM.py`, and DS-SSM YAMLs are likely central to experiment reconstruction.",
        "",
        "## Detailed Lists",
        "",
        "- Full modified-file inventory: `reports/all_modified_files.md`.",
        "- Matrix by server and cross-server hashes: `reports/code_diff_matrix.md`.",
    ]
    write("project_diff_report.md", "\n".join(report))


if __name__ == "__main__":
    main()
