import json
import os
from collections import defaultdict


RAW = os.path.join("inventory", "raw", "all_servers.json")
OUT = "inventory"


def cmd(s, k):
    return s.get("commands", {}).get(k, {}).get("stdout", "").strip()


def lines(text):
    return [x for x in text.splitlines() if x.strip()]


def parse_hashes(text):
    out = {}
    for line in lines(text):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def write(path, text):
    with open(os.path.join(OUT, path), "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def server_report(name, s):
    exp_dirs = lines(cmd(s, "output_experiment_dirs"))
    status = lines(cmd(s, "git_status"))
    summary = lines(cmd(s, "output_summary"))
    text = f"""# {name} Inventory

## Identity

- Host: `{s['meta']['host']}`
- User: `{s['meta']['user']}`
- GPU: `{s['meta']['gpu']}`
- Project dir: `{cmd(s, 'pwd')}`
- Collected at: `{s.get('collected_at', '')}`
- Read-only scope used: `/home/zhangji/Mamba-YOLO`

## Git

- Is Git repo: `{cmd(s, 'git_is_repo')}`
- Branch: `{cmd(s, 'git_branch') or '(detached or unavailable)'}`
- HEAD: `{cmd(s, 'git_head')}`
- Remote: `{cmd(s, 'git_remote').replace(chr(10), ' ; ')}`
- Commit count: `{cmd(s, 'git_commit_count')}`
- Latest commit time: `{cmd(s, 'git_last_commit_iso')}`
- Uncommitted / untracked status lines: `{cmd(s, 'git_status_porcelain_count')}`

### Recent commits

```text
{cmd(s, 'git_last_commits')}
```

### Git status summary

```text
{cmd(s, 'git_status')}
```

## Project Counts

- Python files: `{cmd(s, 'python_count')}`
- YAML/YML config files: `{cmd(s, 'config_count')}`
- Top-level output experiment directories: `{cmd(s, 'output_experiment_count')}`
- Weight-like files under output_dir: `{cmd(s, 'weight_count')}`
- Log/result/config record files under output_dir: `{cmd(s, 'log_count')}`
- output_dir size: `{cmd(s, 'output_size') or '(not available)'}`

## output_dir Top-Level Experiments

```text
{chr(10).join(exp_dirs) if exp_dirs else '(none found)'}
```

## output_dir Key Artifacts Sample

Limited to first 500 matching paths from `find`; no artifact contents were downloaded.

```text
{chr(10).join(summary[:500]) if summary else '(none found)'}
```
"""
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = ["server_1080ti", "server_3090_a", "server_3090_b"]
    for name in names:
        write(f"{name}.md", server_report(name, data[name]))

    hashes = {n: parse_hashes(cmd(data[n], "tracked_file_hashes")) for n in names}
    all_files = sorted(set().union(*(set(hashes[n]) for n in names)))
    diff_files = []
    same_files = []
    for p in all_files:
        vals = {n: hashes[n].get(p, "<missing>") for n in names}
        if len(set(vals.values())) == 1:
            same_files.append(p)
        else:
            diff_files.append((p, vals))

    status_sets = {n: set(lines(cmd(data[n], "git_status"))) for n in names}
    all_status = sorted(set().union(*status_sets.values()))

    overview = ["# Code Diff Overview", ""]
    overview += [
        "## Summary",
        "",
        f"- Compared tracked priority code/config hashes across `{len(all_files)}` paths.",
        f"- Identical tracked priority files across all three servers: `{len(same_files)}`.",
        f"- Differing or missing tracked priority files: `{len(diff_files)}`.",
        "- Scope: tracked `*.py`, `*.yaml`, `*.yml`, and priority folders from Git plus visible untracked Git status. This is a metadata/hash inventory, not a full code download.",
        "",
        "## Differing Tracked Files",
        "",
        "| File | 1080Ti | 3090-A | 3090-B |",
        "|---|---|---|---|",
    ]
    for p, vals in diff_files:
        overview.append(f"| `{p}` | `{vals['server_1080ti'][:12]}` | `{vals['server_3090_a'][:12]}` | `{vals['server_3090_b'][:12]}` |")
    overview += ["", "## Git Status Presence Matrix", "", "| Status line | 1080Ti | 3090-A | 3090-B |", "|---|---:|---:|---:|"]
    for line in all_status:
        overview.append(f"| `{line}` | {'yes' if line in status_sets['server_1080ti'] else ''} | {'yes' if line in status_sets['server_3090_a'] else ''} | {'yes' if line in status_sets['server_3090_b'] else ''} |")
    write("code_diff_overview.md", "\n".join(overview))

    rel = ["# Server Relationship", ""]
    rel += [
        "## Confirmed Facts",
        "",
        "- All three servers are reachable and contain `/home/zhangji/Mamba-YOLO`.",
        "- All three are Git repositories with remote `https://github.com/HZAI-ZJNU/Mamba-YOLO.git`.",
        "- All three have the same HEAD commit `b26cbda230dfa217f96faee8dc7020db3962f3df` and commit count `23`.",
        "- The latest committed history is official/upstream-like and ends at `2024-12-14 22:05:35 +0800`; the research work appears mostly uncommitted/untracked on servers.",
        "",
        "## Relationship Assessment",
        "",
        "- They are very likely three working copies derived from the same official Mamba-YOLO project state.",
        "- They are not code-identical at the working-tree level. Tracked files differ and each server has a different set of untracked experiment modules/backups.",
        "- 3090-A and 3090-B are closest by branch state (`main`), counts, and many identical modified tracked file hashes.",
        "- 1080Ti has no branch name reported, more Git status lines, `.pytest_cache`, PFIM-specific files/backups, and a smaller output experiment directory count.",
        "",
        "## Main Development Environment Likelihood",
        "",
        "- Most likely active development candidates: `server_3090_b` and `server_1080ti`, but no final/master server is selected.",
        "- `server_3090_b` has the largest weight count and an extra modified `ultralytics/utils/metrics.py`, suggesting later metric/result experimentation.",
        "- `server_1080ti` has the largest Git status count and many timestamped backup files around DS-SSM/PFIM changes, suggesting heavy code-edit activity.",
        "- `server_3090_a` looks like a close sibling/training host with one extra `sar_aircraft_test` output directory and fewer dirty files.",
        "",
        "## Possible Roles",
        "",
        "- `server_1080ti`: older or exploratory development host, with PFIM and numerous local backup snapshots.",
        "- `server_3090_a`: training/evaluation host for SAR aircraft and SSDD variants.",
        "- `server_3090_b`: active training/evaluation host with additional metrics modification and most weight artifacts.",
        "",
        "No server is treated as final. This report is only an evidence-based relationship inventory.",
    ]
    write("server_relationship.md", "\n".join(rel))

    po = ["# Project Overview", "", "## Server Summary", "", "| Server | Host | Branch | HEAD | Dirty lines | Experiments | Weights | Logs/results | Py | YAML | output_dir size |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for n in names:
        s = data[n]
        po.append(f"| `{n}` | `{s['meta']['host']}` | `{cmd(s, 'git_branch') or '(none)'}` | `{cmd(s, 'git_head')[:12]}` | `{cmd(s, 'git_status_porcelain_count')}` | `{cmd(s, 'output_experiment_count')}` | `{cmd(s, 'weight_count')}` | `{cmd(s, 'log_count')}` | `{cmd(s, 'python_count')}` | `{cmd(s, 'config_count')}` | `{cmd(s, 'output_size')}` |")
    po += [
        "",
        "## High-Level Findings",
        "",
        "- The committed Git base is identical across all three servers.",
        "- The meaningful research state is in uncommitted tracked modifications and untracked files.",
        "- `output_dir` is non-empty on all servers; 3090-A and 3090-B have six top-level output directories, 1080Ti has five.",
        "- Weight artifacts are numerous, especially on 3090-B; no weights or experiment outputs were downloaded.",
        "- The codebase includes SAR/SSDD dataset prep tools, custom Mamba-YOLO model YAMLs, DS-SSM/PFIM experiment modules, and modified training/model task logic.",
    ]
    write("project_overview.md", "\n".join(po))


if __name__ == "__main__":
    main()
