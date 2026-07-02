import json, os, csv, glob
from collections import Counter

RANK="database/experiment_ranking.json"
REC="downloads/experiment_recovery/recovery_manifest.json"

def load(path, default):
    if os.path.exists(path):
        with open(path,encoding="utf-8") as f: return json.load(f)
    return default

def count_files(root):
    n=0
    for _,_,files in os.walk(root):
        n+=len(files)
    return n

def main():
    os.makedirs("reports",exist_ok=True); os.makedirs("archives",exist_ok=True)
    ranked=load(RANK,[])
    rec=load(REC,[])
    code_versions=[d for d in glob.glob("source_code/server_*") if os.path.isdir(d)]
    # Stage7 mapping
    lines=["# Experiment Code Mapping\n","Baseline Git commit for all recovered server snapshots: `b26cbda230dfa217f96faee8dc7020db3962f3df`.\n",
           "For uncommitted code variants, mapping is by source server snapshot. Exact file-level runtime version is `UNKNOWN` unless inferable from experiment/config name.\n",
           "| Rank | Experiment | Server | Git Commit | Code Version | Config | Score | Mapping Confidence | Notes |",
           "|---:|---|---|---|---|---|---:|---|---|"]
    for r in ranked[:120]:
        cfg=r.get("configs") or "UNKNOWN"
        conf="server_snapshot"
        notes="Exact uncommitted file variant UNKNOWN"
        name=r.get("experiment_name","")
        if "ex23" in name: notes="Likely related to mamba_yolo_ex23; exact runtime code UNKNOWN"
        if "ex24" in name or "ex25" in name or "ex26" in name: notes="Likely related to ex24-26/PFIM lineage; exact runtime code UNKNOWN"
        if "PFIM" in name: notes="Likely PFIM lineage; exact runtime code UNKNOWN"
        lines.append(f"| {r.get('rank','')} | `{name}` | `{r.get('source_server','')}` | `b26cbda230dfa217f96faee8dc7020db3962f3df` | `source_code/{r.get('source_server','')}` | `{cfg if cfg else 'UNKNOWN'}` | `{r.get('score','')}` | `{conf}` | `{notes}` |")
    open("reports/experiment_code_mapping.md","w",encoding="utf-8").write("\n".join(lines)+"\n")
    cv=["# Code Version Map\n","| Code Version | Source | Git Commit | Dirty/Variant Status | Notes |","|---|---|---|---|---|"]
    for v in code_versions:
        name=os.path.basename(v)
        note="Recovered independent server snapshot; contains .git metadata and uncommitted/untracked files where present."
        cv.append(f"| `{v.replace(os.sep,'/')}` | `{name}` | `b26cbda230dfa217f96faee8dc7020db3962f3df` | `not merged` | `{note}` |")
    cv.append("\nSee `reports/code_conflict_report.md` for same-path conflicts and `reports/code_diff_matrix.md` for official/head differences.")
    open("reports/code_version_map.md","w",encoding="utf-8").write("\n".join(cv)+"\n")
    # Stage8 master report
    top=[r for r in ranked if r.get("score") is not None][:20]
    recovered_names={r["experiment"] for r in rec}
    weights=count_files("weights") if os.path.isdir("weights") else 0
    logs=count_files("logs") if os.path.isdir("logs") else 0
    configs=sum(1 for p in glob.glob("experiments/**/*",recursive=True) if os.path.isfile(p) and os.path.basename(p) in ("args.yaml","opt.yaml"))
    mr=["# Master Repository Report\n","## Structure\n","```text\nsource_code/\nexperiments/\nweights/\nlogs/\ndatabase/\nreports/\ninventory/\narchives/\n```\n",
        "## Counts\n",
        f"- Code versions: `{len(code_versions)}`",
        f"- Ranked experiments: `{len(ranked)}`",
        f"- Recovered experiments: `{len(rec)}`",
        f"- Recovered weight files: `{weights}`",
        f"- Recovered log files: `{logs}`",
        f"- Recovered config files: `{configs}`",
        "\n## Top Experiments\n","| Rank | Server | Experiment | Score | Recovered |","|---:|---|---|---:|---|"]
    for r in top:
        mr.append(f"| {r.get('rank')} | `{r.get('source_server')}` | `{r.get('experiment_name')}` | `{r.get('score')}` | `{r.get('experiment_name') in recovered_names}` |")
    mr += ["\n## Still Need Manual Confirmation\n",
           "- Resolve same-path code conflicts in `reports/code_conflict_report.md` before creating a single merged code tree.",
           "- Confirm exact experiment-to-uncommitted-code mapping for PFIM/ex-series experiments.",
           "- Confirm whether `weights/` subdirectories indexed as experiment-like rows should be collapsed in future database views.",
           "- Validate ranking metrics manually where `results.csv` column naming differs."]
    open("reports/master_repository_report.md","w",encoding="utf-8").write("\n".join(mr)+"\n")
    final=["# Final Project Summary\n","## Completed Stages\n",
           "- Stage 0-1: server authenticity and inventory.",
           "- Stage 2-3: code diff analysis and source recovery into independent server snapshots.",
           "- Stage 4: experiment index.",
           "- Stage 5: experiment value/duplicate/quality analysis.",
           "- Stage 6: prioritized experiment recovery.",
           "- Stage 7: experiment-code mapping with UNKNOWN where not provable.",
           "- Stage 8: local master repository structure and report.",
           "\n## Key Outputs\n",
           "- `reports/experiment_ranking.md`",
           "- `reports/duplicate_report.md`",
           "- `reports/experiment_quality_report.md`",
           "- `reports/experiment_recovery_report.md`",
           "- `reports/experiment_code_mapping.md`",
           "- `reports/code_version_map.md`",
           "- `reports/master_repository_report.md`",
           "- `database/experiment_ranking.xlsx`",
           "\n## Summary\n",
           f"- Code versions preserved: `{len(code_versions)}`.",
           f"- Experiments ranked: `{len(ranked)}`.",
           f"- Experiments recovered: `{len(rec)}`.",
           f"- Recovered weights files: `{weights}`.",
           "\nNo server files were modified or deleted."]
    open("reports/final_project_summary.md","w",encoding="utf-8").write("\n".join(final)+"\n")
if __name__=="__main__": main()
