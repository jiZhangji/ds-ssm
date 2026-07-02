import csv, json, os, re, hashlib
from collections import defaultdict

RAW="inventory/raw/experiment_metadata_stage5.json"
IDX="database/experiment_index.csv"
OUTJSON="database/experiment_ranking.json"

def parse_blocks(text):
    exps=[]; cur=None; content_path=None; buf=[]
    for line in text.splitlines():
        if line.startswith("BEGIN_EXP\t"):
            cur={"dir":line.split("\t",1)[1],"texts":{},"text_files":[],"artifacts":[]}
        elif line.startswith("END_EXP\t"):
            if content_path and cur is not None:
                cur["texts"][content_path]="\n".join(buf); content_path=None; buf=[]
            if cur: exps.append(cur)
            cur=None
        elif cur is None:
            continue
        elif line.startswith("DIR_STAT\t"):
            _,c,m=line.split("\t"); cur["ctime_epoch"]=c; cur["mtime_epoch"]=m
        elif line.startswith("TEXT_FILE\t"):
            _,p,s=line.split("\t"); cur["text_files"].append({"path":p,"size":int(s or 0)})
        elif line.startswith("ARTIFACT\t"):
            parts=line.split("\t"); 
            if len(parts)>=4: cur["artifacts"].append({"path":parts[1],"size":int(parts[2] or 0),"mtime":parts[3]})
        elif line.startswith("CONTENT_BEGIN\t"):
            if content_path: cur["texts"][content_path]="\n".join(buf)
            content_path=line.split("\t",1)[1]; buf=[]
        elif line.startswith("CONTENT_END\t"):
            if content_path: cur["texts"][content_path]="\n".join(buf)
            content_path=None; buf=[]
        elif content_path:
            buf.append(line)
    return exps

def best_metrics(texts):
    metrics={}
    for p,t in texts.items():
        if p.endswith("results.csv"):
            rows=list(csv.DictReader(t.splitlines()))
            if rows:
                last=rows[-1]
                for k,v in last.items():
                    try: metrics[k.strip()]=float(str(v).strip())
                    except: pass
                # pick max map columns too
                for k in rows[0].keys():
                    vals=[]
                    for r in rows:
                        try: vals.append(float(str(r.get(k,"")).strip()))
                        except: pass
                    if vals: metrics["max_"+k.strip()]=max(vals)
                metrics["epochs_recorded"]=len(rows)
    return metrics

def config_sig(texts, name):
    relevant=[]
    for p,t in sorted(texts.items()):
        if p.endswith(("args.yaml","opt.yaml")):
            relevant.append(t)
    base="\n---\n".join(relevant) or name
    return hashlib.sha1(base.encode("utf-8","replace")).hexdigest()

def get_metric(m):
    candidates=["max_metrics/mAP50-95(B)","max_metrics/mAP50-95","max_map50-95","max_mAP50-95","metrics/mAP50-95(B)","metrics/mAP50-95"]
    for c in candidates:
        if c in m: return m[c]
    for k,v in m.items():
        kl=k.lower()
        if ("map50-95" in kl or "map50_95" in kl) and not kl.startswith("epoch"):
            return v
    return None

def main():
    os.makedirs("database",exist_ok=True); os.makedirs("reports",exist_ok=True)
    raw=json.load(open(RAW,encoding="utf-8"))
    rows=[]
    for server,payload in raw.items():
        for e in parse_blocks(payload.get("stdout","")):
            arts=e["artifacts"]
            has_best=any(a["path"].endswith("best.pt") for a in arts)
            has_last=any(a["path"].endswith("last.pt") for a in arts)
            best_size=next((a["size"] for a in arts if a["path"].endswith("best.pt")),"")
            metrics=best_metrics(e["texts"])
            score=get_metric(metrics)
            logs=[x["path"] for x in e["text_files"] if x["path"].endswith((".log","results.csv"))]
            configs=[x["path"] for x in e["text_files"] if x["path"].endswith((".yaml",".yml"))]
            complete=bool(has_best and has_last and any(p.endswith("results.csv") for p in logs))
            interrupted=not complete and (has_best or has_last or logs or configs)
            name=e["dir"].replace("output_dir/","",1)
            rows.append({"experiment_name":name,"source_server":server,"remote_dir":e["dir"],"modified_epoch":e.get("mtime_epoch",""),
                         "configs":";".join(configs),"logs":";".join(logs),"has_best_pt":has_best,"has_last_pt":has_last,
                         "best_pt_size":best_size,"score":score,"epochs_recorded":metrics.get("epochs_recorded",""),
                         "training_complete":complete,"interrupted":interrupted,"config_signature":config_sig(e["texts"],name),
                         "metrics":metrics})
    groups=defaultdict(list)
    for r in rows: groups[r["config_signature"]].append(r)
    for r in rows:
        r["duplicate_group_size"]=len(groups[r["config_signature"]])
        r["quality"]="valuable" if r["training_complete"] and r["score"] is not None else ("interrupted" if r["interrupted"] else "low_info")
    ranked=sorted(rows,key=lambda r: (r["score"] is not None, r["score"] or -1, r["training_complete"]), reverse=True)
    for i,r in enumerate(ranked,1): r["rank"]=i
    json.dump(ranked,open(OUTJSON,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    # xlsx/csv
    fields=["rank","experiment_name","source_server","remote_dir","score","epochs_recorded","training_complete","interrupted","quality","duplicate_group_size","has_best_pt","has_last_pt","best_pt_size","configs","logs","config_signature"]
    with open("database/experiment_ranking.csv","w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k,"") for k in fields} for r in ranked])
    try:
        from openpyxl import Workbook
        wb=Workbook(); ws=wb.active; ws.title="ranking"; ws.append(fields)
        for r in ranked: ws.append([r.get(k,"") for k in fields])
        wb.save("database/experiment_ranking.xlsx")
    except Exception as e: open("database/experiment_ranking.xlsx.error.txt","w").write(repr(e))
    top=ranked[:30]
    def table(rs):
        lines=["| Rank | Server | Experiment | Score | Complete | Dup group |","|---:|---|---|---:|---|---:|"]
        for r in rs: lines.append(f"| {r['rank']} | `{r['source_server']}` | `{r['experiment_name']}` | `{'' if r['score'] is None else round(r['score'],6)}` | `{r['training_complete']}` | `{r['duplicate_group_size']}` |")
        return "\n".join(lines)
    open("reports/experiment_ranking.md","w",encoding="utf-8").write("# Experiment Ranking\n\n"+table(top)+"\n")
    dups=[g for g in groups.values() if len(g)>1]
    dupmd=["# Duplicate Report\n",f"- Duplicate config groups: `{len(dups)}`.\n"]
    for i,g in enumerate(dups,1):
        dupmd.append(f"\n## Group {i} `{g[0]['config_signature'][:12]}`\n")
        dupmd.append(table(sorted(g,key=lambda r:r.get("rank",999999))[:20]))
    open("reports/duplicate_report.md","w",encoding="utf-8").write("\n".join(dupmd))
    interrupted=[r for r in ranked if r["interrupted"] and not r["training_complete"]]
    valuable=[r for r in ranked if r["quality"]=="valuable"]
    q=["# Experiment Quality Report\n",f"- Total indexed metadata experiments: `{len(rows)}`.",f"- Valuable complete experiments: `{len(valuable)}`.",f"- Interrupted/partial experiments: `{len(interrupted)}`.","\n## Interrupted Experiments\n",table(interrupted[:80]),"\n## Recommended Retain Experiments\n",table(valuable[:80])]
    open("reports/experiment_quality_report.md","w",encoding="utf-8").write("\n".join(q)+"\n")
if __name__=="__main__": main()
