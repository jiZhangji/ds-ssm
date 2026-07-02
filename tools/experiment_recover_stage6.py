import csv, json, os, posixpath
from datetime import datetime
import paramiko

SERVERS={"server_1080ti":{"host":"172.26.193.188","user":"zhangji","password":"123456"},"server_3090_a":{"host":"172.26.150.36","user":"zhangji","password":"12345"},"server_3090_b":{"host":"172.26.133.16","user":"zhangji","password":"12345"}}
RANK="database/experiment_ranking.json"
REMOTE_BASE="/home/zhangji/Mamba-YOLO"
KEEP_NAMES=("best.pt","last.pt","args.yaml","opt.yaml","results.csv","results.png")
KEEP_SUFFIX=(".log",)
MAX_RECOVER=40

def safe_name(s): return "".join(c if c.isalnum() or c in "._-+" else "_" for c in s.replace("/","__"))[:180]
def connect(m):
    ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(m["host"],username=m["user"],password=m["password"],timeout=15,banner_timeout=15,auth_timeout=15,look_for_keys=False,allow_agent=False)
    return ssh
def list_files(ssh, remote_dir):
    cmd=f"find {remote_dir!r} -maxdepth 2 -type f \\( -name 'best.pt' -o -name 'last.pt' -o -name 'args.yaml' -o -name 'opt.yaml' -o -name 'results.csv' -o -name 'results.png' -o -name '*.log' \\) 2>/dev/null | sort"
    _,out,_=ssh.exec_command(cmd,timeout=60)
    return [x for x in out.read().decode(errors="replace").splitlines() if x]
def main():
    for d in ["experiments","weights","logs","downloads/experiment_recovery","archives"]: os.makedirs(d,exist_ok=True)
    ranked=json.load(open(RANK,encoding="utf-8"))
    selected=[r for r in ranked if r.get("training_complete") and r.get("score") is not None][:MAX_RECOVER]
    meta=[]
    conns={}
    try:
        for r in selected:
            srv=r["source_server"]; conns.setdefault(srv,connect(SERVERS[srv]))
            ssh=conns[srv]; sftp=ssh.open_sftp()
            remote_dir=posixpath.join(REMOTE_BASE,r["remote_dir"])
            local_id=f"{srv}__{safe_name(r['experiment_name'])}"
            exp_dir=os.path.join("experiments",local_id); os.makedirs(exp_dir,exist_ok=True)
            files=list_files(ssh,remote_dir)
            downloaded=[]
            for rp in files:
                base=posixpath.basename(rp)
                if base in ("best.pt","last.pt"):
                    dest=os.path.join("weights",local_id+"__"+base)
                elif base.endswith(".log"):
                    dest=os.path.join("logs",local_id+"__"+base)
                else:
                    dest=os.path.join(exp_dir,base)
                try:
                    sftp.get(rp,dest)
                    downloaded.append({"remote_path":rp,"local_path":dest.replace("\\","/"),"download_time":datetime.now().isoformat(timespec="seconds")})
                except Exception as exc:
                    downloaded.append({"remote_path":rp,"error":repr(exc)})
            sftp.close()
            rec={"experiment":r["experiment_name"],"source_server":srv,"host":SERVERS[srv]["host"],"remote_dir":r["remote_dir"],"rank":r["rank"],"score":r.get("score"),"downloaded":downloaded}
            meta.append(rec)
            with open(os.path.join(exp_dir,"SOURCE.json"),"w",encoding="utf-8") as f: json.dump(rec,f,ensure_ascii=False,indent=2)
    finally:
        for ssh in conns.values(): ssh.close()
    json.dump(meta,open("downloads/experiment_recovery/recovery_manifest.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    md=["# Experiment Recovery Report\n",f"- Recovered experiments: `{len(meta)}`.",f"- Selection rule: top `{MAX_RECOVER}` complete experiments with parsed score.", "\n| Rank | Server | Experiment | Score | Files |","|---:|---|---|---:|---:|"]
    for r in meta: md.append(f"| {r['rank']} | `{r['source_server']}` | `{r['experiment']}` | `{r.get('score')}` | `{len([x for x in r['downloaded'] if 'local_path' in x])}` |")
    open("reports/experiment_recovery_report.md","w",encoding="utf-8").write("\n".join(md)+"\n")
if __name__=="__main__": main()
