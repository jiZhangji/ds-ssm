import json
import os
from datetime import datetime

import paramiko


PROJECT_DIR = "/home/zhangji/Mamba-YOLO"
OUTPUT_DIR = PROJECT_DIR + "/output_dir"

SERVERS = {
    "server_1080ti": {"host": "172.26.193.188", "user": "zhangji", "password": "123456"},
    "server_3090_a": {"host": "172.26.150.36", "user": "zhangji", "password": "12345"},
    "server_3090_b": {"host": "172.26.133.16", "user": "zhangji", "password": "12345"},
}

REMOTE_SCRIPT = r"""
cd /home/zhangji/Mamba-YOLO 2>/dev/null || exit 0
if [ ! -d output_dir ]; then exit 0; fi
find output_dir -mindepth 1 -maxdepth 3 -type d 2>/dev/null | sort | while read d; do
  has_marker=0
  find "$d" -maxdepth 2 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'args.yaml' -o -name 'opt.yaml' -o -name '*.log' -o -name 'results.csv' -o -name 'events.out.tfevents*' \) 2>/dev/null | head -1 | grep -q . && has_marker=1
  if [ "$has_marker" -eq 0 ]; then continue; fi
  ctime=$(stat -c '%W' "$d" 2>/dev/null || echo 0)
  mtime=$(stat -c '%Y' "$d" 2>/dev/null || echo 0)
  best=$(find "$d" -maxdepth 2 -type f -name 'best.pt' 2>/dev/null | head -1)
  last=$(find "$d" -maxdepth 2 -type f -name 'last.pt' 2>/dev/null | head -1)
  best_size=""
  [ -n "$best" ] && best_size=$(stat -c '%s' "$best" 2>/dev/null || echo "")
  configs=$(find "$d" -maxdepth 2 -type f \( -name 'args.yaml' -o -name 'opt.yaml' -o -name '*.yaml' -o -name '*.yml' \) 2>/dev/null | sort | paste -sd ',' -)
  logs=$(find "$d" -maxdepth 2 -type f \( -name '*.log' -o -name 'results.csv' \) 2>/dev/null | sort | paste -sd ',' -)
  tb=$(find "$d" -maxdepth 2 -type f -name 'events.out.tfevents*' 2>/dev/null | sort | paste -sd ',' -)
  complete="unknown"
  if [ -n "$last" ] || [ -n "$best" ]; then complete="likely_yes"; fi
  if find "$d" -maxdepth 2 -type f -name 'results.csv' 2>/dev/null | grep -q .; then complete="likely_yes"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$ctime" "$mtime" "$configs" "$best" "$last" "$best_size" "$logs" "$tb" "$complete"
done
"""


def collect(name, meta):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        meta["host"],
        username=meta["user"],
        password=meta["password"],
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    stdin, stdout, stderr = ssh.exec_command(REMOTE_SCRIPT, timeout=180)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    ssh.close()
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        rows.append(
            {
                "experiment_dir": parts[0],
                "experiment_name": parts[0].replace("output_dir/", "", 1),
                "ctime_epoch": parts[1],
                "mtime_epoch": parts[2],
                "config_files": parts[3],
                "best_pt": parts[4],
                "last_pt": parts[5],
                "best_pt_size": parts[6],
                "log_files": parts[7],
                "tensorboard_files": parts[8],
                "training_complete": parts[9],
            }
        )
    return {"server": name, "host": meta["host"], "collected_at": datetime.now().isoformat(timespec="seconds"), "rows": rows, "stderr": err.strip()}


def main():
    os.makedirs("inventory/raw", exist_ok=True)
    all_data = {}
    for name, meta in SERVERS.items():
        print(f"indexing experiments: {name}", flush=True)
        try:
            all_data[name] = collect(name, meta)
        except Exception as exc:
            all_data[name] = {"server": name, "host": meta["host"], "error": repr(exc), "rows": []}
    with open("inventory/raw/experiment_index_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
