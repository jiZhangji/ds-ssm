import json
import os
from datetime import datetime

import paramiko


SERVERS = {
    "server_1080ti": {"host": "172.26.193.188", "user": "zhangji", "password": "123456"},
    "server_3090_a": {"host": "172.26.150.36", "user": "zhangji", "password": "12345"},
    "server_3090_b": {"host": "172.26.133.16", "user": "zhangji", "password": "12345"},
}

REMOTE_SCRIPT = r"""
cd /home/zhangji/Mamba-YOLO || exit 0
if [ ! -d output_dir ]; then exit 0; fi
find output_dir -mindepth 1 -maxdepth 2 -type d 2>/dev/null | sort | while read d; do
  has=0
  find "$d" -maxdepth 2 -type f \( -name 'results.csv' -o -name 'args.yaml' -o -name 'opt.yaml' -o -name '*.log' -o -name 'best.pt' -o -name 'last.pt' \) 2>/dev/null | head -1 | grep -q . && has=1
  [ "$has" -eq 0 ] && continue
  printf 'BEGIN_EXP\t%s\n' "$d"
  stat -c 'DIR_STAT	%W	%Y' "$d" 2>/dev/null || true
  for f in $(find "$d" -maxdepth 2 -type f \( -name 'args.yaml' -o -name 'opt.yaml' -o -name 'results.csv' -o -name '*.log' \) 2>/dev/null | sort); do
    size=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    printf 'TEXT_FILE\t%s\t%s\n' "$f" "$size"
    if [ "$size" -le 262144 ]; then
      printf 'CONTENT_BEGIN\t%s\n' "$f"
      sed -n '1,220p' "$f" 2>/dev/null || true
      printf '\nCONTENT_END\t%s\n' "$f"
    fi
  done
  for f in $(find "$d" -maxdepth 2 -type f \( -name 'best.pt' -o -name 'last.pt' -o -name 'results.png' -o -name 'events.out.tfevents*' \) 2>/dev/null | sort); do
    stat -c 'ARTIFACT	%n	%s	%Y' "$f" 2>/dev/null || true
  done
  printf 'END_EXP\t%s\n' "$d"
done
"""


def collect(server, meta):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(meta["host"], username=meta["user"], password=meta["password"], timeout=15,
                banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    _, stdout, stderr = ssh.exec_command(REMOTE_SCRIPT, timeout=240)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    ssh.close()
    return {"server": server, "host": meta["host"], "collected_at": datetime.now().isoformat(timespec="seconds"),
            "stdout": out, "stderr": err.strip()}


def main():
    os.makedirs("inventory/raw", exist_ok=True)
    data = {}
    for server, meta in SERVERS.items():
        print(f"collecting experiment metadata: {server}", flush=True)
        try:
            data[server] = collect(server, meta)
        except Exception as exc:
            data[server] = {"server": server, "host": meta["host"], "error": repr(exc), "stdout": ""}
    with open("inventory/raw/experiment_metadata_stage5.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
