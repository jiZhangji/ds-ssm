import hashlib
import json
import os
import posixpath
import sys
from datetime import datetime

import paramiko


PROJECT_DIR = "/home/zhangji/Mamba-YOLO"

SERVERS = {
    "server_1080ti": {"host": "172.26.193.188", "user": "zhangji", "password": "123456", "gpu": "2 x 1080Ti"},
    "server_3090_a": {"host": "172.26.150.36", "user": "zhangji", "password": "12345", "gpu": "1 x RTX 3090"},
    "server_3090_b": {"host": "172.26.133.16", "user": "zhangji", "password": "12345", "gpu": "1 x RTX 3090"},
}

COMMANDS = {
    "pwd": "cd /home/zhangji/Mamba-YOLO && pwd",
    "top_level": "cd /home/zhangji/Mamba-YOLO && find . -maxdepth 2 -type d | sort | head -200",
    "git_is_repo": "cd /home/zhangji/Mamba-YOLO && git rev-parse --is-inside-work-tree 2>/dev/null || true",
    "git_branch": "cd /home/zhangji/Mamba-YOLO && git branch --show-current 2>/dev/null || true",
    "git_head": "cd /home/zhangji/Mamba-YOLO && git rev-parse HEAD 2>/dev/null || true",
    "git_remote": "cd /home/zhangji/Mamba-YOLO && git remote -v 2>/dev/null || true",
    "git_status": "cd /home/zhangji/Mamba-YOLO && git status --short 2>/dev/null || true",
    "git_status_porcelain_count": "cd /home/zhangji/Mamba-YOLO && git status --porcelain 2>/dev/null | wc -l",
    "git_commit_count": "cd /home/zhangji/Mamba-YOLO && git rev-list --count HEAD 2>/dev/null || true",
    "git_last_commits": "cd /home/zhangji/Mamba-YOLO && git log --date=iso --pretty=format:'%h %ad %s' -n 12 2>/dev/null || true",
    "git_last_commit_iso": "cd /home/zhangji/Mamba-YOLO && git log -1 --date=iso --pretty=format:'%ad' 2>/dev/null || true",
    "tracked_file_hashes": "cd /home/zhangji/Mamba-YOLO && git ls-files '*.py' '*.yaml' '*.yml' 'ultralytics/**' 'cfg/**' 'configs/**' 'tools/**' 'scripts/**' 2>/dev/null | sort | while read f; do [ -f \"$f\" ] && printf '%s\\t%s\\n' \"$(sha1sum \"$f\" | awk '{print $1}')\" \"$f\"; done",
    "priority_file_list": "cd /home/zhangji/Mamba-YOLO && find ultralytics cfg configs tools scripts -type f \\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \\) 2>/dev/null | sort | head -1000",
    "config_count": "cd /home/zhangji/Mamba-YOLO && find . -type f \\( -name '*.yaml' -o -name '*.yml' \\) 2>/dev/null | wc -l",
    "python_count": "cd /home/zhangji/Mamba-YOLO && find . -type f -name '*.py' 2>/dev/null | wc -l",
    "output_experiment_dirs": "cd /home/zhangji/Mamba-YOLO && if [ -d output_dir ]; then find output_dir -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort; fi",
    "output_experiment_count": "cd /home/zhangji/Mamba-YOLO && if [ -d output_dir ]; then find output_dir -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; else echo 0; fi",
    "weight_count": "cd /home/zhangji/Mamba-YOLO && if [ -d output_dir ]; then find output_dir -type f \\( -name '*.pt' -o -name '*.pth' -o -name '*.onnx' \\) 2>/dev/null | wc -l; else echo 0; fi",
    "log_count": "cd /home/zhangji/Mamba-YOLO && if [ -d output_dir ]; then find output_dir -type f \\( -name '*.log' -o -name 'events.out.tfevents*' -o -name 'results.csv' -o -name 'args.yaml' -o -name 'opt.yaml' \\) 2>/dev/null | wc -l; else echo 0; fi",
    "output_summary": "cd /home/zhangji/Mamba-YOLO && if [ -d output_dir ]; then find output_dir -maxdepth 2 -type f \\( -name 'results.csv' -o -name 'args.yaml' -o -name 'opt.yaml' -o -name 'best.pt' -o -name 'last.pt' -o -name '*.log' \\) 2>/dev/null | sort | head -500; fi",
    "output_size": "cd /home/zhangji/Mamba-YOLO && du -sh output_dir 2>/dev/null || true",
}


def run(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=90)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return {"stdout": out.strip(), "stderr": err.strip()}


def collect_server(name, meta):
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
    result = {"name": name, "meta": meta, "collected_at": datetime.now().isoformat(timespec="seconds"), "commands": {}}
    for key, command in COMMANDS.items():
        result["commands"][key] = run(ssh, command)
    ssh.close()
    return result


def main():
    os.makedirs("inventory/raw", exist_ok=True)
    all_results = {}
    for name, meta in SERVERS.items():
        print(f"collecting {name} {meta['host']}", flush=True)
        try:
            data = collect_server(name, meta)
        except Exception as exc:
            data = {"name": name, "meta": meta, "collected_at": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)}
        all_results[name] = data
        with open(os.path.join("inventory", "raw", f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    with open(os.path.join("inventory", "raw", "all_servers.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
