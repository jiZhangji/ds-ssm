import json
import os
from datetime import datetime

import paramiko


PROJECT_DIR = "/home/zhangji/Mamba-YOLO"
SERVERS = {
    "server_1080ti": {"host": "172.26.193.188", "user": "zhangji", "password": "123456"},
    "server_3090_a": {"host": "172.26.150.36", "user": "zhangji", "password": "12345"},
    "server_3090_b": {"host": "172.26.133.16", "user": "zhangji", "password": "12345"},
}

CODE_FIND = (
    "find . -type f \\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o "
    "-name '*.sh' -o -name '*.md' -o -name '*.toml' -o -name '*.txt' \\) "
    "-not -path './.git/*' -not -path './output_dir/*' -not -path './datasets/*' "
    "-not -path './*.egg-info/*' -not -path './.pytest_cache/*' "
)

COMMANDS = {
    "git_head": "cd /home/zhangji/Mamba-YOLO && git rev-parse HEAD 2>/dev/null || true",
    "git_diff_name_status": "cd /home/zhangji/Mamba-YOLO && git diff --name-status HEAD -- . ':(exclude)output_dir/**' ':(exclude)datasets/**' 2>/dev/null || true",
    "git_diff_numstat": "cd /home/zhangji/Mamba-YOLO && git diff --numstat HEAD -- . ':(exclude)output_dir/**' ':(exclude)datasets/**' 2>/dev/null || true",
    "git_deleted": "cd /home/zhangji/Mamba-YOLO && git diff --name-only --diff-filter=D HEAD -- . ':(exclude)output_dir/**' ':(exclude)datasets/**' 2>/dev/null || true",
    "git_untracked_code": "cd /home/zhangji/Mamba-YOLO && git ls-files --others --exclude-standard -- '*.py' '*.yaml' '*.yml' '*.sh' '*.md' '*.toml' '*.txt' 2>/dev/null | grep -v '^output_dir/' | grep -v '^datasets/' | sort || true",
    "git_untracked_code_numstat": "cd /home/zhangji/Mamba-YOLO && git ls-files --others --exclude-standard -- '*.py' '*.yaml' '*.yml' '*.sh' '*.md' '*.toml' '*.txt' 2>/dev/null | grep -v '^output_dir/' | grep -v '^datasets/' | sort | while read f; do [ -f \"$f\" ] && printf '%s\\t%s\\t%s\\n' \"$(wc -l < \"$f\")\" \"$(wc -c < \"$f\")\" \"$f\"; done || true",
    "all_code_hashes": "cd /home/zhangji/Mamba-YOLO && " + CODE_FIND + " 2>/dev/null | sort | while read f; do p=${f#./}; printf '%s\\t%s\\t%s\\t%s\\n' \"$(sha1sum \"$f\" | awk '{print $1}')\" \"$(wc -l < \"$f\")\" \"$(wc -c < \"$f\")\" \"$p\"; done",
    "tracked_code_head_list": "cd /home/zhangji/Mamba-YOLO && git ls-files '*.py' '*.yaml' '*.yml' '*.sh' '*.md' '*.toml' '*.txt' 2>/dev/null | grep -v '^output_dir/' | grep -v '^datasets/' | sort || true",
}


def run(ssh, command):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=120)
    return {
        "stdout": stdout.read().decode("utf-8", errors="replace").strip(),
        "stderr": stderr.read().decode("utf-8", errors="replace").strip(),
    }


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
    out = {"name": name, "meta": meta, "collected_at": datetime.now().isoformat(timespec="seconds"), "commands": {}}
    for key, command in COMMANDS.items():
        out["commands"][key] = run(ssh, command)
    ssh.close()
    return out


def main():
    os.makedirs("inventory/raw", exist_ok=True)
    all_data = {}
    for name, meta in SERVERS.items():
        print(f"collecting code recovery metadata: {name}", flush=True)
        try:
            all_data[name] = collect(name, meta)
        except Exception as exc:
            all_data[name] = {"name": name, "meta": meta, "error": repr(exc), "collected_at": datetime.now().isoformat(timespec="seconds")}
    with open("inventory/raw/code_recovery_2a.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
