import json
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/home/zhangji/Mamba-YOLO"

COMMANDS = {
    "pwd": f"cd {PROJECT} && pwd",
    "top": f"cd {PROJECT} && find . -maxdepth 2 -type d | sort | head -120",
    "git_head": f"cd {PROJECT} && git rev-parse HEAD 2>/dev/null || true",
    "git_branch": f"cd {PROJECT} && git branch --show-current 2>/dev/null || true",
    "git_status": f"cd {PROJECT} && git status --short 2>/dev/null || true",
    "train_files": f"cd {PROJECT} && find . -maxdepth 2 -type f -name 'mbyolo_train*.py' | sort",
    "mamba_modules": f"cd {PROJECT} && find ultralytics/nn/modules -maxdepth 1 -type f -name 'mamba_yolo*.py' 2>/dev/null | sort",
    "mamba_cfgs": f"cd {PROJECT} && find ultralytics/cfg/models/mamba-yolo -maxdepth 1 -type f -name '*SSDD*ex27*' -o -name '*ex27*' 2>/dev/null | sort",
    "python": f"cd {PROJECT} && python -V 2>&1 || true",
    "nvidia": "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true",
}


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15,
                look_for_keys=False, allow_agent=False)
    data = {"host": HOST, "project": PROJECT, "collected_at": datetime.now().isoformat(timespec="seconds"), "commands": {}}
    for key, command in COMMANDS.items():
        _, stdout, stderr = ssh.exec_command(command, timeout=60)
        data["commands"][key] = {
            "stdout": stdout.read().decode("utf-8", errors="replace").strip(),
            "stderr": stderr.read().decode("utf-8", errors="replace").strip(),
        }
    ssh.close()
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
