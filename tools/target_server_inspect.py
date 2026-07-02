import json
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"


COMMANDS = {
    "pwd": f"cd {PROJECT} && pwd",
    "top": f"cd {PROJECT} && find . -maxdepth 2 -type d | sort | head -120",
    "git_head": f"cd {PROJECT} && git rev-parse HEAD 2>/dev/null || true",
    "git_branch": f"cd {PROJECT} && git branch --show-current 2>/dev/null || true",
    "git_status": f"cd {PROJECT} && git status --short 2>/dev/null || true",
    "train_files": f"cd {PROJECT} && find . -maxdepth 2 -type f -name 'mbyolo_train*.py' -o -name '*train*.sh' 2>/dev/null | sort",
    "key_hashes": f"cd {PROJECT} && for f in mbyolo_train.py mbyolo_train_PFIM.py ultralytics/nn/modules/mamba_yolo.py ultralytics/nn/modules/mamba_yolo_ex27-E-new.py ultralytics/nn/modules/mamba_yolo_ex24-26.py ultralytics/nn/tasks.py ultralytics/utils/metrics.py ultralytics/engine/trainer.py ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml; do [ -f \"$f\" ] && printf '%s\\t%s\\t%s\\n' \"$(sha1sum \"$f\" | awk '{{print $1}}')\" \"$(wc -l < \"$f\")\" \"$f\"; done",
    "cfgs": f"cd {PROJECT} && find ultralytics/cfg/models/mamba-yolo -maxdepth 1 -type f | sort | sed -n '1,120p'",
}


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    out = {"host": HOST, "project": PROJECT, "collected_at": datetime.now().isoformat(timespec="seconds"), "commands": {}}
    for key, cmd in COMMANDS.items():
        _, stdout, stderr = ssh.exec_command(cmd, timeout=60)
        out["commands"][key] = {
            "stdout": stdout.read().decode("utf-8", errors="replace").strip(),
            "stderr": stderr.read().decode("utf-8", errors="replace").strip(),
        }
    ssh.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
