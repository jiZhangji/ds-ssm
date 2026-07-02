import json
import posixpath
import shlex
import sys
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"
PYTHON = "/home/zhangji/miniconda3/envs/mamba-yolo/bin/python"


def safe_print(text=""):
    sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace") + b"\n")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=20, banner_timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    script = """
import json
mods=['mmdet','mmengine','mmcv','yolox','detectron2']
out={}
for m in mods:
    try:
        mod=__import__(m)
        out[m]={'status':'ok','version':getattr(mod,'__version__','')}
    except Exception as e:
        out[m]={'status':'missing','error':repr(e)}
print(json.dumps(out, indent=2))
"""
    remote = posixpath.join(PROJECT, "complexity_benchmark", "check_detection_env.py")
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote, "w") as f:
            f.write(script)
        cmd = f"cd {shlex.quote(PROJECT)} && {PYTHON} {shlex.quote(remote)}"
        _, stdout, stderr = ssh.exec_command(cmd, timeout=120)
        safe_print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            safe_print(err)
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
