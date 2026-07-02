import posixpath
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    rel = "ultralytics/utils/metrics.py"
    remote = posixpath.join(PROJECT, rel)
    backup = posixpath.join(PROJECT, "codex_backups", f"metrics_numpy_fix_{stamp}.py")
    ssh.exec_command(f"mkdir -p {posixpath.dirname(backup)!r} && cp -p {remote!r} {backup!r}")[1].read()
    with sftp.file(remote, "r") as f:
        text = f.read().decode("utf-8")
    old = 'trapz = getattr(np, "trapz", np.trapezoid)'
    new = 'trapz = np.trapz if hasattr(np, "trapz") else np.trapezoid'
    if old not in text:
        raise RuntimeError("target pattern not found")
    with sftp.file(remote, "w") as f:
        f.write(text.replace(old, new))
    sftp.close()
    _, stdout, stderr = ssh.exec_command(f"cd {PROJECT} && /home/zhangji/miniconda3/envs/mamba-yolo/bin/python -m py_compile {rel}", timeout=60)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    ssh.close()
    print(f"backup={backup}")
    print(out, end="")
    print(err, end="")


if __name__ == "__main__":
    main()
