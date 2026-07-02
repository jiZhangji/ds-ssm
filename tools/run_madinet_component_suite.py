import json
import posixpath
import shlex
import sys
from datetime import datetime
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"
PYTHON = "/home/zhangji/miniconda3/envs/mamba-yolo/bin/python"
LOCAL_RESULTS = ROOT / "downloads" / "complexity_results_172_26_151_143"


def safe_print(text=""):
    sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace") + b"\n")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=20, banner_timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_results = posixpath.join(PROJECT, "complexity_benchmark", "results", f"madinet_component_suite_{stamp}")
    try:
        sftp.put(str(ROOT / "complexity_benchmark" / "madinet_component_suite.py"), posixpath.join(PROJECT, "complexity_benchmark", "madinet_component_suite.py"))
        cmd = (
            f"cd {shlex.quote(PROJECT)} && mkdir -p {shlex.quote(remote_results)} && "
            f"{PYTHON} complexity_benchmark/madinet_component_suite.py "
            f"--madinet-root {shlex.quote(posixpath.join(PROJECT, 'external_repos', 'MaDiNet_official'))} "
            f"--device 0 --warmup 5 --runs 10 "
            f"--out {shlex.quote(posixpath.join(remote_results, 'madinet_component_suite.csv'))}"
        )
        _, stdout, stderr = ssh.exec_command(cmd, timeout=1200)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        safe_print(out)
        safe_print(err)
        local_run = LOCAL_RESULTS / f"madinet_component_suite_{stamp}"
        local_run.mkdir(parents=True, exist_ok=True)
        log = {"remote_results": remote_results, "stdout": out, "stderr": err}
        for fname in ["madinet_component_suite.csv", "madinet_component_suite.json"]:
            try:
                sftp.get(posixpath.join(remote_results, fname), str(local_run / fname))
            except OSError as exc:
                log.setdefault("download_errors", {})[fname] = repr(exc)
        (LOCAL_RESULTS / f"madinet_component_suite_{stamp}_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        safe_print(f"LOCAL_RUN={local_run}")
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
