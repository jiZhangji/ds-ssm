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


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=20, banner_timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    return ssh


def run(ssh, command, timeout=900):
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def main():
    ssh = connect()
    sftp = ssh.open_sftp()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_results = posixpath.join(PROJECT, "complexity_benchmark", "results", f"server_run_{stamp}")
    try:
        for name in ["model_complexity_benchmark.py", "madinet_official_complexity.py"]:
            sftp.put(str(ROOT / "complexity_benchmark" / name), posixpath.join(PROJECT, "complexity_benchmark", name))
        run(ssh, f"mkdir -p {shlex.quote(remote_results)}", timeout=60)
        cmds = {
            "ours": f"cd {shlex.quote(PROJECT)} && {PYTHON} complexity_benchmark/model_complexity_benchmark.py "
            "--configs ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml "
            "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml "
            "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml "
            f"--imgsz 640 --batch 1 --device 0 --warmup 10 --runs 30 --out {shlex.quote(posixpath.join(remote_results, 'ours_complexity.csv'))}",
            "madinet": f"cd {shlex.quote(PROJECT)} && {PYTHON} complexity_benchmark/madinet_official_complexity.py "
            f"--madinet-root {shlex.quote(posixpath.join(PROJECT, 'external_repos', 'MaDiNet_official'))} "
            f"--imgsz 640 --device 0 --warmup 5 --runs 10 --out {shlex.quote(posixpath.join(remote_results, 'madinet_official_complexity.csv'))}",
        }
        log = {"remote_results": remote_results, "commands": {}}
        for key, cmd in cmds.items():
            out, err = run(ssh, cmd, timeout=900)
            log["commands"][key] = {"stdout": out, "stderr": err}
            safe_print(f"===== {key} stdout =====")
            safe_print(out)
            if err:
                safe_print(f"===== {key} stderr =====")
                safe_print(err)
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        local_run = LOCAL_RESULTS / f"server_run_{stamp}"
        local_run.mkdir(parents=True, exist_ok=True)
        for fname in ["ours_complexity.csv", "ours_complexity.json", "madinet_official_complexity.csv", "madinet_official_complexity.json"]:
            remote = posixpath.join(remote_results, fname)
            try:
                sftp.get(remote, str(local_run / fname))
            except OSError as exc:
                log.setdefault("download_errors", {})[fname] = repr(exc)
        (LOCAL_RESULTS / f"server_run_{stamp}_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        safe_print(f"LOCAL_RUN={local_run}")
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
