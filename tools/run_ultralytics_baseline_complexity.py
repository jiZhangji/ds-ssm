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


def run(ssh, command, timeout=900):
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=20, banner_timeout=20, auth_timeout=20, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_results = posixpath.join(PROJECT, "complexity_benchmark", "results", f"ultralytics_baselines_{stamp}")
    local_run = LOCAL_RESULTS / f"ultralytics_baselines_{stamp}"
    try:
        sftp.put(str(ROOT / "complexity_benchmark" / "model_complexity_benchmark.py"), posixpath.join(PROJECT, "complexity_benchmark", "model_complexity_benchmark.py"))
        setup_cmd = f"""
cd {shlex.quote(PROJECT)}
mkdir -p complexity_benchmark/generated_configs {shlex.quote(remote_results)}
cp ultralytics/cfg/models/v5/yolov5.yaml complexity_benchmark/generated_configs/yolov5n.yaml
cp ultralytics/cfg/models/v5/yolov5.yaml complexity_benchmark/generated_configs/yolov5s.yaml
cp ultralytics/cfg/models/v8/yolov8.yaml complexity_benchmark/generated_configs/yolov8n.yaml
"""
        out, err = run(ssh, setup_cmd, timeout=60)
        safe_print(out)
        safe_print(err)
        configs = [
            "ultralytics/cfg/models/v3/yolov3.yaml",
            "complexity_benchmark/generated_configs/yolov5n.yaml",
            "complexity_benchmark/generated_configs/yolov5s.yaml",
            "complexity_benchmark/generated_configs/yolov8n.yaml",
        ]
        cmd = (
            f"cd {shlex.quote(PROJECT)} && {PYTHON} complexity_benchmark/model_complexity_benchmark.py "
            "--configs "
            + " ".join(shlex.quote(c) for c in configs)
            + f" --imgsz 640 --batch 1 --device 0 --warmup 10 --runs 30 --out {shlex.quote(posixpath.join(remote_results, 'ultralytics_baseline_complexity.csv'))}"
        )
        out, err = run(ssh, cmd, timeout=1200)
        safe_print(out)
        safe_print(err)
        local_run.mkdir(parents=True, exist_ok=True)
        log = {"remote_results": remote_results, "stdout": out, "stderr": err, "configs": configs}
        for fname in ["ultralytics_baseline_complexity.csv", "ultralytics_baseline_complexity.json"]:
            try:
                sftp.get(posixpath.join(remote_results, fname), str(local_run / fname))
            except OSError as exc:
                log.setdefault("download_errors", {})[fname] = repr(exc)
        (LOCAL_RESULTS / f"ultralytics_baselines_{stamp}_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        safe_print(f"LOCAL_RUN={local_run}")
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
