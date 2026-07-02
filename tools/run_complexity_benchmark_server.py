import json
import os
import posixpath
import shlex
import stat
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

LOCAL_COMPLEXITY = ROOT / "complexity_benchmark"
LOCAL_MADINET = ROOT / "downloads" / "madinet_official" / "MaDiNet-master"
LOCAL_RESULTS = ROOT / "downloads" / "complexity_results_172_26_151_143"


def safe_print(text=""):
    if not isinstance(text, str):
        text = str(text)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def run(ssh, command, timeout=300):
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def mkdir_p(sftp, path):
    parts = []
    cur = path
    while cur not in ("", "/"):
        parts.append(cur)
        cur = posixpath.dirname(cur)
    for p in reversed(parts):
        try:
            sftp.mkdir(p)
        except OSError:
            pass


def upload_file(sftp, local, remote):
    mkdir_p(sftp, posixpath.dirname(remote))
    sftp.put(str(local), remote)


def upload_tree(sftp, local_root, remote_root, skip_suffixes=None, max_bytes=5_000_000):
    skip_suffixes = skip_suffixes or set()
    uploaded = []
    for path in local_root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() in skip_suffixes:
            continue
        if path.stat().st_size > max_bytes:
            continue
        rel = path.relative_to(local_root).as_posix()
        remote = posixpath.join(remote_root, rel)
        upload_file(sftp, path, remote)
        uploaded.append(rel)
    return uploaded


def download_tree(sftp, remote_root, local_root):
    local_root.mkdir(parents=True, exist_ok=True)
    downloaded = []

    def walk(remote, local):
        local.mkdir(parents=True, exist_ok=True)
        try:
            entries = sftp.listdir_attr(remote)
        except OSError:
            return
        for entry in entries:
            rp = posixpath.join(remote, entry.filename)
            lp = local / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                walk(rp, lp)
            elif stat.S_ISREG(entry.st_mode):
                sftp.get(rp, str(lp))
                downloaded.append(str(lp))

    walk(remote_root, local_root)
    return downloaded


def main():
    if not LOCAL_COMPLEXITY.exists():
        raise SystemExit(f"missing {LOCAL_COMPLEXITY}")
    if not LOCAL_MADINET.exists():
        raise SystemExit(f"missing {LOCAL_MADINET}; download official MaDiNet first")

    ssh = connect()
    sftp = ssh.open_sftp()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_complexity = posixpath.join(PROJECT, "complexity_benchmark")
    remote_madinet = posixpath.join(PROJECT, "external_repos", "MaDiNet_official")
    remote_results = posixpath.join(PROJECT, "complexity_benchmark", "results", f"server_run_{stamp}")
    try:
        run(ssh, f"mkdir -p {shlex.quote(remote_complexity)} {shlex.quote(remote_madinet)} {shlex.quote(remote_results)}", timeout=60)
        uploaded_complexity = upload_tree(sftp, LOCAL_COMPLEXITY, remote_complexity, skip_suffixes={".pyc"})
        uploaded_madinet = upload_tree(
            sftp,
            LOCAL_MADINET,
            remote_madinet,
            skip_suffixes={".zip", ".pt", ".pth", ".pkl", ".onnx", ".engine"},
            max_bytes=10_000_000,
        )

        commands = {
            "env": f"cd {shlex.quote(PROJECT)} && {PYTHON} - <<'PY'\n"
            "import torch\nprint('torch', torch.__version__)\n"
            "for m in ['thop','timm','einops','mamba_ssm','detectron2']:\n"
            "    try:\n        __import__(m); print(m, 'ok')\n    except Exception as e:\n        print(m, 'missing', repr(e))\n"
            "PY",
            "ours": f"cd {shlex.quote(PROJECT)} && {PYTHON} complexity_benchmark/model_complexity_benchmark.py "
            "--configs ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-SSDD.yaml "
            "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml "
            "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml "
            f"--imgsz 640 --batch 1 --device 0 --warmup 20 --runs 100 --out {shlex.quote(posixpath.join(remote_results, 'ours_complexity.csv'))}",
            "madinet": f"cd {shlex.quote(PROJECT)} && {PYTHON} complexity_benchmark/madinet_official_complexity.py "
            f"--madinet-root {shlex.quote(remote_madinet)} --imgsz 640 --device 0 --warmup 10 --runs 30 "
            f"--out {shlex.quote(posixpath.join(remote_results, 'madinet_official_complexity.csv'))}",
            "list": f"find {shlex.quote(remote_results)} -maxdepth 2 -type f -print",
        }
        log = {
            "host": HOST,
            "project": PROJECT,
            "stamp": stamp,
            "remote_results": remote_results,
            "uploaded_complexity_count": len(uploaded_complexity),
            "uploaded_madinet_count": len(uploaded_madinet),
            "commands": {},
        }
        for name, cmd in commands.items():
            out, err = run(ssh, cmd, timeout=900)
            log["commands"][name] = {"stdout": out, "stderr": err}
            safe_print(f"===== {name} stdout =====")
            safe_print(out)
            if err:
                safe_print(f"===== {name} stderr =====")
                safe_print(err)

        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        (LOCAL_RESULTS / f"server_run_{stamp}_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        downloaded = download_tree(sftp, remote_results, LOCAL_RESULTS / f"server_run_{stamp}")
        safe_print(json.dumps({"downloaded": downloaded, "log": str(LOCAL_RESULTS / f"server_run_{stamp}_log.json")}, ensure_ascii=False, indent=2))
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
