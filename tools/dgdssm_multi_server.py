import argparse
import contextlib
import json
import os
import posixpath
import shlex
import sys
from datetime import datetime
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CODE = ROOT / "source_code" / "work_train_ddp"

SERVERS = {
    "3090_5": {"host": "172.26.150.72", "user": "zhangji", "password": "12345", "project_candidates": ["/home/zhangji/Mamba-YOLO"]},
    "3090_4": {"host": "172.26.133.16", "user": "zhangji", "password": "12345", "project_candidates": ["/home/zhangji/Mamba-YOLO"]},
    "3090_3": {
        "host": "172.26.151.143",
        "user": "zhangji",
        "password": "12345",
        "project_candidates": ["/mnt/data5/zhangji/Mamba-YOLO", "/home/zhangji/Mamba-YOLO"],
    },
    "3090_1": {"host": "172.26.150.36", "user": "zhangji", "password": "12345", "project_candidates": ["/home/zhangji/Mamba-YOLO"]},
    "1080ti_2": {"host": "172.26.193.188", "user": "zhangji", "password": "123456", "project_candidates": ["/home/zhangji/Mamba-YOLO"]},
}

FILES = [
    "ultralytics/nn/modules/mamba_yolo_ex30_dg_dsssm.py",
    "ultralytics/nn/modules/__init__.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex30-baseline.yaml",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex31-erase.yaml",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex32-write.yaml",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex33-erase-write.yaml",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml",
]

EXPERIMENTS = {
    "3090_5": {
        "exp": "ex30",
        "config": "Mamba-YOLO-T-DG-DSSSM-SSDD-ex30-baseline.yaml",
        "name": "ssdd_DG-DSSSM_T_300e_32bs_ex30_baseline_20260608",
        "device": "0",
    },
    "3090_4": {
        "exp": "ex32",
        "config": "Mamba-YOLO-T-DG-DSSSM-SSDD-ex32-write.yaml",
        "name": "ssdd_DG-DSSSM_T_300e_32bs_ex32_write_20260608",
        "device": "0",
    },
    "3090_3": {
        "exp": "ex33",
        "config": "Mamba-YOLO-T-DG-DSSSM-SSDD-ex33-erase-write.yaml",
        "name": "ssdd_DG-DSSSM_T_300e_32bs_ex33_erase-write_20260608",
        "device": "0",
    },
    "3090_1": {
        "exp": "ex34",
        "config": "Mamba-YOLO-T-DG-DSSSM-SSDD-ex34-erase-write-delta.yaml",
        "name": "ssdd_DG-DSSSM_T_300e_32bs_ex34_erase-write-delta_20260608",
        "device": "0",
    },
    "1080ti_2": {
        "exp": "ex30",
        "config": "Mamba-YOLO-T-DG-DSSSM-SSDD-ex30-baseline.yaml",
        "name": "ssdd_DG-DSSSM_T_300e_32bs_ex30_baseline_20260608",
        "device": "0,1",
    },
}


def connect(meta):
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
    return ssh


def run(ssh, command, timeout=60):
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out.strip(), err.strip()


def resolve_project(ssh, candidates):
    for project in candidates:
        out, _ = run(ssh, f"test -d {shlex.quote(project)} && echo OK || true")
        if out.strip() == "OK":
            return project
    return None


def inspect_one(name, meta):
    try:
        ssh = connect(meta)
    except Exception as exc:
        return {"server": name, "host": meta["host"], "project": None, "connect_error": repr(exc)}
    try:
        project = resolve_project(ssh, meta["project_candidates"])
        info = {"server": name, "host": meta["host"], "project": project}
        info["nvidia_smi"], info["nvidia_smi_err"] = run(
            ssh,
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
            timeout=30,
        )
        info["gpu_processes"], _ = run(
            ssh,
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
            timeout=30,
        )
        if project:
            info["disk"], _ = run(ssh, f"df -h {shlex.quote(project)} | tail -1", timeout=30)
            info["train_processes"], _ = run(
                ssh,
                "ps -eo pid,etime,cmd | grep -E 'mbyolo_train|torch.distributed.run' | grep -v grep || true",
                timeout=30,
            )
        return info
    finally:
        ssh.close()


def mkdir_p(sftp, path):
    parts = []
    while path not in ("", "/"):
        parts.append(path)
        path = posixpath.dirname(path)
    for p in reversed(parts):
        try:
            sftp.mkdir(p)
        except OSError:
            pass


def upload_one(name, meta, project):
    try:
        ssh = connect(meta)
    except Exception as exc:
        return {"server": name, "error": f"connect failed: {exc!r}"}
    sftp = ssh.open_sftp()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = posixpath.join(project, "codex_backups", f"dgdssm_before_{stamp}")
    try:
        run(ssh, f"mkdir -p {shlex.quote(backup_dir)} {shlex.quote(posixpath.join(project, 'codex_reports'))}", timeout=30)
        uploaded = []
        for rel in FILES:
            local = LOCAL_CODE / Path(rel)
            remote = posixpath.join(project, rel)
            try:
                sftp.stat(remote)
                backup = posixpath.join(backup_dir, rel)
                mkdir_p(sftp, posixpath.dirname(backup))
                sftp.get(remote, str(ROOT / "downloads" / "_tmp_remote_backup"))
                sftp.put(str(ROOT / "downloads" / "_tmp_remote_backup"), backup)
            except OSError:
                pass
            mkdir_p(sftp, posixpath.dirname(remote))
            sftp.put(str(local), remote)
            uploaded.append(rel)
        return {"server": name, "backup_dir": backup_dir, "uploaded": uploaded}
    finally:
        sftp.close()
        ssh.close()


def train_command(project, spec, python_path="/home/zhangji/miniconda3/envs/mamba-yolo/bin/python"):
    config = f"ultralytics/cfg/models/mamba-yolo/{spec['config']}"
    log = f"codex_reports/{spec['name']}.log"
    return (
        f"cd {shlex.quote(project)} && "
        f"nohup {python_path} mbyolo_train.py --task train "
        f"--data datasets/SSDD_YOLO/ssdd.yaml "
        f"--config {shlex.quote(config)} "
        f"--batch_size 32 --imgsz 640 --epochs 300 "
        f"--device {shlex.quote(spec['device'])} --workers 4 --amp "
        f"--project output_dir/ssdd --name {shlex.quote(spec['name'])} "
        f"> {shlex.quote(log)} 2>&1 & echo $!"
    )


def tmux_train_command(project, spec, python_path="/home/zhangji/miniconda3/envs/mamba-yolo/bin/python"):
    config = f"ultralytics/cfg/models/mamba-yolo/{spec['config']}"
    log = f"codex_reports/{spec['name']}.log"
    session = f"dgdssm_{spec['exp']}"
    cuda_prefix = spec.get("cuda_visible")
    train = (
        f"{python_path} mbyolo_train.py --task train "
        f"--data datasets/SSDD_YOLO/ssdd.yaml "
        f"--config {shlex.quote(config)} "
        f"--batch_size 32 --imgsz 640 --epochs 300 "
        f"--device {shlex.quote(spec['device'])} --workers 4 --amp "
        f"--project output_dir/ssdd --name {shlex.quote(spec['name'])} "
        f"> {shlex.quote(log)} 2>&1"
    )
    if cuda_prefix:
        train = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_prefix)} {train}"
    inner = f"cd {shlex.quote(project)} && mkdir -p codex_reports && {train}"
    return (
        f"cd {shlex.quote(project)} && mkdir -p codex_reports && "
        f"(tmux has-session -t {shlex.quote(session)} 2>/dev/null && "
        f"echo SESSION_EXISTS:{shlex.quote(session)} || "
        f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner)} && echo STARTED:{shlex.quote(session)})"
    )


def start_one(name, meta, project):
    spec = EXPERIMENTS[name]
    try:
        ssh = connect(meta)
    except Exception as exc:
        return {"server": name, "exp": spec["exp"], "error": f"connect failed: {exc!r}"}
    try:
        out, err = run(ssh, train_command(project, spec), timeout=20)
        return {"server": name, "exp": spec["exp"], "name": spec["name"], "pid": out, "stderr": err}
    finally:
        ssh.close()


def start_tmux_one(name, meta, project):
    spec = EXPERIMENTS[name]
    try:
        ssh = connect(meta)
    except Exception as exc:
        return {"server": name, "exp": spec["exp"], "error": f"connect failed: {exc!r}"}
    try:
        out, err = run(ssh, tmux_train_command(project, spec), timeout=20)
        return {"server": name, "exp": spec["exp"], "name": spec["name"], "tmux": out, "stderr": err}
    finally:
        ssh.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "upload", "start", "start-tmux", "status"])
    parser.add_argument("--servers", nargs="*", default=list(SERVERS))
    parser.add_argument("--project-map", default=None, help="JSON file produced by inspect with project paths")
    parser.add_argument("--out", default=None, help="Optional UTF-8 JSON output path")
    args = parser.parse_args()

    os.makedirs(ROOT / "downloads", exist_ok=True)
    results = []
    if args.action == "inspect":
        for name in args.servers:
            results.append(inspect_one(name, SERVERS[name]))
    elif args.action == "status":
        for name in args.servers:
            meta = SERVERS[name]
            spec = EXPERIMENTS[name]
            try:
                ssh = connect(meta)
                project = resolve_project(ssh, meta["project_candidates"])
                item = {"server": name, "host": meta["host"], "project": project, "exp": spec["exp"], "name": spec["name"]}
                item["gpu"], _ = run(
                    ssh,
                    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits && "
                    "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
                    timeout=30,
                )
                if project:
                    item["tmux"], _ = run(ssh, "tmux ls 2>/dev/null || true", timeout=30)
                    log = posixpath.join(project, "codex_reports", f"{spec['name']}.log")
                    item["log_tail"], _ = run(ssh, f"tail -60 {shlex.quote(log)} 2>/dev/null || true", timeout=30)
                results.append(item)
            except Exception as exc:
                results.append({"server": name, "host": meta["host"], "error": repr(exc)})
            finally:
                with contextlib.suppress(Exception):
                    ssh.close()
    else:
        if not args.project_map:
            raise SystemExit("--project-map is required for upload/start")
        project_map = {x["server"]: x["project"] for x in json.load(open(args.project_map, encoding="utf-8"))}
        for name in args.servers:
            project = project_map.get(name)
            if not project:
                results.append({"server": name, "error": "missing project path"})
                continue
            if args.action == "upload":
                results.append(upload_one(name, SERVERS[name], project))
            elif args.action == "start":
                results.append(start_one(name, SERVERS[name], project))
            elif args.action == "start-tmux":
                results.append(start_tmux_one(name, SERVERS[name], project))

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
