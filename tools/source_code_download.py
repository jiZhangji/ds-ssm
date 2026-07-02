import json
import os
import stat
from datetime import datetime

import paramiko


REMOTE_ROOT = "/home/zhangji/Mamba-YOLO"
LOCAL_ROOT = "source_code"

SERVERS = {
    "server_1080ti": {"host": "172.26.193.188", "user": "zhangji", "password": "123456"},
    "server_3090_a": {"host": "172.26.150.36", "user": "zhangji", "password": "12345"},
    "server_3090_b": {"host": "172.26.133.16", "user": "zhangji", "password": "12345"},
}

EXCLUDED_DIRS = {
    "output_dir",
    "datasets",
    "weights",
    "runs",
    "__pycache__",
    ".pytest_cache",
    "build",
}

EXCLUDED_SUFFIXES = {
    ".pt",
    ".pth",
    ".onnx",
    ".engine",
    ".torchscript",
    ".tflite",
    ".pb",
    ".bin",
    ".npy",
    ".npz",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".avi",
    ".log",
}

MAX_FILE_BYTES = 5 * 1024 * 1024


def should_skip(rel_path, filename, size):
    parts = rel_path.replace("\\", "/").split("/")
    if any(part in EXCLUDED_DIRS for part in parts):
        return True, "excluded_dir"
    lower = filename.lower()
    if any(lower.endswith(s) for s in EXCLUDED_SUFFIXES):
        return True, "excluded_suffix"
    if size > MAX_FILE_BYTES and ".git/" not in rel_path.replace("\\", "/"):
        return True, "large_file"
    return False, ""


def ensure_dirs():
    for path in ["source_code", "reports", "inventory", "database", "downloads", "downloads/source_code_raw"]:
        os.makedirs(path, exist_ok=True)


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


def download_tree(sftp, remote_dir, local_dir, rel_dir, stats):
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        name = entry.filename
        if name in [".", ".."]:
            continue
        remote_path = remote_dir + "/" + name
        rel_path = f"{rel_dir}/{name}" if rel_dir else name
        local_path = os.path.join(local_dir, name)
        mode = entry.st_mode
        if stat.S_ISDIR(mode):
            skip, reason = should_skip(rel_path, name, 0)
            if skip and name != ".git":
                stats["skipped_dirs"].append({"path": rel_path, "reason": reason})
                continue
            download_tree(sftp, remote_path, local_path, rel_path, stats)
        elif stat.S_ISREG(mode):
            skip, reason = should_skip(rel_path, name, entry.st_size)
            if skip:
                stats["skipped_files"].append({"path": rel_path, "size": entry.st_size, "reason": reason})
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            sftp.get(remote_path, local_path)
            stats["files"] += 1
            stats["bytes"] += entry.st_size
            stats["downloaded"].append({"path": rel_path, "size": entry.st_size})


def main():
    ensure_dirs()
    all_stats = {}
    for name, meta in SERVERS.items():
        print(f"downloading source snapshot: {name}", flush=True)
        local_dir = os.path.join(LOCAL_ROOT, name)
        stats = {
            "server": name,
            "host": meta["host"],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "files": 0,
            "bytes": 0,
            "downloaded": [],
            "skipped_files": [],
            "skipped_dirs": [],
            "error": None,
        }
        try:
            ssh = connect(meta)
            sftp = ssh.open_sftp()
            download_tree(sftp, REMOTE_ROOT, local_dir, "", stats)
            sftp.close()
            ssh.close()
        except Exception as exc:
            stats["error"] = repr(exc)
        stats["finished_at"] = datetime.now().isoformat(timespec="seconds")
        all_stats[name] = stats
        with open(os.path.join("downloads", "source_code_raw", f"{name}_download_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join("downloads", "source_code_raw", "all_download_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
