import json
import os
import posixpath
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"
LOCAL_BEST = r"D:\A-主机文件\Codex\Mamba-YOLO-ds-ssm\source_code\server_3090_b"

FILES_TO_SYNC = [
    "ultralytics/nn/modules/mamba_yolo.py",
    "ultralytics/nn/modules/__init__.py",
    "ultralytics/nn/modules/mamba_yolo_ex27-E-new.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/metrics.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml",
]

TRAIN_SCRIPT = r'''from ultralytics import YOLO
import argparse
import os
from datetime import datetime


ROOT = os.path.abspath(".") + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/SSDD_YOLO/ssdd.yaml")
    parser.add_argument("--config", type=str, default="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-DS-SSM_Backbone-SSDD-ex7.yaml")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--project", default="output_dir/ssdd")
    parser.add_argument("--name", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def dynamic_freeze_callback(trainer):
    freeze_epochs = 20
    if trainer.epoch == 0:
        trainable_keywords = [
            "ds_ssm", "pfim", "lambda_s", "alpha", "beta_i", "rho_i", "eta_i",
            ".dw.", "gamma1", "norm1",
        ]
        trainable = 0
        frozen = 0
        for name, param in trainer.model.named_parameters():
            keep_train = any(k in name for k in trainable_keywords)
            param.requires_grad = keep_train
            if keep_train:
                trainable += param.numel()
            else:
                frozen += param.numel()
        print(f"[IG-DS-SSM Warmup] trainable={trainable / 1e6:.3f}M frozen={frozen / 1e6:.3f}M")
    elif trainer.epoch == freeze_epochs:
        for _, param in trainer.model.named_parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in trainer.model.parameters())
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"[IG-DS-SSM Warmup Done] total={total / 1e6:.3f}M trainable={trainable / 1e6:.3f}M")


if __name__ == "__main__":
    opt = parse_opt()
    name = opt.name or "ddp_PFIM_mambayolo_t_ssdd_backbone_ds-ssm_300e_80bs-ex27-codex-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "imgsz": opt.imgsz,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": name,
        "resume": opt.resume,
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        "warmup_bias_lr": 0.001,
    }
    model = YOLO(opt.config)
    try:
        model.load("yolov8n.pt")
        print("[pretrained] loaded yolov8n.pt")
    except Exception as exc:
        print(f"[pretrained] skipped: {exc}")
    model.add_callback("on_train_epoch_start", dynamic_freeze_callback)
    model.train(**args)
'''


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = posixpath.join(PROJECT, "codex_backups", f"before_best_ex27_ddp_{stamp}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    ssh.exec_command(f"mkdir -p {backup_dir!r} {posixpath.join(PROJECT, 'codex_reports')!r}")[1].read()
    manifest = {"backup_dir": backup_dir, "uploaded_at": datetime.now().isoformat(timespec="seconds"), "files": []}
    for rel in FILES_TO_SYNC:
        remote = posixpath.join(PROJECT, rel)
        local = os.path.join(LOCAL_BEST, *rel.split("/"))
        backup = posixpath.join(backup_dir, rel)
        ssh.exec_command(f"mkdir -p {posixpath.dirname(backup)!r} {posixpath.dirname(remote)!r} && if [ -f {remote!r} ]; then cp -p {remote!r} {backup!r}; fi")[1].read()
        sftp.put(local, remote)
        manifest["files"].append({"rel": rel, "local_source": local, "remote": remote, "backup": backup})
    train_remote = posixpath.join(PROJECT, "mbyolo_train_ddp_ex27.py")
    backup_train = posixpath.join(backup_dir, "mbyolo_train_ddp_ex27.py")
    ssh.exec_command(f"if [ -f {train_remote!r} ]; then cp -p {train_remote!r} {backup_train!r}; fi")[1].read()
    with sftp.file(train_remote, "w") as f:
        f.write(TRAIN_SCRIPT)
    manifest["files"].append({"rel": "mbyolo_train_ddp_ex27.py", "remote": train_remote, "backup": backup_train})
    report_remote = posixpath.join(PROJECT, "codex_reports", f"best_ex27_ddp_patch_{stamp}.json")
    with sftp.file(report_remote, "w") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    sftp.close()
    ssh.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
