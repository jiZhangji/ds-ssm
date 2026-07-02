import posixpath
from datetime import datetime

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"


HELPER = r'''

# ---- DS-SSM/PFIM warmup freeze patch for DDP/single-GPU ----
def _dsssm_freeze_warmup(trainer, freeze_epochs=20):
    """
    DDP/single-GPU compatible warmup freeze.

    First freeze_epochs:
        freeze old YOLO/Mamba backbone parameters;
        train only IG-DS-SSM/PFIM related new parameters.

    epoch == freeze_epochs:
        unfreeze all parameters for joint fine-tuning.
    """
    import os

    epoch = getattr(trainer, "epoch", 0)
    model = getattr(trainer, "model", None)
    if model is None:
        return

    raw_model = model.module if hasattr(model, "module") else model
    named_params = list(raw_model.named_parameters())
    if not named_params:
        return

    has_dsssm = any(("ds_ssm" in n or "pfim" in n or "lambda_s" in n) for n, _ in named_params)
    if not has_dsssm:
        return

    trainable_keywords = (
        "ds_ssm",
        "pfim",
        "lambda_s",
        "alpha",
        "beta_i",
        "rho_i",
        "eta_i",
        ".dw.",
        "gamma1",
        "norm1",
    )

    rank = os.environ.get("RANK", "-1")
    is_main = rank in ("-1", "0")

    if epoch == 0:
        trainable, frozen = 0, 0
        kept_names = []
        for name, param in named_params:
            keep_train = any(k in name for k in trainable_keywords)
            if keep_train:
                param.requires_grad = True
                trainable += param.numel()
                kept_names.append(name)
            else:
                param.requires_grad = False
                frozen += param.numel()

        if is_main:
            print(f"\n[IG-DS-SSM Warmup] Freeze old backbone; first {freeze_epochs} epochs train only DS-SSM/PFIM modules.")
            for name in kept_names:
                print(f"  -> [train] {name}")
            print(f"[IG-DS-SSM Warmup] trainable={trainable / 1e6:.3f}M frozen={frozen / 1e6:.3f}M")

    elif epoch == freeze_epochs:
        for _, param in named_params:
            param.requires_grad = True
        total = sum(p.numel() for _, p in named_params)
        trainable = sum(p.numel() for _, p in named_params if p.requires_grad)
        if is_main:
            print(f"\n[IG-DS-SSM Warmup Done] Epoch {epoch}: unfreeze all parameters for joint fine-tuning.")
            print(f"[IG-DS-SSM Warmup Done] total={total / 1e6:.3f}M trainable={trainable / 1e6:.3f}M")
'''


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rel = "ultralytics/engine/trainer.py"
    remote = posixpath.join(PROJECT, rel)
    backup = posixpath.join(PROJECT, "codex_backups", f"trainer_before_dsssm_warmup_{stamp}.py")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    sftp = ssh.open_sftp()
    ssh.exec_command(f"mkdir -p {posixpath.dirname(backup)!r} && cp -p {remote!r} {backup!r}")[1].read()
    with sftp.file(remote, "r") as f:
        text = f.read().decode("utf-8")

    if "_dsssm_freeze_warmup" not in text:
        text = text.rstrip() + HELPER + "\n"

    call = '            _dsssm_freeze_warmup(self, freeze_epochs=int(__import__("os").environ.get("DSSSM_FREEZE_EPOCHS", "20")))\n'
    marker = "            self.run_callbacks(\"on_train_epoch_start\")\n"
    if call.strip() not in text:
        if marker not in text:
            raise RuntimeError("epoch callback marker not found")
        text = text.replace(marker, call + marker, 1)

    with sftp.file(remote, "w") as f:
        f.write(text)
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
