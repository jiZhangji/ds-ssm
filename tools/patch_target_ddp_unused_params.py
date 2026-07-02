from datetime import datetime
import posixpath

import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"

TARGET = "ultralytics/engine/trainer.py"
OLD = "self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])"
NEW = (
    "self.model = nn.parallel.DistributedDataParallel("
    "self.model, device_ids=[RANK], find_unused_parameters=True)"
)


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
        look_for_keys=False,
        allow_agent=False,
    )
    sftp = ssh.open_sftp()
    remote = posixpath.join(PROJECT, TARGET)
    with sftp.open(remote, "r") as f:
        text = f.read().decode("utf-8")

    if NEW in text:
        print("already_patched=true")
        sftp.close()
        ssh.close()
        return
    if OLD not in text:
        raise RuntimeError(f"DDP target line not found in {remote}")

    backup_dir = posixpath.join(PROJECT, "codex_backups")
    ssh.exec_command(f"mkdir -p {backup_dir}")[1].read()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = posixpath.join(backup_dir, f"trainer_before_ddp_find_unused_{timestamp}.py")
    with sftp.open(backup, "w") as f:
        f.write(text)

    text = text.replace(OLD, NEW, 1)
    with sftp.open(remote, "w") as f:
        f.write(text.encode("utf-8"))

    print(f"backup={backup}")
    print("patched=find_unused_parameters=True")
    sftp.close()
    ssh.close()


if __name__ == "__main__":
    main()
