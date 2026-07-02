import argparse
import sys
import paramiko


HOST = "172.26.151.143"
USER = "zhangji"
PASSWORD = "12345"
PROJECT = "/mnt/data5/zhangji/Mamba-YOLO"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    command = " ".join(args.command).strip()
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
    _, stdout, stderr = ssh.exec_command(f"cd {PROJECT} && {command}", timeout=args.timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    ssh.close()
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    if err:
        sys.stdout.buffer.write(err.encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
