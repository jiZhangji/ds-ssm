"""Launch one explicit server trial with a durable process identity receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['cnn_yolov8n', 'original', 'matched_scale075', 'full'], required=True)
p.add_argument('--gpu-uuid', required=True)
p.add_argument('--dataset-root', type=Path, required=True)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--legacy-gpu-lock', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--batch', type=int, choices=[4, 16], required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
if os.name != 'posix' or not a.output.is_absolute() or a.output.exists():
    raise RuntimeError('New absolute Linux task output required')
receipt = a.output.parent / (a.output.name + '.launch.json')
log_path = a.output.parent / (a.output.name + '.log')
# Exclusive creation prevents a second launcher from targeting this run.
with receipt.open('x') as handle:
    command = [sys.executable, '-u', str(here / 'train_ssdd.py'),
               '--source-root', str(a.source_root), '--dataset-root', str(a.dataset_root),
               '--output', str(a.output), '--gpu-uuid', a.gpu_uuid,
               '--legacy-gpu-lock', str(a.legacy_gpu_lock), '--variant', a.variant,
               '--batch', str(a.batch)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu_uuid, PYTHONDONTWRITEBYTECODE='1')
    with log_path.open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(here), env=environment, start_new_session=True)
    time.sleep(0.25)
    stat = Path('/proc') / str(child.pid) / 'stat'
    record = {'pid': child.pid, 'hostname': socket.gethostname(), 'argv': command,
              'gpu_uuid': a.gpu_uuid, 'status_at_launch': 'started' if child.poll() is None else 'exited',
              'exit_code_at_launch': child.poll(), 'proc_start_ticks': stat.read_text().split()[21] if stat.exists() else None,
              'script_sha256': hashlib.sha256((here / 'train_ssdd.py').read_bytes()).hexdigest(),
              'log': str(log_path), 'output': str(a.output), 'time_unix': time.time()}
    json.dump(record, handle, indent=2)
    handle.write('\n')
print(json.dumps(record))
if record['exit_code_at_launch'] is not None:
    sys.exit(record['exit_code_at_launch'] or 1)
