"""Wait for verified transfer, prepare data, then run the GPU-gated queue."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
lock = (root / 'bootstrap.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
archive = root / 'incoming/AIR-SARShip-2.0-data-lan.zip'
expected = '34e6b7530ee46580c3b6e555750eb0b4af54f0350c7c32032a15fe906ed60566'
while True:
    size = archive.stat().st_size if archive.exists() else 0
    (root / 'status.json').write_text(json.dumps({'state': 'WAITING_FOR_DATA_TRANSFER',
        'received_bytes': size, 'expected_bytes': 235867094, 'pid': os.getpid()}))
    if size == 235867094:
        h = hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError('Source image archive SHA256 mismatch; training blocked')
        archive.replace(root / 'incoming/AIR-SARShip-2.0-data.zip')
        break
    time.sleep(60)
env = dict(os.environ, PYTHONPATH=str(root / 'code'))
if not (root / 'data_manifest.json').exists():
    subprocess.run([sys.executable, str(root / 'queue_experiment.py'), '--prepare'],
                   cwd=root / 'code', env=env, check=True)
subprocess.run([sys.executable, str(root / 'queue_experiment.py')],
               cwd=root / 'code', env=env, check=True)
