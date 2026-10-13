"""Explicit D-server sequential probes; no formal experiment is launched."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
UUID = 'GPU-f1b51ae3-c020-9901-c608-cca67f6ab8f1'
os.environ['CUDA_VISIBLE_DEVICES'] = UUID
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for variant, batch in [('cnn_yolov8n', 16), ('original', 4), ('original', 16),
                       ('matched_scale075', 4), ('matched_scale075', 16), ('full', 4), ('full', 16)]:
    output = ROOT / ('probe_' + variant + '_b' + str(batch) + '_v1')
    command = [sys.executable, '-u', str(ROOT / 'probe_training.py'),
               '--source-root', str(ROOT / 'frozen_code'),
               '--train-images', '/home/powerleader/ig-ds-ssm/data/sar4_prepared/SSDD/images/train',
               '--train-labels', '/home/powerleader/ig-ds-ssm/data/sar4_prepared/SSDD/labels/train',
               '--gpu-uuid', UUID, '--legacy-gpu-lock',
               '/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1/locks/' + UUID + '.lock',
               '--variant', variant, '--batch', str(batch), '--output', str(output),
               '--updates', '12', '--warmup-updates', '3']
    with (ROOT / (output.name + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'variant': variant, 'batch': batch, 'exit_code': result.returncode,
                      'result': str(output / 'result.json')}), flush=True)
    if result.returncode:
        sys.exit(result.returncode)
