"""Bind the independently audited SAMPLE data and reviewed runner without model execution."""
import hashlib
import json
from pathlib import Path
import socket

root = Path(__file__).resolve().parent
if socket.gethostname() != 'powerleader-PT6620W' or str(root) != '/home/powerleader/ig-ds-ssm/tasks/rgsp_table_completion_20261012_v1/sample':
    raise RuntimeError('Designated independent A task only')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
expected = json.loads((root / 'support_expected.json').read_text())
for row in expected:
    if sha(root / row['path']) != row['sha256']:
        raise RuntimeError('Support upload mismatch: ' + row['path'])
inventory = json.loads((root / 'source_inventory.json').read_text())
audit = json.loads((root / 'data/audit.json').read_text())
if audit['status'] != 'PASS' or audit['counts'] != {'train': 806, 'test': 539}:
    raise RuntimeError('Source audit not accepted')
source = Path(inventory['source_root'])
for row in inventory['files']:
    if sha(source / row['path']) != row['sha256']:
        raise RuntimeError('Frozen source changed: ' + row['path'])
if sha(root / 'data/manifest_train.json') != audit['manifest_train_sha256']:
    raise RuntimeError('Training manifest changed')
freeze = {'schema': 'sample-run-freeze-v1', 'protocol_sha256': sha(root / 'protocol.json'),
          'manifest_sha256': sha(root / 'data/manifest_train.json'),
          'audit_sha256': sha(root / 'data/audit.json'), 'source_root': str(source),
          'source_files': [{'path': r['path'], 'sha256': r['sha256']} for r in inventory['files']],
          'support_files': expected}
with (root / 'freeze.json').open('x') as out:
    json.dump(freeze, out, indent=2)
    out.write('\n')
print(json.dumps({'status': 'FROZEN', 'source_files': len(freeze['source_files']),
                  'freeze_sha256': sha(root / 'freeze.json'),
                  'training_manifest_sha256': freeze['manifest_sha256'],
                  'support_files': len(expected)}))
