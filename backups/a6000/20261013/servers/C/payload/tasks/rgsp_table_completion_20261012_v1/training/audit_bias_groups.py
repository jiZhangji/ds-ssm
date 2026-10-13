"""Inspect a task-owned trusted checkpoint on CPU; no model forward or GPU use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True, type=Path)
p.add_argument('--source-root', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(a.source_root))
import torch
torch.set_num_threads(2)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir(parents=True)
snapshot = a.output / 'checkpoint_snapshot.pt'
before = sha(a.checkpoint)
shutil.copyfile(a.checkpoint, snapshot)
if sha(snapshot) != before or sha(a.checkpoint) != before:
    raise RuntimeError('Checkpoint changed during snapshot; retry with a new audit directory')
ckpt = torch.load(snapshot, map_location='cpu')
model = ckpt['model']
norms = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k and isinstance(v, type))
versions = {version: {'decay': [], 'norm': [], 'bias': []} for version in ['v1', 'v2']}
for module_name, module in model.named_modules():
    for name, param in module.named_parameters(recurse=False):
        if not param.requires_grad:
            continue
        fullname = module_name + '.' + name if module_name else name
        for version, groups in versions.items():
            is_bias = name == 'bias' if version == 'v1' else 'bias' in name
            group = 'bias' if is_bias else ('norm' if isinstance(module, norms) else 'decay')
            groups[group].append(fullname)
# Match actual saved optimizer group ordering/parameter indices to the exact
# v1 construction. Merely comparing names would not validate the active run.
index_to_name = [n for group in ['decay', 'norm', 'bias'] for n in versions['v1'][group]]
actual = {}
for group in ckpt['optimizer']['param_groups']:
    name = group['group_name']
    actual_names = [index_to_name[i] for i in group['params']]
    if actual_names != versions['v1'][name]:
        raise RuntimeError('Saved optimizer parameter index ordering differs from v1')
    if abs(group['weight_decay'] - (0.0005 if name == 'decay' else 0.0)) > 1e-12:
        raise RuntimeError('Unexpected actual weight decay')
    actual[name] = {'names': actual_names, 'weight_decay': group['weight_decay'],
                    'betas': group['betas'], 'eps': group['eps']}
result = {'status': 'AUDIT_PASS', 'checkpoint_sha256': before, 'checkpoint_epoch': ckpt['epoch'] + 1,
          'v1_v2_parameter_groups_identical': versions['v1'] == versions['v2'],
          'actual_saved_optimizer_groups': actual, 'v1_expected_names': versions['v1'],
          'v2_expected_names': versions['v2'], 'cpu_deserialization_only': True,
          'model_forward_executed': False}
with (a.output / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in ['actual_saved_optimizer_groups', 'v1_expected_names', 'v2_expected_names']}))
