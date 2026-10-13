"""Read-only snapshots of the explicit four-run task and its Full smoke."""
import csv
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

root = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_table_completion_20261012_v1/training')
runs = []
for name in ['ssdd_cnn_yolov8n_seed0', 'ssdd_original_seed0',
             'ssdd_matched_scale075_seed0', 'ssdd_full_seed0', 'smoke_full_b4_v2']:
    directory = root / name
    if not directory.is_dir():
        continue
    item = {'name': name, 'path': str(directory), 'files': {}}
    for filename in ['protocol.json', 'runtime_audit.json', 'data_manifest.json', 'result.json']:
        path = directory / filename
        if path.is_file():
            raw = path.read_bytes()
            entry = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            if filename != 'data_manifest.json':
                entry['data'] = json.loads(raw)
            item['files'][filename] = entry
    launch_path = root / (name + '.launch.json')
    if launch_path.is_file():
        raw = launch_path.read_bytes()
        launch = json.loads(raw)
        item['launch'] = launch
        item['launch_sha256'] = hashlib.sha256(raw).hexdigest()
        proc = Path('/proc') / str(launch['pid'])
        try:
            stat = (proc / 'stat').read_text().split()
            item['process_identity_matches'] = stat[21] == launch['proc_start_ticks']
            item['process_state'] = stat[2]
        except FileNotFoundError:
            item['process_identity_matches'] = False
            item['process_state'] = 'absent'
    ledger = directory / 'optimizer_ledger.jsonl'
    if ledger.is_file():
        raw = ledger.read_bytes()
        valid = []
        for line in raw.splitlines():
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        item['ledger'] = {'sha256_at_snapshot': hashlib.sha256(raw).hexdigest(),
                          'completed_epoch_rows': len(valid), 'last_complete_row': valid[-1] if valid else None,
                          'continuous_epochs': [r['epoch'] for r in valid] == list(range(1, len(valid) + 1))}
    results = directory / 'run/results.csv'
    if results.is_file():
        raw = results.read_text()
        rows = list(csv.DictReader(io.StringIO(raw)))
        item['csv'] = {'rows': len(rows), 'last_row': rows[-1] if rows else None}
    item['complete'] = item['files'].get('result.json', {}).get('data', {}).get('paper_eligible', False)
    if item['complete']:
        final = item['files']['result.json']['data']
        checkpoint = directory / 'run/weights/best.pt'
        actual_best_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        item['completion_checks'] = {
            'status_training_complete': final['status'] == 'TRAINING_COMPLETE',
            'full_300_epochs': final['epoch'] == 300,
            'full_8700_updates': final['optimizer_updates'] == 8700,
            'ledger_has_300_continuous_epochs': item['ledger']['continuous_epochs'] and item['ledger']['completed_epoch_rows'] == 300,
            'best_checkpoint_sha_matches': actual_best_sha == final['best_sha256'],
            'best_checkpoint_actual_sha256': actual_best_sha,
            'serialized_best_fp32_verified': final['serialized_best_fp32_verified'],
            'not_smoke': not final['protocol']['smoke_only'],
        }
    runs.append(item)
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                      '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
print(json.dumps({'hostname': socket.gethostname(), 'utc': datetime.now(timezone.utc).isoformat(),
                  'gpu_compute': gpu, 'runs': runs}, ensure_ascii=False))
