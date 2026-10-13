"""Read-only inventory of explicitly scoped project artifacts and hardware."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from datetime import datetime, timezone

BASE = Path('/home/powerleader/ig-ds-ssm')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {'error': str(exc)}


def main():
    result = {'hostname': socket.gethostname(), 'utc': datetime.now(timezone.utc).isoformat(),
              'disk_free_gib': round(shutil.disk_usage(BASE).free / 1024**3, 2),
              'loadavg': list(os.getloadavg()), 'cpus': os.cpu_count(), 'lanes': [], 'data': []}
    for label, args in [('gpus', ['--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu']),
                        ('gpu_processes', ['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])]:
        p = subprocess.run(['nvidia-smi', *args, '--format=csv,noheader'], capture_output=True, text=True)
        result[label] = p.stdout.strip()
    for campaign in ['rgsp_controls_20261008_v3', 'rgsp_sadd_controls_20261008_v1', 'rgsp_recovery_20261008_v1']:
        root = BASE / 'tasks' / campaign
        if not root.exists():
            continue
        for lane in sorted(root.iterdir()):
            if not lane.is_dir() or not (lane.name.startswith('seed') or lane.name.startswith('ssdd_seed') or lane.name.startswith('sadd_seed')):
                continue
            row = {'root': str(lane), 'states': {}, 'runs': []}
            for name in ['status.json', 'queue_state.json', 'manifest/protocol.json']:
                path = lane / name
                if path.exists():
                    value = read(path)
                    if name.endswith('protocol.json'):
                        value = {key: value.get(key) for key in ['campaign', 'seed', 'epochs', 'patience', 'batch', 'split', 'data_manifest_sha256']}
                    row['states'][name] = value
            for run in sorted((lane / 'runs').glob('*')):
                if not run.is_dir():
                    continue
                entry = {'variant': run.name, 'run_dir': str(run)}
                for name in ['train_complete.json', 'evaluation_complete.json', 'eval_test.json']:
                    file = run / name
                    if file.exists():
                        entry[name] = read(file)
                if (run / 'results.csv').exists():
                    with (run / 'results.csv').open() as stream:
                        lines = stream.readlines()
                    entry['last_result'] = lines[-1].strip() if lines else ''
                for name in ['weights/best.pt', 'weights/last.pt']:
                    file = run / name
                    if file.exists():
                        entry[name] = {'bytes': file.stat().st_size, 'sha256': sha(file)}
                row['runs'].append(entry)
            result['lanes'].append(row)
    for base_name in ['cross_task_prepared_v2', 'cross_task_prepared', 'sar4_prepared', 'ssdd_bs32_source']:
        root = BASE / 'data' / base_name
        if not root.exists():
            continue
        for path in sorted(root.glob('*/*/manifest.json')):
            payload = read(path)
            records = payload.get('records', [])
            counts = {split: sum(row.get('split') == split for row in records) for split in ['train', 'val', 'test']}
            sample = records[:1]
            result['data'].append({'manifest': str(path), 'sha256': sha(path), 'task': payload.get('task'),
                                   'classes': payload.get('classes'), 'counts': counts, 'sample': sample})
        for path in sorted(root.glob('*/*/*.yaml')):
            result['data'].append({'yaml': str(path), 'text': path.read_text()[:2000]})
        result.setdefault('data_dirs', []).append({'root': str(root), 'directories': [str(p) for p in root.glob('*/*') if p.is_dir()]})
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
