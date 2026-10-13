"""Build finite paired SAR-ACD lanes; no training is launched here."""
import json
from pathlib import Path
import worker

ROOT = Path(__file__).resolve().parent
REMOTE = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_table_expansion_20261013_v1')
OLD = REMOTE.parent / 'rgsp_seedwise_20261009_v1'
PYTHON = '/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'
LANES = {
    'b0acd': ('GPU-09b018e1-7140-296a-73d4-9e010ee209a2', ['Original', 'Matched075', 'Full', 'resnet50']),
    'a0acd': ('GPU-12cf4841-46b9-94d1-2715-9932134cf964', ['alexnet', 'googlenet', 'vgg16', 'vit_b_16', 'swin_t', 'swin_b']),
}


def build(lane):
    gpu, methods = LANES[lane]
    jobs = []
    for method in methods:
        for variant in ('full3032', 'clean2992'):
            dataset = 'SAR-ACD-' + variant
            command = [PYTHON, '-B', '-u', str(REMOTE / 'run_cross.py'),
                '--campaign-root', str(OLD), '--source-root', str(OLD / 'code'),
                '--manifest', str(REMOTE / 'taskdata/acd_pair_v1' / variant / 'manifest.json'),
                '--dataset', dataset, '--method', method, '--seed', '{seed}',
                '--output', '{output}', '--gpu-uuid', '{gpu_uuid}']
            jobs.append({'id': variant + '-' + method.lower(), 'task': 'classification',
                'dataset': dataset, 'method': method,
                'stage': 0 if method in {'Original', 'Matched075', 'Full'} else 20,
                'estimated_seconds': 3600, 'min_free_gb': 24, 'cwd': str(REMOTE),
                'command_probe': command + ['--mode', 'probe'],
                'command_train': command + ['--mode', 'train']})
    return worker.validate_plan({'version': 1, 'lane': lane, 'gpu_uuid': gpu,
        'jobs': jobs, 'legacy_lock_required': True, 'idle_seconds': 60,
        'poll_seconds': 5, 'min_disk_free_gb': 30, 'max_gpu_utilization': 5}, lane)


if __name__ == '__main__':
    (ROOT / 'plans').mkdir(exist_ok=True)
    for lane in LANES:
        plan = build(lane)
        path = ROOT / 'plans' / (lane + '.production.json')
        worker.atomic_json(path, plan, overwrite=False)
        print(json.dumps({'lane': lane, 'jobs': len(plan['jobs']), 'plan_sha256': worker.canonical_sha(plan)}))
