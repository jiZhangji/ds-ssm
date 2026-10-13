"""Queue official TorchVision SSDD baselines after the SAR-ACD B lane."""
from pathlib import Path
import worker

ROOT = Path(__file__).resolve().parent
REMOTE = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_table_expansion_20261013_v1')
OLD = REMOTE.parent / 'rgsp_seedwise_20261009_v1'
PYTHON = '/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'

if __name__ == '__main__':
    methods = ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'fcos_resnet50_fpn']
    jobs = []
    for method in methods:
        command = [PYTHON, '-B', '-u', str(REMOTE / 'external_detection/run_torchvision.py'),
            '--source-root', str(OLD / 'code'), '--dataset-manifest', str(OLD / 'data/ssdd/data_content.json'),
            '--data', str(OLD / 'data/ssdd/data.yaml'), '--dataset-id', 'ssdd', '--method', method,
            '--seed', '{seed}', '--output', '{output}', '--gpu-uuid', '{gpu_uuid}']
        jobs.append({'id':'ssdd-' + method, 'task':'detection', 'dataset':'ssdd', 'method':method,
            'stage':20, 'min_free_gb':40, 'estimated_seconds':14400, 'cwd':str(REMOTE),
            'command_probe':command + ['--mode','probe'], 'command_train':command + ['--mode','train']})
    plan = worker.validate_plan({'version':1, 'lane':'b0det', 'gpu_uuid':'GPU-09b018e1-7140-296a-73d4-9e010ee209a2',
        'jobs':jobs, 'legacy_lock_required':True, 'idle_seconds':60, 'poll_seconds':5,
        'min_disk_free_gb':40, 'max_gpu_utilization':5})
    worker.atomic_json(ROOT / 'plans/b0det.json',plan,overwrite=False)
    # The existing gate reads the predecessor's base plan only for schema QA.
    prior = worker.read_json(ROOT / 'plans/b0acd.production.json')
    worker.atomic_json(ROOT / 'plans/b0acd.json',prior,overwrite=False)
    print('Created finite b0det preflight plan; no detector launched by this builder.')
