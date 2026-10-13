"""Finite config-bound CD complexity measurement; registered A6000 A only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-12cf4841-46b9-94d1-2715-9932134cf964'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import json, sys, time, socket, subprocess, hashlib, fcntl
from pathlib import Path
from contextlib import ExitStack

ROOT = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_unified_a6000_20261012_v2')
OUT = Path(__file__).resolve().parent
GPU = os.environ['CUDA_VISIBLE_DEVICES']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def smi(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()

def idle():
    active = smi('--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits')
    if any(line.startswith(GPU) for line in active.splitlines()):
        raise RuntimeError('Assigned GPU has a compute process; no preemption')

def main():
    if socket.gethostname() != 'powerleader-PT6620W' or not sys.platform.startswith('linux'):
        raise RuntimeError('Execute only on registered A6000 A')
    if (OUT / 'results.json').exists():
        raise FileExistsError('Preserve previous measurement')
    plan = json.loads((ROOT / 'plans/a0.production.json').read_text())
    if plan['gpu_uuid'] != GPU:
        raise RuntimeError('GPU identity mismatch')
    with ExitStack() as stack:
        for path in [ROOT / 'locks' / (GPU + '.lock'), Path(plan['legacy_lock']), OUT / (GPU + '.lock')]:
            lock = stack.enter_context(path.open('rb' if path == Path(plan['legacy_lock']) else 'ab'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(12):
            idle()
            time.sleep(5)
        idle()
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(OUT / 'deps'))
        from external.cd.author_models import build_model, get_method_spec
        from detection.covered_flops import profile_model
        # Use the project's validated torch1.13 UUID compatibility setup.
        job = next(j for j in plan['jobs'] if j['id'] == 'ottawa-original')
        cmd = job['command_train']
        campaign = Path(cmd[cmd.index('--campaign-root') + 1])
        sys.path.insert(0, str(campaign))
        from cuda_uuid_compat import ensure_uuid_cuda
        identity = ensure_uuid_cuda(GPU)
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = []
        expected = {'ddnet': 6712, 'lantnet': 24956, 'camixer': 34214, 'wbanet': 7159, 'wasnet': 7360}
        for method, count in expected.items():
            torch.manual_seed(0)
            model = build_model(method, allow_fp32_adaptation=True).eval()
            spec = get_method_spec(method)
            params = sum(p.numel() for p in model.parameters())
            if params != count:
                raise RuntimeError('Parameter count differs from completed production run: ' + method)
            shape = (1, spec['channels'], spec['patch_size'], spec['patch_size'])
            inputs = (torch.rand(shape, device='cuda:0', dtype=torch.float32) * 254 + 1,)
            row = profile_model(model, inputs=inputs)
            row.update(method=method, model_source_audit=model._cd_source_audit,
                       checkpoint_loaded=False, initialization='random seed0, exact frozen author adapter',
                       scope_limit='One native center patch; not image-total FLOPs. Functional OpenCV DCT, wavelet convolutions, attention matmul, activations, pooling/statistics and image preprocessing/postprocessing are not exhaustively counted.')
            (OUT / (method + '.json')).write_text(json.dumps(row, indent=2), encoding='utf-8')
            print(json.dumps({k: row.get(k) for k in ['method', 'status', 'parameters', 'covered_flops_g', 'error']}), flush=True)
            rows.append(row)
            del model, inputs
            torch.cuda.empty_cache()
        sources = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'external/cd/author_models.py', ROOT / 'external/cd/source_registry.json', ROOT / 'detection/covered_flops.py']}
        (OUT / 'results.json').write_text(json.dumps({'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'PARTIAL_FAILURE', 'host': socket.gethostname(), 'gpu_identity': identity, 'created_at': time.strftime('%F %T %z'), 'script_sha256': sha(__file__), 'source_hashes': sources, 'dependency_hashes': {str(p.relative_to(OUT / 'deps')): sha(p) for p in (OUT / 'deps').rglob('*') if p.is_file() and '__pycache__' not in p.parts}, 'results': rows}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
