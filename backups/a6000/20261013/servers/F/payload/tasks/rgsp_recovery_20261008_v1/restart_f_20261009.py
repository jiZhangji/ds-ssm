import json, os, shutil, subprocess, time
from pathlib import Path
import recovery_queue as r

root = Path(__file__).resolve().parent
fixed = root / 'train_redirect_fixed_20261009.py'
assert fixed.exists()
assert not any('recovery_queue.py' in p.read_text(errors='ignore') for p in Path('/proc').glob('[0-9]*/cmdline') if p.exists())
shutil.copy2(root / 'train_redirect.py', root / 'train_redirect.before_fix_20261009.py')
shutil.copy2(fixed, root / 'train_redirect.py')
for key in ('f0', 'f1'):
    cfg = r.layout(key)
    output = Path(cfg['output'])
    manifest = output / 'recovery_manifest.json'
    shutil.copy2(manifest, output / 'recovery_manifest.before_fix_20261009.json')
    shutil.copy2(output / 'queue_state.json', output / 'queue_state.before_fix_20261009.json')
    cfg = r.read(manifest)
    assert r.sha(output / 'runs/matched_scale075/weights/last.pt') == cfg['checkpoint']['sha256']
    r.code_audit(cfg)
    assert r.gpu_idle(r.gpu_state(cfg))
    cfg['redirect_sha256'] = r.sha(root / 'train_redirect.py')
    cfg['wrapper_fix_20261009'] = 'Respect frozen SSDD CLI: fixed internal epochs=150/patience=30/batch=16'
    r.write(manifest, cfg)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    with (output / 'queue.log').open('a') as log:
        p = subprocess.Popen([str(r.PYTHON), str(root / 'recovery_queue.py'), 'queue', '--key', key], cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'key':key,'pid':p.pid,'wrapper_sha256':cfg['redirect_sha256'],'time':time.time()}), flush=True)
