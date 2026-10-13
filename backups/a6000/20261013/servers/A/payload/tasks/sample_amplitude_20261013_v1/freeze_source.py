"""Copy the known model sources into a new task; no training or overwrite."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if sys.platform != 'linux':
        raise RuntimeError('Server only')
    root = Path(__file__).resolve().parent
    previous = root.parent / 'rgsp_seedwise_20261009_v1'
    target = root / 'frozen'
    if target.exists():
        raise FileExistsError('Do not recreate or overwrite a frozen snapshot')
    target.mkdir()
    for name in ('cross_trial.py','cross_data.py','manifest.py','cuda_uuid_compat.py'):
        shutil.copy2(previous/name, target/name)
    src = previous/'code'
    for p in sorted(src.rglob('*')):
        if p.is_file() and p.suffix in ('.py','.yaml','.yml') and '__pycache__' not in p.parts:
            dest = target/'code'/p.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p,dest)
    records=[]
    for p in sorted(target.rglob('*')):
        if p.is_file():
            old = previous/p.relative_to(target)
            h = sha(p)
            if sha(old)!=h:
                raise RuntimeError('Copy differs: '+str(p))
            records.append({'path':str(p),'sha256':h,'source_path':str(old)})
    (root/'source_copy.json').write_text(json.dumps({'files':records,'status':'COPIED_AND_HASH_VERIFIED'},indent=2)+'\n')
    print(json.dumps({'status':'COPIED_AND_HASH_VERIFIED','files':len(records)}))

if __name__=='__main__':
    main()
