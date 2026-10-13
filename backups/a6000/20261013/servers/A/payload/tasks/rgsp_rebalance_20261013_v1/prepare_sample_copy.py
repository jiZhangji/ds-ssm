import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_unified_a6000_20261012_v2')
TASK=Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_rebalance_20261013_v1')
archive=TASK/'sample_source.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='e93f86968b2a4248800cde26d542a4407dadaadd13a48dec2f9cd4d4e24091fb'
raw=TASK/'raw'
assert not raw.exists()
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers()
 assert len(members)==2690 and sum(m.size for m in members)<256*1024*1024
 for m in members:
  p=Path(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts and p.parts[0]=='decibel'
 raw.mkdir()
 tar.extractall(raw,members=members)
python='/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'
subprocess.run([python,'-B',str(ROOT/'data/prepare_classification.py'),'--taskdata-root',str(TASK/'taskdata'),'--datasets','SAMPLE','--sample-root',str(raw/'decibel')],check=True)
sys.path.insert(0,str(ROOT/'data'))
from prepare_classification import verify_classification_manifest
verified=verify_classification_manifest(TASK/'taskdata/SAMPLE/manifest.json')
ready=json.loads((TASK/'taskdata/SAMPLE/readiness.json').read_text())
assert ready['logical_manifest_sha256']=='2c00618ae1e7cd5405f83af3a0aa8640e5c68871f58406251cd83e8c92c39622'
assert ready['counts']=={'train':645,'val':161,'test':539}
receipt={'schema':'rgsp-verified-sample-copy-v1','source_host':'C','destination_host':'A','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'ready':ready,'verification':verified,'same_logical_manifest_as_C':True}
with (TASK/'sample_copy_receipt.json').open('x') as out:json.dump(receipt,out,indent=2)
print(json.dumps({'same_logical_manifest_as_C':True,'readiness':ready,'verification':verified}))
