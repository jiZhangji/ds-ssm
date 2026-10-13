"""Bind a fixed 24-run plan after amplitude data is verified; no GPU access."""
import argparse
import json
import sys
from pathlib import Path
import worker

PYTHON='/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'
METHODS=('Full','Original','Matched075','ResNet50')

def publish_once(path, value):
    if path.exists():
        if worker.read_json(path)!=value:
            raise RuntimeError('Existing frozen file differs: '+str(path))
    else:
        worker.atomic_json(path,value,overwrite=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--lane',default='a0amplitude')
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    a=parser.parse_args()
    if sys.platform!='linux':
        raise RuntimeError('Bind server paths only on the registered Linux server')
    root=a.root.resolve()
    if root!=Path(__file__).resolve().parent or a.lane not in ('a0amplitude','b0amplitude'):
        raise RuntimeError('Wrong deployment root/lane')
    data=root/'data'
    ready=worker.read_json(data/'READY.json')
    if ready.get('status')!='READY' or (data/'FAILED.json').exists():
        raise RuntimeError('Data not ready or failure record unresolved')
    classes=None
    for split,count in (('train',806),('test',539)):
        mp=data/('manifest_'+split+'.json')
        if worker.file_sha(mp)!=ready['manifest_'+split+'_sha256']:
            raise RuntimeError('Ready/manifest hash mismatch')
        m=worker.read_json(mp)
        if len(m['records'])!=count or len(m['classes'])!=10:
            raise RuntimeError('Unexpected source/test count or class count')
        if classes is None:
            classes=m['classes']
        if classes!=m['classes']:
            raise RuntimeError('Class-order mismatch')
        for r in m['records']:
            if m['classes'][r['label']]!=r['class_name']:
                raise RuntimeError('Class label mismatch')
            if split=='train' and (r['domain']!='synth' or r['elevation_deg'] not in (14,15,16)):
                raise RuntimeError('Target in train')
            if split=='test' and (r['domain']!='real' or r['elevation_deg']!=17):
                raise RuntimeError('Unexpected target membership')
    frozen=root/'frozen'
    copied=worker.read_json(root/'source_copy.json')
    for r in copied['files']:
        if worker.file_sha(r['path'])!=r['sha256']:
            raise RuntimeError('Frozen source changed')
    files=[p for p in frozen.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml')]
    files += [root/'run_amplitude.py']
    source_inventory={'schema':'sample-amplitude-source-inventory-v1',
        'files':[{'path':str(p),'sha256':worker.file_sha(p)} for p in sorted(files)]}
    inventory_path=root/'source_inventory.json'
    publish_once(inventory_path,source_inventory)
    inv_sha=worker.file_sha(inventory_path)
    jobs=[]
    if len(a.methods)!=len(set(a.methods)):
        raise ValueError('Duplicate methods are prohibited')
    for method in a.methods:
        for group in ('clean','gaussian'):
            cmd=[PYTHON,'-B','-u',str(root/'run_amplitude.py'),
                 '--campaign-root',str(frozen),'--source-root',str(frozen/'code'),
                 '--source-inventory',str(inventory_path),'--source-inventory-sha256',inv_sha,
                 '--train-manifest',str(data/'manifest_train.json'),
                 '--train-manifest-sha256',ready['manifest_train_sha256'],
                 '--method',method,'--group',group,'--seed','{seed}',
                 '--output','{output}','--gpu-uuid','{gpu_uuid}']
            jobs.append({'id':'amplitude-'+method.lower()+'-'+group,'task':'classification',
                'dataset':'SAMPLE','method':method,
                'stage':0 if method!='ResNet50' else 20,
                'estimated_seconds':3600,'min_free_gb':12,'cwd':str(root),
                'env':{'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'},
                'command_probe':cmd+['--mode','probe','--updates','32'],
                'command_train':cmd+['--mode','train','--updates','10000',
                    '--test-manifest',str(data/'manifest_test.json'),
                    '--test-manifest-sha256',ready['manifest_test_sha256']]})
    plan=worker.validate_plan({'version':1,'lane':a.lane,'gpu_uuid':a.gpu_uuid,
        'jobs':jobs,'legacy_lock_required':True,'idle_seconds':30,'poll_seconds':5,
        'min_disk_free_gb':30,'max_gpu_utilization':5},a.lane)
    (root/'plans').mkdir(exist_ok=True)
    publish_once(root/'plans'/(a.lane+'.production.json'),plan)
    receipt={'schema':'sample-amplitude-queue-definition-v1','status':'PLAN_READY',
        'jobs':len(jobs),'fixed_seeds':[0,1,2],'planned_runs':len(jobs)*3,
        'methods':a.methods,
        'plan_sha256':worker.canonical_sha(plan),'source_inventory_sha256':inv_sha,
        'train_manifest_sha256':ready['manifest_train_sha256'],
        'test_manifest_sha256':ready['manifest_test_sha256'],
        'physical_batch':16,'gradient_accumulation':4,'updates_per_run':10000,
        'scope':'U64 main campaign only; no R36 or legacy EMA auxiliary runs'}
    publish_once(root/'PLAN_READY.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
