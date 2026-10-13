"""Generate finite plans from verified per-host data; never start a model."""
import argparse
import json
from pathlib import Path
import sys
import worker

ROOT = Path(__file__).resolve().parent
OLD = worker.LEGACY_ROOT
PY = '/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python'
GPUS = {'a0':'GPU-12cf4841-46b9-94d1-2715-9932134cf964',
        'c0':'GPU-e5f8a5d2-306d-582e-c926-f795df943b1b',
        'd0':'GPU-f1b51ae3-c020-9901-c608-cca67f6ab8f1',
        'f0':'GPU-9d27eec3-fd30-8534-8341-b983bea3832b',
        'f1':'GPU-ce57e76d-ee2e-ff91-dbdd-ec0938874c40'}

def job(identifier, task, dataset, method, script, arguments, seconds, stage):
    command = [PY, '-B', '-u', str(ROOT / script)] + arguments
    command += ['--seed','{seed}','--output','{output}','--gpu-uuid','{gpu_uuid}']
    return {'id':identifier, 'task':task,'dataset':dataset,'method':method,
            'command_probe':command+['--mode','probe'],
            'command_train':command+['--mode','train'],
            'estimated_seconds':seconds,'stage':stage,'min_free_gb':18,'cwd':str(ROOT)}

def ready(dataset):
    path=ROOT/'taskdata'/dataset/'readiness.json'
    value=worker.read_json(path)
    if value.get('ready') is not True: raise ValueError('Dataset is not ready: '+dataset)
    return ROOT/'taskdata'/dataset/'manifest.json'

def cross(dataset,method,stage=0):
    manifest=ready(dataset)
    return job(dataset.lower()+'-'+method.lower(),'change_detection' if stage==0 else 'classification',
        dataset,method,'run_cross.py',['--campaign-root',str(OLD),'--source-root',str(OLD/'code'),
        '--manifest',str(manifest),'--dataset',dataset,'--method',method],900 if stage==0 else 2400,stage)

def detection(ds,method):
    folder='sar_shipdataset' if ds=='sar_ship' else ds
    manifest=OLD/'data'/folder/'data_content.json'
    data=OLD/'data'/folder/'data.yaml'
    if not manifest.is_file() or not data.is_file():raise ValueError('Missing detection data '+ds)
    return job(ds+'-'+method.lower(),'detection',ds,method,'detection/run_detection.py',
        ['--source-root',str(OLD/'code'),'--dataset-manifest',str(manifest),'--data',str(data),
         '--dataset-id',ds,'--method',method],200000 if ds=='sar_ship' else 3600,90 if ds=='sar_ship' else 20)

def author(method):
    manifest=worker.read_json(ready('Ottawa'))
    row=manifest['records'][0]
    return job('ottawa-author-'+method,'change_detection','Ottawa',method,'external/cd/run_cd_author.py',
        ['--dataset','Ottawa','--method',method,'--image1',row['image1'],'--image2',row['image2'],
         '--ground-truth',row['mask'],'--cuda-compat',str(OLD/'cuda_uuid_compat.py')],900,0)

def build(lane):
    if lane=='a0':jobs=[cross(ds,m) for ds in ['Ottawa','Farmland','Yellow_River'] for m in ['Original','Matched075','Full']]
    elif lane=='d0':jobs=[author(m) for m in ['ddnet','lantnet','camixer','wbanet','wasnet']]
    elif lane=='c0':jobs=[cross('SAMPLE',m,10) for m in ['Original','Matched075','Full','resnet50']]
    elif lane=='f0':jobs=[detection('ssdd',m) for m in ['YOLOv8n','Original']]+[detection('sar_ship','Original')]
    elif lane=='f1':jobs=[detection('ssdd',m) for m in ['Matched075','Full']]+[detection('sar_ship',m) for m in ['Matched075','Full']]
    else:raise ValueError(lane)
    plan={'version':1,'lane':lane,'gpu_uuid':GPUS[lane],'jobs':jobs,'min_disk_free_gb':25,
          'idle_seconds':60,'poll_seconds':5}
    if lane=='d0':plan.update(legacy_lock_required=False,legacy_lock=None)
    return worker.validate_plan(plan,lane)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lane',choices=GPUS,required=True)
    args=parser.parse_args();plan=build(args.lane);directory=ROOT/'plans';directory.mkdir(exist_ok=True)
    path=directory/(args.lane+'.json')
    worker.atomic_json(path,plan,overwrite=False)
    print(json.dumps({'status':'PLAN_VALID','path':str(path),'jobs':len(plan['jobs']),
                      'plan_sha256':worker.canonical_sha(plan)}))

if __name__=='__main__':main()
