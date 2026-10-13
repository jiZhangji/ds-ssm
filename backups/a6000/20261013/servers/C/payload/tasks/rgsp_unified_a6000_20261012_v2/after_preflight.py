"""Finite detached controller: await probes, retain failures, enqueue admitted jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import worker
from launch_lane import identity

ROOT=Path(__file__).resolve().parent

def controller(lane):
    directory=ROOT/'dispatch';directory.mkdir(exist_ok=True)
    with (directory/(lane+'.lock')).open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=worker.validate_plan(worker.read_json(ROOT/'plans'/(lane+'.json')),lane)
        deadline=time.monotonic()+14*86400
        path=ROOT/'qa_smoke_01/lanes'/lane/'state.json'
        while True:
            if time.monotonic()>deadline:raise RuntimeError('Finite preflight wait expired')
            state=worker.read_json(path)
            if state['plan_sha256']!=worker.canonical_sha(plan):raise RuntimeError('Preflight plan mismatch')
            if state['status'].startswith('PROBES_COMPLETE'):break
            if state['status'] in {'STOPPED','BLOCKED_WORKER','BLOCKED_INVALID_STATE'}:
                raise RuntimeError('Preflight stopped: '+state['status'])
            worker.atomic_json(directory/(lane+'.state.json'),{'state':'WAITING_PREFLIGHT','lane':lane,'time':time.time()})
            time.sleep(10)
        admitted=[];blocked=[]
        for job in plan['jobs']:
            result=state['jobs'][job['id']]
            if result['probe']['status']=='COMPLETE':
                payload=result['probe']['payload']
                worker.validate_probe(payload)
                job['estimated_seconds']=payload['estimated_full_seconds']
                job['protocol_sha256']=payload['protocol_sha256']
                admitted.append(job['id'])
            else:
                job['blocked_reason']='Preflight failure preserved: '+str(result.get('reason'))
                blocked.append(job['id'])
        production=ROOT/'plans'/(lane+'.production.json')
        if not production.exists():worker.atomic_json(production,worker.validate_plan(plan),overwrite=False)
        elif worker.canonical_sha(worker.read_json(production))!=worker.canonical_sha(plan):
            raise RuntimeError('Prior production plan differs; no overwrite')
        receipt={'lane':lane,'state':'ADMITTED_FOR_TRAINING','admitted':admitted,'blocked':blocked,
                 'preflight_state_sha256':worker.file_sha(path),'production_sha256':worker.file_sha(production)}
        worker.atomic_json(directory/(lane+'.admission.json'),receipt,overwrite=False)
        subprocess.run([sys.executable,'-B',str(ROOT/'launch_lane.py'),'--lane',lane],check=True,cwd=ROOT)
        worker.atomic_json(directory/(lane+'.state.json'),{**receipt,'state':'QUEUED','time':time.time()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lane',required=True)
    parser.add_argument('--foreground',action='store_true');args=parser.parse_args()
    if args.foreground:return controller(args.lane)
    worker.safe_name(args.lane,'lane')
    directory=ROOT/'dispatch';directory.mkdir(exist_ok=True)
    with (directory/(args.lane+'.launch.lock')).open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=directory/(args.lane+'.launch.json')
        if receipt.exists():
            prior=worker.read_json(receipt);live=identity(prior['pid'])
            print(json.dumps({'status':'ALREADY_DISPATCHED','lane':args.lane,
                'running':bool(live and live['start_ticks']==prior['start_ticks'])}));return
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--lane',args.lane,'--foreground']
        with (directory/(args.lane+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        live=identity(child.pid)
        if not live:raise RuntimeError('Dispatcher exited before receipt')
        value={'lane':args.lane,'pid':child.pid,'start_ticks':live['start_ticks'],'command':command}
        worker.atomic_json(receipt,value,overwrite=False)
        print(json.dumps({'status':'DISPATCHED',**value}))

if __name__=='__main__':main()
