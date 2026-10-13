"""Idempotent detached launcher; refuses uncertain or failed earlier attempts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import worker

ROOT=Path(__file__).resolve().parent

def identity(pid):
    try:
        value=Path('/proc')/str(pid)
        stat=(value/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':stat[19],'argv':(value/'cmdline').read_bytes().split(b'\0')[:-1]}
    except FileNotFoundError:return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lane',required=True)
    parser.add_argument('--probe-only',action='store_true');args=parser.parse_args()
    plan_path=ROOT/'plans'/(args.lane+('.json' if args.probe_only else '.production.json'))
    plan=worker.validate_plan(worker.read_json(plan_path),args.lane)
    launch=ROOT/'launch';launch.mkdir(exist_ok=True)
    with (launch/(args.lane+'.lock')).open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suffix=args.lane+('.probe01' if args.probe_only else '')
        receipt=launch/(suffix+'.json')
        if receipt.exists():
            previous=worker.read_json(receipt);live=identity(previous['pid'])
            if live and live['start_ticks']==previous['start_ticks']:
                print(json.dumps({'status':'ALREADY_RUNNING','lane':args.lane,'pid':previous['pid']}));return
            print(json.dumps({'status':'PRIOR_ATTEMPT_PRESERVED','lane':args.lane,
                  'state':str(ROOT/'lanes'/args.lane/'state.json')}));return
        output_root=ROOT/'qa_smoke_01' if args.probe_only else ROOT
        command=[sys.executable,'-B','-u',str(ROOT/'worker.py'),'--root',str(output_root),
                 '--plan',str(plan_path),'--lane',args.lane]
        if args.probe_only:command.append('--probe-only')
        with (launch/(suffix+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
        live=identity(child.pid)
        if live is None:raise RuntimeError('Worker exited before its identity was recorded')
        record={'lane':args.lane,'pid':child.pid,'start_ticks':live['start_ticks'],
                'command':command,'plan_sha256':worker.canonical_sha(plan)}
        worker.atomic_json(receipt,record,overwrite=False)
        print(json.dumps({'status':'STARTED','lane':args.lane,**record}))

if __name__=='__main__':main()
