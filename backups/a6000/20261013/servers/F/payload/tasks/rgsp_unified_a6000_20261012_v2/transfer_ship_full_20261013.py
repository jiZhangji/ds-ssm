import json,os,signal,sys,time,hashlib
from pathlib import Path
ROOT=Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_unified_a6000_20261012_v2')
sys.path.insert(0,str(ROOT))
import worker
from launch_lane import identity
receipt=json.loads((ROOT/'launch/f1.json').read_text())
assert receipt['pid']==218379 and receipt['start_ticks']=='36805740'
assert worker.canonical_sha(json.loads((ROOT/'plans/f1.production.json').read_text()))=='09b035fbffb8d9055543334aac4c816de89391ac932e9a9f326081a9912c3b5d'
assert worker.file_sha(ROOT/'worker.py')=='3b42a9d336d355fdcb06316585d6d721a3fbbd4a753c8e16447d5a262b85c294'
assert identity(receipt['pid'])['start_ticks']==receipt['start_ticks']
state=json.loads((ROOT/'lanes/f1/state.json').read_text())
assert state['jobs']['sar_ship-full']['runs']['0']['status']=='PENDING'
run=state['jobs']['sar_ship-matched075']['runs']['0']
assert run['status']=='RUNNING' and run['pid']==247821
child=identity(run['pid'])
assert child
child['argv']=[v.decode('utf-8',errors='replace') for v in child['argv']]
caught=int(next(x.split()[1] for x in Path('/proc/218379/status').read_text().splitlines() if x.startswith('SigCgt:')),16)
assert caught & (1<<(signal.SIGTERM-1))
dest=ROOT/'redistribution_20261013_v1'
dest.mkdir(exist_ok=True)
intent={'schema':'rgsp-scheduling-transfer-v1','from_host':'F','from_lane':'f1','to_host':'C','to_lane':'c0shipfull','job':'sar_ship-full','reason':'Use idle C A6000; preserve current Matched075 child','action':'SIGTERM scheduling worker only; handler finishes active child then exits without starting pending jobs','worker':receipt,'active_child':child,'old_state_sha256':worker.file_sha(ROOT/'lanes/f1/state.json'),'new_probe_protocol_sha256':'caed014dc9803c4aa888caa8330ee4d7b5528aa0f95ed5a2990c0a1b40e2ed82','created_at':time.time()}
worker.atomic_json(dest/'ship_full_transfer.intent.json',intent,overwrite=False)
os.kill(receipt['pid'],signal.SIGTERM)
time.sleep(2)
assert identity(child['pid'])['start_ticks']==child['start_ticks']
post={'signal_delivered':True,'child_still_alive':True,'worker_still_alive_finishing_child':bool(identity(receipt['pid'])),'checked_at':time.time(),'pending_original_state_retained':True}
worker.atomic_json(dest/'ship_full_transfer.signal.json',post,overwrite=False)
print(json.dumps({'intent':intent,'post':post}))
