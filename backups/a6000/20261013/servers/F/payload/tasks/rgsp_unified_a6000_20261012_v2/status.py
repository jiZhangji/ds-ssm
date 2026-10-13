"""Read-only compact snapshot of this campaign; no model imports."""
import json
from pathlib import Path
import socket
ROOT=Path(__file__).resolve().parent
rows=[]
for group in ('qa_smoke_01','.'):
    for path in (ROOT/group/'lanes').glob('*/state.json'):
        value=json.loads(path.read_text())
        rows.append({'phase':'probe' if group!='.' else 'production','lane':value['lane'],
            'state':value['status'],'updated_at':value.get('updated_at'),
            'jobs':[{'id':key,'state':job['status'],'probe':job['probe']['status'],
                     'seed_limit':job['seed_limit'],'estimate_seconds':job['estimated_seconds'],
                     'seeds':{s:r['status'] for s,r in job['runs'].items()},'reason':job.get('reason')}
                    for key,job in value['jobs'].items()]})
gates=[]
for path in (ROOT/'dispatch').glob('*.gate.state.json'):
    gates.append({'lane':path.name.removesuffix('.gate.state.json'),**json.loads(path.read_text())})
print(json.dumps({'hostname':socket.gethostname(),'campaign':ROOT.name,'lanes':rows,'gates':gates}))
