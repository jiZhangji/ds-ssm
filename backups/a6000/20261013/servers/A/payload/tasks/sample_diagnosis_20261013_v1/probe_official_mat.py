"""Fetch two pinned public MAT examples and inspect data only, on a server."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
import numpy as np
from scipy.io import loadmat

if not sys.platform.startswith('linux'):
    raise RuntimeError('Run on a registered Linux project server')
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
commit = '1bae4156afcba7ad9c9f2eb743ed20675ed14fb1'
base = 'https://raw.githubusercontent.com/benjaminlewis-afrl/SAMPLE_dataset_public/' + commit
names = ['synth/2s1/2s1_synth_A_elevDeg_015_azCenter_010_22_serial_b01.mat',
         'real/2s1/2s1_real_A_elevDeg_017_azCenter_010_22_serial_b01.mat']
report = {'commit': commit, 'scope':'Two-file source format preflight, no model execution; not a full-dataset verification', 'files':[]}
for name in names:
    url = base + '/mat_files/' + name
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(4*1024*1024+1)
    if len(data)>4*1024*1024:
        raise RuntimeError('Example exceeds 4 MiB bound')
    path = root / Path(name).name
    with path.open('xb') as f:
        f.write(data)
    mat = loadmat(path)
    row = {'url':url,'filename':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'fields':{}}
    for key,x in mat.items():
        if key.startswith('__'):
            continue
        field={'shape':list(x.shape),'dtype':str(x.dtype)}
        if key=='complex_img':
            amp=np.abs(x)
            field.update({'complex':bool(np.iscomplexobj(x)),'finite':bool(np.isfinite(x).all()),
                          'amplitude_min':float(amp.min()),'amplitude_max':float(amp.max()),'amplitude_mean':float(amp.mean())})
        row['fields'][key]=field
    if 'complex_img' not in row['fields']:
        raise RuntimeError('No confirmed amplitude source')
    report['files'].append(row)
(root/'mat_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
