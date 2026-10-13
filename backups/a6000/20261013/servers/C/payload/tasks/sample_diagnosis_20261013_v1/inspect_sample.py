"""Read-only SAMPLE data and completed-result audit; run on a project server."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--jobs', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
mp = Path(a.manifest)
m = json.loads(mp.read_text())
summary = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'manifest': str(mp), 'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
           'classes': m['classes'], 'counts': m['counts'], 'splits': {}, 'label_errors': [], 'runs': []}
images = collections.defaultdict(list)
hist = collections.defaultdict(lambda: np.zeros(256, dtype=np.int64))
perimage = collections.defaultdict(list)
counts = collections.defaultdict(collections.Counter)
for r in m['records']:
    path = Path(r['path'])
    if m['classes'][r['label']] != r['class_name'] or path.parent.name != r['class_name'] or path.name.split('_')[0] != r['class_name']:
        summary['label_errors'].append(str(path))
    with Image.open(path) as im:
        x = np.array(im)
        counts[r['split']][str((im.mode, x.dtype.name, x.shape))] += 1
        if x.ndim != 2 or x.dtype != np.uint8:
            raise ValueError('Unexpected decibel source')
        hist[r['split']] += np.bincount(x.ravel(), minlength=256)
        edge = np.concatenate((x[:16].ravel(), x[-16:].ravel(), x[16:-16, :16].ravel(), x[16:-16, -16:].ravel()))
        perimage[r['split']].append([float(x.mean()), float(x.std()), float(edge.mean()), float(x[40:88,40:88].mean())])
        key = (r['split'], r['class_name'])
        if not images[key]:
            images[key].append(im.convert('RGB').copy())
for split, h in hist.items():
    stats = np.asarray(perimage[split])
    mean = float((h * np.arange(256)).sum() / h.sum())
    summary['splits'][split] = {'formats': dict(counts[split]), 'pixels': int(h.sum()), 'pixel_mean_0_255': mean,
        'pixel_std_0_255': float(np.sqrt((h*(np.arange(256)-mean)**2).sum()/h.sum())),
        'image_mean_range': [float(stats[:,0].min()),float(stats[:,0].max())],
        'image_mean_std_edge_center_averages': stats.mean(axis=0).tolist(),
        'zero_fraction': float(h[0]/h.sum()), 'saturated_fraction': float(h[-1]/h.sum()),
        'histogram': h.tolist(), 'class_counts': dict(collections.Counter(r['class_name'] for r in m['records'] if r['split']==split))}
sheet = Image.new('RGB', (10*132, 3*158), 'white')
draw = ImageDraw.Draw(sheet)
for i, cls in enumerate(m['classes']):
    for j, split in enumerate(('train','val','test')):
        sheet.paste(images[(split,cls)][0], (i*132,j*158+26))
        draw.text((i*132,j*158), f'{split} {cls}', fill='black')
sheet.save(out/'examples.png')
for rp in sorted(Path(a.jobs).glob('sample-*/run*/result.json')):
    rd = rp.parent
    test = json.loads((rd/'test.json').read_text())
    result = json.loads(rp.read_text())
    epochs = [json.loads(line) for line in (rd/'epochs.jsonl').read_text().splitlines() if line.strip()]
    cm = np.asarray(test['metrics']['confusion_matrix'])
    support, predicted = cm.sum(1), cm.sum(0)
    diag = np.diag(cm)
    den = support+predicted
    f1 = np.divide(2*diag, den, out=np.zeros(len(diag), dtype=float), where=den!=0)
    best = max(epochs, key=lambda e:e['validation']['macro_f1'])
    summary['runs'].append({'directory':str(rd), 'result_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),
        'test':test, 'accuracy_recomputed':float(diag.sum()/cm.sum()), 'macro_f1_recomputed':float(f1.mean()),
        'support_matches_manifest': support.tolist()==[summary['splits']['test']['class_counts'][c] for c in m['classes']],
        'prediction_counts':dict(zip(m['classes'],predicted.tolist())),
        'best_epoch':best['epoch'], 'best_updates':best['optimizer_updates'], 'best_validation':best['validation'],
        'last_epoch':epochs[-1]['epoch'], 'total_updates':epochs[-1]['optimizer_updates'],
        'checkpoint_sha256':test['checkpoint_sha256']})
(out/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('splits','runs')},ensure_ascii=False))
for s,v in summary['splits'].items():
    print(s, json.dumps({k:x for k,x in v.items() if k!='histogram'}))
for r in summary['runs']:
    print(r['directory'], 'test_acc',r['accuracy_recomputed'],'test_f1',r['macro_f1_recomputed'], 'val_acc',r['best_validation']['accuracy'],'updates',r['best_updates'], 'pred_counts',r['prediction_counts'])
