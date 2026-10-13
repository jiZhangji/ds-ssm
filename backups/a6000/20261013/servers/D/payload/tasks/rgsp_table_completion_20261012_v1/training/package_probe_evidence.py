"""Package existing small probe evidence, never images, labels or weights."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
paths = [root / x for x in ['probe_training.py', 'frozen_config_source.py', 'cuda_uuid_compat.py',
                           'frozen_source_hashes.json', 'run_remaining_probes.py']]
for result in sorted(root.glob('probe_*/result.json')):
    paths.extend([result, result.parent / 'initial_audit.json', result.parent / 'updates.jsonl'])
paths.append(root / 'probe_yolov8n_b4_v2/train_manifest.json')
inventory = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
out = root / 'probe_evidence_v1.zip'
with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, str(p.relative_to(root)))
    z.writestr('inventory_sha256.json', json.dumps(inventory, indent=2))
print(json.dumps({'zip': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'files': len(paths)}))
