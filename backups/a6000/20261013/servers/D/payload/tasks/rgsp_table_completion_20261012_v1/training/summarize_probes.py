"""Read completed probe JSONs only; no model imports or execution."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
keys = ['variant', 'physical_batch', 'mean_update_seconds', 'peak_allocated_gib',
        'common_tensor_sha256', 'estimated_ssdd_300e_training_hours_excluding_validation']
rows = []
for p in sorted(root.glob('probe_*/result.json')):
    value = json.loads(p.read_text())
    rows.append({k: value[k] for k in keys})
print(json.dumps(rows, indent=2))
