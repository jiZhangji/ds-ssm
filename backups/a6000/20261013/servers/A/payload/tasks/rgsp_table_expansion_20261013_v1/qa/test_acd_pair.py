"""Server-only audit and rejection tests; never edits official images."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
from prepare_acd_pair import verify_manifest, conflicts


def main():
    root = ROOT / 'taskdata/acd_pair_v1'
    records = {}
    for variant in ('full3032', 'clean2992'):
        path = root / variant / 'manifest.json'
        verify_manifest(path)
        records[variant] = json.loads(path.read_text())
    full, clean = records['full3032'], records['clean2992']
    assert full['paired_holdout_sha256'] == clean['paired_holdout_sha256']
    for split in ('val', 'test'):
        assert [r for r in full['records'] if r['split'] == split] == [r for r in clean['records'] if r['split'] == split]
    clean_names = {r['source_relative_path'] for r in clean['records']}
    added = [r for r in full['records'] if r['source_relative_path'] not in clean_names]
    assert len(added) == 40 and all(r['split'] == 'train' for r in added)
    assert len(conflicts(full['records'])) == 20 and not conflicts(clean['records'])
    mutations = {
        'relabeled_image': lambda m: m['records'][0].update(label=99),
        'duplicate_image': lambda m: m['records'].append(copy.deepcopy(m['records'][0])),
        'changed_holdout_fingerprint': lambda m: m.update(paired_holdout_sha256='0' * 64),
        'conflict_moved_to_test': lambda m: next(r for r in m['records'] if r['source_relative_path'] == added[0]['source_relative_path']).update(split='test'),
    }
    rejected = []
    for name, mutate in mutations.items():
        with tempfile.TemporaryDirectory(prefix='acd-fixture-', dir=ROOT / 'qa') as temporary:
            folder = Path(temporary)
            for filename in ('source_audit.json', 'readiness.json'):
                shutil.copyfile(root / 'full3032' / filename, folder / filename)
            altered = copy.deepcopy(full)
            mutate(altered)
            (folder / 'manifest.json').write_text(json.dumps(altered))
            try:
                verify_manifest(folder / 'manifest.json')
            except (AssertionError, ValueError):
                rejected.append(name)
            else:
                raise AssertionError('Invalid manifest accepted: ' + name)
    result = {'status': 'PASS', 'verified_variants': list(records), 'shared_holdouts_exact': True,
        'full_only_train_images': 40, 'conflict_groups': 20, 'rejection_tests': rejected,
        'source_files_modified': False, 'model_or_holdout_inference': False}
    with (ROOT / 'qa/acd_pair_validation.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
