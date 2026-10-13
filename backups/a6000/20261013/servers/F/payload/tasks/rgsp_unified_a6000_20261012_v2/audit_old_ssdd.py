"""Read-only producer audit of separate 300-epoch TEST-as-VAL SSDD runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tarfile
import worker

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'rgsp_table_completion_20261012_v1' / 'training'


def audit(snapshot, variants):
    worker.safe_name(snapshot, 'snapshot')
    output = ROOT / 'evidence' / snapshot
    output.mkdir(parents=True, exist_ok=False)
    verified = []
    for variant in variants:
        directory = OLD / ('ssdd_' + variant + '_seed0')
        result = worker.read_json(directory / 'result.json')
        if not (result['status'] == 'TRAINING_COMPLETE' and result['paper_eligible'] is True
                and result['epoch'] == 300 and result['optimizer_updates'] == 8700
                and result['serialized_best_fp32_verified'] is True
                and result['protocol']['variant'] == variant
                and result['protocol']['seed'] == 0):
            raise RuntimeError('Incomplete old run: ' + variant)
        best = directory / 'run' / 'weights' / 'best.pt'
        if worker.file_sha(best) != result['best_sha256']:
            raise RuntimeError('Old selected weight changed: ' + variant)
        if worker.file_sha(directory / 'data_manifest.json') != result['protocol']['data_manifest_sha256']:
            raise RuntimeError('Old data manifest changed')
        ledger = [json.loads(line) for line in (directory / 'optimizer_ledger.jsonl').read_text().splitlines()]
        if len(ledger) != 300:
            raise RuntimeError('Old ledger does not contain300 epochs')
        for epoch, row in enumerate(ledger, 1):
            if row['epoch'] != epoch or row['optimizer_updates'] != epoch * 29:
                raise RuntimeError('Old optimizer ledger cursor mismatch')
        with (directory / 'run' / 'results.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream, skipinitialspace=True))
        if len(csv_rows) != 300:
            raise RuntimeError('Old native results CSV is incomplete')
        if any(int(float(row['epoch'])) != epoch for epoch, row in enumerate(csv_rows, 1)):
            raise RuntimeError('Old CSV epoch mismatch')
        hashes = {}
        destination = output / variant
        destination.mkdir()
        for relative in ('result.json', 'protocol.json', 'runtime_audit.json',
                         'data_manifest.json', 'optimizer_ledger.jsonl', 'run/results.csv'):
            original = directory / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            hashes[relative] = worker.file_sha(target)
        verified.append({'variant': variant, 'result': result, 'hashes': hashes,
            'checkpoint_path': str(best), 'checkpoint_sha256': worker.file_sha(best),
            'producer_fp32_reload_attestation': result['serialized_best_fp32_verified'],
            'audit_scope': 'Live checkpoint/data hashes and300-row ledger/CSV; no new model evaluation.'})
    worker.atomic_json(output / 'verification.json', {'results': verified,
        'protocol_group': 'old300epoch_TEST_as_VAL', 'independent_test': False}, overwrite=False)
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'x:gz') as package:
        package.add(output, arcname=snapshot)
    print(json.dumps({'archive': str(archive), 'sha256': worker.file_sha(archive),
                      'variants': variants, 'verified': len(verified)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--variants', nargs='+', required=True,
        choices=('original', 'matched_scale075', 'full', 'cnn_yolov8n'))
    args = parser.parse_args()
    audit(args.snapshot, args.variants)
