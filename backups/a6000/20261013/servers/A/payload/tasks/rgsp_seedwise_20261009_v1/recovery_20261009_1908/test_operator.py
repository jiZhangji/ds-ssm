#!/usr/bin/env python3
"""Server-only standard-library tests; synthetic files are retained as evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import operator_cross_recover as op


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        base = op.OUT / 'unit_fixtures'
        base.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='case_', dir=base))
        self.plan = {'source_root': '/frozen/source'}
        self.trial = {'id': 'sar_acd_original_seed0', 'seed': 0, 'run_dir': str(self.root),
                      'command': ['/python', '/cross_trial.py', '--variant', 'original', '--dataset', 'SAR-ACD',
                                  '--epochs', '200', '--patience', '40', '--batch', '64', '--imgsz', '224']}
        self.protocol = {'dataset': 'SAR-ACD', 'epochs': 200, 'patience': 40, 'batch': 64, 'imgsz': 224}
        self.cfg = {'seed': 0, 'variant': 'original', 'trial_id': self.trial['id'], 'validate_only': False,
                    'plan_sha256': op.m.digest(self.plan), 'source_root': self.plan['source_root'],
                    'protocol': self.protocol, 'protocol_sha256': op.m.digest(self.protocol)}

    def write(self, name, payload):
        (self.root / name).write_text(json.dumps(payload))

    def complete_fixture(self):
        self.write('config.json', self.cfg)
        self.write('initialization.json', {'fixture_only': True})
        (self.root / 'metrics.jsonl').write_text(json.dumps({'epoch': 1, 'loss': 1.0, 'val': {'macro_f1': 0.1}}) + '\n')
        (self.root / 'best.pt').write_bytes(b'unit-test-not-a-model')
        (self.root / 'last.pt').write_bytes(b'unit-test-not-a-model-last')
        best = op.m.sha(self.root / 'best.pt')
        tested = {'identity_sha256': op.m.digest(self.cfg), 'best_sha256': best,
                  'metric': {'macro_f1': 0.1}, 'threshold': None, 'checkpoint_epoch': 1}
        self.write('test_evaluation.json', tested)
        result = {'best_sha256': best, 'seed': 0, 'variant': 'original', 'dataset': 'SAR-ACD',
                  'checkpoint': str(self.root / 'best.pt'), 'metric': tested['metric'], 'threshold': None,
                  'best_epoch': 1, 'protocol': self.protocol, 'protocol_sha256': op.m.digest(self.protocol)}
        self.write('result.json', result)
        self.refresh_completion()

    def refresh_completion(self):
        names = ('config.json', 'initialization.json', 'metrics.jsonl', 'test_evaluation.json', 'result.json', 'best.pt')
        completion = {'status': 'complete', 'identity_sha256': op.m.digest(self.cfg), 'seed': 0,
                      'variant': 'original', 'protocol_sha256': op.m.digest(self.protocol),
                      'artifacts': [{'path': str(self.root / name), 'sha256': op.m.sha(self.root / name)} for name in names]}
        self.write('completion.json', completion)

    def test_exact_full_argument_forwarding(self):
        planned = self.trial['command'] + ['--workers', '4', '--device', 'cuda:0', '--eval-batch', '7', '--resume']
        original = copy.deepcopy(planned)
        wrapped = op.wrapper_command(planned, self.root / 'compat.json')
        self.assertEqual(wrapped[wrapped.index('--') + 1:], planned[2:])
        self.assertEqual(planned, original)
        self.assertIn(str(op.TASK / 'cuda_uuid_compat.py'), wrapped)

    def test_metadata_only_allowed(self):
        self.write('config.json', self.cfg)
        self.assertEqual(len(op.no_training_output(self.trial)), 1)

    def test_partial_training_rejected(self):
        (self.root / 'last.pt').write_bytes(b'preserve')
        with self.assertRaises(RuntimeError):
            op.no_training_output(self.trial)
        self.assertTrue((self.root / 'last.pt').exists())

    def test_unknown_nested_output_rejected(self):
        (self.root / 'weights').mkdir()
        with self.assertRaises(RuntimeError):
            op.no_training_output(self.trial)

    def test_symlink_rejected(self):
        (self.root / 'config.json').symlink_to('/dev/null')
        with self.assertRaises(RuntimeError):
            op.no_training_output(self.trial)

    def test_valid_deep_ledger(self):
        self.complete_fixture()
        audit = op.verify_cross_completion(self.plan, self.trial)
        self.assertEqual(audit['status'], 'deep_verified')
        self.assertEqual(audit['actual_epochs'], 1)
        self.assertEqual(len(audit['internal_artifacts']), 6)

    def test_best_corruption_rejected(self):
        self.complete_fixture()
        (self.root / 'best.pt').write_bytes(b'corrupted')
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_missing_recursive_artifact_rejected(self):
        self.complete_fixture()
        c = op.m.read_json(self.root / 'completion.json')
        c['artifacts'].pop(1)
        self.write('completion.json', c)
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_nonfinite_loss_rejected_even_with_matching_hash(self):
        self.complete_fixture()
        (self.root / 'metrics.jsonl').write_text('{"epoch":1,"loss":NaN}\n')
        self.refresh_completion()
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_nonfinite_metric_rejected_even_with_matching_hash(self):
        self.complete_fixture()
        for name in ('result.json', 'test_evaluation.json'):
            row = op.m.read_json(self.root / name)
            row['metric'] = {'macro_f1': float('inf')}
            self.write(name, row)
        self.refresh_completion()
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_wrong_seed_rejected(self):
        self.complete_fixture()
        self.cfg['seed'] = 1
        self.write('config.json', self.cfg)
        self.refresh_completion()
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_duplicate_epoch_rejected(self):
        self.complete_fixture()
        line = (self.root / 'metrics.jsonl').read_text()
        (self.root / 'metrics.jsonl').write_text(line + line)
        self.refresh_completion()
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)

    def test_frozen_batch_change_rejected(self):
        self.complete_fixture()
        self.protocol['batch'] = 32
        self.cfg['protocol_sha256'] = op.m.digest(self.protocol)
        self.write('config.json', self.cfg)
        self.refresh_completion()
        with self.assertRaises(RuntimeError):
            op.verify_cross_completion(self.plan, self.trial)


if __name__ == '__main__':
    unittest.main(verbosity=2)
