"""Server-side metamorphic checks of the actual detection protocol expression."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DetectionProtocolTests(unittest.TestCase):
    def protocol(self, seed, mode, manifest='a' * 64, config=None):
        tree = ast.parse((ROOT / 'detection/run_detection.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'protocol' for t in n.targets))
        scope = dict(copy=copy, digest=digest, PROTOCOL={'lr0': 0.0002},
            args=SimpleNamespace(seed=seed, mode=mode, dataset_id='ssdd', method='Original', workers=4),
            budget={'train': 730}, sources=[{'sha256': 'b' * 64}],
            runner_sources={'run_detection.py': 'c' * 64}, config=config or {'width': 1},
            data_audit={'input_manifest': {'sha256': manifest}, 'actual_records_sha256': mode},
            loader_seed=seed + 100)
        return eval(compile(ast.Expression(assignment.value), '<actual-protocol>', 'eval'), scope)

    def test_probe_and_all_seeds_share_protocol(self):
        baseline = self.protocol(0, 'probe')
        for seed in (0, 1, 2):
            self.assertEqual(self.protocol(seed, 'train'), baseline)

    def test_changed_data_or_architecture_cannot_share_protocol(self):
        baseline = self.protocol(0, 'train')
        self.assertNotEqual(self.protocol(0, 'train', manifest='d' * 64), baseline)
        self.assertNotEqual(self.protocol(0, 'train', config={'width': 2}), baseline)

    def test_controller_probe_digest_can_bind_training(self):
        baseline = digest(self.protocol(0, 'probe'))
        self.assertEqual(digest(self.protocol(2, 'train')), baseline)


if __name__ == '__main__':
    unittest.main()
