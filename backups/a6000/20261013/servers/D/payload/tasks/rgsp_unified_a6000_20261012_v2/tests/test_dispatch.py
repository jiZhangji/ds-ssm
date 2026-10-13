"""Check predecessor gating and failure preservation without model execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_after_lane as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('dispatch', 'launch', 'lanes/previous'):
            (self.root / name).mkdir(parents=True)
        (self.root / 'launch/previous.json').write_text(json.dumps({'pid': 123, 'start_ticks': '77'}))
        self.root_patch = mock.patch.object(gate, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def state(self, status):
        (self.root / 'lanes/previous/state.json').write_text(json.dumps({'status': status}))

    def test_waits_for_terminal_process_to_release_locks(self):
        self.state('COMPLETE_WITH_BLOCKERS')
        with mock.patch.object(gate, 'identity', side_effect=[{'start_ticks': '77'}, None]), \
             mock.patch.object(gate.time, 'sleep') as sleep, \
             mock.patch.object(gate.subprocess, 'run') as run:
            gate.controller('next', 'previous')
        sleep.assert_called_once_with(10)
        self.assertEqual(run.call_count, 2)
        self.assertIn('--probe-only', run.call_args_list[0].args[0])
        self.assertTrue((self.root / 'lanes/previous/state.json').exists())

    def test_stopped_predecessor_requires_review(self):
        self.state('STOPPED')
        with mock.patch.object(gate, 'identity', return_value=None), \
             mock.patch.object(gate.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'requires review'):
                gate.controller('next', 'previous')
        run.assert_not_called()

    def test_waits_until_predecessor_creates_production_state(self):
        with mock.patch.object(gate, 'identity', return_value=None), \
             mock.patch.object(gate.time, 'sleep', side_effect=lambda _: self.state('COMPLETE')), \
             mock.patch.object(gate.subprocess, 'run') as run:
            gate.controller('next', 'previous')
        self.assertEqual(run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
