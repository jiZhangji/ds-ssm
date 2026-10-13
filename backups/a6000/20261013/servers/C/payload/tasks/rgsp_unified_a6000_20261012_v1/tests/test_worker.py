"""Stdlib-only worker tests; run on an authorized server, never implicitly locally.

python -m unittest discover -s tests -p test_worker.py -v
No model, CUDA workload, SSH connection, network access or service is used.
GPU inspection, time and child processes are replaced by deterministic fixtures.
"""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


WORKER_PATH = Path(__file__).resolve().parents[1] / 'worker.py'
SPEC = importlib.util.spec_from_file_location('rgsp_unified_worker', WORKER_PATH)
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
GPU = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
DIGEST = 'a' * 64


def job(identifier='cd-original', seconds=100, stage=0):
    return {'id': identifier, 'task': 'change_detection', 'dataset': 'Ottawa',
            'method': 'Original', 'stage': stage, 'estimated_seconds': seconds,
            'min_free_gb': 8,
            'command_probe': ['/env/python', '/task/runner.py', '--mode', 'probe', '--output', '{output}', '--seed', '{seed}'],
            'command_train': ['/env/python', '/task/runner.py', '--mode', 'train', '--output', '{output}', '--seed', '{seed}']}


def plan(jobs=None):
    return w.validate_plan({'version': 1, 'lane': 'd0', 'gpu_uuid': GPU,
                            'legacy_lock_required': False, 'legacy_lock': None,
                            'jobs': jobs if jobs is not None else [job()]})


def mark_probed(state, identifier, seconds):
    current = state['jobs'][identifier]
    current['probe']['status'] = 'COMPLETE'
    current['estimated_seconds'] = seconds
    current['seed_limit'] = w.seed_limit(seconds)
    for seed in range(current['seed_limit'], 3):
        current['runs'][str(seed)]['status'] = 'SKIPPED'


def training_artifact(output, seed=0, metrics=None):
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'best.pt'
    checkpoint.write_bytes(b'fixture checkpoint, not a model')
    result = {'status': 'TRAINING_COMPLETE', 'seed': seed,
              'metrics': metrics if metrics is not None else {'F1': 0.1, 'accuracy': 0.2},
              'protocol_sha256': DIGEST,
              'checkpoint': {'path': 'best.pt', 'sha256': w.file_sha(checkpoint)}}
    w.atomic_json(output / 'result.json', result)
    w.atomic_json(output / 'completion.json', {
        'status': 'TRAINING_COMPLETE', 'result_sha256': w.file_sha(output / 'result.json')})
    return result


class PlanTests(unittest.TestCase):
    def test_default_legacy_lock_is_exact_registered_path(self):
        raw = {'version': 1, 'lane': 'a0', 'gpu_uuid': GPU, 'jobs': [job()]}
        self.assertEqual(w.validate_plan(raw)['legacy_lock'], str(w.LEGACY_ROOT / 'locks' / (GPU + '.lock')))

    def test_exemption_is_explicit_and_does_not_create_legacy_path(self):
        raw = plan()
        raw.pop('legacy_lock')
        with self.assertRaises(w.WorkerError):
            w.validate_plan(raw)

    def test_schema_rejects_duplicate_id_unsafe_path_and_shell_string(self):
        for mutation in ('duplicate', 'path', 'shell', 'placeholder', 'seed'):
            raw = plan()
            if mutation == 'duplicate':
                raw['jobs'].append(copy.deepcopy(raw['jobs'][0]))
            elif mutation == 'path':
                raw['jobs'][0]['id'] = '../old'
            elif mutation == 'shell':
                raw['jobs'][0]['command_train'] = 'python runner.py'
            elif mutation == 'placeholder':
                raw['jobs'][0]['command_train'].append('{output.__class__}')
            else:
                raw['jobs'][0]['command_train'][-1] = '0'
            with self.subTest(mutation=mutation), self.assertRaises(w.WorkerError):
                w.validate_plan(raw)

    def test_no_weakened_disk_or_idle_gate(self):
        for field, value in [('min_disk_free_gb', 14), ('idle_seconds', 29),
                             ('max_gpu_utilization', 20), ('poll_seconds', 61)]:
            raw = plan()
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(w.WorkerError):
                w.validate_plan(raw)

    def test_validate_only_never_queries_gpu_starts_process_or_creates_root(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lane.json'
            w.atomic_json(path, plan())
            absent = Path(temp) / 'not-created'
            with mock.patch.object(w, 'query_gpu', side_effect=AssertionError('GPU touched')), \
                    mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('child started')), \
                    mock.patch.object(w.Worker, 'run', side_effect=AssertionError('run called')), \
                    mock.patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(w.main(['--plan', str(path), '--root', str(absent), '--validate-plan']), 0)
            self.assertFalse(absent.exists())
            self.assertEqual(json.loads(out.getvalue())['status'], 'PLAN_VALID')
            self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ['lane.json'])


class SchedulingTests(unittest.TestCase):
    def test_exact_duration_boundaries(self):
        for seconds, cap in [(0.01, 3), (3600, 3), (3600.01, 2), (18000, 2), (18000.01, 1)]:
            self.assertEqual(w.seed_limit(seconds), cap)
        for bad in [0, -1, float('nan'), float('inf'), True]:
            with self.assertRaises(w.WorkerError):
                w.seed_limit(bad)

    def test_cd_first_long_cd_after_short_ship(self):
        spec = plan([job('ship', 500, 90), job('long-cd', 20000), job('class', 500, 10), job('fast-cd', 100)])
        state = w.new_state(spec, '/fixture')
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'fast-cd')
        state['jobs']['fast-cd']['status'] = 'COMPLETE'
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'class')
        state['jobs']['class']['status'] = 'COMPLETE'
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'ship')
        state['jobs']['ship']['status'] = 'COMPLETE'
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'long-cd')

    def test_all_stage_seed0_before_seed1_and_seed2(self):
        spec = plan([job('a', 10), job('b', 4000)])
        state = w.new_state(spec, '/fixture')
        mark_probed(state, 'a', 10)
        mark_probed(state, 'b', 4000)
        chosen = []
        while True:
            action = w.next_action(spec, state)
            if action is None:
                break
            config, kind, seed = action
            chosen.append((config['id'], seed))
            state['jobs'][config['id']]['runs'][str(seed)]['status'] = 'COMPLETE'
        self.assertEqual(chosen, [('a', 0), ('b', 0), ('a', 1), ('b', 1), ('a', 2)])

    def test_probe_reclassifies_long_task_without_early_training(self):
        spec = plan([job('unknown', 100), job('class', 1000, 10)])
        state = w.new_state(spec, '/fixture')
        mark_probed(state, 'unknown', 22000)
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'class')

    def test_blocker_does_not_hold_other_jobs(self):
        spec = plan([job('blocked'), job('available')])
        state = w.new_state(spec, '/fixture')
        state['jobs']['blocked']['status'] = 'BLOCKED'
        self.assertEqual(w.next_action(spec, state)[0]['id'], 'available')

    def test_accuracy_never_changes_seed_order(self):
        spec = plan([job('a'), job('b')])
        state = w.new_state(spec, '/fixture')
        baseline = w.next_action(spec, state)
        state['jobs']['a']['metrics'] = {'accuracy': -1000}
        state['jobs']['b']['metrics'] = {'accuracy': 1000}
        self.assertEqual(w.next_action(spec, state), baseline)


class ArtifactTests(unittest.TestCase):
    def test_probe_must_explicitly_avoid_test(self):
        for payload in [{'status': 'PROBE_PASS', 'estimated_full_seconds': 1},
                        {'status': 'PROBE_PASS', 'estimated_full_seconds': 1, 'test_evaluated': 0},
                        {'status': 'PROBE_PASS', 'estimated_full_seconds': 1, 'test_evaluated': True},
                        {'status': 'PROBE_PASS', 'estimated_full_seconds': float('nan'), 'test_evaluated': False}]:
            with self.assertRaises(w.WorkerError):
                w.validate_probe(payload)
        w.validate_probe({'status': 'PROBE_PASS', 'estimated_full_seconds': 1, 'test_evaluated': False})

    def test_complete_requires_matching_result_checkpoint_seed_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run0'
            training_artifact(output)
            verified = w.validate_training(output, 0, {}, DIGEST)
            self.assertEqual(verified['protocol_sha256'], DIGEST)
            for seed, digest in [(1, DIGEST), (0, 'b' * 64)]:
                with self.assertRaises(w.WorkerError):
                    w.validate_training(output, seed, {}, digest)
            (output / 'best.pt').write_bytes(b'changed')
            with self.assertRaises(w.WorkerError):
                w.validate_training(output, 0, {})

    def test_result_without_completion_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            w.atomic_json(output / 'result.json', {'status': 'TRAINING_COMPLETE'})
            with self.assertRaises((OSError, w.WorkerError)):
                w.validate_training(output, 0, {})

    def test_completion_digest_and_required_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            training_artifact(output)
            with self.assertRaises(w.WorkerError):
                w.validate_training(output, 0, {'required_fields': ['protocol.dataset_sha256']})
            w.atomic_json(output / 'completion.json', {'status': 'TRAINING_COMPLETE', 'result_sha256': '0' * 64})
            with self.assertRaises(w.WorkerError):
                w.validate_training(output, 0, {})

    def test_checkpoint_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run0'
            result = training_artifact(output)
            result['checkpoint']['path'] = '../external.pt'
            w.atomic_json(output / 'result.json', result)
            with self.assertRaises(w.WorkerError):
                w.validate_training(output, 0, {})

    def test_metrics_are_finite_and_not_empty_or_boolean(self):
        for value in [{}, {'F1': float('nan')}, {'F1': float('inf')}, {'F1': True}, {'F1': None}]:
            self.assertFalse(w.numeric_metrics(value))
        self.assertTrue(w.numeric_metrics({'score': [0, 1.0], 'class': {'a': 0.2}}))


class WorkerStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.spec = plan()
        self.worker = w.Worker(self.spec, Path(self.temp.name) / 'fresh-task')
        self.worker.lane_root.mkdir(parents=True)
        self.worker.initialize_state()
        self.config = self.spec['jobs'][0]

    def test_atomic_state_events_and_plan_mismatch_rejected(self):
        self.worker.persist('fixture')
        self.assertEqual(w.read_json(self.worker.state_path)['revision'], 2)
        events = [json.loads(line) for line in self.worker.events_path.read_text().splitlines()]
        self.assertEqual([e['revision'] for e in events], [1, 2])
        changed = copy.deepcopy(self.spec)
        changed['jobs'][0]['estimated_seconds'] += 1
        other = w.Worker(changed, self.worker.root)
        with self.assertRaises(w.WorkerError):
            other.initialize_state()

    def test_seed0_can_only_reduce_future_budget_and_preserves_success(self):
        current = self.worker.state['jobs'][self.config['id']]
        mark_probed(self.worker.state, self.config['id'], 100)
        current['runs']['0']['status'] = 'COMPLETE'
        self.worker.apply_cap(self.config, 2, 'longer_than_hour')
        self.assertEqual(current['seed_limit'], 2)
        self.assertEqual(current['runs']['2']['status'], 'SKIPPED')
        self.worker.apply_cap(self.config, 3, 'cannot_expand')
        self.assertEqual(current['seed_limit'], 2)
        self.assertEqual(current['runs']['0']['status'], 'COMPLETE')

    def test_finish_measured_seed0_over_five_hours_cancels_only_future_seeds(self):
        current = self.worker.state['jobs'][self.config['id']]
        mark_probed(self.worker.state, self.config['id'], 100)
        output = self.worker.output_dir(self.config, 'train', 0)
        training_artifact(output)
        record = current['runs']['0']
        record.update(status='RUNNING', returncode=0, elapsed_seconds=19000)
        self.worker.finish(self.config, 'train', 0, record)
        self.assertEqual(current['seed_limit'], 1)
        self.assertEqual(current['runs']['0']['status'], 'COMPLETE')
        self.assertEqual(current['status'], 'COMPLETE')
        self.assertTrue((output / 'best.pt').exists())

    def test_preexisting_output_is_preserved_and_not_retried(self):
        output = self.worker.output_dir(self.config, 'probe', 0)
        output.mkdir(parents=True)
        marker = output / 'existing.txt'
        marker.write_text('preserve me')
        with mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            self.worker.execute(self.config, 'probe', 0)
        self.assertEqual(marker.read_text(), 'preserve me')
        self.assertEqual(self.worker.state['jobs'][self.config['id']]['status'], 'BLOCKED')
        self.assertIsNone(w.next_action(self.spec, self.worker.state))

    def test_uncertain_launch_is_blocked_on_reentry_without_retry(self):
        current = self.worker.state['jobs'][self.config['id']]
        current['probe']['status'] = 'RUNNING'
        self.worker.persist('fixture_launch_intent')
        with mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('must not retry')):
            self.worker.reconcile()
        self.assertEqual(current['status'], 'BLOCKED')

    def test_child_receives_all_locks_uuid_and_no_shell(self):
        child = mock.Mock(pid=4242, returncode=1)
        child.poll.return_value = 1
        self.worker.locks = [mock.Mock(fd=71), mock.Mock(fd=72), mock.Mock(fd=73)]
        with mock.patch.object(w.subprocess, 'Popen', return_value=child) as launch:
            self.worker.execute(self.config, 'probe', 0)
        kwargs = launch.call_args.kwargs
        self.assertEqual(kwargs['pass_fds'], (71, 72, 73))
        self.assertFalse(kwargs['shell'])
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], GPU)
        self.assertEqual(self.worker.state['jobs'][self.config['id']]['status'], 'BLOCKED')

    def test_gpu_query_failure_waits_and_never_falls_back_to_cpu(self):
        self.worker.gpu_inspector = mock.Mock(side_effect=w.ResourceUnavailable('fixture unavailable'))
        self.worker.sleep = lambda _: self.worker.stop()
        with mock.patch.object(w.subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            self.assertFalse(self.worker.wait_ready(self.config))
        events = self.worker.events_path.read_text()
        self.assertIn('resource_query_unavailable', events)


class ResourceAndLockTests(unittest.TestCase):
    def test_idle_requires_full_continuous_window_and_resets_on_busy(self):
        gate = w.IdleWindow(30)
        self.assertFalse(gate.observe(0, True))
        self.assertFalse(gate.observe(29, True))
        self.assertFalse(gate.observe(30, False))
        self.assertFalse(gate.observe(31, True))
        self.assertFalse(gate.observe(60, True))
        self.assertTrue(gate.observe(61, True))

    def test_lock_close_never_explicitly_unlocks_shared_description(self):
        lock = w.HeldLock('/fixture.lock', False)
        lock.fd = 123
        with mock.patch.object(w.os, 'close') as close:
            if w.fcntl is not None:
                with mock.patch.object(w.fcntl, 'flock') as flock:
                    lock.close()
                    flock.assert_not_called()
            else:
                lock.close()
        close.assert_called_once_with(123)

    @unittest.skipIf(w.fcntl is None, 'flock available only on Linux servers')
    def test_existing_lock_content_unchanged_conflict_nonblocking(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'gpu.lock'
            path.write_bytes(b'old evidence stays')
            first, second = w.HeldLock(path, False), w.HeldLock(path, False)
            try:
                self.assertTrue(first.try_acquire())
                self.assertFalse(second.try_acquire())
                self.assertEqual(path.read_bytes(), b'old evidence stays')
            finally:
                first.close()
                second.close()
            self.assertTrue(second.try_acquire())
            second.close()
            self.assertEqual(path.read_bytes(), b'old evidence stays')

    @unittest.skipIf(w.fcntl is None, 'flock available only on Linux servers')
    def test_missing_legacy_lock_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'absent.lock'
            with self.assertRaises(FileNotFoundError):
                w.HeldLock(path, False).try_acquire()
            self.assertFalse(path.exists())

    def test_gpu_snapshot_rejects_wrong_uuid_and_counts_compute(self):
        gpu_line = GPU + ', NVIDIA RTX A6000, 47000, 49140, 0\n'
        with mock.patch.object(w.subprocess, 'run', side_effect=[
                mock.Mock(stdout=gpu_line), mock.Mock(stdout=GPU + ', 42\n')]):
            snapshot = w.query_gpu(GPU)
        self.assertEqual(snapshot['compute_pids'], [42])
        with mock.patch.object(w.subprocess, 'run', return_value=mock.Mock(stdout=gpu_line)):
            with self.assertRaises(w.ResourceUnavailable):
                w.query_gpu('GPU-different')


if __name__ == '__main__':
    unittest.main()
