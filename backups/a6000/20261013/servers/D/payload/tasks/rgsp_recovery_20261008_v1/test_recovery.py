"""CPU-only recovery gate and derived-record tests; run on project server."""
import json
import tempfile
import unittest
from pathlib import Path
import recovery_queue as r


class RecoveryTests(unittest.TestCase):
    def test_dependency_requires_entire_queue(self):
        cfg = {"predecessor_signature": "frozen", "gpu_uuid": "gpu"}
        state = {"queue_signature_sha256": "frozen", "gpu_uuid": "gpu", "status": "training",
                 "queue": [{"seed": 1, "arm": "a"}, {"seed": 2, "arm": "a"}],
                 "completed": [{"seed": 1, "arm": "a", "status": "complete"}]}
        self.assertFalse(r.predecessor_ok(state, cfg))
        state["status"] = "complete_shard"
        self.assertFalse(r.predecessor_ok(state, cfg))
        state["completed"].append({"seed": 2, "arm": "a", "status": "complete"})
        self.assertTrue(r.predecessor_ok(state, cfg))
        state["status"] = "failed"
        self.assertFalse(r.predecessor_ok(state, cfg))
        state["queue_signature_sha256"] = "changed"
        with self.assertRaises(RuntimeError):
            r.predecessor_ok(state, cfg)

    def test_gpu_gate(self):
        self.assertTrue(r.gpu_idle({"pids": [], "memory_mib": 500, "utilization": 0}))
        for gpu in ({"pids": ["7"], "memory_mib": 500, "utilization": 0},
                    {"pids": [], "memory_mib": 500, "utilization": 90},
                    {"pids": [], "memory_mib": 3000, "utilization": 0}):
            self.assertFalse(r.gpu_idle(gpu))

    def test_only_derived_trailing_nuls_and_post_checkpoint_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "controls_by_epoch.jsonl").write_bytes(b'{"epoch": 1}\n{"epoch": 2}\n\0\0')
            (run / "results.csv").write_bytes(b'epoch,loss\n1,0.5\n2,0.4\n\0')
            report = r.clean_derived_records(run, 1)
            self.assertEqual(json.loads((run / "controls_by_epoch.jsonl").read_text()), {"epoch": 1})
            self.assertEqual((run / "results.csv").read_text(), 'epoch,loss\n1,0.5\n')
            self.assertEqual(report[0]["trailing_nul_bytes"], 2)
            (run / "controls_by_epoch.jsonl").write_bytes(b'{"epoch":\0 1}\n')
            with self.assertRaises(RuntimeError):
                r.clean_derived_records(run, 1)


if __name__ == "__main__":
    unittest.main()
