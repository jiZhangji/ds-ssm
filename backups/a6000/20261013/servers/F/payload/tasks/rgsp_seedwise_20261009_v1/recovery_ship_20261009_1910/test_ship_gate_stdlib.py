#!/usr/bin/env python3
"""Server-only, tiny stdlib fixtures; never run full_audit_gate/stage/publish.

The actual source-snapshot ZIP is read only to check source/provenance bindings.
No source image, full dataset ZIP, real dataset descriptor, or live readiness is
created or changed. Publication is tested only as an os.link primitive on names
under the independent fixture directory, never by calling the gate's publish.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import ast
import collections
import datetime
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import random
import socket
import types
import unittest
import zipfile


TASK = Path("/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1")
ROOT = TASK / "recovery_ship_20261009_1910"
GATE = None
AUDITOR = None
STAGER = None
MANIFEST = None
DEFERRED = None
FIXTURES = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_source(name, path):
    raw = path.read_bytes()
    ast.parse(raw, filename=str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def pure_reference(path, names):
    """Extract only selected stdlib functions, not the auditor's PIL imports/main."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    scope = {"collections": collections, "Decimal": Decimal, "hashlib": hashlib,
             "json": json, "SPLITS": ("train", "val", "test")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return types.SimpleNamespace(**scope)


def row(number, split, image_key, pixel_key=None, label_key="a"):
    return {"name": "fixture_%03d.jpg" % number, "split": split, "source_group": "synthetic/group",
            "image_sha256": sha(image_key.encode()), "pixel_sha256": sha((pixel_key or image_key).encode()),
            "label_sha256": sha(label_key.encode()), "label_semantic_sha256": sha(label_key.encode()),
            "canonical_labels": [["0", ".5", ".5", ".2", ".2"]],
            "width": 2, "height": 2, "gray_mean": 0.0, "gray_std": 0.0,
            "gray_min": 0, "gray_max": 0}


class GateStdlibTests(unittest.TestCase):
    def setUp(self):
        self.work = FIXTURES / self._testMethodName
        self.work.mkdir()

    def test_strict_json_rejects_duplicates_and_nan(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(data=data), self.assertRaises((RuntimeError, ValueError)):
                GATE.parse_json(data)
        self.assertEqual(GATE.parse_json(b'{"a":[1,2]}'), {"a": [1, 2]})

    def test_frozen_runtime_context_is_read_only(self):
        context = GATE.context(types.SimpleNamespace(expected_hostname="a6000-c"))
        self.assertEqual(context["plan"]["datasets"]["sar_shipdataset"], context["entry"])
        self.assertEqual(context["helper"]["sha256"], sha((ROOT / "ship_gate.py").read_bytes()))
        self.assertEqual(len(context["files"]), len(GATE.PINNED) + 2)
        self.assertEqual(context["paths"]["readiness"], GATE.DATASET / "readiness.json")

    def test_write_new_is_no_clobber_and_serializer_matches_stager(self):
        value = [{"name": "fixture", "pixel_sha256": "0" * 64, "value": 1}]
        path = self.work / "gate.json"
        record = GATE.write_new(path, value)
        expected = STAGER.write_new(self.work / "original_stage_serializer.json", value)
        self.assertEqual(record["sha256"], expected)
        with self.assertRaises(FileExistsError):
            GATE.write_new(path, {"different": True})
        self.assertEqual(sha(path.read_bytes()), record["sha256"])

    def test_paths_and_hashes_reject_substitution(self):
        path = self.work / "original.json"
        record = GATE.write_new(path, {"fixture": True})
        with self.assertRaises(RuntimeError):
            GATE.file_record(path, "0" * 64)
        alias = self.work / "symlink.json"
        alias.symlink_to(path)
        with self.assertRaises(RuntimeError):
            GATE.file_record(alias)
        with self.assertRaises(RuntimeError):
            GATE.explicit_path(str(alias), self.work)
        with self.assertRaises(RuntimeError):
            GATE.explicit_path(str(self.work / ".." / "escape.json"), self.work)
        self.assertEqual(GATE.file_record(path, record["sha256"])[0], record)

    def test_sorted_decimal_semantics_match_auditor(self):
        one = b"0 .5 .50 .200 .2\n0 .6 .6 .1 .1\n"
        two = b"0 0.600 0.6000 0.10 0.10\n0 0.50 0.5 0.2 0.2\n"
        self.assertEqual(GATE.canonical_labels(one), GATE.canonical_labels(two))
        self.assertEqual(GATE.canonical_labels(one), AUDITOR.canonical_label(one, 1)[:2])
        for data in (b"", b"0 .5 .5 .2 .2\n0 .5 .5 .2 .2\n", b"0 NaN .5 .2 .2\n",
                     b"1 .5 .5 .2 .2\n", b"0 0 0 .5 .5\n"):
            with self.subTest(data=data), self.assertRaises((RuntimeError, ValueError)):
                GATE.canonical_labels(data)

    def test_duplicate_summaries_match_frozen_auditor(self):
        rng = random.Random(901)
        for count in (0, 1, 4, 20, 100):
            rows = [row(index, rng.choice(GATE.SPLITS), str(rng.randrange(8)),
                        str(rng.randrange(6)), str(rng.randrange(4))) for index in range(count)]
            for key in ("image_sha256", "pixel_sha256"):
                summary, groups = GATE.groups_for(rows, key)
                expected = AUDITOR.duplicate_audit(rows, key)
                self.assertEqual(summary, {k: v for k, v in expected.items() if k != "groups"})
                self.assertEqual(groups, expected["groups"])
                self.assertEqual(GATE.compact_groups(rows, key), STAGER.compact_groups(rows, key))

    def test_test_priority_and_test_conflict_scope(self):
        rows = [row(1, "train", "same"), row(2, "val", "same"), row(3, "test", "same"),
                row(4, "test", "same", label_key="contradiction"), row(5, "train", "other")]
        summary, _ = GATE.groups_for(rows, "image_sha256")
        self.assertEqual(summary["candidate_exclusion_counts"], {"train": 1, "val": 1, "test": 0})
        self.assertEqual(summary["test_internal_duplicate_groups"], 1)
        self.assertEqual(summary["test_internal_semantic_disagreement_groups"], 1)
        # Two differently encoded files can be decoded-identical with conflicting labels.
        rows = [row(6, "test", "encoding_1", "pixel", "a"), row(7, "test", "encoding_2", "pixel", "b")]
        self.assertEqual(GATE.groups_for(rows, "image_sha256")[0]["test_internal_semantic_disagreement_groups"], 0)
        self.assertEqual(GATE.groups_for(rows, "pixel_sha256")[0]["test_internal_semantic_disagreement_groups"], 1)

    def test_selected_manifest_depends_on_labels_pixels_and_order(self):
        rows = [row(1, "train", "a"), row(2, "val", "b"), row(3, "test", "c")]
        selected = [{key: value[key] for key in GATE.MANIFEST_FIELDS} for value in rows]
        predicted = sha((json.dumps(selected, sort_keys=True, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n").encode())
        self.assertEqual(predicted, STAGER.write_new(self.work / "selected.json", selected))
        variants = [list(reversed(selected)), [dict(value) for value in selected], [dict(value) for value in selected]]
        variants[1][0]["label_sha256"] = "0" * 64
        variants[2][0]["pixel_sha256"] = "0" * 64
        for index, changed in enumerate(variants):
            self.assertNotEqual(predicted, STAGER.write_new(self.work / ("changed_%s.json" % index), changed))

    def test_fixed_snapshot_source_and_provenance_binding(self):
        path = TASK / "external_inputs/source_snapshot_v1.zip"
        self.assertEqual(sha(path.read_bytes()), GATE.SNAPSHOT_SHA256)
        with zipfile.ZipFile(path) as archive:
            metadata_bytes = archive.read(next(name for name in archive.namelist() if name.endswith("/source_snapshot.json")))
            source_bytes = archive.read(next(name for name in archive.namelist() if name.endswith("/sar_shipdataset_source.json")))
        metadata, source = json.loads(metadata_bytes), json.loads(source_bytes)
        provenance = {"snapshot_archive": str(path), "snapshot_archive_sha256": GATE.SNAPSHOT_SHA256,
                      "snapshot_metadata_sha256": sha(metadata_bytes), "source_manifest_sha256": sha(source_bytes),
                      "source_host": metadata["hostname"], "source_audited_utc": source["audited_utc"]}
        checked, _ = GATE.source_snapshot({"provenance": provenance})
        self.assertEqual(len(checked["rows"]), 39729)
        for key in ("snapshot_archive_sha256", "snapshot_metadata_sha256", "source_manifest_sha256", "source_host"):
            altered = dict(provenance)
            altered[key] = "incorrect"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                GATE.source_snapshot({"provenance": altered})

    def test_deferred_candidate_without_live_publication(self):
        root = self.work / "synthetic_data"
        inventory = []
        for split in GATE.SPLITS:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            # Deliberately not real image encodings: no decoder/model is invoked.
            image_bytes = ("synthetic-" + split).encode()
            label_bytes = b"0 .5 .5 .2 .2\n"
            with (root / "images" / split / "fixture.jpg").open("xb") as stream:
                stream.write(image_bytes)
            with (root / "labels" / split / "fixture.txt").open("xb") as stream:
                stream.write(label_bytes)
            inventory.append({"split": split, "name": "fixture.jpg", "image_sha256": sha(image_bytes),
                              "label_sha256": sha(label_bytes)})
        manifest = GATE.write_new(self.work / "inventory.json", inventory)
        yaml_bytes = ("path: " + str(root) + "\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: [ship]\n").encode()
        with (self.work / "fixture.yaml").open("xb") as stream:
            stream.write(yaml_bytes)
        descriptor = {"protocol": "ssdd", "root": str(root), "yaml": str(self.work / "fixture.yaml"),
                      "yaml_sha256": sha(yaml_bytes), "manifest": manifest["path"],
                      "manifest_sha256": manifest["sha256"], "nc": 1, "names": ["ship"],
                      "counts": {split: 1 for split in GATE.SPLITS}}
        descriptor_record = GATE.write_new(self.work / "descriptor.fixture.json", descriptor)
        audit = {"status": "verified", "dataset_identity": GATE.IDENTITY, "nc": 1, "names": ["ship"],
                 "manifest_sha256": manifest["sha256"], "counts": descriptor["counts"]}
        audit_record = GATE.write_new(self.work / "audit.fixture.json", audit)
        uncreated = self.work / "never_created_live.fixture.json"
        entry = {"deferred": True, "expected_identity": GATE.IDENTITY, "expected_nc": 1, "protocol": "ssdd",
                 "descriptor": descriptor_record["path"], "audit": audit_record["path"], "readiness": str(uncreated)}
        ready = {"status": "verified_ready", "dataset_identity": GATE.IDENTITY, "nc": 1,
                 "descriptor_sha256": descriptor_record["sha256"], "source_audit_sha256": audit_record["sha256"],
                 "manifest_sha256": manifest["sha256"]}
        candidate = GATE.write_new(self.work / "candidate.fixture.json", ready)
        paths = MANIFEST.validate_deferred_declaration(entry)
        result, _, _ = DEFERRED.resolve_ready(entry, paths, ready, candidate)
        self.assertEqual(result, descriptor)
        self.assertFalse(uncreated.exists())
        for field in ("descriptor_sha256", "source_audit_sha256", "manifest_sha256"):
            bad = dict(ready)
            bad[field] = "0" * 64
            bad_record = GATE.write_new(self.work / ("bad_" + field + ".json"), bad)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                DEFERRED.resolve_ready(entry, paths, bad, bad_record)
        self.assertFalse(uncreated.exists())

    def test_atomic_link_primitive_is_no_clobber_in_fixture_only(self):
        candidate = self.work / "candidate.marker.json"
        record = GATE.write_new(candidate, {"fixture_only": True})
        target = self.work / "published.marker.json"
        os.link(candidate, target, follow_symlinks=False)
        GATE.sync_directory(self.work)
        self.assertTrue(os.path.samefile(candidate, target))
        self.assertEqual(sha(target.read_bytes()), record["sha256"])
        other = self.work / "other.marker.json"
        GATE.write_new(other, {"different": True})
        with self.assertRaises(FileExistsError):
            os.link(other, target, follow_symlinks=False)
        self.assertEqual(sha(target.read_bytes()), record["sha256"])


def main():
    global GATE, AUDITOR, STAGER, MANIFEST, DEFERRED, FIXTURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-hostname", required=True, choices=("a6000-c",))
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert os.name == "posix" and socket.gethostname() == args.expected_hostname
    assert Path(__file__).resolve().parent == ROOT
    assert args.output_dir.is_absolute() and args.output_dir.resolve() == args.output_dir
    assert args.output_dir.is_relative_to(ROOT / "evidence") and args.output_dir != ROOT / "evidence"
    helper_path = ROOT / "ship_gate.py"
    assert sha(helper_path.read_bytes()) == args.expected_helper_sha256
    GATE = load_source("ship_gate_under_test", helper_path)
    for name, expected in GATE.PINNED.items():
        assert sha((TASK / name).read_bytes()) == expected, name
    AUDITOR = pure_reference(TASK / "audit_external_archives.py", ("sha", "canonical_label", "duplicate_audit"))
    STAGER = pure_reference(TASK / "stage_shipdataset.py", ("sha", "write_new", "compact_groups"))
    MANIFEST = load_source("manifest", TASK / "manifest.py")
    DEFERRED = load_source("deferred_data", TASK / "deferred_data.py")
    live = GATE.DATASET / "readiness.json"
    live_before = sha(live.read_bytes()) if live.exists() else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    FIXTURES = args.output_dir / "fixtures"
    FIXTURES.mkdir()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateStdlibTests))
    live_after = sha(live.read_bytes()) if live.exists() else None
    no_gpu_imports = not any(name == "torch" or name.startswith("torch.") or name == "PIL" for name in sys.modules)
    summary = {"status": "passed" if result.wasSuccessful() and live_before == live_after and no_gpu_imports else "failed",
               "hostname": socket.gethostname(), "python": sys.executable,
               "started_utc": started_utc, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "helper_sha256": args.expected_helper_sha256, "tests_run": result.testsRun,
               "test_harness_sha256": sha(Path(__file__).read_bytes()), "pinned_sources_verified": GATE.PINNED,
               "failures": len(result.failures), "errors": len(result.errors),
               "live_readiness_before": live_before, "live_readiness_after": live_after,
               "no_torch_or_pillow_import": no_gpu_imports, "full_data_audit_stage_or_publish_called": False,
               "fixture_scope": "stdlib only; source snapshot metadata, tiny synthetic file hashes, candidate validation, and os.link primitive",
               "test_log": buffer.getvalue()}
    record = GATE.write_new(args.output_dir / "result.json", summary)
    print(json.dumps({"result": record, **summary}, sort_keys=True), flush=True)
    raise SystemExit(0 if summary["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
