# Independent ShipDataset recovery gate

This directory adds an operator helper only. It does not modify the frozen
114-slot plan, its code hashes, any original script, source labels, source split,
or historical results. The initial review was static; subsequent C/F syntax
checks and C-only lightweight standard-library tests are recorded below. No full
dataset gate, actual staging, or readiness publication has run in this subtask.

## Scope and hard stop

The allowed dataset is **SAR-ShipDataset**, not SAR-AIRcraft1.0 or AIR-SARShip.
The allowed hosts are C (`a6000-c`) and F (`powerleader-a6000-f`), with the exact
existing task root and Python below. The gate does not connect to any host,
probe/allocate a GPU, start training, install software, or create a service.

The new copy retains the existing seed2026 image split, applying exact-byte and
decoded-RGB cross-split exclusions with priority **test > val > train**. Original
test membership, image bytes, and label bytes are unchanged. Sensor/polarization/
date filename groups remain unverified scene proxies; this is not a new grouped
or scene-disjoint split. Within-split duplicates are retained.

The gate independently recomputes every byte/pixel duplicate group and exclusion
from the reviewed audit rows, and checks the grouped member ledgers. It rehashes
every extracted original image and label against the fixed source snapshot and
reconstructs the label semantics. Pixel identities come from the SHA-bound full
Pillow archive audit, not a second independent image decoder.

**Any test-internal byte/pixel-identical image group with different semantic
label multisets stops before staging and publication.** No preferred label,
merged box, test deletion, alternate split, or class correction is chosen.
An absent readiness remains absent: the helper never writes `failed`, `invalid`,
or `rejected` into the live readiness path, since those would stop waiting lanes.
Unexpected label bounds, row multiplicity, orientations, hashes, membership,
group counts, or missing evidence also stop the gate.

## Fixed inputs

| Artifact | SHA256 |
| --- | --- |
| Canonical plan | `9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4` |
| Plan file bytes | `37e1e2254643541118e4bab29d11cb93f1483847fc73aa69be8d63be4907cbe0` |
| source_snapshot_v1.zip, 5,795,637 bytes | `aea74cd1d3a55b750004b72249a8896c13a7751025b5398d8eb2e37098bb9b69` |
| sar_shipdataset_source.json payload | `a69d2acf8fcf2d90cf691b15c007549002be5036c7155493e5cdcd809f25dc4a` |
| Complete source ZIP, 426,990,175 bytes | `6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e` |
| Original archive auditor | `772fdada1542a3bc2c644859a6cc82ae76dac150ddf2459a83462e6695473b0b` |
| Original stage script | `81e621a130f8f6e66c3785c27a411ebe46dc4cb9733f1299411a35d0352f41f8` |
| Frozen manifest.py | `ff1f6708e735d5bd91d9c341edfcfdcc714829c5ab5ee0c6110df263d5b6fed7` |
| Frozen deferred_data.py | `dbfd4cadcdbe6a7533bfb679ed540d2739c8ee6b961b4558bd5e66e6e05a7fb1` |

The helper also invokes the unchanged `manifest.verify_code` for the complete
frozen runtime list. A different existing file is a stop, not permission to
overwrite it. Record and compare the helper's own SHA after deployment.

## Full archive audit (root operator only)

The original `audit_external_archives.py` reads a hard-coded input location:
`TASK/external_inputs/raw_archives/ship_dataset_v0.zip`. Recovery transfer names
ending in `.recovery_20261009_1910_c` or `_f` are not automatically selected.
The root operator must first verify the completed transfer's full size/SHA and
resolve the canonical path without overwriting an existing partial. Retain the
partial and its provenance. If a separate audit wrapper is needed to select an
explicit new filename, review and record that wrapper separately; this helper
does not change the archive auditor's constants or choose a partial implicitly.

Use a new output directory; never retry over an incomplete audit directory:

```bash
RGSP_TASK=/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1
RGSP_PY=/home/powerleader/ig-ds-ssm/envs/ig-ds-ssm-env/bin/python
RGSP_AUDIT="$RGSP_TASK/external_inputs/audits/ship_recovery_20261009_1910_c"

"$RGSP_PY" -B "$RGSP_TASK/audit_external_archives.py" \
  --dataset sar_shipdataset \
  --snapshot-zip "$RGSP_TASK/external_inputs/source_snapshot_v1.zip" \
  --snapshot-sha256 aea74cd1d3a55b750004b72249a8896c13a7751025b5398d8eb2e37098bb9b69 \
  --output "$RGSP_AUDIT"
```

The reviewed report must have `status=audit_only_not_ready`, CRC passed for every
member, zero source/archive image-label mismatches, all 39,729 rows, and original
31,783/3,972/3,974 counts. Its `audited_rows.json`, `byte_groups.json`, and
`pixel_groups.json` must match the report's hashes. The gate enforces these and
additional source/class/exclusion checks. The source-byte forecast is 48 train
and 2 val exclusions; decoded-pixel exclusions may be larger and must be reviewed
from the actual full report. Never use the byte-only forecast as the staged count.

## Read-only gate, staging, and final publication

Set `RGSP_AUDIT_SHA` to the exact full report hash reviewed by the root operator;
the text placeholders below are not valid command arguments. C's hostname is
shown; on F use `powerleader-a6000-f` and its exact reviewed audit directory.

```bash
RGSP_GATE="$RGSP_TASK/recovery_ship_20261009_1910/ship_gate.py"
RGSP_AUDIT_SHA=<reviewed-archive_audit.json-sha256>

"$RGSP_PY" -B "$RGSP_GATE" --validate-only \
  --expected-hostname a6000-c \
  --audit "$RGSP_AUDIT/archive_audit.json" \
  --reviewed-audit-sha256 "$RGSP_AUDIT_SHA"
```

`--validate-only` creates no directories, locks, receipts, bytecode, or candidates.
If a stage exists, it also verifies that stage and any already published gate
readiness. Only `status=validated_only` is a successful read-only result.

The original stage destination must be absent. Use one new evidence directory:

```bash
"$RGSP_PY" -B "$RGSP_GATE" --stage \
  --expected-hostname a6000-c \
  --audit "$RGSP_AUDIT/archive_audit.json" \
  --reviewed-audit-sha256 "$RGSP_AUDIT_SHA" \
  --evidence-dir "$RGSP_TASK/recovery_ship_20261009_1910/evidence/stage_c_01"
```

This first runs the complete gate, then calls the unchanged SHA-pinned
`stage_shipdataset.py`. It validates exact manifest rows/bytes/order, image and
label file sets and hashes, YAML path/class order, exclusions and priority,
`source_audit.json`, and every `stage_complete.json` binding. Staging never
publishes readiness. Success is `staged_verified_not_published` and the output
includes the exact stage-completion and manifest hashes.

On F, pass `--expected-manifest-sha256` with the selected C manifest hash to both
validation and staging. The predicted manifest is compared **before** creation,
so a different decoder-derived exclusion set stops without staging. C/F source
audit hashes may differ due to host/timestamp/archive provenance; their content
manifest, YAML, and descriptor must represent the same chosen dataset.

Only after the root operator has reviewed the stage evidence, run publication
with the explicit reviewed stage-completion hash and selected manifest hash:

```bash
RGSP_STAGE_SHA=<reviewed-stage_complete.json-sha256>
RGSP_MANIFEST_SHA=<selected-identical-C-and-F-data_content.json-sha256>

"$RGSP_PY" -B "$RGSP_GATE" --publish \
  --expected-hostname a6000-c \
  --audit "$RGSP_AUDIT/archive_audit.json" \
  --reviewed-audit-sha256 "$RGSP_AUDIT_SHA" \
  --reviewed-stage-complete-sha256 "$RGSP_STAGE_SHA" \
  --expected-manifest-sha256 "$RGSP_MANIFEST_SHA" \
  --evidence-dir "$RGSP_TASK/recovery_ship_20261009_1910/evidence/publish_c_01"
```

Publication redoes the gate and stage verification, writes/fsyncs an independent
candidate, calls the unchanged `deferred_data.resolve_ready` against that
candidate (including all referenced image/label hashes), rechecks bindings/code,
then uses same-filesystem `os.link` to create the live readiness atomically.
Readers cannot observe a partially written live JSON. No `os.replace`, unlink,
overwrite, or deletion is used. The evidence candidate remains as a hard link.
Separate local writer locking coordinates this helper only, not unrelated tools.

Success is `published` or a fully reverified `already_published`. A different
existing readiness is refused. If an IO error occurs after the atomic link but
before `publication_complete.json`, the live marker may already exist; inspect
the preserved candidate/validation and run read-only validation. Do not relaunch
training, overwrite readiness, or infer that an error automatically unpublishes
an already linked marker.

## Static review performed

- Manually compared JSON fields/serialization against the original auditor and
  stager, including Decimal-normalized sorted label multisets and list order.
- Confirmed the original stager and frozen deferred checker do not themselves
  reject test-internal pixel semantic conflicts; this gate adds that precondition
  without modifying them.
- Confirmed publication remains a separate explicit mode, requires selected
  hashes, and never uses a live-file streaming write.
- No local Python, parser, unit test, data audit, or model execution was used.
  A requested independent review agent failed immediately with a token-rate
  limit; it is not counted as a completed independent review.

## Server verification, 2026-10-09

Both C and F received byte-identical, exclusively created copies of the original
auditor/stager at the task root and the independent helper/test harness in this
directory. Existing files would have been reused only with exact hashes; no
existing file was replaced. On both hosts, `ast.parse` and `compile` passed for
the four scripts and the pinned frozen manifest/deferred files. These checks
did not execute the auditor, stager, or helper CLI. Deployment receipts are in
`deployment_receipts_20261009.json` in the local independent directory.

C was selected for the lightweight CPU tests: 32 CPUs, load 0.23/0.24/0.18 and
846,449,872,896 free bytes at deployment. F was also reachable (32 CPUs, load
0.28/0.16/0.12 and 231,297,331,200 free bytes), but did not run the test suite.
The prior read-only resource check found no GPU compute processes on either
host; no GPU was allocated, and no existing service or task was changed.

On C, `test_ship_gate_stdlib.py` passed all **11 tests**, zero errors/failures,
in 0.765 seconds, ending at `2026-10-09T11:42:38.375134+00:00` (19:42 China time).
It checked strict JSON parsing; path/hash/symlink rejection; exclusive writes;
Decimal label semantics; duplicate summary/member parity against the pinned
auditor; test-first exclusions and mixed-split/test-internal semantic conflicts;
selected-manifest serializer parity and hash sensitivity; the fixed source
snapshot/provenance; the complete frozen runtime context/hash list; the frozen
deferred resolver with a tiny synthetic descriptor; and the atomic no-clobber
hard-link primitive on fixture-only filenames.

| Verified artifact | SHA256 |
| --- | --- |
| ship_gate.py | `1bb625fb0a8262f27ecaced01b25177b3775f4de6dc42684446bc2756a083c29` |
| test_ship_gate_stdlib.py | `30dd4900c497c7f31f6b5d1aa2fef00883e46fbabf2af4fc624f66ea69faa957` |
| C evidence/stdlib_c_01/result.json, 2,200 bytes | `1fffab06f70b08415072fac646ae5a5752b60b02e5d4194fc625079a0f4525f9` |

The server result remains at
`/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1/recovery_ship_20261009_1910/evidence/stdlib_c_01/result.json`.
Its local evidence copy is `server_stdlib_c_01_result.json`. Fixture files are
retained under that server evidence directory; nothing was deleted. Actual live
readiness was absent before and after the tests. No torch or Pillow import
occurred, and `full_audit_gate`, `run_stage`, and `publish` were never called.

The fixture suite is not an end-to-end dataset approval, decoding audit, actual
staging test, or actual readiness publication test. The complete source ZIP,
full all-image archive audit, independent gate validation, verified staging, and
separate reviewed publication remain root-operator work. At the last scoped
inventory, C/F had partial source ZIPs only; those transfers are owned by the
root operator and were neither changed nor restarted by this subtask.

For a deliberately requested repeat, use C's existing Python and a new evidence
directory (never reuse `stdlib_c_01`):

```bash
"$RGSP_PY" -B "$RGSP_TASK/recovery_ship_20261009_1910/test_ship_gate_stdlib.py" \
  --expected-hostname a6000-c \
  --expected-helper-sha256 1bb625fb0a8262f27ecaced01b25177b3775f4de6dc42684446bc2756a083c29 \
  --output-dir "$RGSP_TASK/recovery_ship_20261009_1910/evidence/stdlib_c_02"
```
