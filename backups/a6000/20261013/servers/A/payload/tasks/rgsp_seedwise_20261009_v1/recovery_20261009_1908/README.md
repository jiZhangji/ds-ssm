# A seed0 cross-task operator recovery, 2026-10-09

This is an independent recovery attempt, not an edit to the frozen experiment.
The original `plan.json`, its 360-entry hash inventory, all frozen runtime code,
five already completed A receipts, old failed lane/dispatcher status and logs
remain unchanged. The formal plan canonical hash is
`9c6b1afe025155c2209e56ecd745a7fccccd82dc96b4bd75987c66a709bfd7e4`.

The 19:08 Beijing check found A (`powerleader-PT6620W`) idle: its assigned A6000
UUID `GPU-12cf4841-46b9-94d1-2715-9932134cf964` used 530MiB desktop memory,
0% utilization and had no CUDA compute process; disk free was about237GiB.
The failed SAR-ACD Original run contained only `config.json`, with no training
checkpoint, metrics, initialization file or completed evaluation. The other
fourteen cross-task seed0 runs had not started. All 360 frozen source hashes
and the five existing receipt artifact sets were rechecked successfully.

Cause: the frozen cross CLI imports torch1.13 with a UUID mask but does not
invoke the already frozen UUID device-count adapter. The original launch died
at `torch.cuda.get_device_name` before any optimizer/epoch. The operator uses
the frozen `cuda_uuid_compat.py` CLI, whose `runpy` executes the same unchanged
cross script in the same process. The full planned argument list, including
workers/device/resume flags, is forwarded exactly. Original versus actual argv,
operator/helper/cross/plan hashes, GPU identity and compatibility report are
recorded separately. It never claims the original executable argv ran directly.

Scope is exactly SAR-ACD, SAMPLE, Ottawa, Farmland and Yellow_River, each with
Original/Matched075/Full and seed0. No AIR slot is run or marked complete, no
second seed starts and no global barrier or original lane-completion file is
written. Any existing checkpoint/metrics requires a separate resume audit;
this attempt refuses it. Failures stop the finite queue without retry/signals.

On A only, using the existing project Python:

1. `test_operator.py` exercises synthetic standard-library-only fixtures, retained
   under `unit_fixtures`. No CUDA/model is used by these tests.
2. `operator_cross_recover.py verify-only` verifies immutable inputs read-only.
3. `operator_cross_recover.py probe` runs the frozen validation-only CLI on two
   training samples per arm for classification and CD. Outputs/checkpoints are
   strictly below this independent recovery directory, never formal run outputs.
4. `operator_cross_recover.py run` launches the fifteen formal trials sequentially.

Both execution modes acquire the original dispatch, lane and campaign GPU locks;
all are inherited by the child, preventing a duplicate even if the operator dies.
Every child observes the original finite idle gate and a fresh no-compute check.
Existing services/jobs are never stopped. Software is not installed or modified.

Before writing each new receipt, the operator recursively verifies the six
internal completion artifacts and their hashes, config identity, seed, protocol,
best checkpoint/test/result agreement, ordered real epochs and finite loss/metrics.
The independent deep audit, compatibility report and launch contract enter the
receipt's details as hashed evidence; the frozen outer result/completion artifact
order remains unchanged. This does not establish bitwise equivalence, scene
independence or annotation correctness; earlier data limitations remain in force.

Actual probe/launch/epoch evidence and final outcome must be read from this
directory on A. This README describes the audited procedure, not completed training.
