# Method Definition

## Problem framing

SAR imagery contains speckle, local clutter and low-contrast targets. A vanilla visual state-space scan may propagate local residuals and background responses as if they were useful tokens. The project therefore studies whether SAR priors can be injected into the state update itself while keeping the Mamba-YOLO detector compact.

## DS-SSM

The active code implements a SAR-aware state-space block with:

- local residual/de-speckle correction, represented in the current code as an input correction of the form `x_tilde = x - lambda_s * residual`;
- energy-aware damping through a soft retention factor `gamma`;
- selective scan over the corrected/damped input;
- residual output involving `D * x_tilde` outside the CUDA scan call.

Exact scalar defaults must be read from the YAML and the code for each experiment; do not infer an experiment-specific `lambda_init` from the general method description.

## PFIM

PFIM estimates a token-level importance value from local trend and local variation. With a valid 2-D shape, it can use a local 2-D statistic in addition to sequence-local information. The importance is used to guide:

- `gamma`: how much scan input/state contribution is retained or damped;
- `write_gate`: how strongly the current token is written into the state;
- `delta`: transition step sensitivity.

In the executed CUDA path, `gamma` and `write_gate` are multiplicative factors on the same scan drive. `delta` is the separate transition-control path. This implementation detail must be reflected in diagrams and claims.

## Placement

For detection, the project focuses on shallow high-resolution stages because SAR small targets and speckle are visible there. P3 is the current paper version, while earlier sweeps also examined P2/P4/P5/P2+P3. Placement is task-dependent; segmentation evidence shows that P3 can damage thin road structures.

## Initialization

- Strict scratch causal comparison: baseline and ours both use `weights=none`.
- Optional pretrained comparison: both receive the same real Mamba-YOLO/appropriate backbone initialization; newly introduced parameters are separately initialized and documented.
- Historical `yolov8n.pt` loading is not equivalent to Mamba-YOLO pretrained transfer.

## PFIM initialization nuance

The code applies `delta_logits = delta_proj(u) + eta_i * importance` with the **raw** (uncentered) importance, not `imp_center`. At initialization importance ≈ 0.5 and `eta_init = 0.1`, so delta receives a small positive bias (~+0.05). Paper wording: gamma/write branches are neutral around 0.5; delta carries a small initialized bias. Do not change the equation without rerunning experiments.

The CPU fallback was aligned to CUDA semantics: `h = A_t * h + B_t * (gamma * write_gate * u)_t` (verified max abs error 0 via `tools/check_dsssm_cpu_fallback.py`).

Parameter-count口径: frozen code baseline 5,985,937 / full·global·no-2dstd 5,965,147 / no-PFIM 5,963,962; historical logs baseline 5,984,593 / PFIM 5,963,803. The fixed 1,344 difference comes from historical code state; never mix the two conventions — use `model_snapshot.json` for new tables.

## Validated and unvalidated parts

- Implemented: residual correction, energy-aware damping, soft gamma, PFIM map, gamma/write/delta controls and ablation adapters.
- Indirectly supported: full PFIM in recovered top runs.
- Still needing publication-grade isolation: no-PFIM/global/no-2D-std and gamma/write/delta branch matrix under frozen OGSOD protocol.
- Candidate only: separate local-scale/GDN-style extension; do not merge it into the mainline without source and experiment evidence.
