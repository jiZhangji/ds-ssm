# Detector Complexity Evidence (2026-07-24)

## Protocol

- Server: `172.26.133.16` (`lab540-3090-4`)
- Input: `1 x 3 x 640 x 640`
- Precision/device: FP32 on CPU; no training GPU was used
- Framework: PyTorch 2.1.1, Detectron2 0.6, fvcore JIT analysis
- Reported GFLOPs: `2 x fvcore raw operation count`, matching the multiply-add convention used by the existing Ultralytics rows
- Parameters: total and trainable values are both retained in the evidence CSV; paper tables use total parameters

Raw JSON and profiler logs are under `artifacts/complexity_20260724/`.

## Server-Measured Results

| Method | Classes | Total Params (M) | Trainable (M) | Raw fvcore Gops | Table GFLOPs |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN R50-FPN | 1 | 41.295061 | 41.072661 | 75.171579 | 150.343158 |
| RetinaNet R50-FPN | 1 | 36.276717 | 36.054317 | 79.670842 | 159.341683 |
| DiffusionDet R50-FPN | 1 | 110.551518 | 110.329118 | 102.131496 | 204.262993 |
| DiffusionDet R50-FPN | 7 | 110.560770 | 110.338370 | 102.136104 | 204.272209 |
| ConsistencyDet R50-FPN | 1 | 110.551518 | 110.329118 | 1871.846994 | 3743.693988 |
| ConsistencyDet R50-FPN | 7 | 110.560770 | 110.338370 | 1872.031314 | 3744.062628 |
| DiffDet4SAR R50-FPN | 1 | 111.141342 | 110.918942 | 102.393640 | 204.787281 |
| DiffDet4SAR R50-FPN | 7 | 111.150594 | 110.928194 | 102.398248 | 204.796497 |

Faster R-CNN and RetinaNet use canonical Detectron2 R50-FPN configurations because MaDiNet does not identify the exact comparison config. They are architecture references, not claims about an unreleased MaDiNet training setup.

ConsistencyDet's current official forward executes 40 head passes (`n_steps=40`) despite `SAMPLE_STEP=1`; the measured FLOPs reflect actual execution at commit `0f6e44a`. Its only staging change replaces hard-coded CUDA transfers with transfers to the input tensor's device. DiffDet4SAR's published override YAML is syntactically invalid; the profiler loads the valid base and applies the same file's explicit R50, 500-proposal, and class-count settings.

Unsupported operations are listed in each raw JSON. The reported totals therefore remain profiler-supported lower estimates for operations such as ROIAlign, top-k, padding, random sampling, trigonometric embeddings, and elementwise activations.

## Literature-Only Results

| Method | Params (M) | GFLOPs | Evidence |
|---|---:|---:|---|
| Improved FCOS | 40.6 | NR | Author Table I; final all-module model |
| MLSDNet | 0.92825 | 1.42 | Author abstract; original input/counting convention |
| SAR-ShipNet | NR | NR | No public implementation or numerical report found |
| GFECSI-Net | NR | NR | No public implementation or numerical report found |
| CMFT | NR | NR | No public implementation or numerical report found |
| SA-Net | NR | NR | Full public article contains no numerical complexity report |

`NR` means not reported and not reproducible from a public implementation. These cells must not be replaced with guessed backbone values.

## MaDiNet Limitation

The official MaDiNet repository releases `MambaSAR.py` but not the complete detector integration or a model config. The measured `223.2648M / 345.283 GFLOPs` value covers the official MambaSAR res4+res5 core and is a lower bound, not complete-detector complexity.
