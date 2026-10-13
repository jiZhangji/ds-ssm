#!/usr/bin/env python3
"""Check the local fused selective scan against the differentiable reference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.common_utils_mbyolo import SelectiveScanCore, selective_scan_reference


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    batch, channels, groups, state, length = 2, 8, 4, 4, 64
    originals = [
        torch.randn(batch, channels, length, device=device),
        0.5 * torch.rand(batch, channels, length, device=device),
        -0.5 * torch.rand(channels, state, device=device),
        torch.randn(batch, groups, state, length, device=device),
        torch.randn(batch, groups, state, length, device=device),
        torch.randn(channels, device=device),
        0.5 * torch.rand(channels, device=device),
    ]
    fused_inputs = [value.detach().clone().requires_grad_() for value in originals]
    reference_inputs = [value.detach().clone().requires_grad_() for value in originals]
    fused = SelectiveScanCore.apply(*fused_inputs, True, 1, 1, True)
    reference = selective_scan_reference(*reference_inputs, delta_softplus=True)
    upstream = torch.randn_like(fused)
    fused_grads = torch.autograd.grad(fused, fused_inputs, upstream)
    reference_grads = torch.autograd.grad(reference, reference_inputs, upstream)

    output_diff = float((fused - reference).abs().max())
    gradient_diffs = [float((left - right).abs().max()) for left, right in zip(fused_grads, reference_grads)]
    output_ok = torch.allclose(fused, reference, rtol=6e-4, atol=2e-3)
    gradient_ok = all(
        torch.allclose(left, right, rtol=1e-3, atol=2e-3)
        for left, right in zip(fused_grads, reference_grads)
    )
    print({"output_max_abs_diff": output_diff, "gradient_max_abs_diffs": gradient_diffs})
    if not output_ok or not gradient_ok:
        raise SystemExit("Fused selective scan does not match the reference implementation")
    print("selective_scan_backend_validation=PASS")


if __name__ == "__main__":
    main()
