#!/usr/bin/env python3
"""Verify that the DS-SSM CPU fallback matches the CUDA scan-input semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics.nn.modules.mamba_yolo as mamba_yolo


def explicit_cuda_semantics(module, x, hw_shape):
    """Reference recurrence for the executed CUDA path: u_scan = gamma * write * u."""
    input_dtype = x.dtype
    batch, tokens, channels = x.shape
    x_bar = module.dw(x.transpose(1, 2)).transpose(1, 2)
    x_tilde = x - module.lambda_s.to(dtype=input_dtype) * (x - x_bar)

    importance = module.pfim(x_tilde, x_bar, hw_shape=hw_shape)
    centered_importance = importance - 0.5
    u, z = module.in_proj(x_tilde).chunk(2, dim=-1)
    delta_logits = module.delta_proj(u) + module.eta_i.to(dtype=input_dtype) * importance
    delta = F.softplus(delta_logits)
    b_t = module.B_proj(u)
    c_t = module.C_proj(u)
    a = -torch.exp(module.A_log.to(dtype=input_dtype))

    energy = torch.norm(x_bar, p=2, dim=-1, keepdim=True) / (channels**0.5)
    gamma_logits = (
        -module.alpha.to(dtype=input_dtype) * module.sigma_n * energy
        + module.beta_i.to(dtype=input_dtype) * centered_importance
    )
    gamma = module.min_retain + (1.0 - module.min_retain) * torch.sigmoid(gamma_logits)
    write_gate = torch.clamp(
        1.0 + torch.tanh(module.rho_i.to(dtype=input_dtype)) * centered_importance,
        min=0.1,
    )

    h = x.new_zeros(batch, channels, module.state_dim)
    outputs = []
    for index in range(tokens):
        dt = delta[:, index, :].unsqueeze(-1)
        a_t = torch.exp(dt * a.unsqueeze(0))
        b_timestep = dt * b_t[:, index, :].unsqueeze(1)
        c_timestep = c_t[:, index, :].unsqueeze(1)
        u_scan = gamma[:, index, :] * write_gate[:, index, :] * u[:, index, :]
        h = a_t * h + b_timestep * u_scan.unsqueeze(-1)
        outputs.append((c_timestep * h).sum(dim=-1) + module.D.to(dtype=input_dtype) * x_tilde[:, index, :])

    y = torch.stack(outputs, dim=1) * F.silu(z)
    return module.out_proj(y)


def main() -> None:
    torch.manual_seed(7)
    old_use_cuda_scan = mamba_yolo.USE_CUDA_SCAN
    mamba_yolo.USE_CUDA_SCAN = False
    try:
        module = mamba_yolo.DeSpeckleSSM(dim=8, state_dim=3).eval()
        inputs = torch.randn(2, 12, 8)
        with torch.no_grad():
            actual = module(inputs, hw_shape=(3, 4))
            expected = explicit_cuda_semantics(module, inputs, hw_shape=(3, 4))
        error = (actual - expected).abs().max().item()
    finally:
        mamba_yolo.USE_CUDA_SCAN = old_use_cuda_scan

    print(f"max_abs_error={error:.8e}")
    if error > 1e-6:
        raise SystemExit("DS-SSM CPU fallback does not match CUDA scan-input semantics")
    print("PASS: CPU fallback implements u_scan = gamma * write_gate * u")


if __name__ == "__main__":
    main()
