#!/usr/bin/env python3
"""GPU numerical/control tests plus read-only historical checkpoint audit."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import ultralytics.nn.modules.mamba_yolo as m
from ultralytics import YOLO
from run_rgsp_controls import VARIANTS, config, sha
from train_rgsp_control import controls, reset_bn_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--historical-root", type=Path)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    report = {}
    if args.historical_root:
        report["historical_pfim_only"] = []
        for seed in range(3):
            path = args.historical_root / f"seed{seed}_ssdd_pfim_only_p3/weights/best.pt"
            if not path.exists():
                continue
            model = YOLO(str(path))
            entries = []
            for name, module in model.model.named_modules():
                if isinstance(module, m.DeSpeckleSSM):
                    entries.append({"module": name, "lambda_s": float(module.lambda_s),
                                    "alpha": float(module.alpha)})
            report["historical_pfim_only"].append({"seed": seed, "path": str(path),
                                                   "sha256": sha(path), "scalars": entries})
            del model
    if not args.audit_only:
        assert torch.cuda.is_available(), "No local/CPU fallback"
        assert m.USE_CUDA_SCAN and m.selective_scan_fn is not None
        report["backend"] = m.selective_scan_fn.__module__
        report["gpu"] = torch.cuda.get_device_name()
        device = "cuda"
        torch.manual_seed(401)
        old = m.DS_VSSBlockSweep(16, 16, 4).to(device)
        torch.manual_seed(401)
        full = m.DS_VSSBlockSweep(16, 16, 4, "full").to(device)
        assert all(torch.equal(v, full.state_dict()[k]) for k, v in old.state_dict().items())
        x = torch.randn(2, 16, 4, 4, device=device)
        assert torch.equal(old(x), full(x)), "Legacy full forward changed"
        # Validate the full module fast path against its differentiable recurrence.
        ref = copy.deepcopy(full)
        y = full(x)
        m.USE_CUDA_SCAN = False
        try:
            y_ref = ref(x)
        finally:
            m.USE_CUDA_SCAN = True
        torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=2e-5)
        y.square().mean().backward()
        y_ref.square().mean().backward()
        gradient_max = 0.0
        for (name, a), (_, b) in zip(full.named_parameters(), ref.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-3, atol=2e-5, msg=name)
                gradient_max = max(gradient_max, float((a.grad - b.grad).abs().max()))
        report["module_fast_reference"] = {"output_max_abs": float((y - y_ref).abs().max()),
                                            "gradient_max_abs": gradient_max}
        report["controls"] = []
        template = None
        for variant in VARIANTS[1:]:
            torch.manual_seed(401)
            block = m.DS_VSSBlockSweep(16, 16, 4, variant).to(device)
            s = block.ds_ssm
            state = block.state_dict()
            # Apart from intentional scalar zeroing, common initial tensors identical.
            if template is None:
                template = {key: value.clone() for key, value in state.items()}
            for key, value in state.items():
                if key not in {"ds_ssm.lambda_s", "ds_ssm.alpha"}:
                    assert torch.equal(value, template[key]), (variant, key, "RNG confound")
            captured = {}
            scan = m.selective_scan_fn
            def capture(*a, **kw):
                captured["u"], captured["delta"] = a[0], a[1]
                return scan(*a, **kw)
            m.selective_scan_fn = capture
            tokens = torch.randn(2, 16, 16, device=device)
            try:
                out, importance = s(tokens, hw_shape=(4, 4), return_importance=True)
            finally:
                m.selective_scan_fn = scan
            trend = s.dw(tokens.transpose(1, 2)).transpose(1, 2)
            corrected = tokens - s.lambda_s * (tokens - trend)
            drive, _ = s.in_proj(corrected).chunk(2, dim=-1)
            energy = trend.norm(dim=-1, keepdim=True) / 4
            gamma_logits = -s.alpha * s.sigma_n * energy
            if s.use_pfim and s.pfim_on_gamma:
                gamma_logits = gamma_logits + s.beta_i * (importance - 0.5)
            gamma = s.min_retain + (1 - s.min_retain) * gamma_logits.sigmoid()
            write = 1 + s.rho_i.tanh() * (importance - 0.5) if s.use_pfim and s.pfim_on_write else 1
            expected_u = (drive * gamma * write).transpose(1, 2)
            expected_delta = s.delta_proj(drive)
            if s.use_pfim and s.pfim_on_delta:
                expected_delta = expected_delta + s.eta_i * importance
            torch.testing.assert_close(captured["u"], expected_u)
            torch.testing.assert_close(captured["delta"], torch.nn.functional.softplus(expected_delta).transpose(1, 2))
            if variant in {"matched_identity", "matched_scale075", "residual_scale075"}:
                expected_scale = 1.0 if variant == "matched_identity" else 0.75
                torch.testing.assert_close(gamma, torch.full_like(gamma, expected_scale))
            # Simulate legacy trainer unfreezing, then take real optimizer steps.
            for parameter in block.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(block.parameters(), lr=0.001)
            for step in range(2):
                optimizer.zero_grad()
                loss = block(x).square().mean()
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)
                optimizer.step()
            row = controls(block)[0]
            row["parameters"] = sum(p.numel() for p in block.parameters())
            row["buffer_elements"] = sum(p.numel() for p in block.buffers())
            row["status"] = "PASS"
            report["controls"].append(row)
        # Test model parser and seeded common initialization across the full detector.
        detector_template = None
        parser_rows = []
        for variant in VARIANTS:
            cfg = args.output.parent / "validation_configs" / f"{variant}.yaml"
            cfg.parent.mkdir(parents=True, exist_ok=True)
            import yaml
            cfg.write_text(yaml.safe_dump(config(variant), sort_keys=False))
            torch.manual_seed(51)
            detector = YOLO(str(cfg)).model
            reset_bn_statistics(detector)
            blocks = controls(detector)
            assert len(blocks) == (0 if variant == "original" else 1)
            if variant != "original":
                state = detector.state_dict()
                if detector_template is None:
                    detector_template = {k: v.clone() for k, v in state.items()}
                for k, v in state.items():
                    if not k.endswith((".lambda_s", ".alpha")):
                        assert torch.equal(v, detector_template[k]), (variant, k, "detector RNG changed")
            parser_rows.append({"variant": variant, "controls": blocks})
            del detector
        report["detector_parser_and_initialization"] = parser_rows
        report["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
