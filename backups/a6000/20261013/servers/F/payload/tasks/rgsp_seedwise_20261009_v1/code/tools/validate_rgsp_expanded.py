#!/usr/bin/env python3
"""Server-only numerical and serialization audit for the expanded RGSP controls.

The training entry point may import ``audit_controls(model, optimizer=None)``
for setup/epoch/reload checks. The CLI requires a CUDA device and never falls
back to a local or CPU model run. Pass the preserved pre-change module through
--legacy-source to additionally compare every historical control with its actual
old implementation, not merely with the independent mathematical reference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import ultralytics.nn.modules.mamba_yolo as m


# Independent specification: residual, energy, importance factors, delta.
CONTROL_SPECS = {
    "matched_identity": (False, False, False, False),
    "matched_scale075": (False, False, False, False),
    "residual_scale075": (True, False, False, False),
    "energy_only": (False, True, False, False),
    "local_energy": (True, True, False, False),
    "local_energy_input": (True, True, True, False),
    "local_energy_delta": (True, True, False, True),
    "full": (True, True, True, True),
    "full_fixed_half": (True, True, True, True),
    "guidance_only": (False, False, True, True),
    "global": (True, True, True, True),
    "full_no_lrc": (False, True, True, True),
    "const_guidance": (True, True, True, True),
    "post_gate": (True, True, True, False),
    "lsgm_trend_only": (True, True, True, True),
    "lsgm_std_only": (True, True, True, True),
    "lsgm_point_only": (True, True, True, True),
}
LEGACY_CONTROLS = tuple(CONTROL_SPECS)[:11]
DESCRIPTORS = {
    "lsgm_trend_only": "trend_only",
    "lsgm_std_only": "std_only",
    "lsgm_point_only": "point_only",
}
CASES = [(name, name, {}) for name in CONTROL_SPECS]
CASES += [
    ("full_trend_k5", "full", {"trend_kernel_size": 5}),
    ("full_trend_k7", "full", {"trend_kernel_size": 7}),
    ("full_stats_k5", "full", {"stats_kernel_size": 5}),
    ("full_stats_k7", "full", {"stats_kernel_size": 7}),
    ("full_independent_k5_k7", "full", {"trend_kernel_size": 5, "stats_kernel_size": 7}),
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_controls(model, optimizer=None):
    """Fail closed on changed controls, buffer registration or optimizer leakage.

    Parameter *registration* is checked separately from requires_grad: stripped
    evaluation checkpoints legitimately mark all Parameters non-trainable. In
    particular const-guidance eta must remain a Parameter, even in such a file.
    """
    optimizer_ids = None if optimizer is None else {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    rows = []
    for name, block in model.named_modules():
        if not isinstance(block, m.DS_VSSBlockSweep) or not hasattr(block, "control"):
            continue
        control = block.control
        require(control in CONTROL_SPECS, f"{name}: unknown control {control}")
        residual, energy, input_path, delta_path = CONTROL_SPECS[control]
        s = block.ds_ssm
        fixed = 0.5 if control in {"full_fixed_half", "const_guidance"} else None
        dynamic = (input_path or delta_path) and fixed is None
        lrc = control != "full_no_lrc"
        descriptor = DESCRIPTORS.get(control, "trend_std")
        require(s.use_pfim == (input_path or delta_path), f"{control}: importance enabled")
        require(s.pfim_on_gamma == input_path and s.pfim_on_write == input_path,
                f"{control}: importance factor switches")
        require(s.pfim_on_delta == delta_path, f"{control}: delta switch")
        require(getattr(s, "fixed_importance", None) == fixed, f"{control}: fixed importance")
        require(getattr(s, "pfim_tokenwise", True) == (control != "global"),
                f"{control}: token/global importance")
        require(getattr(s, "lrc_enabled", True) == lrc, f"{control}: LRC bypass")
        require(getattr(s, "post_gate_mode", False) == (control == "post_gate"),
                f"{control}: post-scan gate switch")
        require(s.min_retain == (1.0 if control == "matched_identity" else 0.5),
                f"{control}: minimum retain")
        require(getattr(s.pfim, "descriptor_mode", "trend_std") == descriptor,
                f"{control}: descriptor")
        require(tuple(s.pfim.mlp[0].normalized_shape) == (2 * s.dim,),
                f"{control}: predictor LayerNorm is not 2C")
        require(s.pfim.mlp[1].in_features == 2 * s.dim, f"{control}: predictor input is not 2C")
        active = {
            "lambda_s": residual,
            "alpha": energy,
            "beta_i": input_path and fixed is None,
            "rho_i": input_path and fixed is None,
            "eta_i": delta_path and control != "full_fixed_half",
        }
        scalars = {}
        for key, enabled in active.items():
            tensor = getattr(s, key)
            require((key in s._parameters) == enabled, f"{control}: {key} parameter registration")
            require((key in s._buffers) == (not enabled), f"{control}: {key} buffer registration")
            if not enabled:
                require(not tensor.requires_grad, f"{control}: disabled {key} has requires_grad")
            scalars[key] = float(tensor.detach())
            require(math.isfinite(scalars[key]), f"{control}: nonfinite {key}")
        if not residual:
            require(scalars["lambda_s"] == 0.0, f"{control}: disabled residual drifted")
        if not energy:
            require(scalars["alpha"] == 0.0, f"{control}: disabled energy drifted")
        dw_active = lrc and (residual or energy or dynamic)
        require(bool(list(s.dw.parameters())) == dw_active, f"{control}: trend parameter registration")
        require(bool(list(s.pfim.parameters())) == dynamic, f"{control}: predictor parameter registration")
        if not dw_active:
            require(not any(t.requires_grad for t in s.dw.buffers()), f"{control}: trend buffer gradients")
        if not dynamic:
            require(not any(t.requires_grad for t in s.pfim.buffers()), f"{control}: predictor buffer gradients")
        options = getattr(block, "options", {"trend_kernel_size": 3, "stats_kernel_size": 3})
        require(s.dw.kernel_size == (options["trend_kernel_size"],), f"{control}: trend kernel")
        require(s.dw.padding == (options["trend_kernel_size"] // 2,), f"{control}: trend padding")
        require(s.pfim.kernel_size == options["stats_kernel_size"], f"{control}: statistics kernel")
        require(s.pfim.pad == options["stats_kernel_size"] // 2, f"{control}: statistics padding")
        if optimizer_ids is not None:
            for buffer_name, tensor in block.named_buffers():
                require(id(tensor) not in optimizer_ids, f"{control}: {buffer_name} entered optimizer")
        rows.append({
            "name": name, "control": control, "options": dict(options), **scalars,
            "buffer_scalars": [key for key, enabled in active.items() if not enabled],
            "parameter_scalars": [key for key, enabled in active.items() if enabled],
            "requires_grad": {key: bool(getattr(s, key).requires_grad) for key in active},
            "lrc_enabled": lrc, "trend_parameters_active": dw_active,
            "predictor_parameters_active": dynamic, "descriptor_mode": descriptor,
            "predictor_input_features": 2 * s.dim, "fixed_importance": fixed,
            "importance_factor_location": "scan_output_before_shortcut" if control == "post_gate" else "scan_input",
            "input_path": input_path and control != "post_gate", "delta_path": delta_path,
            "min_retain": s.min_retain,
        })
    return rows


def _block(control, options, device, seed=401):
    torch.manual_seed(seed)
    return m.DS_VSSBlockSweep(16, 16, 4, control, options).to(device)


def _excite(s):
    """Exercise non-neutral importance; zero-final-layer initialization hides bugs."""
    with torch.no_grad():
        for key, value in (("lambda_s", 0.37), ("alpha", 0.71), ("beta_i", 0.83),
                           ("rho_i", 0.63), ("eta_i", 0.23)):
            if key in s._parameters:
                getattr(s, key).fill_(value)
        if list(s.pfim.parameters()):
            weight = s.pfim.mlp[-1].weight
            weight.copy_(torch.linspace(-0.27, 0.31, weight.numel(), device=weight.device).reshape_as(weight))
            s.pfim.mlp[-1].bias.fill_(0.07)


def _reference(s, x, control, options):
    """Independent equations and recurrence, all on the selected CUDA device."""
    residual, energy_enabled, input_path, delta_path = CONTROL_SPECS[control]
    batch, length, channels = x.shape
    height = width = 4
    require(length == height * width, "Reference probe uses a fixed 4x4 token map")
    trend_kernel = options.get("trend_kernel_size", 3)
    stats_kernel = options.get("stats_kernel_size", 3)
    if control == "full_no_lrc":
        trend = corrected = x
    else:
        trend = F.conv1d(x.transpose(1, 2), s.dw.weight, padding=trend_kernel // 2,
                         groups=channels).transpose(1, 2)
        corrected = x - s.lambda_s * (x - trend)
    descriptor = None
    if (input_path or delta_path) and control not in {"full_fixed_half", "const_guidance"}:
        if control in {"lsgm_trend_only", "lsgm_point_only"}:
            std = torch.zeros_like(corrected)
        else:
            image = corrected.transpose(1, 2).reshape(batch, channels, height, width)
            mean = F.avg_pool2d(image, stats_kernel, 1, stats_kernel // 2, count_include_pad=False)
            mean2 = F.avg_pool2d(image.square(), stats_kernel, 1, stats_kernel // 2, count_include_pad=False)
            std = (torch.clamp(mean2 - mean.square(), min=0) + s.eps).sqrt().flatten(2).transpose(1, 2)
        first = corrected if control == "lsgm_point_only" else trend
        if control == "lsgm_std_only":
            first = torch.zeros_like(first)
        descriptor = torch.cat((first, std), dim=-1)
        importance = s.pfim.mlp(descriptor).sigmoid()
        if control == "global":
            importance = importance.mean(dim=1, keepdim=True).expand(-1, length, -1)
    else:
        importance = torch.full((batch, length, 1), 0.5, dtype=x.dtype, device=x.device)
    drive, z = s.in_proj(corrected).chunk(2, dim=-1)
    delta_logits = s.delta_proj(drive)
    if delta_path:
        delta_logits = delta_logits + s.eta_i * importance
    delta = F.softplus(delta_logits)
    energy = trend.norm(p=2, dim=-1, keepdim=True) / math.sqrt(channels)
    energy_logits = -s.alpha * s.sigma_n * energy
    minimum = 1.0 if control == "matched_identity" else 0.5
    gamma_energy = minimum + (1 - minimum) * energy_logits.sigmoid()
    gamma_logits = energy_logits + s.beta_i * (importance - 0.5) if input_path else energy_logits
    gamma_full = minimum + (1 - minimum) * gamma_logits.sigmoid()
    write = (1 + s.rho_i.tanh() * (importance - 0.5)).clamp(min=0.1) if input_path else 1.0
    ratio = gamma_full * write / gamma_energy if control == "post_gate" else None
    scan_drive = drive * gamma_energy if control == "post_gate" else (drive * write) * gamma_full
    b_tokens, c_tokens = s.B_proj(drive), s.C_proj(drive)
    a = -s.A_log.float().exp()
    state = x.new_zeros(batch, channels, s.state_dim)
    output = []
    for index in range(length):
        dt = delta[:, index].unsqueeze(-1)
        transition = (dt * a.unsqueeze(0)).exp()
        write_state = dt * b_tokens[:, index].unsqueeze(1) * scan_drive[:, index].unsqueeze(-1)
        state = transition * state + write_state
        output.append((c_tokens[:, index].unsqueeze(1) * state).sum(-1))
    scan_output = torch.stack(output, dim=1)
    y_state = scan_output * ratio if ratio is not None else scan_output
    y = s.out_proj((y_state + s.D * corrected) * F.silu(z))
    return y, importance, {
        "descriptor": descriptor, "trend": trend, "corrected": corrected,
        "drive": scan_drive, "delta": delta, "scan_output": scan_output,
        "post_gate_ratio": ratio, "z": z, "energy_enabled": energy_enabled,
    }


def _gradient_comparison(left, right, left_x, right_x, label, exact=False, tolerance=None):
    if tolerance is None:
        tolerance = {"rtol": 0.0, "atol": 0.0} if exact else {"rtol": 4e-3, "atol": 3e-5}
    torch.testing.assert_close(left_x.grad, right_x.grad, msg=label + ": input gradient", **tolerance)
    largest = float((left_x.grad - right_x.grad).abs().max())
    left_parameters, right_parameters = dict(left.named_parameters()), dict(right.named_parameters())
    require(left_parameters.keys() == right_parameters.keys(), label + ": parameter names")
    for name, parameter in left_parameters.items():
        other = right_parameters[name]
        require((parameter.grad is None) == (other.grad is None), f"{label}: gradient presence {name}")
        if parameter.grad is not None:
            require(bool(torch.isfinite(parameter.grad).all()), f"{label}: nonfinite gradient {name}")
            torch.testing.assert_close(parameter.grad, other.grad, msg=f"{label}: {name}", **tolerance)
            largest = max(largest, float((parameter.grad - other.grad).abs().max()))
    return largest


def _load_buffer(buffer, device):
    buffer.seek(0)
    try:
        return torch.load(buffer, map_location=device, weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        buffer.seek(0)
        return torch.load(buffer, map_location=device)


def _roundtrip(block, x, device):
    buffer = io.BytesIO()
    torch.save(block, buffer)
    reloaded = _load_buffer(buffer, device)
    audit_controls(reloaded)
    require(block.state_dict().keys() == reloaded.state_dict().keys(), "Serialized state keys")
    for key, value in block.state_dict().items():
        require(torch.equal(value, reloaded.state_dict()[key]), "Serialized tensor " + key)
    with torch.no_grad():
        require(torch.equal(block(x), reloaded(x)), "Serialized full-module forward changed")
    restored = _block(block.control, block.options, device)
    restored.load_state_dict(block.state_dict(), strict=True)
    audit_controls(restored)
    with torch.no_grad():
        require(torch.equal(block(x), restored(x)), "Strict state_dict reload changed forward")
    # Match the legacy optimizer-strip behavior and verify type-based audit.
    for parameter in reloaded.parameters():
        parameter.requires_grad_(False)
    audit_controls(reloaded)
    return {"full_module": "PASS", "state_dict": "PASS", "stripped_parameters_audit": "PASS"}


def _case_test(name, control, options, device, common_state, common_rng):
    block = _block(control, options, device)
    require(torch.equal(torch.get_rng_state(), common_rng), name + ": constructor changed shared CPU RNG")
    initial_state = block.state_dict()
    require(initial_state.keys() == common_state.keys(), name + ": initial state keys")
    for key, tensor in initial_state.items():
        if key in {"ds_ssm.lambda_s", "ds_ssm.alpha"}:
            continue
        if key == "ds_ssm.dw.weight" and options.get("trend_kernel_size", 3) != 3:
            torch.testing.assert_close(tensor, torch.full_like(tensor, 1 / options["trend_kernel_size"]))
            continue
        require(torch.equal(tensor, common_state[key]), name + ": shared initialization differs at " + key)
    audit_controls(block)
    _excite(block.ds_ssm)
    s = block.ds_ssm
    reference = copy.deepcopy(s)
    fallback = copy.deepcopy(s)
    torch.manual_seed(813)
    x = torch.randn(2, 16, 16, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    x_fallback = x.detach().clone().requires_grad_(True)
    observed = {"trend_calls": 0, "predictor_calls": 0}

    def trend_hook(*unused):
        observed["trend_calls"] += 1

    def descriptor_hook(unused_module, inputs):
        observed["predictor_calls"] += 1
        observed["descriptor"] = inputs[0].detach()

    hooks = [s.dw.register_forward_hook(trend_hook), s.pfim.mlp[0].register_forward_pre_hook(descriptor_hook)]
    original_scan = m.selective_scan_fn

    def capture(*arguments, **keywords):
        result = original_scan(*arguments, **keywords)
        observed["drive"] = arguments[0].detach().transpose(1, 2)
        observed["delta"] = arguments[1].detach().transpose(1, 2)
        observed["scan_output"] = result.detach().transpose(1, 2)
        return result

    m.selective_scan_fn = capture
    try:
        actual, actual_importance = s(x, hw_shape=(4, 4), return_importance=True)
    finally:
        m.selective_scan_fn = original_scan
        for hook in hooks:
            hook.remove()
    expected, expected_importance, components = _reference(reference, x_ref, control, options)
    torch.testing.assert_close(actual_importance, expected_importance, rtol=1e-6, atol=1e-7)
    for key in ("drive", "delta"):
        torch.testing.assert_close(observed[key], components[key], rtol=2e-6, atol=2e-7, msg=name + ": " + key)
    torch.testing.assert_close(observed["scan_output"], components["scan_output"], rtol=1e-3, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-5, msg=name)
    require(observed["trend_calls"] == (0 if control == "full_no_lrc" else 1), name + ": trend bypass")
    if components["descriptor"] is None:
        require(observed["predictor_calls"] == 0, name + ": disabled predictor was executed")
    else:
        require(observed["predictor_calls"] == 1, name + ": predictor call count")
        torch.testing.assert_close(observed["descriptor"], components["descriptor"], rtol=2e-6, atol=2e-7)
        require(float(actual_importance.std()) > 0 if control != "global" else True,
                name + ": non-neutral probe failed to exercise spatial importance")
    original_flag = m.USE_CUDA_SCAN
    m.USE_CUDA_SCAN = False
    try:
        fallback_output = fallback(x_fallback, hw_shape=(4, 4))
    finally:
        m.USE_CUDA_SCAN = original_flag
    torch.testing.assert_close(fallback_output, expected, rtol=1e-3, atol=2e-5, msg=name + ": fallback")
    probe = torch.randn_like(actual)
    for output in (actual, expected, fallback_output):
        ((output * probe).sum() + 0.2 * output.square().mean()).backward()
    grad_max = _gradient_comparison(s, reference, x, x_ref, name)
    fallback_grad_max = _gradient_comparison(fallback, reference, x_fallback, x_ref, name + ": fallback")
    row = audit_controls(block)[0]
    for key in row["parameter_scalars"]:
        gradient = getattr(s, key).grad
        require(gradient is not None and float(gradient.abs()) > 0, name + ": no active gradient for " + key)
    row["scalar_gradient_abs"] = {
        key: float(getattr(s, key).grad.abs()) for key in row["parameter_scalars"]
    }
    if control == "post_gate":
        without_shortcut = copy.deepcopy(s)
        with torch.no_grad():
            without_shortcut.D.zero_()
            output_without_shortcut = without_shortcut(x.detach(), hw_shape=(4, 4))
            shortcut = s.out_proj(s.D * components["corrected"].detach() * F.silu(components["z"].detach()))
        torch.testing.assert_close(actual.detach() - output_without_shortcut, shortcut, rtol=1e-3, atol=2e-5)
        row["ungated_shortcut"] = "PASS"
    # Reproduce the trainer's unconditional re-enabling of Parameters.
    for parameter in block.parameters():
        parameter.requires_grad_(True)
    frozen_buffers = {key: tensor.detach().clone() for key, tensor in block.named_buffers()}
    optimizer = torch.optim.AdamW(block.parameters(), lr=0.003, weight_decay=0.01)
    image = torch.randn(2, 16, 4, 4, device=device)
    eta_before = float(s.eta_i.detach())
    for unused_step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = block(image).square().mean()
        require(bool(torch.isfinite(loss)), name + ": nonfinite optimizer loss")
        loss.backward()
        for parameter in block.parameters():
            if parameter.grad is not None:
                require(bool(torch.isfinite(parameter.grad).all()), name + ": nonfinite optimizer gradient")
        optimizer.step()
        audit_controls(block, optimizer)
        for key, tensor in block.named_buffers():
            require(torch.equal(tensor, frozen_buffers[key]), name + ": inactive buffer drifted: " + key)
    if control == "const_guidance":
        require(s.eta_i.requires_grad and float(s.eta_i.detach()) != eta_before,
                "Const-guidance eta did not train")
        row["eta_trained_after_trainer_unfreeze"] = True
    row.update({
        "case": name, "status": "PASS", "common_initialization": "PASS",
        "output_max_abs": float((actual - expected).detach().abs().max()),
        "gradient_max_abs": grad_max, "fallback_gradient_max_abs": fallback_grad_max,
        "inactive_buffers_after_optimizer": "PASS", "serialized_reload": _roundtrip(block, image, device),
        "parameters": sum(p.numel() for p in block.parameters()),
        "buffer_elements": sum(t.numel() for t in block.buffers()),
    })
    return row


def _legacy_checks(device, legacy_source):
    torch.manual_seed(411)
    default = m.DS_VSSBlockSweep(16, 16, 4).to(device)
    full = _block("full", {}, device, seed=411)
    require(default.state_dict().keys() == full.state_dict().keys(), "Default/full state layout")
    for key, tensor in default.state_dict().items():
        require(torch.equal(tensor, full.state_dict()[key]), "Default/full initialization: " + key)
    x = torch.randn(2, 16, 4, 4, device=device)
    _excite(default.ds_ssm)
    _excite(full.ds_ssm)
    require(torch.equal(default(x), full(x)), "Default/full non-neutral forward")
    missing = copy.deepcopy(full)
    for owner, attribute in ((missing, "options"), (missing.ds_ssm, "lrc_enabled"),
                             (missing.ds_ssm, "post_gate_mode"), (missing.ds_ssm.pfim, "descriptor_mode")):
        delattr(owner, attribute)
    serialized = io.BytesIO()
    torch.save(missing, serialized)
    missing_reloaded = _load_buffer(serialized, device)
    require(torch.equal(full(x), missing_reloaded(x)), "Checkpoint missing new attributes changed behavior")
    audit_controls(missing_reloaded)
    report = {"default_equals_full": "PASS", "missing_new_attributes_reload": "PASS"}
    if legacy_source is None:
        report["old_source_comparison"] = "NOT_RUN: supply --legacy-source preserved_pre_change_mamba_yolo.py"
        return report
    spec = importlib.util.spec_from_file_location("ultralytics.nn.modules._rgsp_legacy_validation", legacy_source)
    require(spec is not None and spec.loader is not None, "Cannot load preserved legacy source")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    rows = []
    for control in (None, *LEGACY_CONTROLS):
        torch.manual_seed(717)
        before = legacy.DS_VSSBlockSweep(16, 16, 4, control).to(device)
        after = _block(control, None, device, seed=717)
        require(before.state_dict().keys() == after.state_dict().keys(), f"Legacy {control}: state layout")
        for key, tensor in before.state_dict().items():
            require(torch.equal(tensor, after.state_dict()[key]), f"Legacy {control}: initialization {key}")
        _excite(before.ds_ssm)
        _excite(after.ds_ssm)
        left_x = x.detach().clone().requires_grad_(True)
        right_x = x.detach().clone().requires_grad_(True)
        left, right = before(left_x), after(right_x)
        require(torch.equal(left, right), f"Legacy {control}: forward changed")
        left.square().sum().backward()
        right.square().sum().backward()
        # CUDA scan backward uses floating-point reductions, so even the old
        # implementation is not necessarily bitwise-repeatable. Measure its
        # own repeatability before allowing bounded FP32 gradient roundoff.
        repeated = copy.deepcopy(before)
        repeated.zero_grad(set_to_none=True)
        repeated_x = x.detach().clone().requires_grad_(True)
        repeated_output = repeated(repeated_x)
        require(torch.equal(left.detach(), repeated_output.detach()),
                f"Legacy {control}: old-source forward is not repeatable")
        repeated_output.square().sum().backward()
        self_roundoff = _gradient_comparison(before, repeated, left_x, repeated_x,
                                             f"Legacy {control}: old-vs-old repeatability")
        require(self_roundoff <= 7.5e-6, f"Legacy {control}: backward noise exceeds the strict audit bound")
        tolerance = {"rtol": 1e-5, "atol": max(2e-7, 4 * self_roundoff)}
        maximum = _gradient_comparison(before, after, left_x, right_x, f"Legacy {control}",
                                       tolerance=tolerance)
        rows.append({"control": control, "forward_bitwise_equal": True, "gradient_max_abs": maximum,
                     "old_vs_old_gradient_max_abs": self_roundoff, "gradient_tolerance": tolerance})
    report.update({"old_source_comparison": "PASS", "source_sha256": sha256(legacy_source), "controls": rows})
    return report


def _parser_checks(device):
    from ultralytics.nn.tasks import parse_model

    rows = []
    next_layer_template = None
    for case, control, options in CASES:
        # The literal wrapper also covers the historical Python keyword global.
        cfg = {"nc": 1, "depth_multiple": 1.0, "width_multiple": 1.0,
               "backbone": [[-1, 1, "DS_VSSBlockSweep", [16, 4, repr(control), options]]],
               "head": [[-1, 1, "Conv", [16, 1, 1]]]}
        torch.manual_seed(102)
        model, unused_save = parse_model(copy.deepcopy(cfg), ch=16, verbose=False)
        current = model[1].state_dict()
        if next_layer_template is None:
            next_layer_template = {key: tensor.clone() for key, tensor in current.items()}
        for key, tensor in current.items():
            require(torch.equal(tensor, next_layer_template[key]), case + ": downstream layer RNG changed")
        require(model[0].ds_ssm.state_dim == 4, case + ": parser changed state size")
        audit_controls(model)
        model = model.to(device)
        with torch.no_grad():
            output = model(torch.randn(2, 16, 4, 4, device=device))
        require(output.shape == (2, 16, 4, 4) and bool(torch.isfinite(output).all()), case + ": parser forward")
        rows.append({"case": case, "parser": "PASS", "downstream_initialization": "PASS"})
    rejected = []
    for options in ({"unknown": 3}, {"trend_kernel_size": 4}, {"stats_kernel_size": 0},
                    {"stats_kernel_size": True}, {"stats_kernel_size": "5"}, []):
        try:
            m.DS_VSSBlockSweep(16, 16, 4, "full", options)
        except (ValueError, TypeError):
            rejected.append(options)
        else:
            raise AssertionError(f"Invalid options were accepted: {options!r}")
    return {"cases": rows, "invalid_options_rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace validation evidence: {args.output}")
    if args.legacy_source is not None and not args.legacy_source.is_file():
        raise FileNotFoundError(args.legacy_source)
    require(torch.cuda.is_available(), "CUDA required; no local/CPU fallback")
    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(m.USE_CUDA_SCAN and m.selective_scan_fn is not None, "Selective scan backend required")
    if m.selective_scan_fn.__module__ == m.__name__:
        require("selective_scan_cuda_core" in m.__dict__, "Compiled local CUDA scan backend missing")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device), "cuda_device": args.cuda_device,
        "backend": m.selective_scan_fn.__module__, "source_sha256": sha256(Path(m.__file__)),
        "validator_sha256": sha256(Path(__file__)), "control_cases": len(CASES), "controls": [],
    }
    try:
        report["legacy_equivalence"] = _legacy_checks(device, args.legacy_source)
        template = _block("full", {}, device)
        common_state = {key: tensor.detach().clone() for key, tensor in template.state_dict().items()}
        common_rng = torch.get_rng_state().clone()
        for name, control, options in CASES:
            report["controls"].append(_case_test(name, control, options, device, common_state, common_rng))
            print(f"PASS {name}", flush=True)
        report["parser"] = _parser_checks(device)
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": report.get("status", "FAIL"), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
