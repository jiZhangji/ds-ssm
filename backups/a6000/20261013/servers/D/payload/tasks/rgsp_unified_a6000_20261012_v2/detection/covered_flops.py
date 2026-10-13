"""Isolated, batch-one FP32 covered-operation profiling of an unfused model.

Call only on an approved, available server, serialized with training/inference in
the calling process. The caller owns device selection and resource admission.
This module never moves a CPU model to CUDA, installs a dependency, or fuses a
model. A CUDA deepcopy is profiled; the supplied model is never forwarded or put
in eval mode. Explicit ``inputs=(tensor,)`` or ``inputs=(before, after)`` also
supports classification/change detection. Inputs are cloned before forwarding.
Temporary global wrappers make concurrent forwards unsafe.

The counting contract matches the 2026-10-12 ``profile_*_scan_run1.py`` ledger:
legacy THOP counters times two, plus reviewed einsum/scan/CrossMerge arithmetic.
It is deliberately NOT a complete FLOP, hardware-instruction, or latency count.
Missing dependencies and failed coverage checks return a null total, not zero.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import importlib
import math
import random
import sys
import threading


_PROFILE_LOCK = threading.Lock()
_MISSING = object()
_UNCOUNTED = (
    "Scan exponentials and softplus special functions; symbolic counts are separate.",
    "Other functional arithmetic, activations, statistics, interpolation and dense-head decoding not counted by THOP.",
    "DS-SSM external softplus, energy/gamma/write/z gates and D*x residual; PFIM functional pooling/statistics/gates.",
    "CUDA prefix-scan overhead, memory/address operations, preprocessing, NMS and postprocessing.",
)


class _CoverageError(RuntimeError):
    """The requested counting contract cannot be certified for this forward."""


def _type_name(module):
    cls = type(module)
    return cls.__module__ + "." + cls.__name__


def _kind(module):
    names = {cls.__name__ for cls in type(module).__mro__}
    if "SS2D" in names:
        return "SS2D"
    if "DeSpeckleSSM" in names:
        return "DeSpeckleSSM"
    if "SemanticSSM" in names:
        return "SemanticSSM"
    return None


def _argument(args, kwargs, name, index, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


@contextmanager
def _preserve_runtime(torch, device):
    """Restore RNGs and backend flags, including when deepcopy/import/forward fails."""
    python_rng = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_rng = numpy.random.get_state() if numpy is not None else None
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        # Only touch the already-initialized device holding the input model.
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index], enabled=True):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
        random.setstate(python_rng)
        if numpy_rng is not None:
            numpy.random.set_state(numpy_rng)


@contextmanager
def _preserve_modules(model):
    """THOP does not clean every module, or clean hooks after every failure."""
    hook_names = (
        "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
        "_backward_pre_hooks", "_forward_hooks_with_kwargs",
        "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called",
    )
    snapshots = []
    for module in model.modules():
        snapshots.append((
            module, module.training, dict(module._buffers),
            set(module._non_persistent_buffers_set),
            {name: dict(getattr(module, name)) for name in hook_names if hasattr(module, name)},
            {name: module.__dict__.get(name, _MISSING) for name in ("total_ops", "total_params")},
        ))
    try:
        yield
    finally:
        for module, training, buffers, nonpersistent, hooks, attributes in snapshots:
            module.training = training  # Preserve mixed child train/eval states.
            module._buffers.clear()
            module._buffers.update(buffers)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(nonpersistent)
            for name, original in hooks.items():
                getattr(module, name).clear()
                getattr(module, name).update(original)
            for name, original in attributes.items():
                if original is _MISSING:
                    module.__dict__.pop(name, None)
                else:
                    module.__dict__[name] = original


class FunctionalLedger:
    """Capture actual reviewed calls; certify each SSM module independently."""

    def __init__(self, torch, model):
        self.torch = torch
        self.model = model
        self.scan_calls = []
        self.einsum_calls = []
        self.cross_merge_calls = []
        self.module_calls = []
        self._restore = []
        self._active = []
        self.expected_modules = {name: _kind(module) for name, module in model.named_modules() if _kind(module)}

    def _patch(self, obj, name, value):
        original = vars(obj).get(name, _MISSING)

        def restore():
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)

        self._restore.append(restore)
        setattr(obj, name, value)

    def _owner(self):
        return self._active[-1]["name"] if self._active else None

    def __enter__(self):
        try:
            self._patch_einsum()
            namespaces = {}
            for name, module in self.model.named_modules():
                kind = _kind(module)
                if kind is None:
                    continue
                namespace = sys.modules.get(type(module).__module__)
                if namespace is None:
                    raise _CoverageError("SSM module namespace is unavailable: " + _type_name(module))
                namespaces[id(namespace)] = namespace
                for symbol in ("cross_selective_scan", "selective_scan_fn"):
                    function = getattr(namespace, symbol, None)
                    owner = sys.modules.get(getattr(function, "__module__", ""))
                    if owner is not None:
                        namespaces[id(owner)] = owner
                if kind != "SS2D" and not getattr(namespace, "USE_CUDA_SCAN", False):
                    raise _CoverageError("Python-loop SSM fallback is outside this CUDA profile: " + name)
                self._audit_module(name, module, kind)

            scan_classes, merge_classes = {}, {}
            for namespace in namespaces.values():
                for symbol, backend in (("SelectiveScanCore", "core"), ("SelectiveScanFn", "mamba")):
                    cls = getattr(namespace, symbol, None)
                    if isinstance(cls, type):
                        scan_classes[id(cls)] = (cls, backend)
                cls = getattr(namespace, "CrossMerge", None)
                if isinstance(cls, type):
                    merge_classes[id(cls)] = cls
                # Never silently execute the very expensive Python reference path.
                if callable(getattr(namespace, "selective_scan_reference", None)):
                    self._patch(namespace, "selective_scan_reference", self._reject_reference)
            if self.expected_modules and not scan_classes:
                raise _CoverageError("No reviewed selective-scan entry point found for this model")
            for cls, backend in scan_classes.values():
                self._patch_scan(cls, backend)
            for cls in merge_classes.values():
                self._patch_merge(cls)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *_):
        while self._restore:
            self._restore.pop()()

    @staticmethod
    def _reject_reference(*args, **kwargs):
        raise ImportError("Compiled CUDA selective scan is unavailable; Python reference fallback refused")

    def _audit_module(self, name, module, kind):
        def before(_module, _inputs):
            self._active.append({"name": name, "kind": kind, "start": self._counts()})

        def after(_module, _inputs, _output):
            if not self._active or self._active[-1]["name"] != name:
                raise _CoverageError("Unexpected nested SSM module execution")
            current = self._active.pop()
            observed = tuple(end - start for end, start in zip(self._counts(), current["start"]))
            expected = (1, 2, 1) if kind == "SS2D" else (1, 0, 0)
            self.module_calls.append({"module": name, "kind": kind, "scan_calls": observed[0],
                                      "einsum_calls": observed[1], "cross_merge_calls": observed[2]})
            if observed != expected:
                raise _CoverageError(f"Incomplete {kind} capture at {name}: scan/einsum/merge={observed}, expected={expected}")

        self._restore.append(module.register_forward_pre_hook(before).remove)
        self._restore.append(module.register_forward_hook(after).remove)

    def _counts(self):
        return len(self.scan_calls), len(self.einsum_calls), len(self.cross_merge_calls)

    def _patch_einsum(self):
        original = self.torch.einsum

        def einsum(equation, *operands):
            if not isinstance(equation, str):
                raise _CoverageError("Einsum sublist syntax is not covered")
            tensors = operands[0] if len(operands) == 1 and isinstance(operands[0], (tuple, list)) else operands
            compact = "".join(equation.split())
            if compact.count("->") != 1 or "." in compact:
                raise _CoverageError("Unreviewed einsum: " + equation)
            left, right = compact.split("->")
            terms = left.split(",")
            if len(terms) != 2 or len(tensors) != 2 or len(set(right)) != len(right):
                raise _CoverageError("Only explicit binary dense contractions are covered: " + equation)
            dimensions = {}
            for term, tensor in zip(terms, tensors):
                if not self.torch.is_tensor(tensor) or len(term) != tensor.ndim or len(set(term)) != len(term):
                    raise _CoverageError("Unreviewed einsum shape/diagonal: " + equation)
                if tensor.dtype != self.torch.float32:
                    raise _CoverageError("Einsum must remain FP32")
                for letter, size in zip(term, tensor.shape):
                    if letter in dimensions and dimensions[letter] != int(size):
                        raise _CoverageError("Einsum broadcasting is outside this counter")
                    dimensions[letter] = int(size)
            reduction = set(dimensions) - set(right)
            if not set(right) <= set(dimensions) or not reduction or not reduction <= (set(terms[0]) & set(terms[1])):
                raise _CoverageError("Unreviewed non-contraction einsum: " + equation)
            flops = 2 * math.prod(dimensions.values())
            output = original(equation, *operands)
            self.einsum_calls.append({"module": self._owner(), "equation": equation,
                                      "shapes": [list(t.shape) for t in tensors], "flops": flops})
            return output

        self._patch(self.torch, "einsum", einsum)

    def _patch_scan(self, cls, backend):
        original = cls.forward

        def scan(ctx, *args, **kwargs):
            u, delta, a, b, c = [_argument(args, kwargs, name, index) for index, name in enumerate(("u", "delta", "A", "B", "C"))]
            if any(not self.torch.is_tensor(t) for t in (u, delta, a, b, c)):
                raise _CoverageError("Selective scan requires tensor u/delta/A/B/C")
            if u.ndim != 3 or a.ndim != 2 or tuple(delta.shape) != tuple(u.shape):
                raise _CoverageError("Unexpected selective-scan input shapes")
            batch, channels, length = map(int, u.shape)
            states = int(a.shape[-1])
            if int(a.shape[0]) != channels or states < 1:
                raise _CoverageError("Selective-scan state/channel mismatch")
            for tensor in (u, delta, a, b, c):
                if tensor.dtype != self.torch.float32 or tensor.device != u.device:
                    raise _CoverageError("Only real FP32 same-device scans are covered")
            for tensor in (b, c):
                if tensor.ndim not in (3, 4) or tuple(tensor.shape[-2:]) != (states, length) or int(tensor.shape[0]) != batch:
                    raise _CoverageError("Only variable-B/C selective scans are covered")
                if tensor.ndim == 4 and (int(tensor.shape[1]) < 1 or channels % int(tensor.shape[1])):
                    raise _CoverageError("Invalid variable-B/C scan groups")
            d = _argument(args, kwargs, "D", 5)
            z = _argument(args, kwargs, "z", 6) if backend == "mamba" else None
            bias_index, softplus_index = (7, 8) if backend == "mamba" else (6, 7)
            bias = _argument(args, kwargs, "delta_bias", bias_index)
            softplus = bool(_argument(args, kwargs, "delta_softplus", softplus_index, False))
            if z is not None:
                raise _CoverageError("Internal selective-scan z gating is not covered")
            for tensor in (d, bias):
                if tensor is not None and (not self.torch.is_tensor(tensor) or tuple(tensor.shape) != (channels,)):
                    raise _CoverageError("Unexpected selective-scan D/delta_bias shape")
            elements = batch * channels * length
            # (delta*u):1; delta*A:N; B*(delta*u):N; a*h+b:2N;
            # dot(C,h):2N-1 => 6N factored basic arithmetic per token/channel.
            arithmetic = elements * (6 * states + 2 * int(d is not None) + int(bias is not None) + int(softplus))
            special = elements * (states + 2 * int(softplus))
            output = original(ctx, *args, **kwargs)
            self.scan_calls.append({"module": self._owner(), "backend": backend,
                                    "u_shape": list(u.shape), "a_shape": list(a.shape),
                                    "b_shape": list(b.shape), "c_shape": list(c.shape),
                                    "has_D": d is not None, "has_delta_bias": bias is not None,
                                    "delta_softplus": softplus, "basic_arithmetic_flops": arithmetic,
                                    "special_function_evaluations_symbolic_upper_bound": special})
            return output

        self._patch(cls, "forward", staticmethod(scan))

    def _patch_merge(self, cls):
        original = cls.forward

        def merge(ctx, value):
            if value.ndim != 5 or int(value.shape[1]) != 4:
                raise _CoverageError("Only four-direction CrossMerge is covered")
            batch, _directions, channels, height, width = map(int, value.shape)
            output = original(ctx, value)
            self.cross_merge_calls.append({"module": self._owner(), "shape": list(value.shape),
                                          "flops": 3 * batch * channels * height * width})
            return output

        self._patch(cls, "forward", staticmethod(merge))

    def verify(self):
        seen = {row["module"] for row in self.module_calls}
        missing = sorted(set(self.expected_modules) - seen)
        if self._active or missing:
            raise _CoverageError("SSM module forward coverage incomplete: " + repr(missing))
        if not self.expected_modules and (self.scan_calls or self.cross_merge_calls):
            raise _CoverageError("Selective scan was called outside a reviewed SSM module")


def _validate_model(torch, model, detection):
    if not isinstance(model, torch.nn.Module):
        raise _CoverageError("Pass the raw unfused torch detector, not a predictor/result wrapper")
    tensors = list(model.parameters()) + list(model.buffers())
    parameters = list(model.parameters())
    if not parameters:
        raise _CoverageError("A detector with registered parameters is required")
    device = parameters[0].device
    if device.type != "cuda" or device.index is None:
        raise _CoverageError("A model already on one approved CUDA device is required; no CPU fallback")
    if any(t.device != device for t in tensors):
        raise _CoverageError("All parameters and buffers must be on the same CUDA device")
    if any((t.is_floating_point() and t.dtype != torch.float32) or t.is_complex() for t in tensors):
        raise _CoverageError("All floating parameters and buffers must already be FP32")
    modules = list(model.modules())
    heads = [m for m in modules if type(m).__name__ == "Detect"]
    if detection and (len(heads) != 1 or getattr(heads[0], "end2end", False)):
        raise _CoverageError("Exactly one standard dense Detect head is required")
    if any(type(m).__name__ in ("NMS", "AutoShape", "DataParallel", "DistributedDataParallel") for m in modules):
        raise _CoverageError("Pass an unwrapped dense single-device detector without NMS")
    if detection and not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in modules):
        raise _CoverageError("No BatchNorm modules found; unfused detector contract cannot be established")
    if any(getattr(getattr(m, "forward", None), "__name__", "") in ("forward_fuse", "fuseforward") for m in modules):
        raise _CoverageError("Fused detector found; provide the original unfused model")
    # Deepcopy does not isolate closures in existing hooks. Do not invoke them.
    if any(m._forward_hooks or m._forward_pre_hooks for m in modules):
        raise _CoverageError("Existing forward hooks must be removed by the caller before isolated profiling")
    hook_module = torch.nn.modules.module
    if getattr(hook_module, "_global_forward_hooks", {}) or getattr(hook_module, "_global_forward_pre_hooks", {}):
        raise _CoverageError("Global forward hooks prevent isolated profiling")
    return device


def _output_tensors(torch, value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _output_tensors(torch, item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _output_tensors(torch, item)]
    return []


def profile_model(model, imgsz=640, inputs=None) -> dict:
    """Return a JSON-safe profile; non-PASS results always have a null total.

    Dependencies are imported lazily. The caller must check ``status == 'PASS'``
    before using ``covered_flops_g``. PASS certifies the stated partial coverage,
    not complete FLOPs. Extra CUDA memory for a deepcopy and one forward is needed.
    Without ``inputs``, use a raw dense detector and ``(1, 3, imgsz, imgsz)``.
    With ``inputs``, pass a nonempty tuple of batch-one CUDA FP32 tensors for the
    complete registered classification/CD/task model. Shared parameters count
    once; repeated encoder forwards count once per actual call. ``imgsz`` is then
    ignored. No weights, BN buffers, gradients, module modes, input values or RNG
    state are changed. Independent training/inference threads must be inactive.
    """
    coverage = {
        "version": "rgsp-covered-operations-v1",
        "scope": "Full unfused registered model; eval, batch=1, FP32, no TF32, no task postprocessing/NMS.",
        "exhaustive": False,
        "thop": "2 * existing THOP counters (legacy convention, including THOP normalization approximations).",
        "einsum": "2 * product(unique index dimensions) for real FP32 binary dense contractions, MAC=2.",
        "scan": "6*B*D*L*N factored variable-B/C recurrence; +2BDL for skip D*u; +BDL for delta bias; +BDL for softplus addition.",
        "cross_merge": "3*B*D*H*W additions for four-direction merging; CrossScan movement has no arithmetic charge.",
        "not_included": list(_UNCOUNTED),
        "interpretation": "Covered-operation estimate, not total FLOPs or hardware work; not directly interchangeable with THOP-only reports.",
    }
    result = {"status": "FAILED", "parameters": None, "trainable_parameters": None,
              "covered_flops_g": None, "coverage": coverage, "unsupported": list(_UNCOUNTED),
              "thop_legacy_x2_flops": None, "scan_basic_arithmetic_flops": None,
              "functional_einsum_mac2_flops": None, "cross_merge_add_flops": None,
              "scan_calls": [], "einsum_calls": [], "cross_merge_calls": [], "ssm_module_calls": [],
              "unregistered_leaf_module_types": [], "finite_forward": None}
    if not _PROFILE_LOCK.acquire(blocking=False):
        result["error"] = {"type": "ConcurrentProfile", "message": "A profile is already active in this process"}
        return result
    try:
        import torch

        result["parameters"] = int(sum(p.numel() for p in model.parameters()))
        result["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        detection = inputs is None
        device = _validate_model(torch, model, detection=detection)
        if detection:
            size = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
            if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
                raise _CoverageError("imgsz must be a positive integer or pair of positive integers")
            input_shapes = [[1, 3, *size]]
            coverage["scope"] = "Full unfused detector including dense head; eval, batch=1, FP32, no TF32, no NMS."
        else:
            if not isinstance(inputs, tuple) or not inputs:
                raise _CoverageError("Explicit inputs must be a nonempty tuple of CUDA FP32 tensors")
            for tensor in inputs:
                if not torch.is_tensor(tensor) or tensor.ndim == 0 or int(tensor.shape[0]) != 1:
                    raise _CoverageError("Each explicit input must be a batch-one tensor")
                if tensor.dtype != torch.float32 or tensor.device != device:
                    raise _CoverageError("Explicit inputs must already be FP32 on the model's CUDA device")
            input_shapes = [list(t.shape) for t in inputs]
            coverage["parameter_sharing"] = "Registered shared parameters counted once; repeated module forwards counted per actual execution."
        result.update({"input_shapes": input_shapes, "device": str(device), "torch_version": str(torch.__version__),
                       "batch": 1, "precision": "FP32", "fused": False, "nms_included": False})
        with _preserve_runtime(torch, device):
            thop = importlib.import_module("thop")
            registry = importlib.import_module("thop.profile").register_hooks
            result["thop_version"] = str(getattr(thop, "__version__", "unreported"))
            probe = deepcopy(model)
            with _preserve_modules(probe):
                probe.eval()
                for module in probe.modules():
                    if detection and type(module).__name__ == "Detect":
                        module.export = False
                    if _kind(module) and type(module) in registry and registry[type(module)].__name__ != "zero_ops":
                        raise _CoverageError("A THOP SSM hook would double-count the functional ledger")
                unknown = sorted({_type_name(m) for m in probe.modules() if not list(m.children()) and type(m) not in registry})
                result["unregistered_leaf_module_types"] = unknown
                result["unsupported"].extend("Unregistered THOP leaf: " + name for name in unknown)
                outputs = []
                probe.register_forward_hook(lambda _module, _inputs, output: outputs.append(output))
                profile_inputs = ((torch.zeros(1, 3, *size, dtype=torch.float32, device=device),) if detection
                                  else tuple(tensor.detach().clone() for tensor in inputs))
                with FunctionalLedger(torch, probe) as ledger:
                    macs, _unused_thop_parameters = thop.profile(probe, inputs=profile_inputs, verbose=False)
                    ledger.verify()
                if len(outputs) != 1:
                    raise _CoverageError("THOP did not perform exactly one complete model forward")
                output = outputs[0]
                if detection:
                    dense = output[0] if isinstance(output, (tuple, list)) else output
                    if not torch.is_tensor(dense) or dense.ndim != 3 or int(dense.shape[0]) != 1 or dense.dtype != torch.float32:
                        raise _CoverageError("Expected batch-one FP32 decoded dense detector output")
                    result["dense_output_shape"] = list(dense.shape)
                    checked_outputs = [dense]
                else:
                    checked_outputs = [tensor for tensor in _output_tensors(torch, output) if tensor.is_floating_point()]
                    if not checked_outputs or any(tensor.dtype != torch.float32 for tensor in checked_outputs):
                        raise _CoverageError("Expected FP32 task logits/output tensors")
                result["output_shapes"] = [list(tensor.shape) for tensor in checked_outputs]
                if len(checked_outputs) == 1:
                    result["output_shape"] = list(checked_outputs[0].shape)
                result["finite_forward"] = all(bool(torch.isfinite(tensor).all().item()) for tensor in checked_outputs)
                if not result["finite_forward"]:
                    raise _CoverageError("Non-finite model output")
                base = 2.0 * float(macs)
                scan = sum(call["basic_arithmetic_flops"] for call in ledger.scan_calls)
                einsum = sum(call["flops"] for call in ledger.einsum_calls)
                merge = sum(call["flops"] for call in ledger.cross_merge_calls)
                total = base + scan + einsum + merge
                if not math.isfinite(total) or total <= 0 or base <= 0:
                    raise _CoverageError("Invalid or empty THOP/covered operation count")
                result.update({"thop_legacy_x2_flops": base, "scan_basic_arithmetic_flops": scan,
                               "functional_einsum_mac2_flops": einsum, "cross_merge_add_flops": merge,
                               "covered_flops_g": total / 1e9, "scan_calls": ledger.scan_calls,
                               "einsum_calls": ledger.einsum_calls, "cross_merge_calls": ledger.cross_merge_calls,
                               "ssm_module_calls": ledger.module_calls})
        result["status"] = "PASS"
    except ImportError as exc:
        result["status"] = "MISSING_DEPENDENCY"
        result["covered_flops_g"] = None
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "dependency": getattr(exc, "name", None)}
    except Exception as exc:
        result["covered_flops_g"] = None
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, _CoverageError):
            result["unsupported"].append(str(exc))
    finally:
        _PROFILE_LOCK.release()
    return result
