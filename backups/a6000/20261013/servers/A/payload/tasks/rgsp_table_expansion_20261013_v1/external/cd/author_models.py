"""Hash-locked author definitions, without executing notebook workflow cells.

This module is for the registered A6000 experiment environment. Importing it
does not build a model, preprocess an image, download anything or start training.
The original files stay unchanged. Every transformed AST is hashed at build time.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "source_registry.json").read_text(encoding="utf-8"))
RUNNABLE = ("ddnet", "camixer", "lantnet", "wbanet", "wasnet")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_method_spec(method):
    return copy.deepcopy(REGISTRY["methods"][method.lower()])


def _source(relative):
    path = ROOT / relative
    expected = REGISTRY["source_files"][relative]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Author source SHA mismatch: {relative}: {actual}")
    return path.read_text(encoding="utf-8-sig")


class _PrecisionOnly(ast.NodeTransformer):
    """Change explicit FP16 casts only. Never repair an author gradient path."""
    def __init__(self):
        self.changes = 0

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "float16":
            node.attr = "float32"
            self.changes += 1
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "half":
            if node.args or node.keywords:
                raise RuntimeError("Unexpected half() signature; source requires review")
            node.func.attr = "float"
            self.changes += 1
        return node


class _LantHookOnce(ast.NodeTransformer):
    """Deduplicate the binary diagonal mask: mask**n == mask for n >= 1."""
    def __init__(self):
        self.changes = 0

    def visit_Expr(self, node):
        self.generic_visit(node)
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return node
        owner = call.func.value
        if (call.func.attr == "register_hook" and isinstance(owner, ast.Attribute)
                and owner.attr == "weight" and isinstance(owner.value, ast.Name)
                and owner.value.id == "self"):
            self.changes += 1
            test = ast.UnaryOp(op=ast.Not(), operand=ast.Call(
                func=ast.Name(id="getattr", ctx=ast.Load()),
                args=[ast.Name(id="self", ctx=ast.Load()),
                      ast.Constant(value="_cd_diagonal_hook_registered"), ast.Constant(value=False)],
                keywords=[]))
            mark = ast.Assign(targets=[ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="_cd_diagonal_hook_registered", ctx=ast.Store())],
                value=ast.Constant(value=True))
            return ast.copy_location(ast.If(test=test, body=[node, mark], orelse=[]), node)
        return node


def _definitions(namespace, relative, selected, names, audit, *, fp32=False, hook_once=False):
    text = _source(relative)
    if relative.endswith(".ipynb"):
        cells = json.loads(text)["cells"]
        sources = [(f"{relative}:cell{index}", "".join(cells[index]["source"])) for index in selected]
    else:
        sources = [(relative, text)]
    nodes, found, cell_hashes = [], set(), {}
    for location, source in sources:
        cell_hashes[location] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        parsed = ast.parse(source, filename=location)
        for node in parsed.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                if node.name in found:
                    raise RuntimeError(f"Duplicate requested definition {node.name}")
                nodes.append(node)
                found.add(node.name)
    if found != set(names):
        raise RuntimeError(f"Missing author definitions: {set(names) - found}")
    module = ast.Module(body=nodes, type_ignores=[])
    precision = _PrecisionOnly()
    dedup = _LantHookOnce()
    if fp32:
        module = precision.visit(module)
    if hook_once:
        module = dedup.visit(module)
        if dedup.changes != 1:
            raise RuntimeError("LANTNet hook adaptation must match exactly one source statement")
    ast.fix_missing_locations(module)
    ast_hash = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    audit.append({"source": relative, "source_sha256": REGISTRY["source_files"][relative],
                  "cell_source_sha256": cell_hashes, "definitions": sorted(names),
                  "transformed_ast_sha256": ast_hash, "precision_cast_rewrites": precision.changes,
                  "hook_once_rewrites": dedup.changes})
    exec(compile(module, str(ROOT / relative), "exec"), namespace)


def _namespace(method, device):
    return {"__name__": f"cd_author_{method}", "np": np, "torch": torch,
            "nn": nn, "F": F, "math": math, "device": device,
            "patch_size": get_method_spec(method)["patch_size"]}


def build_model(method, device="cuda:0", *, allow_fp32_adaptation=False,
                deduplicate_lant_hooks=True):
    method = method.lower()
    if method not in RUNNABLE:
        raise RuntimeError(f"No author adapter for {method}: {get_method_spec(method)['status']}")
    device = torch.device(device)
    # LANTNet explicitly addresses cuda:0. Bind a physical GPU through
    # CUDA_VISIBLE_DEVICES at process launch, not by silently rewriting its code.
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Use logical cuda:0 on a registered server; bind the physical GPU before launch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no local/CPU fallback is permitted")
    torch.cuda.set_device(0)
    spec = get_method_spec(method)
    namespace = _namespace(method, device)
    audit = []
    if method == "ddnet":
        namespace["cv2"] = importlib.import_module("cv2")
        _definitions(namespace, spec["notebook"], [5, 6], ["MRC", "DCT", "DDNet"], audit)
        model = namespace["DDNet"]()
    elif method == "lantnet":
        _definitions(namespace, spec["notebook"], [5, 7], ["LayerAttention", "LANTNet"],
                     audit, hook_once=deduplicate_lant_hooks)
        model = namespace["LANTNet"]()
    elif method == "camixer":
        from einops import rearrange, repeat
        from einops.layers.torch import Rearrange
        namespace.update(rearrange=rearrange, repeat=repeat, Rearrange=Rearrange)
        _definitions(namespace, spec["notebook"], [7, 8],
                     ["pair", "Attention", "ViT", "shift", "CSC", "SGU", "CAMixer"], audit)
        namespace["vit"] = namespace["ViT"](image_size=9, patch_size=3,
            num_classes=2, dim=27, depth=6, heads=4, mlp_dim=32).to(device)
        model = namespace["CAMixer"]()
    else:
        if not allow_fp32_adaptation:
            raise RuntimeError("Wavelet author sources force FP16; explicit dtype-only adaptation required")
        from einops.layers.torch import Rearrange
        namespace.update(Rearrange=Rearrange, pywt=importlib.import_module("pywt"),
                         Function=torch.autograd.Function,
                         trunc_normal_=importlib.import_module("timm.models.layers").trunc_normal_)
        wave_names = ["DWT_Function", "IDWT_Function", "DWT_2D", "IDWT_2D"]
        if method == "wbanet":
            _definitions(namespace, "vendor/wbanet/torch_wavelets.py", None, wave_names, audit, fp32=True)
            _definitions(namespace, spec["notebook"], [3, 6, 7, 8],
                         ["pair", "WaveAttention", "WSM", "BiAttn", "ConcactFeature", "WBANet"],
                         audit, fp32=True)
        else:
            _definitions(namespace, spec["notebook"], [5, 8, 12, 13, 14],
                         ["pair", *wave_names, "WaveAttention", "WSM", "SEBlock", "MHSA",
                          "BiAttn", "ConcactFeature", "WASNet"], audit, fp32=True)
        expected_casts = 7 if method == "wbanet" else 1
        if sum(item["precision_cast_rewrites"] for item in audit) != expected_casts:
            raise RuntimeError("Unexpected precision adaptation count; review author source")
        namespace["wsm"] = namespace["WSM"](image_size=8, patch_size=2,
            num_classes=2, dim=12, depth=6, heads=4, mlp_dim=32).to(device)
        model = namespace["WBANet" if method == "wbanet" else "WASNet"]()
    model = model.to(device=device, dtype=torch.float32)
    model._cd_source_audit = {"method": method, "repository": spec["repository"],
        "commit": spec["commit"], "definitions": audit,
        "adapter_sha256": sha256(__file__), "registry_sha256": sha256(ROOT / "source_registry.json"),
        "author_source_unmodified": True, "runtime_equivalence": "PENDING_SERVER_VALIDATION"}
    return model


def extract_logits(method, output):
    logits = output[-1] if method.lower() == "camixer" else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        raise RuntimeError(f"Invalid author logits for {method}")
    return logits


def build_criterion(method, device="cuda:0"):
    method = method.lower()
    if method in ("ddnet", "camixer", "wbanet"):
        return nn.CrossEntropyLoss().to(device)
    spec = get_method_spec(method)
    namespace = _namespace(method, torch.device(device))
    audit = []
    if method == "lantnet":
        _definitions(namespace, spec["notebook"], [6], ["Loss"], audit)
        criterion = namespace["Loss"]().to(device)
        criterion._cd_source_audit = audit
        return criterion
    if method == "wasnet":
        _definitions(namespace, spec["notebook"], [15], ["mse_kl_loss"], audit)
        namespace["mse_kl_loss"]._cd_source_audit = audit
        return namespace["mse_kl_loss"]
    raise RuntimeError(f"No author loss for {method}")


def native_preclassification(method, im1, im2):
    """Only two image arrays are accepted. There is intentionally no GT argument."""
    from scipy import signal
    from scipy.linalg import norm
    from scipy.spatial.distance import cdist
    method = method.lower()
    first = np.asarray(im1, dtype=np.float32)
    second = np.asarray(im2, dtype=np.float32)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("Expected equally shaped grayscale time-1/time-2 arrays")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Non-finite input image")
    namespace = {"__name__": "cd_author_preclassification", "np": np, "torch": torch,
                 "nn": nn, "signal": signal, "norm": norm, "cdist": cdist}
    audit = []
    names = ["del2", "srad", "dicomp", "hcluster", "FCM"]
    if method == "wasnet":
        _definitions(namespace, get_method_spec(method)["notebook"], [6, 7], names, audit)
    else:
        relative = "vendor/wbanet/preclassify.py" if method == "wbanet" else "vendor/camixer/preclassify.py"
        _definitions(namespace, relative, None, names, audit)
    difference = namespace["dicomp"](first.copy(), second.copy())
    if not np.isfinite(difference).all():
        raise RuntimeError("Native SRAD yielded non-finite DI. Source retained; no silent clipping/substitution")
    labels = namespace["hcluster"](difference.reshape(-1, 1), difference)
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({1.0, 1.5, 2.0}):
        raise RuntimeError("Invalid native HFCM mask; fail closed without GT assistance")
    features = np.stack([first, second, difference.astype(np.float32)], axis=-1)
    return features, labels.astype(np.float32), {"source": audit, "uses_ground_truth": False,
        "transductive": True, "intensity_scaling": "original_scale_no_divide255",
        "label_counts": {str(value): int((labels == value).sum()) for value in (1.0, 1.5, 2.0)}}
