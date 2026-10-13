"""Auditable, scratch-only classification models for the unified campaign.

Importing this module does not import torch/torchvision, construct a network,
download weights, or run any computation.  The caller owns preprocessing,
device/precision, loss, optimizer, scheduling, and model selection.

Inputs: N x 3 x input_size x input_size; SAR grayscale is replicated into RGB
by the shared data pipeline.  Outputs: one N x num_classes logits tensor.
Only exact model IDs are accepted.  Ambiguous literature labels (e.g. VGGNet,
ViT, ERM) deliberately are not aliases for a guessed architecture.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from torch import nn


_BUILDERS = {
    "alexnet": "alexnet",
    "vgg16": "vgg16",
    "googlenet": "googlenet",
    "resnet50": "resnet50",
    "vit_b_16": "vit_b_16",
    "swin_t": "swin_t",
    "swin_b": "swin_b",
    # An explicitly named new CE/ERM baseline, not the ambiguous paper row.
    "erm_resnet50": "resnet50",
}

_MIN_INPUT_SIZE = {
    "alexnet": 63,
    "vgg16": 32,
    "googlenet": 32,
    "resnet50": 32,
    "vit_b_16": 32,
    "swin_t": 32,
    "swin_b": 32,
    "erm_resnet50": 32,
}


class ModelNotReadyError(ValueError):
    """A requested method has no verified implementation in this factory."""


def available_methods() -> Tuple[str, ...]:
    """Exact constructible IDs (source-ready; not a claim of runtime QA)."""
    return tuple(_BUILDERS)


def _scratch_kwargs(builder: Any) -> Dict[str, Any]:
    """Use the 0.14 weights API and safely support older explicit APIs.

    Do not use a broad TypeError retry: that could mask a broken constructor.
    Both branches explicitly disable pretrained weights.  No hub loader,
    checkpoint path, default weights enum, or package installation is used.
    """
    parameters = inspect.signature(builder).parameters
    if "weights" in parameters:
        return {"weights": None, "progress": False}
    if "pretrained" in parameters:
        return {"pretrained": False, "progress": False}
    raise RuntimeError(
        "Cannot prove the constructor's scratch-only contract: expected an "
        "explicit weights or pretrained argument. Use a supported torchvision "
        "installation (the campaign targets torchvision 0.14.x)."
    )


def create_model(method: str, num_classes: int, input_size: int) -> nn.Module:
    """Return the specified randomly initialized torchvision classifier.

    GoogLeNet deliberately disables auxiliary heads/losses and input
    transformation for this single-logit, shared-cross-entropy protocol.  That
    is a disclosed protocol adaptation, not an exact original training recipe.
    ViT is exactly the base/16 architecture; its positional embedding is
    initialized for the supplied input resolution, without interpolation from
    a pretrained checkpoint.  VGG16 is the non-BatchNorm variant.
    """
    if method not in _BUILDERS:
        raise ModelNotReadyError(
            "No source-verified implementation for {!r}. Available exact IDs: "
            "{}. See source_registry.json; blocked methods must not fall back "
            "to a different network.".format(method, ", ".join(available_methods()))
        )
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2")
    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise ValueError("input_size must be an integer")
    if input_size < _MIN_INPUT_SIZE[method]:
        raise ValueError(
            "{} requires input_size >= {} in this campaign adapter".format(
                method, _MIN_INPUT_SIZE[method]
            )
        )
    if method == "vit_b_16" and input_size % 16:
        raise ValueError("vit_b_16 requires input_size divisible by patch_size=16")

    # Deliberately lazy: manifest/planning tools can import this file without
    # initializing either PyTorch or CUDA, including on the controller host.
    try:
        import torchvision
        from torchvision import models
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "The selected server must provide a compatible torch/torchvision "
            "environment. No dependency is installed by this adapter."
        ) from exc

    builder_name = _BUILDERS[method]
    builder = getattr(models, builder_name, None)
    if builder is None:
        raise RuntimeError(
            "torchvision {} does not expose {}".format(torchvision.__version__, builder_name)
        )
    kwargs = _scratch_kwargs(builder)
    kwargs["num_classes"] = num_classes
    if method == "googlenet":
        kwargs.update(aux_logits=False, transform_input=False, init_weights=True)
    elif method == "vit_b_16":
        kwargs["image_size"] = input_size

    model = builder(**kwargs)
    # Plain provenance metadata: never a learned state or a hidden input/loss
    # transform. The shared runner should persist this together with its source
    # hashes, actual installed-library version and model parameter count.
    model._rgsp_external_metadata = {
        "method_id": method,
        "builder": "torchvision.models." + builder_name,
        "torchvision_version": torchvision.__version__,
        "initialization": "scratch",
        "pretrained_weights": None,
        "num_classes": num_classes,
        "input_size": input_size,
        "input_channels": 3,
        "output": "single_logits_tensor",
        "auxiliary_heads": False if method == "googlenet" else None,
        "literature_result_reproduction_claimed": False,
    }
    return model
