"""Official VMamba-Tiny backbone adapter for the shared SAR runner."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .mamba_yolo import DS_VSSBlockSweep
from .vmamba_official.vmamba import Backbone_VSSM


class VMambaTinyBackbone(nn.Module):
    """VMamba-Tiny feature pyramid with an optional SAR-aware P3 replacement."""

    output_channels = (96, 192, 384, 768)

    def __init__(self, ours: bool, state_dim: int, pretrained: str | None = None):
        super().__init__()
        checkpoint = str(Path(pretrained).expanduser().resolve()) if pretrained else None
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            patch_size=4,
            in_chans=3,
            depths=[2, 2, 8, 2],
            dims=96,
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            norm_layer="ln2d",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
        )
        self.init_report = {
            "path": checkpoint,
            "loaded_tensors": 0,
            "model_tensors": len(self.encoder.state_dict()),
            "coverage": 0.0,
        }
        if checkpoint:
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            source = checkpoint_data.get("model", checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
            if hasattr(source, "state_dict"):
                source = source.state_dict()
            if not isinstance(source, dict):
                raise TypeError(f"Unsupported VMamba checkpoint format: {checkpoint}")
            incompatible = self.encoder.load_state_dict(source, strict=False)
            missing = set(incompatible.missing_keys)
            model_tensors = len(self.encoder.state_dict())
            loaded = max(model_tensors - len(missing), 0)
            self.init_report = {
                "path": checkpoint,
                "loaded_tensors": loaded,
                "model_tensors": model_tensors,
                "coverage": loaded / max(model_tensors, 1),
                "missing_tensors": len(incompatible.missing_keys),
                "unexpected_tensors": len(incompatible.unexpected_keys),
            }
            print(f"VMAMBA PRETRAIN TRANSFER: {self.init_report}", flush=True)
        self.p3 = DS_VSSBlockSweep(192, 192, state_dim) if ours else nn.Identity()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = list(self.encoder(image))
        features[1] = self.p3(features[1])
        return tuple(features)
