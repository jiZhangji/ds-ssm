# Classification adapters — unified A6000 campaign

This directory belongs only to `rgsp_unified_a6000_20261012_v1`. It neither
replaces historical runs nor reproduces the heterogeneous literature scores
by default. `source_registry.json` distinguishes model identity, source
readiness, adapter readiness and server validation.

## Interface

`factory.create_model(method: str, num_classes: int, input_size: int) -> nn.Module`

The shared runner supplies `N x 3 x 224 x 224` tensors (RGB conversion/three
identical channels for grayscale SAR) and owns preprocessing, device, precision,
cross-entropy, optimization, selection and evaluation. The factory returns one
logits tensor in eager execution. It does not move the model to a device,
initialize CUDA, download weights, install dependencies or start training.
Importing `factory.py` or this package does not import torch/torchvision.

Initial exact IDs are `alexnet`, `vgg16`, `googlenet`, `resnet50`, `vit_b_16`,
`swin_t`, `swin_b`, and `erm_resnet50`. All use scratch initialization.
Unsupported names fail closed; no fallback network is substituted.

The eight initial models use the installed official TorchVision library. The
source audit targets tag `v0.14.1`, commit
`5e8e2f125f140d1e908cf424a6a85cacad758125`; the actual installed version is
attached to each model's `_rgsp_external_metadata` and must be saved in its run
record. Reference source SHA256 values are in `source_registry.json`; they do
not prove an installed server copy is unmodified. The server runner remains
responsible for environment/source hashes and forward/backward/serialization
validation. No model was executed by the classification-source audit agent.

## Deliberate identity and protocol choices

- `vgg16` is configuration D without BatchNorm, **not** an inferred meaning of
  the original table's unspecified `VGGNet` label.
- `resnet50` is TorchVision ResNet-50 **v1.5**: bottleneck downsampling stride
  belongs to the 3x3 convolution. `ResNet` without a depth is not an alias.
- `vit_b_16` is base/16, 12 blocks, width 768, 12 heads and MLP width 3072.
  Its positional embeddings are randomly initialized for the requested image
  size. The original table's bare `ViT` label is not sufficient to establish
  an exact variant match.
- `swin_t` is Swin V1 tiny (width 96, depths 2/2/6/2); `swin_b` is Swin V1 base
  (width 128, depths 2/2/18/2), both using 4x4 patches and 7x7 windows.
- `googlenet` disables auxiliary heads/losses and internal input transformation
  for the shared single-logit CE protocol. This differs from training with
  auxiliary classifiers and is disclosed rather than hidden. `init_weights`
  is explicitly true, preserving the library's scratch initializer.
- `erm_resnet50` is an explicitly named new uniform-CE ResNet-50 baseline. It
  must not silently replace the paper's `ERM` result, whose training recipe and
  augmentations are separate provenance questions.

New runs share the parent campaign's physical batch 16, accumulation 4,
AdamW learning rate 2e-4, weight decay 5e-4, 150-epoch cap, patience 30,
three-epoch warmup, linear LR schedule and FP32. Architecture-specific defaults
such as dropout/stochastic depth remain the library's documented model
defaults; they are not silently homogenized.

## External methods

SAR-specific and domain-generalization candidates stay in the registry even
when blocked. A title, acronym or third-party implementation is not sufficient
to label a replacement model as the author method. Methods requiring masks,
special augmentations, adversarial/variance/CORAL objectives, extra branches or
target-domain access cannot be reproduced by merely passing their backbone
through the shared cross-entropy runner. Exact author source/variant and all
such requirements must be resolved before admission.

Source-only means that target images/labels are not used during fitting or
hyperparameter/model selection; final evaluation on a target test split does
not by itself constitute training access. Dataset splits and selection rules
are determined by the parent campaign, not by this adapter.

The registry records public source observations only. GitHub anonymous REST
was rate-limited during this audit; raw public sources and read-only
`git ls-remote` were used without looking for credentials. Missing code is
reported as unconfirmed/not located, never as proof that no author code exists.
