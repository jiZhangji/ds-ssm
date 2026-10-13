# External SAR change-detection source audit

This directory is an isolated source-and-adapter audit for the 2026-10-12 unified
experiment request. No model has been run by this audit. Original notebooks and
source files are retained without execution; they are not experiment results.

The author-reported, per-image-pair pseudo-label protocol is separate from the
project's existing supervised spatial split. Ground-truth images are evaluation
inputs only for an unsupervised reproduction. Never use their labels for
pseudo-label generation, validation, checkpoint selection or threshold tuning.

Source registry and implementation notes are being assembled here. Only exact
author architectures with documented provenance will be exposed by adapters.
Missing implementations remain blocked rather than being replaced by an
unattributed approximate model.
