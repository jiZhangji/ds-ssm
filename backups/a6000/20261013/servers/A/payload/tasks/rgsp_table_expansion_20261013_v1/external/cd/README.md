# External SAR change-detection source audit

This directory contains source-audited adapters for the 2026-10-12 unified
experiment request. The operator has now run the five implemented methods on D:
all d0r3 preflight probes passed; production is queued. Earlier d0/d0r2 failures
remain in their original attempts. See the campaign README and server receipts.
Original notebooks/source files retain their audited hashes. Short probes are
not final scientific results.

The author-reported, per-image-pair pseudo-label protocol is separate from the
project's existing supervised spatial split. Ground-truth images are evaluation
inputs only for an unsupervised reproduction. Never use their labels for
pseudo-label generation, validation, checkpoint selection or threshold tuning.

Source registry and implementation notes are being assembled here. Only exact
author architectures with documented provenance will be exposed by adapters.
Missing implementations remain blocked rather than being replaced by an
unattributed approximate model.
