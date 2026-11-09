# contracts.py

## External Interface

This module exports the full Gaussian-state, canonical-update, projection,
propagation, residency, issue, render-trace, and target-pixel records used by
every oracle stage. A render trace carries its complete current projection, so
the model preserves the training view that becomes the next SIU summary.
Propagation holds the prior training summary, while CIU accepts a separate
same-frame current view. The record widths match the public degree-3 SH
contract.

## Internal Helpers

The enums encode the public structural, tier, request, path, and propagation
tags. They intentionally mirror the Chisel protocol names. The public frame
constants mirror the default 16-bit Chisel frame identifier domain.
