# SIU Tests

`SiuSpec` first verifies the shared 3x3 FMA contraction, then drives five equal
low-impact updates to verify warm-up protection and the fifth-frame
confirmation path. It finally verifies that an insertion marker bypasses
confirmation and preserves effective strength.

The malformed-pair test verifies that a ready/valid transfer with mismatched
Gaussian IDs is consumed and counted as invalid, but does not produce SIU's
accepted-input pulse or a propagation record. Top-level bootstrap storage uses
that pulse, so such a pair cannot establish a prior-render summary.

The suite also rejects a user-tagged projection at the SIU boundary. Only a
training render can supply the prior summary consumed by Stage I.
