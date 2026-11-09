# TUU Specification

`TrainingUpdateUnit` consumes one matching target pixel and trajectory at a
time. It forms a scalar surrogate from residual, coverage, and transmittance,
updates the StateSPM Gaussian and Adam-vector state with the documented fixed
shape, and produces a canonical update for the next SIU transaction.

The trajectory must carry a training request and a `full` or `light` path.
`filter` identifies user-side work that bypasses TRU's trajectory output, so
TUU rejects direct filter-tag or user-view inputs without consuming its
buffered target. A trajectory at frame `65535` is likewise rejected without
consuming the target, because its feedback frame would otherwise wrap to zero.
