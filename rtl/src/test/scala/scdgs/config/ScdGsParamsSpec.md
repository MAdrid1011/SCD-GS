# ScdGsParamsSpec.scala

## Test Cases

`row and GID fields represent the complete public Gaussian domain` verifies
that the default Gaussian count fits both the public GID encoding and the
candidate-row encoding. It also rejects a row width that cannot represent the
configured domain.

`rejects parameterizations outside the public mechanism contract` verifies
that `ScdGsParams` rejects unsupported frame, binary32, tile, capacity, SH,
window, line-size, render-queue, and logical-lane configurations.
