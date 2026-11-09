# Memory Tests

`MemorySpec` retains a projection summary and training trajectory in SplatSPM,
target and rendered representative-pixel records in FrameSPM, and an FRU
metadata snapshot in MetaSPM. FrameSPM readback checks retain the full pixel
tag, so two pixels that share a compact-array index cannot alias as a hit. It
also writes a resident state row with Adam moment and variance,
reads it through the active path, and then requests candidate state with
deliberately nonmatching row metadata. This proves that candidate lookup is
GID-indexed. A duplicate external state seed is rejected and cannot overwrite
the live Gaussian or its Adam vectors. A physical-row reuse test verifies that the displaced state and
Adam vectors are marked as requiring a matching candidate refill before FRU
can provide the next active or resident placement. It also verifies that a
candidate seed and an unfilled GID selected for an active row take the
candidate-refill path rather than returning empty or stale state, and that each
refill restores its Adam vectors. The candidate request is asserted as a
64-byte aligned memory-line transaction rather than a synthetic performance
access.
