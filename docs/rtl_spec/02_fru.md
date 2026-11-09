# FRU Specification

`FusedResidencyUnit` accepts SIU propagation events and TRU render reads. Its
direct-indexed tier directory stores `h[3:0]`, `d[7:0]`, tier, and row. Active
and resident rows identify
physical partition locations. Candidate row metadata is the full logical `gid`
index for the refill path and is not a physical candidate-store row. The unit
probes active, resident, then candidate regions and performs priority
allocation only after the fifth frame of a window.

For every accepted SIU propagation, the recent-strength field uses
`floor(255 * clamp(effectiveStrength, 0, 1))`, with `1.0` represented by
`8'hff`. Confirmations update `d`, energy, sample count, and touched metadata
without incrementing `h` or producing a residency transfer.
