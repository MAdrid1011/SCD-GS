# TUU Training Update Formation

TUU holds one live target pixel and consumes a matching TRU trajectory. It
forms a scalar surrogate gradient from RGB residual, coverage, and
transmittance. The optimizer applies that scalar to the documented geometry,
opacity, SH, and optimizer-state update shape. Each matched pair produces one
scalar-surrogate transaction without aggregation across pairs.

The public interface exposes a StateSPM write and a canonical feedback record.
TUU completes the StateSPM write before it exposes the corresponding feedback
record to SIU, so a backpressured write cannot let feedback observe the
pre-update state. An unmatched trajectory increments an error counter,
produces no state update, and leaves the target buffered for a later matching
trajectory. A training trajectory is `full` or `light` and carries a training
view; a `filter` tag or user-view record is rejected because user work has no
TRU trajectory. A terminal 16-bit frame trajectory is also rejected: feedback
is assigned to the source frame plus one and may not wrap to frame zero. The
implementation uses sixteen logical residual and backward lanes. The fixture
drives a deterministic subset of those lanes.

`TuuSpec` verifies matched pairing, the StateSPM write, feedback frame advance,
and rejection of mismatched records. The semantic oracle verifies the same
state-transition rule without claiming numerical convergence.
