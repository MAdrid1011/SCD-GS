# Mechanism Parameters

`ScdGsParams.scala` contains public widths, capacities, thresholds, the
five-observation SIU history, five-frame FRU window, 64-byte transaction size,
and enumerations for structure,
tier, request, propagation, and issue-path tags. `rowWidth` must represent
every candidate Gaussian ID as well as every active and resident physical row.
The executable contract fixes binary32 fields, degree-3 SH, a five-observation
SIU history, a five-frame FRU window,
window, 64-byte candidate lines, and sixteen logical training lanes. Active
and resident StateSPM partitions and the TRU frame buffer must be nonempty.

`maxGaussians`, active/resident capacities, `tilePixels`, and queue depths are
reference-artifact bounds needed to elaborate executable RTL. They are not
paper-reported capacities, platform settings, or PPA measurements, and no
capacity, throughput, or storage claim should be inferred from their defaults.
The public default `frameWidth` is 16, so trace frame identifiers are limited
to `0..65535`; TUU reserves `65535` from training trajectories because their
feedback update uses the next frame identifier.
