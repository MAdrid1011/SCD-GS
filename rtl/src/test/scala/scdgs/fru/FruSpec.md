# FRU Tests

`FruSpec` sends a commit, confirmation, and repeated training-render reads to
prove 4-bit hotness saturation and post-event metadata snapshots. It verifies
8-bit update quantization including `0.5 -> 127` (truncate toward zero) and
the `1.0 -> 255` boundary, five-frame priority allocation, `d` clearing, and
right-shift aging of an untouched entry. It also verifies that the allocator
and aging passes publish the final active/resident placement and cleared window
fields through the MetaSPM snapshot stream. An equal-hotness,
equal-recent-strength tie is resolved by ascending Gaussian ID.
