# TRU

`TieredRenderUnit.scala` schedules tiered render work, preserves compositing
order, stops saturated tiles, and emits training trajectories. A trajectory's
representative-pixel key is its zero-extended tile ID because the public path
does not implement a pixelwise coverage kernel. It contains no platform-specific
timing or performance model.

The only legal issue paths are `full`, `light`, and `filter`. An undefined
2-bit path tag is consumed, counted as invalid input, and never reaches the
frame buffer or rendering outputs.
