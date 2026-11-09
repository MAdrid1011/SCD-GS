# Memory Specification

StateSPM owns active and resident physical state rows. A `StateSeed` initializes
one previously unseen Gaussian only; a duplicate seed is rejected rather than
overwriting live state or Adam vectors outside the canonical-update loop. Any
FRU-selected candidate placement must complete a 64-byte aligned
candidate-refill request before its state can reach CIU; active or resident
rows do the same when no local state is present. The module-boundary fill
returns a complete StateSPM record (Gaussian state plus Adam moment and
variance vectors) addressed by `gid` and installs it at the selected placement.
FRU owns the live tier
directory and five-frame window state. MetaSPM exposes a published readback
snapshot, including the final allocation and aging state for each valid entry,
and is not the execution owner. SplatSPM owns prior projections and
forward trajectories. FrameSPM owns bounded target and rendered
representative-pixel records, indexed by the zero-extended tile ID and
full-key-tagged on readback. It is not a pixelwise framebuffer. These roles
must remain separate at their public interfaces.
