# TRU Multi-Path Rendering

TRU accepts CIU issues from one frame at a time. It latches the first queued
record's frame ID and backpressures a different frame until the current batch
has drained. At `frameEnd`, it drains the globally earliest `(tile, depth,
gid)` record, so adding a later-tier record cannot change compositing order. A
tile whose transmittance falls below `0.05` skips later work.

The one-frame buffer does not accept new issues while it drains. A later empty
`frameEnd` therefore cannot reset the blend order of the already closed frame.

Full work evaluates degree-3 SH in binary32. Light work reuses the CIU
view-color estimate through the public FP16 conversion boundary. The public renderer uses
the CIU bounded support proxy as coverage rather than a pixelwise coverage
kernel. Act-Unit opacity times that coverage forms alpha and updates tile
transmittance. Filtered user work enters neither shading, sorting, coverage, nor
blending. A training request is never filtered: CIU converts it to light work
before TRU.

Only the three public issue paths are legal. An undefined path encoding is
discarded before it can affect ordering, compositing, SplatSPM, or FRU.
A training request at terminal frame `65535` is discarded at the same input
boundary: its mandatory TUU feedback would require an unrepresentable next
frame. Terminal-frame user work remains legal because it has no feedback edge.
TRU also rejects non-finite depth before queueing work, so every accepted
record has a defined numerical position in the global compositing order.

For training work, the transferred trajectory is written to SplatSPM, its
rendered-pixel record is written to FrameSPM, and it emits a render-read event
to FRU without waiting for a target. The public renderer has no pixelwise
kernel, so its trajectory `pixel` is the zero-extended tile ID: a
representative-pixel key used by FrameSPM and by TUU's matching target. It is
not a raster coordinate. The top level holds one trajectory for TUU until the
matching target has been buffered. The trajectory carries the current training
view stamped with its render frame, which SplatSPM retains for the next SIU
update. `TruSpec` checks priority dispatch, representative-pixel emission,
early stop, trace emission, and terminal-frame rejection. The oracle checks
global `(tile, depth, gid)` compositing order across tiers independently of
queue timing, including non-finite-depth rejection.
