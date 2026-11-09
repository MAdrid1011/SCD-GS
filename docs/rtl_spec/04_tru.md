# TRU Specification

`TieredRenderUnit` accepts CIU issue records from one frame at a time. It
latches the first queued frame ID and backpressures a different frame until the
closed batch drains. After `frameEnd`, it drains the globally earliest `(tile,
depth, gid)` record across all tiers. Residency cannot change that order. Full work evaluates degree-3 SH in
binary32; light work crosses the public FP16 conversion boundary. The unit
uses the CIU support proxy as coverage and multiplies it by Act-Unit opacity
for alpha before updating transmittance. It skips deeper records after
transmittance drops below `0.05`. A training trajectory writes SplatSPM,
retaining its current training view as the next SIU summary. Its rendered-pixel
record uses the zero-extended tile ID as its representative-pixel key and is
written to FrameSPM; it emits the matching render-read to FRU. The public path
does not implement a pixelwise coverage kernel. The top-level one-entry bridge
waits for TUU's buffered matching target before TUU consumes that trajectory.

While the closed frame drains, `issueIn` is backpressured. A later empty
`frameEnd` does not restart the drain or reset its blend order.

The public path encoding admits only `full`, `light`, and `filter`. An
undefined encoding is rejected before it reaches the frame buffer or produces
a render-side event. Training work at terminal frame `65535` is rejected at the
same boundary, because its required next-frame feedback record is not
representable. Terminal-frame user work remains valid. Non-finite depth is
rejected before queueing so the global order remains numerically defined.
