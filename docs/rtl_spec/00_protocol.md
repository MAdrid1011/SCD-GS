# Five-Stage Protocol

All streaming interfaces use `Decoupled` ready/valid transfer. A transaction is
identified by `(frameId, gid)` and retains its structural marker and request
kind until TUU emits the next canonical update.

The top level accepts canonical updates, strictly earlier prior-training
summaries, same-frame current views with embedded request kind, state seeds,
candidate fills, target pixels, and frame edges. A training trajectory emitted
by TRU is the sole render-read source for FRU; it is not an independent
top-level input. `currentViewIn` is the Stage III user/training view interface.
The top level exposes mechanism counters and a 64-byte candidate-refill request
interface. `frameEnd` closes the TRU issue batch for the global cross-tier
`(tile, depth, gid)` merge and advances FRU five-frame state. Its rising edge
is one boundary event; holding it high never creates additional frames.

A canonical bootstrap update and its prior-training summary transfer together.
SIU may consume an invalid pair to report it through the invalid counter, but
only an SIU-accepted pair writes the bootstrap summary into SplatSPM. This
prevents a mismatched, same-frame, or nonmonotonic external summary from
becoming feedback state. Once SplatSPM has a summary for a `gid`, the external
prior-summary interface is not accepted for that Gaussian; subsequent canonical
updates use the retained training summary. CIU consumes a current view only
when it has the same `(gid, frame)` as its completed residency response.
