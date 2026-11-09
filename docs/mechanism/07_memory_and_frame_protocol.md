# Memory And Frame Protocol

StateSPM owns live Gaussian state and optimizer state for active and resident
tiers, plus candidate payload storage. Every FRU-selected candidate placement
must emit a 64-byte-aligned `MemoryLineReqBundle` unless that candidate has
already completed its matching refill; an active or resident placement also
emits one when it has no local state. The pending residency event cannot reach
CIU until the matching candidate fill produces a complete StateSPM response.
The request is a 64-byte unit, while the module-boundary fill carries Gaussian
state and Adam moment and variance vectors. This is a transaction-size
contract, not an LPDDR timing model. The fill is installed at the active,
resident, or candidate placement that caused the lookup and authorizes later
local candidate reads only until that placement changes. MetaSPM
receives the post-event FRU snapshot for every accepted commit, confirmation,
and training-render read, plus the placement and final-aging snapshot for each
valid allocator entry. `ScdGsTopIO.metadataReadGid` selects that readback
record and `ScdGsTopIO.metadataSnapshot` returns its residency and window
fields. `metadataSnapshotValid` distinguishes an unwritten entry; MetaSPM is
not the live directory owner. SIU and FRU retain the live window registers that
drive their decisions. SplatSPM stores prior training summaries and forward
traces. CIU accepts a current view only when its `gid` and `frameId` match the
complete residency response. A TRU training trajectory writes that current view
back as the next prior summary. Its `render-read` goes directly to FRU, while a
bounded top-level bridge delivers the trajectory to SplatSPM, FrameSPM, and TUU
with independent ready/valid transfers. These links do not impose a same-cycle
fan-out requirement. The target stream independently fills TUU's one-entry
target buffer and writes the same target to FrameSPM; consequently, a target can
arrive before the corresponding trajectory. The input handoff never accepts a
second target while TUU owns its one target buffer.

FrameSPM is a bounded 256-entry direct-mapped record store. It indexes target
and rendered records with the low bits of their 32-bit representative-pixel key
and retains the full key as a tag, so a collision replaces the prior record and
the prior key reads invalid. The public TRU defines that key as the
zero-extended tile ID because this artifact models the paper's bounded support
proxy rather than a raster coverage kernel. A target that is intended to pair
with a training trajectory must use that same `(pixel, frame)` key. This is a
storage and pairing convention, not an image-buffer capacity or quality claim.

All stream links use ready/valid transfer. A rising edge on `frameEnd` advances
the FRU five-frame window and closes the TRU batch used for cross-tier depth
merge; a held-high signal remains one boundary. TRU latches the first
non-filtered issue's frame ID and backpressures a different frame until that
closed batch drains. The top-level test covers candidate refill, blocked queues,
frame edges, and the feedback connection from TUU to SIU.
