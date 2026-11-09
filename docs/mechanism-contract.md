# SCD-GS Mechanism Contract

This document is the public behavioral contract for SCD-GS. It defines the
mechanism reproduced by the repository, not a performance model or a claim
about any particular accelerator platform.

## Global Rules

- Canonical updates carry `gid`, `frameId`, and `structural`. Projection
  summaries carry `gid`, `frameId`, and `requestKind`. Derived records preserve
  the identifiers needed to join the two streams. Public frame identifiers are
  unsigned 16-bit values in `0..65535`.
- Floating-point payload fields at external RTL interfaces are IEEE-754
  binary32. Gaussian identifiers, frame and tile identifiers, counters, and
  control tags are unsigned integer or enumerated fields. The design uses
  HardFloat-backed binary32 FMA, comparison, conversion, square-root, and
  division units. The Python model evaluates the paper equations semantically;
  RTL tests check the protocol and decision contract rather than claim that
  every intermediate floating-point word is bit-identical to Python.
- `requestKind` is carried by `ProjectionSummaryBundle`. A summary consumed by
  SIU must be tagged `training`, because it represents the preceding training
  render. CIU separately pairs a complete residency response with the current
  view for the same `(gid, frame)`; that view may be `user` or `training`.
- SIU accepts one monotonic source-frame observation per Gaussian and averages
  its latest five observations. An update and its retained summary must share a
  `gid`, and the summary frame must strictly precede the update frame. Trace
  records bind every observation to an open `FRAME`
  interval. FRU allocation advances only on every fifth completed `frameEnd`
  boundary.
- A projection radius is a positive normal IEEE-754 binary32 value. This
  makes SIU geometric normalization well-defined; zero, subnormal, negative,
  infinite, and NaN radius encodings are malformed inputs and are rejected
  before they can establish SplatSPM state.
- A structural marker is one of insert, delete, invalidate, or reset. It
  bypasses SIU suppression and CIU precision reduction.

## Canonical Update

`CanonicalUpdate` contains `deltaMu[3]`, symmetric `deltaSigma[6]`,
`deltaOpacity`, `deltaSh`, scalar strength `n`, a structural marker, and the
originating frame. SIU consumes it and TUU produces it. TUU-generated feedback
sets `n = min(1, abs(adamStep))`. No stage is allowed to replace this record
with a scalar-only update.

## SIU / Splat Impact

The public Act Unit uses the saturating affine binary32 function
`clamp(0.5 + 0.25 * x, 0, 1)`. SIU, CIU, TRU, and the oracle use the same
function at the public RTL boundary.

SIU reads a `training`-tagged previous training-render summary `(Rcam, Jpi,
JCov, r, A, o, d, Y, requestKind, tile, depth)` and computes:

```text
[deltaU, deltaV] = Jpi * Rcam * deltaMu
deltaR = frobeniusNorm(JCov * deltaSigma * transpose(JCov))
deltaAlpha = abs(act(o + deltaOpacity) - act(o))
deltaQ = abs(luminance(sum(deltaSh[b] * Y[b](d))))
C = min(A / tileArea, 1)
psi = C * ((abs(deltaU) + abs(deltaV) + abs(deltaR)) / r + deltaAlpha + deltaQ)
Phi = average(psi over the latest five accepted observations)
```

Before five samples, a non-structural update remains protected. Afterwards,
`Phi >= tauPhi` creates a state-change commit with the original strength.
Otherwise SIU emits a confirmation record with zero effective strength. FRU
records that confirmation in its metadata but does not send it to CIU or TRU.

## FRU / Fused Residency

The direct-indexed tier directory records `{valid, gid, tier, row, h[3:0], d[7:0]}`. `row` is a
partition-local physical row for active and resident entries. Candidate `row`
is logical metadata because candidate state is addressed by `gid`. A commit
allocation initializes `h=1` and `d=floor(255 * clamp(effectiveStrength, 0,
1))`, including `d=255` at `effectiveStrength=1.0`. A commit hit saturating-
increments `h` and updates
`d=max(d, quantizedEffectiveStrength)`. A TRU
training-render read only saturating-increments `h`.

The lookup checks active first, resident second, and candidate last. At a
five-frame boundary the allocator compares priority `(h, d, -gid)`, swaps a
higher-priority lower-tier entry upward, right-shifts untouched `h`, and clears
`d`. The final term makes an equal-metadata choice deterministic by selecting
the smaller `gid`.
For CIU, `E` is the mean effective strength over observed commit or
confirmation records in the current five-frame window. A candidate placement,
or a candidate payload not yet authorized by its matching fill, uses a refill
request and response before it enters CIU. Active and resident state use the
StateSPM read-response path directly; a completed candidate fill may do so
until its placement changes.

## CIU / Training-Conditioned Render Issue

CIU forms the screen covariance `C = JCov * Sigma * transpose(JCov)` and the
bounded support proxy `Omega = saturate(9 * abs(det(C)))`. It
derives `b(v)` from degree-0 and degree-1 SH, then:

```text
P = saturate(average(act(opacity) * Omega * b(v)))
kappa = 1 - clamp(E, 0, 1)
rho = 1 - kappa * (1 - P)
omega = rho * Omega
```

`b(v)` is retained as the raw low-order SH estimate. The saturation in `P`
applies after the channelwise product has been averaged; the implementation
does not insert a per-channel color clamp.

Before CIU passes a record to TRU, it writes the current live-state projection
fields into the separate current view: projected center and depth, opacity, and
the degree-0/1 `b(v)` color. The prior SIU summary remains in the residency
record. `conic`, `radius`, and `area` remain current-view fields because the
paper does not define a separate conic-inversion step in CIU.

Structural state, and insufficient samples without a candidate marker, use
full work. `rho > tauRho` also uses full work. If `omega < tauD`, user work is
filtered while training work takes the conservative light path. All remaining
records take light work.

## TRU / Tiered Render

TRU drains every closed frame in global `(tile, depth, gid)` order. Residency
controls issue and refill priority but cannot change compositing order. TRU
stops later work for a tile as soon as its transmittance is below `0.05`.
Full work uses degree-3 SH in binary32. Light work reuses `b(v)` through the
public FP16 conversion boundary. The bounded support proxy supplies coverage for the public blend model; Act-Unit opacity times coverage supplies alpha.
It is not a pixelwise coverage kernel. A filtered user record reaches neither
SH, sorting, coverage, nor blending.
Only the `full`, `light`, and `filter` issue tags are legal. An undefined path
tag is discarded and cannot create a render-read event or trajectory.
Training work at terminal frame `65535` is likewise discarded by TRU before
rendering side effects, because its mandatory feedback has no representable
successor frame. Terminal-frame user work remains valid because it does not
form a feedback edge.
TRU also discards non-finite depth before queueing work, preserving a defined
numerical `(tile, depth, gid)` ordering for every accepted record.
Training work writes `(gid, blendOrder, coverage, transmittance, currentView)`
to SplatSPM. The current view is stamped with the render frame and becomes the
prior training summary for the next SIU update. Its zero-extended tile ID is
the representative-pixel key for the FrameSPM record and matching target; it
is not a raster coordinate. Its transferred trajectory emits the matching
training-render read to FRU.

## TUU / Training Update

TUU holds one target pixel and consumes a matching TRU trajectory. Sixteen
logical residual and backward lanes form a scalar surrogate gradient from RGB
residual, coverage, and transmittance. The optimizer applies that scalar across
the documented geometry, opacity, SH, and optimizer-state update shape in
StateSPM. A mismatched trajectory, user-view record, or non-`full`/`light`
path is counted and discarded while the target remains buffered. TUU completes
the matching StateSPM write before it emits
the next `CanonicalUpdate` to SIU. Each matched pair produces one
scalar-surrogate transaction without aggregation across pairs.

TUU feedback has `frameId = trajectory.frameId + 1`. TRU rejects terminal-frame
training work before it can create a trajectory; TUU repeats that validation at
its direct input and retains any buffered target rather than allowing the
unsigned identifier to wrap to zero.

## Storage And Transaction Boundaries

- StateSPM: live Gaussian state and optimizer state, with active, resident,
  and candidate/refill transaction paths. A state seed establishes only the
  initial StateSPM copy. FRU creates its directory entry when it observes the
  first propagation event for that Gaussian.
- MetaSPM: a public readback snapshot of FRU residency and short-window
  metadata. `ScdGsTopIO.metadataReadGid` selects the entry and
  `ScdGsTopIO.metadataSnapshot` returns its `{tier, row, h, d, energy,
  samples, touched}` fields with `metadataSnapshotValid`. FRU publishes event
  snapshots and the placement/final-aging state of every valid allocator
  entry. FRU owns that live execution state, while SIU retains its own local
  window registers at the Stage I boundary.
- SplatSPM: projection summaries and forward trajectories.
- FrameSPM: bounded direct-mapped target and rendered representative-pixel
  records, each protected by the full pixel-key tag.
- Candidate refill uses 64-byte aligned memory-line requests. Every
  FRU-selected candidate placement must complete this transaction before CIU
  receives its state; active or resident placement without local state does so
  too. The matching `CandidateFillBundle` restores the complete StateSPM
  record: Gaussian state plus Adam moment and variance vectors. It is an RTL
  module-boundary record, not a modeled 64-byte response payload, and is
  installed at the requested active, resident, or candidate placement. The
  public model verifies request semantics only. It makes no bandwidth or
  latency claim.
