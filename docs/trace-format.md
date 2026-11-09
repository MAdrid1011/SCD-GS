# SCD-GS Trace Format

Public traces use UTF-8 line records with pipe-separated fields:

```text
KIND|key=value|key=value
```

`TraceFormat.parseLine` rejects unknown and duplicate keys. `TraceFormat.validate`
also rejects malformed values, invalid enums, out-of-width or nonmonotonic
frame identifiers, tile identifiers, and TRU blend orders, inconsistent frame
boundaries, missing prior-summary or current-view state, and a `gid` outside
the supplied configuration range. The grammar is pinned to the public default
`ScdGsParams` shape: degree-3 SH has 16 bases and 48 RGB coefficients,
12-bit tiles, and a 64-entry TRU frame buffer. `ReplayTrace` validates
command-line fixtures against that default configuration's 4096-Gaussian
identifier domain.

`PROJECTION.radius` is one positive normal binary32 word. This is the valid
denominator domain of the SIU geometric-normalization equation; zero,
subnormal, negative, infinite, and NaN encodings are rejected.
`PROJECTION.depth` must be a finite binary32 value. This preserves TRU's
defined numerical `(tile, depth, gid)` compositing order; negative and
subnormal finite depths remain representable.

## Record Kinds

| Kind | Required fields | Public-interface role |
| --- | --- | --- |
| `FRAME` | `id`, `edge` | `begin` or `end` frame boundary |
| `STATE` | `gid`, `frame`, `mean`, `covariance`, `opacity`, `sh`, `tier` | Complete `StateSeedBundle` input |
| `UPDATE` | `gid`, `frame`, `dmu`, `dsigma`, `do`, `dsh`, `n`, `struct` | Complete `CanonicalUpdateBundle` input, including TUU feedback |
| `PROJECTION` | `gid`, `frame`, `r_cam`, `j_pi`, `j_cov`, `view_direction`, `sh_basis`, `center`, `conic`, `radius`, `area`, `opacity`, `view_color`, `request`, `role`, `tile`, `depth` | Complete `ProjectionSummaryBundle`: `role=prior` seeds SIU's retained training summary; `role=current` is the CIU view for its same-frame update |
| `EVENT` | `gid`, `frame`, `source` | The SIU propagation result (`commit` or `confirmation`) for its preceding same-`(gid, frame)` update, or a `render-read` companion of a training trajectory |
| `REQUEST` | `gid`, `frame`, `kind` | CIU request observation; must match the same-frame current projection `request` and a preceding commit |
| `TRAJECTORY` | `gid`, `frame`, `pixel`, `blend`, `coverage`, `transmittance`, `color`, `path` | TRU training forward trace; `pixel` is the same-frame current projection's tile ID, used as the public representative-pixel key |
| `TARGET` | `pixel`, `frame`, `rgb` | Complete `TargetPixelBundle` input |

`STATE` supplies `stateSeedIn`; its `frame` field establishes trace ordering and
is not a field of `StateSeedBundle`. Exactly one `STATE` record initializes a
given Gaussian before its first `UPDATE`; a later duplicate would overwrite
live StateSPM/Adam state outside the canonical-update loop and is rejected.
`tier` is `active`, `resident`, or `candidate`, and selects the initial StateSPM
placement only. FRU creates its directory entry when it sees the first
propagation event for the Gaussian. A candidate `STATE` record seeds only a
payload; it cannot authorize Stage III until a matching candidate-fill
transaction supplies the complete Gaussian and Adam-vector state.

`CANDIDATE_FILL` is deliberately not a causal trace-record kind. It is the
complete response transaction on `ScdGsTopIO.candidateFillIn`, carrying the
same StateSPM-owned Gaussian and Adam-vector fields described by
`CandidateFillBundle`; it has no SIU, FRU, CIU, TRU, or TUU logical-event
counterpart. `MemorySpec` checks that full bundle and `FixtureReplaySpec`
drives it only after the corresponding candidate request has transferred.

## Binary32 Vectors

All floating-point values are comma-separated binary32 hexadecimal words in
row-major order, written as `0x` followed by one to eight hexadecimal digits.
The required widths map directly to `ScdGsBundles.scala`:

| Record field | Width | Bundle field |
| --- | ---: | --- |
| `STATE.mean` | 3 | `GaussianStateBundle.mean` |
| `STATE.covariance` | 6 | `GaussianStateBundle.covariance` |
| `STATE.opacity` | 1 | `GaussianStateBundle.opacity` |
| `STATE.sh` | 48 | `GaussianStateBundle.sh` |
| `UPDATE.dmu` | 3 | `CanonicalUpdateBundle.deltaMu` |
| `UPDATE.dsigma` | 6 | `CanonicalUpdateBundle.deltaSigma` |
| `UPDATE.do` | 1 | `CanonicalUpdateBundle.deltaOpacity` |
| `UPDATE.dsh` | 48 | `CanonicalUpdateBundle.deltaSh` |
| `UPDATE.n` | 1 | `CanonicalUpdateBundle.strength` |
| `PROJECTION.r_cam` | 9 | `ProjectionSummaryBundle.rCam` (3 by 3) |
| `PROJECTION.j_pi` | 6 | `ProjectionSummaryBundle.jPi` (2 by 3) |
| `PROJECTION.j_cov` | 6 | `ProjectionSummaryBundle.jCov` (2 by 3) |
| `PROJECTION.view_direction` | 3 | `ProjectionSummaryBundle.viewDirection` |
| `PROJECTION.sh_basis` | 16 | `ProjectionSummaryBundle.shBasis` |
| `PROJECTION.center` | 2 | `ProjectionSummaryBundle.center` |
| `PROJECTION.conic` | 3 | `ProjectionSummaryBundle.conic` |
| `PROJECTION.radius`, `area`, `opacity`, `depth` | 1 each | Matching projection scalar |
| `PROJECTION.view_color` | 3 | `ProjectionSummaryBundle.viewColor` |
| `TRAJECTORY.coverage`, `transmittance` | 1 each | Matching `RenderTraceBundle` scalar |
| `TRAJECTORY.color`, `TARGET.rgb` | 3 each | Matching RGB vector |

`struct` is `none`, `insert`, `delete`, `invalidate`, or `reset`. `request`
and `kind` are `user` or `training`. `role` is `prior` or `current`. `source`
is `commit`, `confirmation`, or `render-read`.
`path` is `full` or `light` for a trajectory. User filter work bypasses TRU and
does not produce a trajectory record. `gid` is a non-negative decimal integer
within the selected configuration. A `tile` is a 12-bit unsigned decimal value
in `0..4095`. `blend` is in `0..63`, the range generated by the public
64-entry TRU frame buffer. Frame identifiers are 16-bit unsigned decimal
values in `0..65535`, matching `ScdGsParams.frameWidth`. A `TRAJECTORY` must
use `0..65534`: TUU emits its canonical feedback at source frame plus one, so
a terminal-frame trajectory has no representable successor and is rejected.

The public TRU has no pixelwise coverage kernel. It emits the zero-extended
current `tile` as `TRAJECTORY.pixel`, so validation requires those two values to
match. This representative-pixel key is the FrameSPM record key and the
`TARGET.pixel` key used by TUU; it is not a raster coordinate.

## Record Order

Every non-`FRAME` record belongs to its enclosing open frame. A `STATE` record
must precede the first mechanism input that consumes that state. A bootstrap
`PROJECTION role=prior` record appears exactly once for a `gid` and pairs with
that Gaussian's first `UPDATE`; it must request `training` and its frame must
strictly precede the update frame. This is the logical form of the top-level
bootstrap pair, so an unconsumed or duplicate prior record is invalid. Later
updates consume the prior training summary retained by SplatSPM.
`UPDATE.frame` must strictly increase for each `gid`, matching SIU's
one-observation-per-source-frame contract.

A `PROJECTION role=current` follows the matching commit and supplies CIU's
current view. It must have the same `(gid, frame)` as that update and may
request `user` or `training`; confirmations cannot reach CIU. Exactly one
`REQUEST` must match every current view. A `TRAJECTORY` requires that same
commit and a same-frame training current view, and its `pixel` must equal that
view's `tile`; its full view fields are not duplicated in the trajectory record,
and the resulting training trajectory becomes SplatSPM's prior summary at that
frame. A user filter has no trajectory and therefore cannot replace the
retained summary.

Each `UPDATE` has exactly one same-`(gid, frame)` SIU propagation `EVENT`:
either `commit` or `confirmation`. The first four accepted observations for a
Gaussian, and every structural update, must use `commit`; later non-structural
observations may use either propagation result. Each `(gid, frame)` admits at
most one current view, propagation event, request, trajectory, and render-read
record. A current view has exactly one request. Confirmations terminate at FRU
metadata and therefore cannot have a current view, request, or trajectory.
These cardinalities match the single accepted Stage-I propagation and the
one-to-one causal relation between a TRU trajectory and its render-read.
Across each `(pixel, frame)`, the number of `TARGET` records must equal the
number of training trajectories. This represents TUU's one-buffer, one-pair
contract. The trace is causal rather than cycle-accurate: `TARGET` and
`TRAJECTORY` records may appear in either order, while the ready/valid driver
may transfer the target before the matching trajectory. `ScdGsTop` retains that
target in TUU until the trajectory can be accepted.

An `EVENT` is causal evidence rather than a free-standing annotation: each SIU
propagation event follows a validated `UPDATE` with the same `(gid, frame)`,
and every training `TRAJECTORY` has exactly one same-frame `render-read`
companion. The trajectory path is `full` or `light`. The trace grammar does
not encode the ready/valid arrival order of `TARGET` and `TRAJECTORY`, but it
requires every training trajectory to have exactly one matching target by
`(pixel, frame)`.

## Replay Output

The command-line replay validator emits only parsed record counts and does not
instantiate a timing model. `TraceReplaySpec` checks the full record shape;
`FixtureReplaySpec` uses the same fixture for a top-level feedback exercise.
