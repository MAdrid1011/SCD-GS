# Canonical Update And Trace

## Canonical Update

TUU emits one `CanonicalUpdateBundle` for each matched scalar-surrogate update. SIU
consumes it with the projection summary from the preceding training render.
The record retains `gid`, `frameId`, all update components, scalar strength,
and the structural marker until SIU makes the propagation decision.

An external bootstrap update and prior-training summary are one transaction.
The top level retains the summary in SplatSPM only after SIU accepts the pair:
their GIDs must match, the retained frame must strictly precede the update
frame, and the source frame must advance monotonically for that Gaussian. The
summary must be tagged `training`; a user view is not a preceding
training-render summary. A consumed invalid pair is counted but cannot
establish a prior-render summary. After that bootstrap, the external summary
port is not accepted for the same Gaussian; SIU uses only the training summary
retained by SplatSPM.

Structural markers are `insert`, `delete`, `invalidate`, and `reset`. They
bypass low-impact suppression and select the protected full path in CIU.

## Forward Trace

CIU pairs each complete residency record with a current view that has the same
`(gid, frame)`. The current view carries the request kind for Stage III and is
not the prior summary consumed by SIU. For each training render, TRU records
Gaussian ID, frame, representative pixel, blend order, coverage, transmittance,
RGB color, selected path, and that full current view. The representative pixel
is the zero-extended current tile because the public renderer has no pixelwise
coverage kernel. SplatSPM retains the trajectory and the current training view
as the next prior summary, while FrameSPM receives the corresponding record.
TUU pairs the trajectory with a target of the same `(pixel, frame)` key and
does not manufacture an update for a mismatched pair.

The public trace records causal mechanism records rather than a cycle-by-cycle
ready/valid waveform. Every canonical update has one SIU propagation event:
`commit` or zero-strength `confirmation`. The first four accepted observations
and structural updates are commits. No request or trajectory can follow without
a commit; confirmations terminate at FRU metadata. A trajectory requires a
same-frame training current view and uses full or light work; it becomes the
retained training summary for the next update. User filter work has no
trajectory. The accompanying `render-read` shares the trajectory's `(gid,
frame)`. These are causal companion records, not a same-cycle transfer
requirement: their ready/valid links may complete independently exactly once.
TRU transfers the render-read directly to FRU without waiting for a target. A
bounded top-level bridge retains the trajectory until SplatSPM, FrameSPM, and
TUU each accept their own transfer. The target stream independently fills TUU's
one-entry buffer and writes FrameSPM, so it may arrive before the trajectory. A
skipped early-stop record has no trajectory and therefore produces no TUU pair.

`docs/trace-format.md` defines the portable text form. `ReplayTrace` validates
the format. `TraceReplaySpec` and the fixture test its required records.
