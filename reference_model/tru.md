# tru.py

## External Interface

`TieredRenderModel.process(issues)` accepts one closed, non-filtered frame and
returns training trajectories plus the count of later records stopped after
tile transmittance falls below the public threshold. It rejects a batch that
would combine multiple frame IDs, and rejects terminal-frame training work
because it cannot produce a representable next-frame feedback update.

## Internal Helpers

The oracle excludes filtered user work before checking frame identity, because
the Chisel TRU never queues it. It then orders the closed frame by `(tile,
depth, gid)`, matching the Chisel TRU. Tile and depth come from the retained
projection in each issue; residency controls issue and refill order but cannot
change compositing order. Each emitted training trajectory preserves that full
current projection for SplatSPM's next-SIU summary. Coverage is the bounded CIU
support proxy, and alpha is the Act-Unit opacity times that coverage, matching
the RTL blend path. The oracle evaluates all degree-3 SH bases for full work,
while the RTL uses the documented FP32 path. Light work round-trips the CIU
color estimate through IEEE-754 binary16.
