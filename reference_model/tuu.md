# tuu.py

## External Interface

`TrainingUpdateModel.buffer_target()` accepts one target when its buffer is
empty. `process_trace()` consumes that target only for a matching trajectory
with an existing optimizer state, a training view, and a `full` or `light`
path; it rejects a `filter` tag or user-view record while retaining the target.
It also rejects terminal-frame trajectories, because the emitted feedback uses
the source frame plus one and must not wrap to zero.
`process(trajectories, targets)` is a
one-target convenience wrapper for deterministic tests. `optimizer_state(gid)`
exposes seeded semantic state and rejects a missing Gaussian identifier.

## Internal Helpers

The implementation mirrors the public 16-lane configuration at transaction
granularity. It is a deterministic semantic oracle, not a workload or timing
model.
