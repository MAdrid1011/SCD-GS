# TUU Tests

`TuuSpec` supplies matching trace and target records, then verifies the
per-Gaussian Adam state write and next-frame canonical update. It also checks
the sixteen-lane configuration, ready/valid holding behavior of both outputs,
and rejection of filter-tagged, user-view, or key-mismatched inputs while the
target stays buffered. It also proves that a matching terminal-frame trace is
rejected, retains that target, and emits no wrapped feedback update.
