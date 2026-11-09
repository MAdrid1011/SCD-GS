# CIU

`ConditionedIssueUnit.scala` atomically pairs a completed residency record with
its same-`(gid, frame)` current view, then implements the projection-conditioned
full, light, and filter decision. Its issue record retains the prior SIU
summary separately from the current view and exposes the intermediate values
needed to audit the path rule.
