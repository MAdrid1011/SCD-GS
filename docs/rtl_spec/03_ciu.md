# CIU Specification

`ConditionedIssueUnit` atomically pairs a completed residency response with a
same-`(gid, frame)` current view, then derives projected support and the `P`,
`kappa`, `rho`, and `omega` decision fields. It emits full, light, or filter
work. Training filter candidates are emitted as light work, while user filter
candidates bypass TRU.
