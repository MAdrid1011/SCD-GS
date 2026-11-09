# CIU And Training-Conditioned Issue

CIU atomically pairs a complete residency record with current view metadata for
the same `(gid, frame)`. The residency's propagation record keeps the prior
training summary used by SIU; CIU does not reinterpret it as the current view.
It forms screen covariance
`C = JCov * Sigma * transpose(JCov)`, the bounded support proxy
`Omega = saturate(9 * abs(det(C)))`, the degree-0/degree-1 SH color estimate
`b(v)`, and:

`requestKind` belongs to the current view. `ScdGsTop.currentViewIn` admits both
user and training views, so the user filter and conservative training-light
branches are executable through the public five-stage integration.

```text
P = saturate(avg(act(opacity) * Omega * b(v)))
kappa = 1 - clamp(E, 0, 1)
rho = 1 - kappa * (1 - P)
omega = rho * Omega
```

`b(v)` remains the raw degree-0/degree-1 SH estimate. `P` is saturated only
after averaging the channelwise product, with no per-channel color clamp.
`Omega` is carried as `IssueBundle.supportArea` for TRU. It does not overwrite
the retained projection-summary `area` or `radius` consumed by the next SIU
decision.

Before issuing, CIU writes the current live-state center, depth, opacity, and
degree-0/1 color into `IssueBundle.view`. The prior summary remains in
`IssueBundle.residency.propagation.projection` for auditability. The current
view's `conic`, `radius`, and `area` fields remain unchanged because CIU has no
separate conic-inversion mechanism in the paper.

Structural records, and records with insufficient samples that are not
candidates, use full work. A record with `rho > tauRho` also uses full work. If
`omega < tauD`, a user request is filtered and a training request becomes light
work. Every remaining record is light work.

`ConditionedIssueUnit.issueOut` carries the selected path and the intermediate
decision fields so tests can validate the rule rather than infer it from a
counter. `CiuSpec` and the oracle cover all three paths.
