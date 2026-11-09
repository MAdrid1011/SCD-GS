# SIU And Temporal Splat Impact

SIU joins a canonical update with SplatSPM metadata from the previous training
render. It computes projected-center displacement, the Frobenius norm of the
2x2 projected covariance change, opacity-activation difference, SH luminance
change, and coverage. The
public Act Unit uses `clamp(0.5 + 0.25 * x, 0, 1)` as its binary32 activation
function. The oracle uses the same function. Their weighted sum is `psi`.
`phi` is the per-Gaussian mean of its latest five accepted observations.

The prior summary must carry the `training` request tag and have a strictly
earlier frame than its canonical update. SIU rejects a user view or a
same-frame summary as bootstrap metadata, because neither can be the preceding
training render from which Stage I derives its retained basis and geometry.

The retained radius is a positive normal binary32 denominator. SIU rejects a
bootstrap pair with a zero, subnormal, negative, infinite, or NaN radius
before it can update SplatSPM or its observation history.

For the first four observations of a Gaussian, SIU protects the record and
propagates its effective strength. After warm-up, `phi >= tauPhi` propagates a
state change. Otherwise SIU emits a confirmation with effective strength zero.
Insertion, deletion, invalidation, and reset always retain effective strength.

The implementation boundary is `SplatImpactUnit.updateIn`,
`priorSummaryIn`, and `propagationOut`. The unit test covers warm-up, low-impact
confirmation, structural protection, and invalid-ID handling. The oracle uses
the same state-machine contract without modeling execution time.
