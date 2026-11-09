# SIU Specification

`SplatImpactUnit` joins canonical updates with strictly earlier, training-tagged
SplatSPM projection summaries.
It produces a propagation record containing `psi`, `Phi`, effective strength,
and a confirmation/state-change tag. Five samples are required for a
non-structural confirmation. Structural records always propagate. Its Act Unit
uses the specified saturating affine binary32 activation function.
