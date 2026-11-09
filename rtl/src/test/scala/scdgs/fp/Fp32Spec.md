# FP32 Test

`Fp32Spec.scala` checks binary32 add, multiply, clamping, ordered comparison,
and numerical equality helpers used by SIU, CIU, TRU, and TUU decision logic.
It verifies signed-zero equality before TRU's GID tie break and rejects an
unordered NaN greater-than-or-equal comparison. It also distinguishes NaN and
infinity from the largest finite binary32 value for TRU depth validation.
