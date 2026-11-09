# Binary32 Helpers

`Fp32.scala` wraps Berkeley HardFloat operations used by mechanism decisions.
Its threshold comparison is ordered, so an unordered NaN operand cannot satisfy
a greater-than-or-equal decision.
It provides binary32 arithmetic, comparison, conversion, and packed-FP16
conversion helpers. `scdgs.math.DivF32` supplies the sequential binary32
division unit used where the paper equations require dynamic division.
