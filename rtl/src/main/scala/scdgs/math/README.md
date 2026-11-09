# Math Units

`SqrtF32.scala` contains one-at-a-time binary32 square-root and division
wrappers. SIU uses them for the Frobenius norm of projected covariance change
and its radius normalization; TUU uses them for the Adam denominator.
