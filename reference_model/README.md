# SCD-GS Semantic Oracle

This package is a dependency-free executable specification for the public
five-stage mechanism contract. It uses synthetic full Gaussian records, prior
training summaries, current views, and canonical updates. It is a semantic test oracle, not
a bit-exact numerical oracle for every intermediate RTL binary32 result, a
performance simulator, or a model implementation.

`tests/test_mechanisms.py` exercises each stage and the canonical-feedback
transition from one matched trace and target.
