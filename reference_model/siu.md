# siu.py

## External Interface

`SplatImpactModel.process(update, projection)` returns a propagation record
containing `psi`, five-observation `phi`, effective strength, and propagation tag.

The projection must be tagged `training`; a user view cannot supply SIU's prior
training-render metadata.

## Internal Helpers

`_activation()` supplies the oracle opacity activation. The matrix helpers
evaluate the same camera, projection, covariance, and SH contractions as the
public SIU boundary.
