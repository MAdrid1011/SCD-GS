# ciu.py

## External Interface

`ConditionedIssueModel.process(residency, current_view)` requires a complete
StateSPM response and a current view with the residency update's `(gid, frame)`.
It returns a full, light, or filter issue record with `P`, `kappa`, `rho`, and
`omega` fields.

## Internal Helpers

Threshold evaluation follows the exact request fallback rule in the mechanism
contract.
