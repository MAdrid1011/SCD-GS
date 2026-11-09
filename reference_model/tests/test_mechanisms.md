# Mechanism Oracle Tests

`test_siu_warmup_confirmation_and_structural_protection` checks projected impact,
the five-sample gate, and structural protection. `test_fru_fuses_events_allocates_ages_and_averages_energy`
checks the 4-bit/8-bit residency state, including `0.5 -> 127` and `1.0 -> 255`
toward-zero quantization boundary, accepted
candidate refill accounting, and window aging. The CIU tests cover path
selection plus a non-diagonal `JCov * Sigma * transpose(JCov)` vector and its
`P/kappa/rho/omega` fields while preserving the retained SIU `area` and
`radius` fields. The TRU tests derive tile/depth, coverage, and
Act-Unit alpha from the issue record, then cover tile early stop, global
`(tile, depth, gid)` compositing order across three tiers, complete degree-3 SH
shading, binary16 light-color rounding, preservation of the full current
projection in the training trajectory, and rejection of a closed batch that
would combine frame IDs.
The TUU suite checks Adam feedback, one-target buffering, filter-tag, user-view,
key-mismatch, and terminal-frame retention, plus rejection of an unseeded
optimizer state.
