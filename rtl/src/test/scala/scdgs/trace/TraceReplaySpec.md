# Trace Replay Tests

`TraceReplaySpec` validates complete Gaussian state, full camera and projection,
and 48-coefficient feedback-update vectors in the six-frame public fixture. It
also rejects unknown record fields, short SH vectors, mismatched requests, and
an out-of-range Gaussian identifier. It rejects a repeated `UPDATE` source
frame for one Gaussian, duplicate state initialization, a malformed prior
training summary, a current view without an SIU commit, a render-read that is
missing or has a mismatched source frame, an unmatched target/trajectory pair,
duplicate stage handoffs, a trajectory whose representative-pixel key does not
match its current-view tile, and a user current view attempting a trajectory.
It also requires one SIU propagation event per update, including commits for
warm-up and structural updates. Every validated trace therefore satisfies SIU's
strict training-summary and propagation rules, CIU/TRU view-causality, and TUU
pairing rules. The tests also reject frame identifiers outside the public
16-bit domain and a terminal-frame training trajectory that cannot produce
next-frame TUU feedback.
It also rejects values outside the public 12-bit tile field and the 64-entry
TRU blend-order range.
It rejects NaN and infinity depth encodings that would make numerical depth
ordering undefined.
