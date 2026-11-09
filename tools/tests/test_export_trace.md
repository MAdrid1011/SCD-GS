# Export Trace Tests

`test_export_trace.py` loads `tools/export_trace.py` directly and converts the
public six-frame text fixture into the exporter's JSON record shape. It verifies
that the exporter accepts the fixture and rejects a duplicate `UPDATE` for the
same Gaussian and source frame, duplicate state initialization, a non-training
prior summary, a current view without a commit, a user current view attempting
a trajectory, a trajectory whose representative pixel disagrees with its
current-view tile, a missing or mismatched render-read, and an unmatched
target/trajectory pair. It also rejects duplicate stage-handoff records. These
cases match `TraceFormat.validate`. It requires one SIU propagation per update,
preserves the required warm-up and structural commits, and accepts only a later
non-structural confirmation. It rejects out-of-width frame identifiers and a
terminal-frame training trajectory, matching the Scala validator's feedback
causality rule.
It also rejects trace tiles and blend orders that cannot be represented by the
public RTL configuration.
It rejects non-finite depth encodings before they can reach TRU ordering.
