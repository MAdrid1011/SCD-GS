# Public Trace Validation

`ReplayTrace.scala` parses and validates the portable synthetic trace grammar,
including complete Gaussian state, 48-coefficient SH updates, and complete
camera and prior/current projection vectors, then reports record counts.
`FixtureReplaySpec` uses the fixture bootstrap and current-view records for a
top-level feedback exercise. Neither component is a
workload or result simulator. The grammar uses the default 16-bit public frame
domain and excludes a terminal-frame training trajectory because TUU feedback
uses source frame plus one.
