# Public Trace Fixtures

This directory holds small synthetic traces used by format validation and
bootstrap/current-view interface tests. They cover selected causal mechanism
branches and contain no dataset, model, or experimental-result artifact.
Candidate fills are complete external StateSPM response transactions rather
than causal trace records; `FixtureReplaySpec` and `MemorySpec` exercise that
interface directly.
