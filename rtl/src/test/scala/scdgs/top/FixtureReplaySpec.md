# Fixture Bootstrap-To-RTL Test

`FixtureReplaySpec.scala` parses and validates the checked-in public trace. It
uses the first prior summary, update, current view, and target records to
parameterize a deterministic top-level bootstrap exercise. It completes the
matching candidate refill before issuing the current view, then closes the
frame drain, verifies that no second target is accepted while TUU owns its
one-entry buffer, and observes the feedback handoff through a second SIU
commit. It is not a record-by-record or six-frame timing replay.
