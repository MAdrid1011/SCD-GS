# Top-Level Loop Tests

`ScdGsLoopSpec` verifies ready/valid propagation across the public stage graph,
candidate refill behavior, rejection of a user-tagged SIU bootstrap, a
same-frame current user-view filter transaction, and a frame-end transaction.
It also proves that a later external prior summary cannot replace an accepted
SplatSPM bootstrap summary. TUU feedback is independently checked in `TuuSpec`
with its trace/target pair.
