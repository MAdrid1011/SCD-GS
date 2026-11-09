# Top-Level Integration

`ScdGsTop.scala` connects the five stages, StateSPM, MetaSPM, SplatSPM, and
FrameSPM, the candidate line interface, frame boundary, counters, canonical
feedback, and valid-tagged MetaSPM readback. A completed training trajectory
is retained in a bounded bridge and delivered once to SplatSPM, FrameSPM, and
TUU; its matching render-read goes directly to FRU. These ready/valid links are
independent, and there is no external render-read port. `priorTrainingSummaryIn`
supplies only a strictly earlier training
summary to SIU, while `currentViewIn` pairs a same-`(gid, frame)` current user
or training view with CIU's completed residency response. Its `invalidInput`
counter combines rejected identifiers and rejected cross-record transactions.
`ScdGsPublicTop` is the named SystemVerilog emission wrapper for the same public
interface.
