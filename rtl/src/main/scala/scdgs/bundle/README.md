# Bundle Interfaces

`ScdGsBundles.scala` defines every public record exchanged by SIU, FRU, CIU,
TRU, TUU, StateSPM, and SplatSPM. It is the single source for canonical
updates, prior summaries, current views, residency records, issue records,
render traces, and memory lines. `IssueBundle` preserves the SIU prior summary
inside its residency record while carrying the CIU current view separately.
