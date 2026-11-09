# Paper-To-Repository Traceability

This matrix is the reviewer entry point. Each row must remain linked to a
design contract, source module, deterministic trace, and test.

| Mechanism | Contract | RTL | Oracle and test |
| --- | --- | --- | --- |
| SIU impact and five-observation confirmation | `mechanism/02_siu.md` | `rtl/src/main/scala/scdgs/siu/SplatImpactUnit.scala` | `reference_model/siu.py`, `SiuSpec.scala` |
| FRU event fusion and allocation | `mechanism/03_fru.md` | `rtl/src/main/scala/scdgs/fru/FusedResidencyUnit.scala` | `reference_model/fru.py`, `FruSpec.scala` |
| CIU issue equations and request rule | `mechanism/04_ciu.md` | `rtl/src/main/scala/scdgs/ciu/ConditionedIssueUnit.scala` | `reference_model/ciu.py`, `CiuSpec.scala` |
| TRU global compositing order and trajectory | `mechanism/05_tru.md` | `rtl/src/main/scala/scdgs/tru/TieredRenderUnit.scala` | `reference_model/tru.py`, `TruSpec.scala` |
| TUU feedback update | `mechanism/06_tuu.md` | `rtl/src/main/scala/scdgs/tuu/TrainingUpdateUnit.scala` | `reference_model/tuu.py`, `TuuSpec.scala` |
| Five-stage wiring and feedback handoff | `mechanism/07_memory_and_frame_protocol.md` | `rtl/src/main/scala/scdgs/top/ScdGsTop.scala` | `ScdGsLoopSpec.scala`, `FixtureReplaySpec.scala` |
| State, trace, metadata, frame storage, and candidate refill | `mechanism/07_memory_and_frame_protocol.md` | `rtl/src/main/scala/scdgs/memory/StateSpm.scala`, `MetaSpm.scala`, `SplatSpm.scala`, `FrameSpm.scala` | `MemorySpec.scala` |

The trace fixture uses only synthetic values chosen to validate record shape and
selected control branches. It is not a cycle-accurate top-level replay, is not
derived from a dataset, and carries no paper-result measurement.
