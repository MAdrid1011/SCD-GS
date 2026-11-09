# Paper Mechanism Contract

The public artifact implements the runtime mechanism in the paper as a
five-stage feedback stream:

```text
canonical update -> SIU -> FRU -> CIU -> TRU -> TUU -> canonical update
```

`CanonicalUpdateBundle` is the sole state-update record passed between stages.
It carries a Gaussian identifier, frame identifier, geometry, covariance,
opacity, SH, scalar strength, and structural marker. A scalar-only update is
not a substitute for this record.

| Paper mechanism | Public contract | Implementation | Test evidence |
| --- | --- | --- | --- |
| Stage I temporal impact | `02_siu.md` | `scdgs/siu/SplatImpactUnit.scala` | `SiuSpec.scala`, `reference_model/siu.py` |
| Stage II residency | `03_fru.md` | `scdgs/fru/FusedResidencyUnit.scala` | `FruSpec.scala`, `reference_model/fru.py` |
| Stage III issue | `04_ciu.md` | `scdgs/ciu/ConditionedIssueUnit.scala` | `CiuSpec.scala`, `reference_model/ciu.py` |
| Stage IV rendering | `05_tru.md` | `scdgs/tru/TieredRenderUnit.scala` | `TruSpec.scala`, `reference_model/tru.py` |
| Stage V update | `06_tuu.md` | `scdgs/tuu/TrainingUpdateUnit.scala` | `TuuSpec.scala`, `reference_model/tuu.py` |
| Storage and frame protocol | `07_memory_and_frame_protocol.md` | `scdgs/memory/`, `scdgs/top/ScdGsTop.scala` | `MemorySpec.scala`, `ScdGsLoopSpec.scala` |

The fixed six-frame trace is a record-shape and branch-oriented fixture.
`TraceReplaySpec` validates its grammar, while `TuuSpec` and top-level tests
validate the feedback handoff. It validates grammar and selected top-level
bootstrap records, not timing of every fixture record. It does not establish
image quality, throughput, energy, area, or any reported measurement.
