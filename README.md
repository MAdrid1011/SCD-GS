# SCD-GS: A State-Conditioned Dataflow Accelerator for Streaming Gaussian Systems

This repository is a mechanism-faithful Chisel companion for SCD-GS. It makes the
paper's five-stage training and rendering loop inspectable and executable:

```text
canonical update -> SIU -> FRU -> CIU -> TRU -> TUU -> canonical update
```

The public artifact is a reference implementation of the dataflow, interfaces,
state transitions, deterministic trace validation, and bounded top-level
stimulus. It is not an experimental
reproduction package: it does not include model sources, datasets, checkpoints,
platform-specific profiling artifacts, or paper-result tables.

## Mechanisms

| Stage | Unit | Public evidence |
| --- | --- | --- |
| I | Splat Impact Unit (SIU) | Screen-space impact, five-observation confirmation, structural protection |
| II | Fused Residency Unit (FRU) | Propagation/read event fusion, direct-indexed tier directory, residency allocation and aging |
| III | Conditioned Issue Unit (CIU) | Projection-derived full/light/filter selection |
| IV | Tiered Render Unit (TRU) | Global compositing order across tiers, transmittance early stop, depth-preserving render trace |
| V | Training Update Unit (TUU) | Matched trace/target surrogate update, optimizer update, canonical feedback |

Start at [`docs/mechanism/00_paper_contract.md`](docs/mechanism/00_paper_contract.md)
for the paper-mechanism map, then use
[`docs/traceability.md`](docs/traceability.md) to jump to the implementation
and tests.

## Verify The Public Artifact

Requirements are a JDK supported by SBT, SBT 1.10.11, Python 3.10 or newer,
and Verilator. The ChiselSim regression suite uses Verilator as its simulation
backend. The separate `verilator --lint-only` command is an optional additional
structural check.

The optional default-configuration lint elaborates all 4096 direct-indexed
Gaussian records and is memory-intensive. Plan for roughly 10 GiB of available
working memory for that command; the semantic Python and Chisel tests do not
require the lint step.

The serial ChiselSim suite builds one local Verilator simulation per test. It
can take tens of minutes and uses roughly 1 GiB under `rtl/build/`; that
generated directory is ignored and is not part of the public artifact.

```bash
python -m unittest discover -s reference_model/tests
python -m unittest discover -s tools/tests
python tools/check_public_tree.py
(cd rtl && sbt test)
(cd rtl && sbt "runMain scdgs.trace.ReplayTrace ../docs/fixtures/six_frame.scdgs-trace")
(cd rtl && sbt "runMain scdgs.emit.EmitScdGsTop --target-dir ../out/sv")
(cd out/sv && verilator --lint-only --top-module ScdGsPublicTop -f filelist.f)
```

The fixture is deliberately small and deterministic. Its counters demonstrate
mechanism behavior; they do not represent dataset accuracy, throughput, energy,
area, or a reproduction of the paper's measurements.

## Layout

```text
docs/                 Mechanism contract, trace format, verification guide
docs/fixtures/        Public deterministic trace fixtures
reference_model/      Executable semantic oracle with stage-level checks
rtl/                  Chisel implementation and SystemVerilog emitter
tools/                Public-tree audit and external-trace conversion helpers
```

## Scope And Provenance

The private manuscript is the source of truth used to derive this public
mechanism contract. It is intentionally not part of the repository. Reported
experimental results remain paper provenance and are neither regenerated nor
represented as public-repository output.

## License

SCD-GS source is licensed under Apache-2.0. Third-party Berkeley HardFloat is
retained under its BSD-3-Clause license; see `NOTICE` and
`rtl/vendor/berkeley-hardfloat/LICENSE`.
