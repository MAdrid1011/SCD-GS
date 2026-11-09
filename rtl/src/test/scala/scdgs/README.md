# SCD-GS RTL Tests

Tests are organized by the five public stages plus top-level, memory, trace,
and generated-SystemVerilog coverage. Each test uses deterministic synthetic
records documented by `docs/trace-format.md`. `TestSupport.scala` provides the
complete-record drivers and common simulation configuration used by the suites.
