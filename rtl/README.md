# SCD-GS RTL

This directory contains the Chisel implementation of the public five-stage
mechanism contract. `ScdGsTop` connects SIU, FRU, CIU, TRU, and TUU with the
documented StateSPM, SplatSPM, FrameSPM, and MetaSPM storage boundaries.

## Commands

Run `sbt test` for deterministic mechanism tests. Generate inspectable
SystemVerilog with:

```bash
sbt "runMain scdgs.emit.EmitScdGsTop --target-dir ../out/sv"
```

Generated files are intentionally local-only.
