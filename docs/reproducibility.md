# Reproducing The Mechanism

The public artifact reproduces the dataflow on synthetic traces. It does not
attempt to recreate the paper's workloads, platform measurements, or result
figures.

```bash
python -m unittest discover -s reference_model/tests
python -m unittest discover -s tools/tests
python tools/check_public_tree.py
(cd rtl && sbt test)
(cd rtl && sbt "runMain scdgs.trace.ReplayTrace ../docs/fixtures/six_frame.scdgs-trace")
(cd rtl && sbt "runMain scdgs.emit.EmitScdGsTop --target-dir ../out/sv")
(cd out/sv && verilator --lint-only --top-module ScdGsPublicTop -f filelist.f)
```

The Python oracle and Chisel suite independently check the documented stage
rules. The ChiselSim suite uses Verilator as its simulation backend, so
Verilator is required for `sbt test`; the separate `verilator --lint-only`
command is optional. The six-frame fixture is validated for full record shape
and trace ordering. `FixtureReplaySpec` uses its first prior summary, update,
current view, and target records to parameterize a deterministic top-level
bootstrap and feedback exercise; it is not a record-by-record replay of every
fixture frame. Separate
Chisel tests cover candidate refill, queue backpressure, invalid inputs,
frame boundaries, Adam state, and generated SystemVerilog. Optional Verilator
lint is an additional structural check of generated RTL, not a timing or power evaluation.
At the default 4096-Gaussian configuration, lint elaborates the direct-indexed
state arrays and requires roughly 10 GiB of working memory. It is optional and
is separate from the semantic test sequence.

The serial ChiselSim suite creates local per-test Verilator work directories
under `rtl/build/`. Expect roughly 1 GiB of generated files and a run time of
tens of minutes; this directory is ignored and never part of a public release.
The default backend uses four compilation workers. On a host with sufficient
CPU and memory, `sbt -Dscdgs.testParallelism=<positive-integer> test` raises
only the test compilation parallelism.

The external trace converter accepts only caller-provided local records and
writes the public trace format. It does not import or download model source,
datasets, weights, checkpoints, or experimental output; a caller-provided
synthetic `TARGET` record is serialized only as the documented mechanism input.
