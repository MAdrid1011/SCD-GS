# TestSupport.scala

## External Interface

`TestSupport` provides deterministic binary32 constants, a bounded
`stepUntil(clock, maximum)(condition)` wait helper, and complete-record drivers
for every public bundle used by the Chisel tests. `regressionSimulator` selects
the shared Verilator configuration used by suites that import this object.

`driveState`, `driveUpdate`, `driveProjection`, `drivePropagation`,
`driveResidency`, and `driveIssue` populate the complete record hierarchy for
the five-stage dataflow. `driveOptimizerState`, `driveStateWrite`, and
`driveCandidateFill` populate the full StateSPM and optimizer payloads. Each
helper accepts identifiers and frame IDs explicitly so test records preserve
the public transaction identity.

## Internal Configuration

The simulator output split and parallelism settings bound C++ compilation work
for the large binary32 combinational networks. The default uses four workers.
On a host with more available CPU and memory, pass a positive JVM property such
as `sbt -Dscdgs.testParallelism=16 test` to use that many workers for both the
ChiselSim compiler budget and Verilator. This affects only test execution and
does not alter generated hardware or test stimulus.
