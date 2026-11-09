# Contributing To SCD-GS

Contributions must preserve the five-stage mechanism contract. A change is not
complete until its design document, deterministic oracle, Chisel interface, and
test all agree.

## Required Order

1. Update `docs/mechanism-contract.md`, the relevant `docs/rtl_spec` file, and
   `docs/traceability.md`.
2. Add a failing oracle or Chisel test that states the expected mechanism rule.
3. Implement the RTL and update the trace fixture if a public input changed.
4. Run the public verification commands in `docs/reproducibility.md`.

## Interface Discipline

- Preserve `gid`, frame identity, structural marker, and request kind across
  every stage boundary.
- Do not add a scalar-only compatibility path for canonical updates.
- Do not change a threshold, window, tier priority, or fallback rule without a
  matching contract and fixture update.
- Candidate memory requests must remain 64-byte aligned.

## Scope Discipline

Do not add model packages, datasets, weights, benchmark runners, result tables,
or platform measurements. Public trace tools may consume a local export but
must emit only the documented mechanism records and counters.

## Verification

```bash
python -m unittest discover -s reference_model/tests
python tools/check_public_tree.py
(cd rtl && sbt test)
```

Generated RTL is checked locally and is not committed. Pull requests must state
which traceability row changed and include the exact verification output.
