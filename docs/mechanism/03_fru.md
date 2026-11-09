# FRU And Event-Fused Residency

FRU accepts SIU commit events and training-render read events. Each
direct-indexed tier-directory entry records validity, Gaussian ID, tier, row,
`h[3:0]`, and `d[7:0]`. Active and resident rows are physical partition rows.
Candidate row metadata is the logical `gid` location used by the refill store.
A commit allocates with `h=1` and `d = floor(255 * clamp(effectiveStrength,
0, 1))`; the inclusive `1.0` endpoint maps to `255`. A hit saturates `h` and
takes the maximum `d`. A render read only increments `h`.

A zero-strength SIU confirmation remains a metadata observation: it updates
the `d`/energy/sample window fields and touched bit, but neither increments
`h` nor emits a StateSPM residency transfer.

The energy accumulator sums the effective-strength field itself. CIU clamps
only the resulting mean when it forms `kappa`; FRU does not introduce an
additional nonnegative clamp.

Lookup observes active, then resident, then candidate state. An active hit does
not require a later-region lookup. A candidate placement, or a candidate
payload that has not completed its matching refill, carries an explicit refill
request and must receive a complete StateSPM response before CIU can issue it.
After that transaction, a candidate payload may be served locally until its
placement changes. At each
completed five-frame window, priority is `h` first, `d` second, and ascending
`gid` last. `E` is the mean effective strength over the observed records in
that window. Untouched entries age by right shift and every `d`, energy
accumulator, and sample count is cleared for the new window.

FRU publishes a snapshot after each accepted event and while it applies and
ages every valid allocator entry. MetaSPM therefore converges to the final
tier, row, hotness, and cleared short-window fields after each allocation;
MetaSPM is readback storage, not the live directory owner.

`FusedResidencyUnit` emits the selected tier, row, refill flag, and lookup
stage. `FruSpec` checks event fusion, saturation, priority allocation, reset of
the recent-update field, and untouched aging.
