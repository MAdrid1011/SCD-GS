# export_trace.py

## External Interface

```bash
python tools/export_trace.py --input-json local_export.json --output trace.scdgs-trace
```

`local_export.json` is a list of pre-extracted mechanism records. The converter
validates the public grammar, including complete Gaussian state, 48-coefficient
SH updates, complete camera and projection records, and frame ordering before
writing UTF-8 trace lines. It does not load a model, data, weight, result
table, or platform profiler.

Use `record_type: "REQUEST"` for JSON request records because `kind` is the
request field. All other records use their `kind` field as the record type.

## Internal Helpers

`format_record()` validates required and forbidden keys before serializing a
single record. `validate_records()` validates vector widths, retained state and
projection dependencies, prior/current view roles, request/view agreement,
frame boundaries, and strictly increasing `UPDATE.frame` values per Gaussian.
It also preserves the pipeline's causal edges: every update requires one SIU
`commit` or `confirmation` event, the first four accepted observations and
structural updates require commits, requests and trajectories require their
matching commits, updates require a strictly earlier prior training summary,
trajectories require a same-frame training current view, whose tile matches the
trajectory pixel, and use a full or light path. Render-read events require a
same-frame training trajectory for the same Gaussian. It rejects duplicate
current-view, propagation, request, trajectory, and render-read records for one
`(gid, frame)`.
`main()` performs only file conversion.

The pinned public shape rejects tile identifiers outside the 12-bit
`ProjectionSummaryBundle.tile` field and blend orders outside the 64-entry TRU
frame buffer's `0..63` output range. It also rejects non-finite
`ProjectionSummaryBundle.depth` encodings before trace ordering is evaluated.
