# fru.py

## External Interface

`FusedResidencyModel.commit()`, `confirmation()`, `render_read()`, and
`finish_frame()` expose the public event and five-frame allocation protocol.
`confirmation()` advances the metadata sample window without raising hotness.

## Internal Helpers

`_quantize()` creates the eight-bit update-strength field as
`floor(255 * clamp(strength, 0, 1))`; `1.0` is `255`. `_lookup()` records the direct-indexed
active/resident/candidate lookup outcome. `candidate_request_accepted()` counts
only an accepted cold line request. Window allocation assigns partition-local
rows to active and resident entries. Candidate rows remain the logical Gaussian
identifier used by the refill path.
