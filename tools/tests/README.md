# Tool Tests

This directory verifies public command-line helper contracts without model,
dataset, platform, or network dependencies.

## Files

- `test_export_trace.py` checks that the JSON exporter accepts the public
  fixture shape and rejects a repeated per-Gaussian update frame.
