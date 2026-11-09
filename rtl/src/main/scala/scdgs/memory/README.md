# On-Chip Storage

`StateSpm.scala` holds live Gaussian and optimizer state and produces 64-byte
aligned candidate-refill requests. Its matching fill carries the complete
StateSPM record, including Gaussian and Adam-vector state. `SplatSpm.scala`
retains prior projection summaries and training trajectories. `FrameSpm.scala`
stores target and rendered representative-pixel records in a 256-entry,
full-tagged direct-mapped store; it is not a pixelwise framebuffer. `MetaSpm.scala`
publishes valid-tagged post-event FRU metadata snapshots, while FRU remains the
live directory owner.
