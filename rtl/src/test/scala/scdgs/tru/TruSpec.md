# TRU Tests

`TruSpec` closes a frame before execution, checks global `(tile, depth, gid)`
compositing order across three tiers, then
checks that resident work for a saturated tile is discarded after the active
blend drives transmittance below the public stop threshold. A separate numeric
vector checks all 16 degree-3 SH bases on full work and binary16 rounding on
the light path. It also proves that an empty later frame boundary cannot reset
the blend order of a frame that is still draining. A signed-zero regression
verifies that depth is compared numerically before the Gaussian-ID tie break.
It also verifies that a next-frame issue is backpressured until the preceding
closed frame drains, and that a ready FRU render-read can transfer once while
the matching trajectory remains independently backpressured. An undefined path
tag is discarded before it can enter the frame buffer or emit either output.
Terminal-frame training work is likewise rejected before it can create a
trajectory, render-read, or blend side effect.
Non-finite depth is rejected before it can enter the global compositing order.
