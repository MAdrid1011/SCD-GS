"""
OpQUEEN — Stage-0 operator slot for QUEEN.

QUEEN's training has a per-frame outer loop (frame_idx) with an inner
iteration loop. Each frame:
  - The GaussianModel holds Gaussian attributes in standard groups
  - A latent LatentDecoder decodes per-frame residuals (group "decoder")
  - Gate modules (group "gate") control which attributes are updated
  - Rendering uses render_mask() or render_mask_shift() instead of render()

The optimizer combines Gaussian attrs + decoder + gate into one optimizer
(gaussians.optimizer in QUEEN's train.py). No separate NTC optimizer.

Param groups (as declared in QUEEN/scene/gaussian_model.py)
-----------------------------------------------------------
  Gaussian: "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"
  Network:  "decoder", "gate"

Hook installation
-----------------
  gaussians.optimizer.step → pipeline.pre_optimizer_step(optimizer)
  begin_frame() must be called externally by the ablation runner at each
  new frame boundary (QUEEN's frame loop is in train.py, not in the model).

QUEEN render_mask() compatibility
----------------------------------
During FPS measurement the ablation runner calls render_mask() (or render())
with the MaskedGaussianProxy. QUEEN's render_mask() accepts a `mask`
argument that is ANDed with its internal rasterisation settings.
When using MaskedGaussianProxy, the proxy already represents the filtered
subset, so `mask=None` is correct (render all elements of the proxy).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class OpQUEEN:
    """Stage-0 operator slot for QUEEN compressed Gaussian splatting."""

    gaussian_attr_groups: list[str] = [
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
    ]
    network_attr_groups: list[str] = [
        "decoder", "gate",
        # QUEEN names latent decoder groups as "ldec_<attr>" (e.g. "ldec_f_rest").
        # The list above is kept as a logical fallback; the actual measurement
        # in run_queen.py filters groups NOT in gaussian_attr_groups, which
        # captures all ldec_* and gate groups regardless of exact naming.
    ]

    def __init__(self) -> None:
        self._gaussians    = None
        self._optimizer    = None
        self._orig_step    = None
        self._pipeline: "GLASSPipeline | None" = None

    # ── Hook installation ─────────────────────────────────────────────────────

    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        gaussians=None,
        optimizer=None,
        **_,
    ) -> None:
        """Install optimizer hook into QUEEN's combined optimizer.

        Parameters
        ----------
        pipeline  : GLASSPipeline
        gaussians : QUEEN GaussianModel
        optimizer : gaussians.optimizer (the unified QUEEN optimizer)
        """
        if gaussians is None or optimizer is None:
            raise ValueError(
                "OpQUEEN.install_hooks requires gaussians= and optimizer="
            )

        self._gaussians = gaussians
        self._optimizer = optimizer
        self._pipeline  = pipeline

        orig = optimizer.step

        def _step(closure=None):
            pipeline.pre_optimizer_step(optimizer)
            return orig(closure)

        optimizer.step  = _step
        self._orig_step = orig

    def uninstall_hooks(self) -> None:
        if self._optimizer is not None and self._orig_step is not None:
            self._optimizer.step = self._orig_step
        self._gaussians = None
        self._optimizer = None
        self._pipeline  = None

    # ── Gaussian access ───────────────────────────────────────────────────────

    def n_gaussians(self) -> int:
        if self._gaussians is None:
            return 0
        return int(self._gaussians.get_xyz.shape[0])

    def get_gaussians(self):
        return self._gaussians

    def __repr__(self) -> str:
        n = self.n_gaussians()
        return f"OpQUEEN(n_gaussians={n})"
