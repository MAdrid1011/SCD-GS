"""
OpSwift4D — Stage-0 operator slot for Swift4D.

Swift4D jointly trains on all timestamps simultaneously (unlike per-frame
incremental baselines). Its training loop is a standard PyTorch outer
iteration loop (no separate per-frame step functions).

Adaptation for the ASIC simulator
-----------------------------------
Since there is no per-frame structure, we treat each optimizer iteration as
one logical "frame" for the Correlation Window Buffer (CWB). This means:
  - pipeline.begin_frame() fires inside the wrapped optimizer.step()
  - CWB history spans the last cwb_window *iterations*, not calendar frames
  - Training time is reported as total_time / n_iterations (per-step average)
    and flagged as incremental=False in the output JSON

Param groups (Swift4D GaussianModel, same layout as standard 3DGS)
-------------------------------------------------------------------
  Gaussian: "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"
  Network:  "hash" (hash-grid dynamic branch / deformation network)

Swift4D sometimes names its hash grid group "hash_grid" or "deform_hash".
AttrClass.from_group_name handles these via the NETWORK fall-through path.

Hooks installed
---------------
  optimizer.step → pipeline.begin_frame() + pipeline.pre_optimizer_step()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class OpSwift4D:
    """Stage-0 operator slot for Swift4D joint multi-frame Gaussian training."""

    gaussian_attr_groups: list[str] = [
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
    ]
    network_attr_groups: list[str] = [
        # Swift4D's dynamic latent branch
        "hash", "hash_grid", "deform_hash",
        # Coarse/fine MLP branches (named differently across configs)
        "coarse_mlp", "fine_mlp", "deform",
    ]

    def __init__(self) -> None:
        self._gaussians    = None
        self._optimizer    = None
        self._orig_step    = None
        self._pipeline: "GLASSPipeline | None" = None
        self._step_count   = 0

    # ── Hook installation ─────────────────────────────────────────────────────

    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        gaussians=None,
        optimizer=None,
        **_,
    ) -> None:
        """Wrap the joint optimizer so each step is treated as a logical frame.

        Parameters
        ----------
        pipeline  : GLASSPipeline
        gaussians : Swift4D GaussianModel
        optimizer : the unified optimizer from Swift4D's train.py
        """
        if gaussians is None or optimizer is None:
            raise ValueError(
                "OpSwift4D.install_hooks requires gaussians= and optimizer="
            )

        self._gaussians  = gaussians
        self._optimizer  = optimizer
        self._pipeline   = pipeline
        self._step_count = 0

        orig = optimizer.step

        def _step(closure=None):
            self._step_count += 1
            # Each iteration is a logical "frame" for CWB and hotness
            pipeline.begin_frame()
            pipeline.pre_optimizer_step(optimizer)
            return orig(closure)

        optimizer.step  = _step
        self._orig_step = orig

    def uninstall_hooks(self) -> None:
        if self._optimizer is not None and self._orig_step is not None:
            self._optimizer.step = self._orig_step
        self._gaussians  = None
        self._optimizer  = None
        self._pipeline   = None

    # ── Gaussian access ───────────────────────────────────────────────────────

    def n_gaussians(self) -> int:
        if self._gaussians is None:
            return 0
        return int(self._gaussians.get_xyz.shape[0])

    def get_gaussians(self):
        return self._gaussians

    @property
    def step_count(self) -> int:
        """Total optimizer steps executed since install_hooks()."""
        return self._step_count

    def __repr__(self) -> str:
        n = self.n_gaussians()
        return f"OpSwift4D(n_gaussians={n}, steps={self._step_count})"
