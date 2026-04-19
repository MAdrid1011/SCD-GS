"""
BaseOperatorSlot — abstract base class for Stage-0 operator slots.

Each concrete slot (op_3dgstream.py, op_queen.py, op_hicom.py, op_swift4d.py)
is responsible for exactly three things:

  1. Declaring which param-group names are per-Gaussian Gaussian attributes
     and which are non-per-Gaussian network parameters.
     (Used by GLASSPipeline for descriptor capture and cold-mask application.)

  2. Installing and uninstalling runtime hooks into the baseline's training
     loop so that GLASSPipeline.pre_optimizer_step() and post_render() are
     called at the correct points.

  3. Exposing the current Gaussian count and the Gaussian model object so
     GLASSPipeline can build MaskedGaussianProxy instances.

The operator slot does NO computation — all Stage 1-5 logic lives in
GLASSPipeline and the core/ modules. The slot is just plumbing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class BaseOperatorSlot(ABC):
    """Abstract Stage-0 operator slot.

    Subclasses must fill in the two class-level group lists and implement
    the four abstract methods. All other logic belongs in GLASSPipeline.

    Class variables (override in subclasses)
    -----------------------------------------
    gaussian_attr_groups : list[str]
        Optimizer param-group names that contain per-Gaussian attributes.
        GLASSPipeline uses this list to:
        - Capture per-Gaussian descriptors (Stage 0)
        - Apply per-Gaussian gradient scales (Stage 1)
        - Zero cold-Gaussian gradients (Stage 2+3)

    network_attr_groups : list[str]
        Optimizer param-group names that contain non-per-Gaussian network
        parameters (NTC, latent decoders, gating networks, hash grids, etc.).
        These receive the harder ntc_grad_threshold instead of per-Gaussian
        similarity-based scaling.
    """

    gaussian_attr_groups: list[str] = []
    network_attr_groups:  list[str] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        **model_ctx,
    ) -> None:
        """Install runtime hooks into the baseline training loop.

        Must monkey-patch the relevant optimizer.step() calls and any
        per-frame boundary markers so that:
          - pipeline.pre_optimizer_step(optimizer, ...) fires before each step
          - pipeline.post_render(render_pkg) fires after each render call
          - pipeline.begin_frame() fires at the start of each new frame

        model_ctx contains model objects (gaussians, optimizer, trainer,
        module, render_fn, etc.) as keyword arguments specific to each model.

        The slot must store enough internal state to support uninstall_hooks().
        """
        ...

    @abstractmethod
    def uninstall_hooks(self) -> None:
        """Restore all monkey-patched methods to their original callables.

        Called after each ablation run to ensure the next mode starts from
        a clean baseline state.
        """
        ...

    @abstractmethod
    def n_gaussians(self) -> int:
        """Return the current total Gaussian count.

        For models with separate base + incremental Gaussian sets
        (e.g. HiCoM), this must return the combined count.
        """
        ...

    @abstractmethod
    def get_gaussians(self) -> object:
        """Return the Gaussian model (or merged model) for proxy wrapping.

        The returned object must expose the standard property API:
          .get_xyz, .get_opacity, .get_scaling, .get_rotation,
          .get_features, .max_radii2D, .active_sh_degree, .max_sh_degree
        """
        ...

    # ── Optional override ─────────────────────────────────────────────────────

    def notify_structural_change(
        self,
        added: list[int],
        removed: list[int],
    ) -> None:
        """Called when Gaussians are spawned, split, or pruned.

        The default implementation is a no-op. Subclasses that need to
        track structural events (e.g. to accumulate struct_changed IDs
        for the next optimizer hook call) should override this.

        The pipeline itself calls salience.mark_structural() from
        pre_optimizer_step(); this method is a convenience entry point
        for the hook to pass the IDs before that call.
        """
        pass

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}("
            f"gaussian_groups={self.gaussian_attr_groups}, "
            f"network_groups={self.network_attr_groups})"
        )
