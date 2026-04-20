"""
OpHiCoM — Stage-0 operator slot for HiCoM.

HiCoM's training loop lives in the libgs-based Trainer + Module stack:
  Trainer.frame_training_loop()          — outer frame loop
    Module.pre_training_step()
    loss, metrics, render_pkg = Module.training_step(viewpoint, step)
    loss.backward()
    Module.post_training_step(render_pkg) — optimizer.step() lives here

The deformation network (Module.deform) has its own optimizer accessed via
Module.deform_optimizer or resolved from Module.deform.parameters().

Two Gaussian model objects coexist:
  Module.gaussians       — base Gaussians (loaded/carried across frames)
  Module.gaussians_incr  — incremental Gaussians added per frame

For the proxy, we need their combined state. We construct a lightweight
"merged" accessor (MergedGaussianAccessor) that concatenates their attrs
without importing libgs's MergedGaussianModel (which requires the hicom
conda env). The proxy operates on the concatenated tensors.

Hooks installed
---------------
  trainer.frame_training_loop → inject pipeline.begin_frame() at each frame
  module.post_training_step   → inject pipeline.pre_optimizer_step() for each
                                optimizer (gaussians.optimizer,
                                gaussians_incr.optimizer if set,
                                deform optimizer if present)
                                then pipeline.post_render(render_pkg)
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class _MergedGaussianAccessor:
    """Lightweight merged-Gaussian view for HiCoM (no libgs import needed).

    Concatenates base and incremental Gaussian attribute tensors along dim 0.
    Compatible with the standard get_xyz / get_opacity / ... property API
    required by MaskedGaussianProxy.
    """

    def __init__(self, base, incr) -> None:
        self._base = base
        self._incr = incr

    def _cat(self, attr: str) -> torch.Tensor:
        b = getattr(self._base, attr)
        # ``_incr`` may fall back to ``_base`` when HiCoM has not yet spawned
        # an incremental GaussianModel (e.g. frame 1 stage 1).  Concatenating
        # in that case would double the row count and break downstream code
        # that expects ``shape[0] == n_total_gaussians``.
        if self._incr is self._base:
            return b
        try:
            i = getattr(self._incr, attr)
            if i is not None and i.numel() > 0:
                return torch.cat([b, i], dim=0)
        except Exception:
            pass
        return b

    @property
    def get_xyz(self):      return self._cat("get_xyz")
    @property
    def get_opacity(self):  return self._cat("get_opacity")
    @property
    def get_scaling(self):  return self._cat("get_scaling")
    @property
    def get_rotation(self): return self._cat("get_rotation")
    @property
    def get_features(self): return self._cat("get_features")
    @property
    def max_radii2D(self):  return self._cat("max_radii2D")
    @property
    def active_sh_degree(self): return self._base.active_sh_degree
    @property
    def max_sh_degree(self):    return self._base.max_sh_degree


class OpHiCoM:
    """Stage-0 operator slot for HiCoM hierarchical motion Gaussians."""

    gaussian_attr_groups: list[str] = [
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
    ]
    network_attr_groups: list[str] = [
        "deform",           # deformation MLP weight group name in libgs
        "deform_optimizer", # sentinel in case group is named differently
    ]

    def __init__(self) -> None:
        self._module   = None
        self._trainer  = None
        self._pipeline: "GLASSPipeline | None" = None
        self._orig_frame_loop = None
        self._orig_post_step  = None

    # ── Hook installation ─────────────────────────────────────────────────────

    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        trainer=None,
        module=None,
        **_,
    ) -> None:
        """Install frame-boundary and post-training-step hooks.

        Parameters
        ----------
        pipeline : GLASSPipeline
        trainer  : HiCoM Trainer instance (has frame_training_loop)
        module   : HiCoM Module instance  (has post_training_step)
        """
        if trainer is None or module is None:
            raise ValueError(
                "OpHiCoM.install_hooks requires trainer= and module="
            )

        self._module   = module
        self._trainer  = trainer
        self._pipeline = pipeline

        # ── Frame boundary: begin_frame() at each new frame ───────────────────
        orig_frame = trainer.frame_training_loop

        def _frame_loop():
            pipeline.begin_frame()
            orig_frame()

        trainer.frame_training_loop = _frame_loop
        self._orig_frame_loop = orig_frame

        # ── Post-training-step: intercept all optimizers ──────────────────────
        orig_post = module.post_training_step

        # HiCoM owns three optimizers that live *inside* sub-modules of the
        # libgs Module, not as direct attributes on it.  Only one of them
        # actually calls ``.step()`` on a given iteration, and which one
        # depends on the frame / stage (see ``post_training_step_incr``):
        #
        #   frame 0                           → module.gaussians.optimizer
        #   frame 1+, stage 1 (step ≤ N1)     → module.deform.optimizer
        #   frame 1+, stage 2 (step >  N1)    → module.gaussians_incr.optimizer
        #
        # where ``N1 = module.deform.config.num_stage1_steps``.
        #
        # ``pipeline.pre_optimizer_step`` must only fire on the optimizer
        # that is really about to step; otherwise the base-Gaussian .grad
        # (which stays populated via the shared rasteriser backward but is
        # never applied in frame 1+) would be double-counted as extra ASIC
        # update traffic that the hardware never emits.
        #
        # The deform optimizer's param_groups have no ``name`` field and
        # hold only MLP parameters — not per-Gaussian attributes — so even
        # if we called pre_optimizer_step on it, the pipeline's descriptor
        # capture would match nothing in ``gaussian_attr_groups``.  We skip
        # it explicitly to keep the hook free of no-op pipeline calls.
        def _stepping_gaussian_optimizer():
            if getattr(trainer, "is_first_frame", False):
                return getattr(getattr(module, "gaussians", None),
                               "optimizer", None)
            deform = getattr(module, "deform", None)
            if deform is None:
                return None
            stage1_steps = int(getattr(deform.config, "num_stage1_steps", 0))
            if module.global_step > stage1_steps:
                incr = getattr(module, "gaussians_incr", None)
                if incr is None:
                    return None
                return getattr(incr, "optimizer", None)
            return None  # stage 1: only the deform MLP steps — no GS traffic

        def _post_step(render_pkg):
            opt = _stepping_gaussian_optimizer()
            if opt is not None:
                pipeline.pre_optimizer_step(opt)
            orig_post(render_pkg)
            pipeline.post_render(render_pkg)
            # Feed scene statistics to the ASIC cycle model so that the
            # runner can later account for the render + loss + backward
            # cycles of this frame via add_gaussian_training_iters()
            # (lag-one design: stats from frame N size the cycles of
            # frame N's training loop, flushed at begin_frame N+1).
            try:
                sh_deg = int(getattr(module.gaussians, "active_sh_degree", 0))
            except Exception:
                sh_deg = 0
            try:
                pipeline.set_render_stats_from_render_pkg(render_pkg, sh_deg)
            except Exception:
                pass

        module.post_training_step = _post_step
        self._orig_post_step = orig_post

    def uninstall_hooks(self) -> None:
        if self._trainer is not None and self._orig_frame_loop is not None:
            self._trainer.frame_training_loop = self._orig_frame_loop
        if self._module is not None and self._orig_post_step is not None:
            self._module.post_training_step = self._orig_post_step
        self._module   = None
        self._trainer  = None
        self._pipeline = None

    # ── Gaussian access ───────────────────────────────────────────────────────

    def n_gaussians(self) -> int:
        if self._module is None:
            return 0
        n_base = self._module.gaussians.get_xyz.shape[0]
        incr   = getattr(self._module, "gaussians_incr", None)
        n_incr = 0
        if incr is not None:
            try:
                n_incr = incr.get_xyz.shape[0]
            except Exception:
                pass
        return n_base + n_incr

    def get_gaussians(self) -> _MergedGaussianAccessor:
        """Return a merged accessor over base + incremental Gaussians."""
        return _MergedGaussianAccessor(
            self._module.gaussians,
            getattr(self._module, "gaussians_incr", self._module.gaussians),
        )

    def __repr__(self) -> str:
        n = self.n_gaussians()
        return f"OpHiCoM(n_gaussians={n})"
