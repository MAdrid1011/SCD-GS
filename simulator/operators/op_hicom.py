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

        def _post_step(render_pkg):
            # Call pre_optimizer_step for every optimizer in the module
            for opt_name in ("optimizer", "gaussians_incr_optimizer",
                             "deform_optimizer"):
                opt = getattr(module, opt_name, None)
                if opt is None:
                    continue
                # Also check via deform sub-module
                if opt_name == "deform_optimizer" and opt is None:
                    opt = getattr(getattr(module, "deform", None), "optimizer", None)
                if opt is not None:
                    pipeline.pre_optimizer_step(opt)

            orig_post(render_pkg)
            pipeline.post_render(render_pkg)

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
