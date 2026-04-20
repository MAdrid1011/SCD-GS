"""
Op4DGC — Stage-0 operator slot for 4DGC (CVPR 2025).

4DGC ("Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint
Video", Hu et al.) is a direct descendant of 3DGStream: it replaces 3DGStream's
NTC with a **Motion Estimation Module (MEM)** built from three multi-resolution
3D grids + a tiny MLP (Φ_μ, Φ_R) + per-level entropy bottlenecks.  Per-frame
training has the same two stages:

  Stage 1 (MEM):
    gaussians.mem_optimizer (Adam over Motion_Grid params only) trains the MEM
    to predict per-Gaussian (Δxyz, Δrot).  ``gaussians.query_mem()`` is called
    every iteration to refresh ``_new_xyz`` / ``_new_rot`` from the current
    MEM.  The per-Gaussian attrs (``_xyz`` / ``_rotation``) are **not updated
    per-iter** — they're committed once, at the end of Stage 1, via
    ``gaussians.update_by_mem()``.  This is the exact analogue of
    ``update_by_ntc()`` in 3DGStream and is where we ingest per-Gaussian
    deltas into the GLASS pipeline.

  Stage 2 (compensated Gaussians):
    gaussians.optimizer trains the newly-added ``_added_*`` Gaussians
    (xyz / f_dc / f_rest / opacity / scaling / rotation), with
    ``gaussians.adding_and_prune`` as the structural-change hook.  Same layout
    as 3DGStream's Stage 2.

Hook installation — two-phase
-------------------------------
  Phase 1  install_hooks(pipeline, gaussians):
    Called right after ``training_one_frame_setup()``; only
    ``gaussians.mem_optimizer`` exists at this point.  Installs:
      • gaussians.mem_optimizer.step → _pre_mem_step() (grad-threshold suppression)

  Phase 1.5  install_mem_delta_hook(gaussians, n_mem_iters):
    Called immediately after ``install_hooks``.  Wraps
    ``gaussians.update_by_mem()`` to snapshot backing tensors before the
    bulk commit and compute per-Gaussian L2 delta norms after, feeding them
    into ``pipeline.ingest_ntc_deltas(...)``.  Note: MEM only writes to
    ``_xyz`` and ``_rotation`` (unlike 3DGStream's NTC, which writes to four
    attrs); the snapshot therefore covers just those two tensors.

    Architecture parameters (grid params, MLP MACs-per-Gaussian, encoding
    dim, output dim) are measured once from the live Motion_Grid module
    and re-used in every frame's ``ingest_ntc_deltas()`` call — no magic
    numbers, everything is derived from tensor shapes / module config.

  Phase 2  install_s2_hooks(gaussians):
    Called right after ``training_one_frame_s2_setup()``.  Installs:
      • gaussians.optimizer.step      → pipeline.pre_optimizer_step()
      • gaussians.adding_and_prune    → struct-change ID capture

  uninstall_hooks() restores all patches from all phases.

The ablation runner is responsible for calling ``pipeline.post_render(...)``
after each training render; the slot itself only wraps the optimizers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class Op4DGC:
    """Stage-0 operator slot for 4DGC.

    Attribute group lists cover both the base Gaussians carried over from the
    previous frame and the compensated Gaussian set spawned in Stage 2.
    """

    # ── Param group declarations (used by GLASSPipeline) ─────────────────────

    gaussian_attr_groups: list[str] = [
        # Base Gaussians (inherited from previous frame via scene.load)
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
        # Compensated Gaussians spawned in Stage 2 (training_one_frame_s2_setup
        # + adding_and_prune)
        "added_xyz", "added_f_dc", "added_f_rest",
        "added_opacity", "added_scaling", "added_rotation",
    ]
    network_attr_groups: list[str] = [
        # MEM (Motion_Grid + tiny MLP + entropy bottlenecks) — aggregate.
        # The mem_optimizer param groups are unnamed, so we treat the whole
        # thing as a single network-level group and use the same
        # ntc_grad_threshold knob 3DGStream's NTC uses (both are per-frame
        # network updates serving the same role: predict a per-Gaussian
        # residual).
        "__mem__",
    ]

    def __init__(self) -> None:
        self._gaussians          = None
        self._pipeline: "GLASSPipeline | None" = None
        # Saved originals for uninstall
        self._orig_mem_step      = None
        self._orig_gs_step       = None
        self._orig_adding_prune  = None
        self._orig_update_mem    = None
        # Structural change IDs to pass to the next gs optimizer call
        self._pending_struct_ids: list[int] = []
        # MEM architecture parameters — measured once at hook install from
        # the live Motion_Grid module; re-used across frames.
        self._mem_n_grid_params: int = 0
        self._mem_enc_dim:       int = 0
        self._mem_macs_per_g:    int = 0
        self._mem_out_dim:       int = 0
        # Maximum MEM training iterations per frame (= args.iterations from runner).
        self._mem_n_iters_max:   int = 0

    # ── Phase 1: MEM optimizer hook (after training_one_frame_setup) ──────────

    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        gaussians=None,
        **_,
    ) -> None:
        """Install the MEM optimizer hook into a 4DGC GaussianModel.

        Must be called *after* ``training_one_frame_setup()`` so that
        ``gaussians.mem_optimizer`` exists.  ``gaussians.optimizer`` is
        created later in ``training_one_frame_s2_setup`` and is handled by
        ``install_s2_hooks`` below.
        """
        if gaussians is None:
            raise ValueError("Op4DGC.install_hooks requires gaussians=<GaussianModel>")

        self._gaussians = gaussians
        self._pipeline  = pipeline
        self._pending_struct_ids = []

        orig_mem = gaussians.mem_optimizer.step

        def _mem_step(closure=None):
            self._pre_mem_step(gaussians.mem_optimizer)
            return orig_mem(closure)

        gaussians.mem_optimizer.step = _mem_step
        self._orig_mem_step = orig_mem

    # ── Phase 1.5: MEM delta hook (capture bulk commit deltas) ────────────────

    def install_mem_delta_hook(self, gaussians, n_mem_iters: int = 0) -> None:
        """Wrap ``update_by_mem()`` to capture per-Gaussian delta norms.

        Must be called after ``install_hooks()`` so that ``self._pipeline``
        is set.  ``update_by_mem`` is invoked exactly once per frame (right
        before Stage 2), so the hook fires once per frame.

        The hook snapshots ``_xyz`` and ``_rotation`` immediately before the
        MEM commit, then computes per-Gaussian L2 delta norms after.  All
        magnitudes and ``total_d`` are derived from actual tensor shapes.

        MEM architecture parameters are measured from the live
        ``Motion_Grid`` module:
          • grid params — total floats across the three multi-res grids
          • enc_dim     — concatenated interpolation output (= 20 for the
                          paper's 4+4+4+4+2+2 layout, per Motion_Grid.interpolate)
          • MACs/g      — tinymlp forward MACs per query
          • out_dim     — 7 (Δxyz 3 + Δrot 4)

        Parameters
        ----------
        gaussians    : GaussianModel from 4DGC scene/gaussian_model.py
        n_mem_iters  : maximum MEM training iterations per frame (= args.iterations).
                       Passed to ``pipeline.ingest_ntc_deltas`` so the cycle
                       model can account for MEM forward + backward + Adam
                       across all iters of Stage 1.
        """
        self._mem_n_iters_max = n_mem_iters
        import torch

        pipeline = self._pipeline
        if pipeline is None:
            return

        # ── Measure MEM architecture parameters from the live model ──────────
        try:
            motion_grid = gaussians.mem.model  # Motion_Grid instance

            # Grid params — sum over the three multi-res grids
            n_grid_params = sum(p.numel() for p in motion_grid.grid.parameters())

            # tinymlp: Sequential(Linear(20,64) → LeakyReLU → Linear(64,7))
            # We count weight MACs only (biases are negligible vs MACs):
            mlp_macs = 0
            mlp_layers = [l for l in motion_grid.tinymlp
                          if isinstance(l, torch.nn.Linear)]
            for lin in mlp_layers:
                mlp_macs += int(lin.in_features) * int(lin.out_features)

            # Encoding dim — the concatenated output of interpolate() before
            # the tinymlp.  Read from the MLP's first Linear in_features.
            enc_dim = int(mlp_layers[0].in_features) if mlp_layers else 0

            # Output dim — last Linear out_features.  Matches (Δxyz 3 + Δrot 4).
            out_dim = int(mlp_layers[-1].out_features) if mlp_layers else 0

            self._mem_n_grid_params = n_grid_params
            self._mem_enc_dim       = enc_dim
            self._mem_macs_per_g    = mlp_macs
            self._mem_out_dim       = out_dim
        except Exception as exc:
            print(f"  [Op4DGC] MEM param measurement failed: {exc}; "
                  "MEM cycles will not be counted.")
            self._mem_n_grid_params = 0
            self._mem_enc_dim       = 0
            self._mem_macs_per_g    = 0
            self._mem_out_dim       = 0

        orig_update = gaussians.update_by_mem

        # Capture measured params in closure (values are stable across frames)
        _n_grid  = self._mem_n_grid_params
        _enc     = self._mem_enc_dim
        _macs    = self._mem_macs_per_g
        _out     = self._mem_out_dim
        _n_iters = self._mem_n_iters_max

        def _hooked_update():
            # Snapshot backing tensors BEFORE the MEM commit.
            # MEM writes only to _xyz and _rotation (see update_by_mem in
            # baselines/4DGC/scene/gaussian_model.py); features are untouched.
            old = {k: getattr(gaussians, k).clone()
                   for k in ("_xyz", "_rotation")}
            orig_update()
            parts = []
            for key in ("_xyz", "_rotation"):
                nv, ov = getattr(gaussians, key), old[key]
                if nv.shape == ov.shape and nv.shape[0] > 0:
                    parts.append((nv - ov).reshape(nv.shape[0], -1))
            if parts:
                with torch.no_grad():
                    delta   = torch.cat(parts, dim=1)  # [N, 3+4]
                    norms   = delta.norm(dim=1)         # [N] — real magnitudes
                    total_d = delta.shape[1]            # 7 (3 + 4)
                # Feed the bulk commit into the pipeline using the same
                # ingest_ntc_deltas() entry point 3DGStream uses — it is
                # baseline-agnostic and correctly routes network-level
                # params (grid + MLP) vs per-Gaussian deltas to separate
                # accumulators.
                pipeline.ingest_ntc_deltas(
                    norms, total_d,
                    ntc_macs_per_g=_macs,
                    n_hash_params=_n_grid,   # MEM's "hash"-like grid storage
                    enc_dim=_enc,
                    ntc_output_dim=_out,
                    n_ntc_iters_max=_n_iters,
                )

        gaussians.update_by_mem = _hooked_update
        self._orig_update_mem   = orig_update

    # ── Phase 2: GS optimizer + densification hooks (after Stage-2 setup) ─────

    def install_s2_hooks(self, gaussians) -> None:
        """Install Stage-2 hooks after ``training_one_frame_s2_setup()``.

        At this point ``gaussians.optimizer`` (for the compensated Gaussians)
        and ``gaussians.adding_and_prune`` are both available.  The spawned
        Gaussians from the Stage-2 setup itself are captured as an initial
        structural change on the first optimizer step.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return

        self._gaussians = gaussians

        # ── Stage-2 Gaussian optimizer hook ──────────────────────────────────
        orig_gs = gaussians.optimizer.step

        def _gs_step(closure=None):
            pipeline.pre_optimizer_step(
                gaussians.optimizer,
                struct_changed=self._pending_struct_ids,
            )
            self._pending_struct_ids = []
            return orig_gs(closure)

        gaussians.optimizer.step = _gs_step
        self._orig_gs_step = orig_gs

        # ── Record initial compensated-Gaussian spawn as a structural change ─
        # training_one_frame_s2_setup creates the first batch of _added_* via
        # adding_postfix.  We've already been called, so the added tensors
        # exist; record their IDs as pending-structural for the first step.
        n_base  = int(getattr(gaussians, "_xyz").shape[0])
        n_added = int(getattr(gaussians, "_added_xyz").shape[0]
                      if getattr(gaussians, "_added_xyz", None) is not None
                      else 0)
        if n_added > 0:
            new_ids = list(range(n_base, n_base + n_added))
            self._pending_struct_ids = new_ids
            self.notify_structural_change(new_ids, [])

        # ── adding_and_prune hook (structural change detection) ───────────────
        if hasattr(gaussians, "adding_and_prune"):
            orig_ap = gaussians.adding_and_prune

            def _adding_and_prune(training_args, extent):
                n_before = int(gaussians.get_xyz.shape[0])
                result   = orig_ap(training_args, extent)
                n_after  = int(gaussians.get_xyz.shape[0])
                if n_after > n_before:
                    new_ids = list(range(n_before, n_after))
                    self._pending_struct_ids.extend(new_ids)
                    self.notify_structural_change(new_ids, [])
                return result

            gaussians.adding_and_prune = _adding_and_prune
            self._orig_adding_prune = orig_ap

    def uninstall_hooks(self) -> None:
        """Restore all monkey-patched methods from all phases."""
        g = self._gaussians
        if g is None:
            return
        if self._orig_mem_step is not None and hasattr(g, "mem_optimizer"):
            g.mem_optimizer.step = self._orig_mem_step
        if self._orig_gs_step is not None and hasattr(g, "optimizer"):
            g.optimizer.step = self._orig_gs_step
        if self._orig_adding_prune is not None:
            g.adding_and_prune = self._orig_adding_prune
        if self._orig_update_mem is not None:
            g.update_by_mem = self._orig_update_mem
        self._gaussians          = None
        self._pipeline           = None
        self._orig_mem_step      = None
        self._orig_gs_step       = None
        self._orig_adding_prune  = None
        self._orig_update_mem    = None

    # ── MEM-specific pre-step ─────────────────────────────────────────────────

    def _pre_mem_step(self, mem_optimizer) -> None:
        """Apply MEM gradient threshold (consolidation mode >= 1 only).

        The MEM optimizer contains Motion_Grid + MLP + entropy-bottleneck
        weights — all network-level (not per-Gaussian).  We apply the same
        ntc_grad_threshold hard cutoff 3DGStream uses for its NTC optimizer,
        which for MEM plays the same role.
        """
        if self._pipeline is None or self._pipeline.cfg.mode < 1:
            return
        thresh = self._pipeline.cfg.ntc_grad_threshold
        for group in mem_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.norm().item() < thresh:
                    p.grad.zero_()

    # ── Structural change notification ────────────────────────────────────────

    def notify_structural_change(
        self,
        added: list[int],
        removed: list[int],
    ) -> None:
        """Accumulate new Gaussian IDs for the next optimizer hook call."""
        self._pending_struct_ids.extend(added)

    # ── Gaussian access ───────────────────────────────────────────────────────

    def n_gaussians(self) -> int:
        if self._gaussians is None:
            return 0
        return int(self._gaussians.get_xyz.shape[0])

    def get_gaussians(self):
        return self._gaussians

    def __repr__(self) -> str:
        n = self.n_gaussians()
        return f"Op4DGC(n_gaussians={n})"
