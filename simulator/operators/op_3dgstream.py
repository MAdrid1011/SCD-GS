"""
Op3DGStream — Stage-0 operator slot for 3DGStream.

3DGStream per-frame training has two stages:
  Stage 1 (NTC): trains a Neural Texture Cache (NTC) network that maps each
    Gaussian to a set of predicted attribute updates. Optimizer: ntc_optimizer.
    Param groups are named arbitrarily (network weights), so we treat the
    entire NTC optimizer as network-level (one aggregate descriptor).

  Stage 2 (Gaussian): after NTC is dumped via dump_NTC(), per-Gaussian attrs
    are updated by update_by_ntc() and then further refined with densification.
    Optimizer: gaussians.optimizer. Param groups:
      "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"  (base)
      "added_xyz", "added_f_dc", "added_f_rest",                 (added Gs)
      "added_opacity", "added_scaling", "added_rotation"

Hook installation — two-phase
-------------------------------
  Phase 1  install_hooks(pipeline, gaussians):
    Called right after training_one_frame_setup().  At this point only
    ntc_optimizer exists.  Installs:
      • gaussians.ntc_optimizer.step  → _pre_ntc_step() (NTC threshold)

  Phase 2  install_s2_hooks(gaussians):
    Called right after training_one_frame_s2_setup().  At this point
    gaussians.optimizer (the Stage-2 Gaussian optimizer) exists.  Installs:
      • gaussians.optimizer.step      → pipeline.pre_optimizer_step()
      • gaussians.adding_and_prune   → struct-change ID capture

  Phase 1.5  install_ntc_delta_hook(gaussians):
    Called immediately after install_hooks() (in _patched_s1).  Wraps
    gaussians.update_by_ntc() to snapshot backing tensors before the NTC
    commit and compute per-Gaussian L2 delta norms after, feeding them into
    pipeline.ingest_ntc_deltas() for ASIC cycle and stream-stats accounting.

  uninstall_hooks() restores all patches from both phases.

Render post-hook
----------------
  The ablation runner calls pipeline.post_render() after each training render.
  The slot itself only wraps the optimizers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class Op3DGStream:
    """Stage-0 operator slot for 3DGStream.

    Attribute group lists cover both base Gaussians and the added-Gaussian
    sets created during Stage-2 densification.
    """

    # ── Param group declarations (used by GLASSPipeline) ─────────────────────

    gaussian_attr_groups: list[str] = [
        # Base Gaussians (loaded from previous frame checkpoint)
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
        # Added/spawned Gaussians in Stage 2
        "added_xyz", "added_f_dc", "added_f_rest",
        "added_opacity", "added_scaling", "added_rotation",
    ]
    network_attr_groups: list[str] = [
        # NTC network weights — treated as a single network-level group.
        # The NTC optimizer's group names vary; we declare a sentinel here
        # and handle NTC separately via the _pre_ntc_step method.
        "__ntc__",
    ]

    def __init__(self) -> None:
        self._gaussians          = None
        self._pipeline: "GLASSPipeline | None" = None
        # Saved originals for uninstall
        self._orig_ntc_step      = None
        self._orig_gs_step       = None
        self._orig_adding_prune  = None
        self._orig_update_ntc    = None   # NTC delta hook
        # Structural change IDs to pass to the next gs optimizer call
        self._pending_struct_ids: list[int] = []
        # NTC architecture parameters — measured at hook install time from
        # the live tcnn model; passed to pipeline.ingest_ntc_deltas() each frame.
        self._ntc_n_hash_params:  int = 0
        self._ntc_enc_dim:        int = 0
        self._ntc_macs_per_g:     int = 0
        self._ntc_out_dim:        int = 0
        # Maximum NTC training iterations per frame (= args.iterations from runner).
        self._ntc_n_iters_max:    int = 0

    # ── Phase 1: NTC hook (called after training_one_frame_setup) ─────────────

    def install_hooks(
        self,
        pipeline: "GLASSPipeline",
        gaussians=None,
        **_,
    ) -> None:
        """Install the NTC optimizer hook into a 3DGStream GaussianModel.

        Must be called *after* training_one_frame_setup() so that
        gaussians.ntc_optimizer exists.  gaussians.optimizer is NOT yet
        available at this point (it is created in training_one_frame_s2_setup).

        Parameters
        ----------
        pipeline  : GLASSPipeline instance
        gaussians : GaussianModel from 3DGStream scene/gaussian_model.py
        """
        if gaussians is None:
            raise ValueError("Op3DGStream.install_hooks requires gaussians=<GaussianModel>")

        self._gaussians = gaussians
        self._pipeline  = pipeline
        self._pending_struct_ids = []

        # ── NTC optimizer hook ────────────────────────────────────────────────
        orig_ntc = gaussians.ntc_optimizer.step

        def _ntc_step(closure=None):
            self._pre_ntc_step(gaussians.ntc_optimizer)
            return orig_ntc(closure)

        gaussians.ntc_optimizer.step = _ntc_step
        self._orig_ntc_step = orig_ntc

    # ── Phase 1.5: NTC delta hook (called after install_hooks, before Stage 2) ──

    def install_ntc_delta_hook(self, gaussians, n_ntc_iters: int = 0) -> None:
        """Wrap update_by_ntc() to capture per-Gaussian NTC delta norms.

        Must be called after install_hooks() so that self._pipeline is set.
        update_by_ntc() is invoked between Stage 1 and Stage 2 (inside
        train_one_frame), so the hook fires exactly once per frame.

        The hook snapshots the raw backing tensors (_xyz, _rotation,
        _features_dc, _features_rest) immediately before the NTC commit, then
        computes per-Gaussian L2 delta norms after the commit.  All magnitudes
        and total_d are derived from actual tensor shapes — no magic numbers.

        NTC architecture parameters (n_hash_params, enc_dim, n_mlp_macs_per_g,
        out_dim) are measured once from the live tcnn model and stored on the
        slot for use in every subsequent ingest_ntc_deltas() call.

        Parameters
        ----------
        gaussians    : GaussianModel instance from 3DGStream.
        n_ntc_iters  : maximum NTC training iterations per frame (= args.iterations).
                       Passed to pipeline.ingest_ntc_deltas() so the cycle model
                       can account for NTC forward + backward + Adam across all iters.
                       0 means NTC training cycles will not be counted.
        """
        self._ntc_n_iters_max = n_ntc_iters
        import torch

        pipeline = self._pipeline
        if pipeline is None:
            return

        # ── Measure NTC architecture parameters from the live model ──────────
        # All values are read directly from the tcnn.NetworkWithInputEncoding
        # instance — no hard-coded constants.
        try:
            ntc_model = gaussians.ntc.model  # tcnn.NetworkWithInputEncoding

            # Total fp16 parameters (encoding + network combined)
            n_total_params = sum(p.numel() for p in ntc_model.parameters())

            # Output dimension from the tcnn model attribute
            out_dim = int(ntc_model.n_output_dims)

            # Encoding output dimension from the stored config dict
            enc_cfg = ntc_model.encoding_config
            n_levels = int(enc_cfg.get("n_levels", 0))
            n_feat   = int(enc_cfg.get("n_features_per_level", 0))
            enc_dim  = n_levels * n_feat  # e.g. 16 × 4 = 64

            # MLP layer counts from the stored network config dict
            net_cfg     = ntc_model.network_config
            n_neurons   = int(net_cfg.get("n_neurons", 0))
            n_hidden    = int(net_cfg.get("n_hidden_layers", 0))

            # MLP MACs per Gaussian = number of MLP weights (one MAC per weight):
            #   input layer  : enc_dim × n_neurons
            #   hidden layers: (n_hidden - 1) × n_neurons²  (if n_hidden > 1)
            #   output layer : n_neurons × out_dim
            n_mlp_macs = (enc_dim * n_neurons
                          + max(0, n_hidden - 1) * n_neurons * n_neurons
                          + n_neurons * out_dim)

            # Hash parameters = total minus MLP weights (tcnn packs them together)
            n_hash_params = max(0, n_total_params - n_mlp_macs)

            self._ntc_n_hash_params = n_hash_params
            self._ntc_enc_dim       = enc_dim
            self._ntc_macs_per_g    = n_mlp_macs
            self._ntc_out_dim       = out_dim
        except Exception as exc:
            print(f"  [Op3DGStream] NTC param measurement failed: {exc}; "
                  "NTC cycles will not be counted.")
            self._ntc_n_hash_params = 0
            self._ntc_enc_dim       = 0
            self._ntc_macs_per_g    = 0
            self._ntc_out_dim       = 0

        orig_update = gaussians.update_by_ntc

        # Capture measured params in closure (values are stable across frames)
        _n_hash   = self._ntc_n_hash_params
        _enc      = self._ntc_enc_dim
        _macs     = self._ntc_macs_per_g
        _out      = self._ntc_out_dim
        _n_iters  = self._ntc_n_iters_max

        def _hooked_update():
            # Snapshot backing tensors BEFORE the NTC commit
            old = {k: getattr(gaussians, k).clone()
                   for k in ("_xyz", "_rotation", "_features_dc", "_features_rest")}
            orig_update()
            # delta = new_value - old_value for every NTC-modified attr
            parts = []
            for key in ("_xyz", "_rotation", "_features_dc", "_features_rest"):
                nv, ov = getattr(gaussians, key), old[key]
                if nv.shape == ov.shape and nv.shape[0] > 0:
                    parts.append((nv - ov).reshape(nv.shape[0], -1))
            if parts:
                with torch.no_grad():
                    delta   = torch.cat(parts, dim=1)  # [N, total_d]
                    norms   = delta.norm(dim=1)         # [N] — real magnitudes
                    total_d = delta.shape[1]            # from tensor shapes
                pipeline.ingest_ntc_deltas(
                    norms, total_d,
                    ntc_macs_per_g=_macs,
                    n_hash_params=_n_hash,
                    enc_dim=_enc,
                    ntc_output_dim=_out,
                    n_ntc_iters_max=_n_iters,
                )

        gaussians.update_by_ntc = _hooked_update
        self._orig_update_ntc   = orig_update

    # ── Phase 2: GS optimizer + densification hooks ───────────────────────────

    def install_s2_hooks(self, gaussians) -> None:
        """Install Stage-2 hooks after training_one_frame_s2_setup().

        At this point gaussians.optimizer (the added-Gaussian optimizer) and
        gaussians.adding_and_prune are both available.

        Parameters
        ----------
        gaussians : GaussianModel instance (same one passed to install_hooks)
        """
        pipeline = self._pipeline
        if pipeline is None:
            return

        # Update reference in case this is called on a new frame's gaussians
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

        # ── adding_and_prune hook (structural change detection) ───────────────
        if hasattr(gaussians, "adding_and_prune"):
            orig_ap = gaussians.adding_and_prune

            def _adding_and_prune(training_args, extent):
                n_before = gaussians.get_xyz.shape[0]
                result   = orig_ap(training_args, extent)
                n_after  = gaussians.get_xyz.shape[0]
                if n_after > n_before:
                    new_ids = list(range(n_before, n_after))
                    self._pending_struct_ids = new_ids
                    self.notify_structural_change(new_ids, [])
                return result

            gaussians.adding_and_prune = _adding_and_prune
            self._orig_adding_prune = orig_ap

    def uninstall_hooks(self) -> None:
        """Restore all monkey-patched methods from both phases."""
        g = self._gaussians
        if g is None:
            return
        if self._orig_ntc_step is not None and hasattr(g, "ntc_optimizer"):
            g.ntc_optimizer.step = self._orig_ntc_step
        if self._orig_gs_step is not None and hasattr(g, "optimizer"):
            g.optimizer.step = self._orig_gs_step
        if self._orig_adding_prune is not None:
            g.adding_and_prune = self._orig_adding_prune
        if self._orig_update_ntc is not None:
            g.update_by_ntc = self._orig_update_ntc
        self._gaussians          = None
        self._pipeline           = None
        self._orig_ntc_step      = None
        self._orig_gs_step       = None
        self._orig_adding_prune  = None
        self._orig_update_ntc    = None

    # ── NTC-specific pre-step ─────────────────────────────────────────────────

    def _pre_ntc_step(self, ntc_optimizer) -> None:
        """Apply NTC gradient threshold (consolidation mode >= 1 only).

        The NTC optimizer contains network weights, not per-Gaussian attrs.
        We apply the ntc_grad_threshold hard cutoff to each parameter group.
        """
        if self._pipeline is None or self._pipeline.cfg.mode < 1:
            return
        thresh = self._pipeline.cfg.ntc_grad_threshold
        for group in ntc_optimizer.param_groups:
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
        return f"Op3DGStream(n_gaussians={n})"
