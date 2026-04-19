"""
GLASSPipeline — shared Stages 1-5 of the GLASS ASIC simulator.

This is the single class that implements all ASIC optimisation logic.
It is instantiated once per ablation run and operates identically regardless
of which baseline model is under evaluation.

Model-specific knowledge is entirely isolated in the BaseOperatorSlot
(operators/op_*.py). The pipeline only calls the slot for:
  1. slot.gaussian_attr_groups  — list of per-Gaussian param-group names
  2. slot.network_attr_groups   — list of network-level param-group names
  3. slot.n_gaussians()          — current Gaussian count
  4. slot.get_gaussians()        — Gaussian model object for proxy wrapping

Stage mapping
-------------
  Stage 0  Operator Slot (in op_*.py)   — model-specific descriptor capture entry
  Stage 1  ConsolidationUnit            — gradient fold / scale
  Stage 2  IncrementalCommitEngine      — apply scales + cold-grad zero (in pipeline)
  Stage 3  LiveStateFabric              — hotness tracking + render mask
  Stage 4  SalienceFirstIssue           — PROMOTE / FULL / SKIP classifier
  Stage 5  Render Backend               — baseline render() called with proxy

Ablation mode gates
-------------------
  mode >= 1  → Stage 1 active (consolidation scales applied)
  mode >= 2  → Stage 3 active (cold-mask zeroing + fabric render mask)
  mode >= 3  → Stage 4 active (salience filter added to render mask)

At mode=0 (Base ASIC) the pipeline only captures descriptors and calls
post_render() (hotness/salience bookkeeping), adding a small Python overhead
but applying no gradient modifications and no render masking.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch

from simulator.config import ASICConfig
from simulator.core.consolidation import ConsolidationUnit
from simulator.core.cycle_model import CycleAccumulator
from simulator.core.live_state_fabric import LiveStateFabric
from simulator.core.proxy import MaskedGaussianProxy
from simulator.core.salience_issue import SalienceFirstIssue
from simulator.core.update_descriptor import AttrClass, UpdateDescriptor, UpdateStream

if TYPE_CHECKING:
    from simulator.operators.base_operator import BaseOperatorSlot


class GLASSPipeline:
    """Shared ASIC pipeline — Stages 1 through 5.

    The same instance handles 3DGStream, QUEEN, HiCoM, and Swift4D. The only
    model-specific wiring is provided by the BaseOperatorSlot injected at
    construction.

    Usage (in ablation runner)
    --------------------------
        slot     = Op3DGStream()
        pipeline = GLASSPipeline(config, slot)
        slot.install_hooks(pipeline, gaussians=g, ...)

        for frame in frames:
            pipeline.begin_frame()
            with metrics.frame_timer():
                train_one_frame(...)       # hooks fire automatically
            pipeline.post_render(render_pkg)

        fps = pipeline.measure_fps(render_fn, cameras, pipe, bg)
    """

    def __init__(self, config: ASICConfig, slot: "BaseOperatorSlot") -> None:
        self.cfg  = config
        self.slot = slot

        # Core compute units — instantiated once, reused every frame
        self.consolidation = ConsolidationUnit(config)
        self.fabric        = LiveStateFabric(config)
        self.salience      = SalienceFirstIssue(config)

        # Cycle-accurate ASIC timing model
        self._cycles       = CycleAccumulator(config.asic_spec)
        # Count of Gaussians with grad_scale > 0 after consolidation;
        # used to compute mode-1 optimizer savings in pre_optimizer_step.
        self._n_gs_active: int = 0

        # Suppression ratio from the previous frame, used by the adaptive NTC
        # training scheduler: mode >= 1 uses this to scale down n_ntc_iters.
        # Initialised to 0.0 so frame 0 always runs the full iteration count.
        self._prev_suppression_ratio: float = 0.0

        # Render stats captured from the previous frame's inference render_pkg.
        # Used to parameterise add_render_training_iter() in ingest_ntc_deltas().
        # None until the first post_render / set_render_stats_from_render_pkg call.
        self._prev_render_stats: "dict | None" = None

        self._version = 0           # monotonic frame counter
        self._stream  = UpdateStream()  # reset each begin_frame()

        # Cumulative traffic stats across the ablation run
        self._total_bytes_all:      int = 0
        self._suppressed_bytes_all: int = 0
        # Per-frame traffic stats used by the mode-0 fast path.  In mode >= 1,
        # stream_stats() still reads directly from self._stream descriptors.
        self._frame_total_bytes:      int = 0
        self._frame_suppressed_bytes: int = 0
        self._frame_unique_targets:   int = 0

    # ── Frame lifecycle ────────────────────────────────────────────────────────

    def begin_frame(self) -> None:
        """Advance the frame counter and reset per-frame state.

        Must be called at the start of every incremental training frame.
        The LiveStateFabric per-frame delta cache is also reset here.

        When cfg.reset_cwb_each_frame is True the CWB history is also
        cleared so that stale cross-frame history from a previous training
        phase does not carry over into the new frame's consolidation step.
        This is appropriate for baselines that re-initialise their optimiser
        at every frame (e.g. QUEEN); NTC-based models leave it False so that
        temporal stability detection spans multiple frames.
        """
        self._version += 1
        self._stream = UpdateStream()
        self.fabric.reset_frame_deltas()
        if self.cfg.frame_level_ucn:
            # Flush the per-iteration accumulator into one CWB entry per
            # (Gaussian, attr), then clear the accumulator for the new frame.
            # Both the dict path (slow) and the tensor path (mode-1 fast)
            # keep their own accumulator; only the one actually populated
            # this frame pushes anything into its CWB.
            self.consolidation.flush_frame_accum()
            self.consolidation.flush_frame_accum_batched()
        elif self.cfg.reset_cwb_each_frame:
            self.consolidation.cwb.clear()
            self.consolidation.cwb_batched.clear()

        # ILSF tier sort fires once per frame; it operates on a disjoint memory
        # region and completes well within one ICE step, so it is hidden by the
        # gradient pipeline (see parallelism analysis in cycle_model.py).
        # We record the cycles for transparency but they do NOT enter the critical path.
        n_g = self.slot.n_gaussians()
        if n_g > 0:
            self._cycles.add_ilsf_sort(n_g)
        self._cycles.reset_frame()
        self._frame_total_bytes = 0
        self._frame_suppressed_bytes = 0
        self._frame_unique_targets = 0

    # ── Stage 0: Descriptor capture ───────────────────────────────────────────

    def _capture_stream(
        self,
        optimizer: object,
        struct_changed: list[int] | None = None,
    ) -> UpdateStream:
        """Read current .grad from all param groups → UpdateDescriptor stream.

        Uses AttrClass.from_group_name() to normalise group names; this
        handles all naming conventions across the 4 baselines without any
        per-model branching in the pipeline.

        Parameters
        ----------
        optimizer      : any PyTorch optimizer with .param_groups
        struct_changed : list of Gaussian IDs that had structural changes
                         this frame (spawn/prune/split)
        """
        struct_set = set(struct_changed or [])
        for group in optimizer.param_groups:
            name       = group.get("name", "")
            attr_cls   = AttrClass.from_group_name(name)
            is_network = name in self.slot.network_attr_groups

            for param in group["params"]:
                if param.grad is None:
                    continue
                g = param.grad.detach()
                if g.dim() == 0:
                    continue

                if is_network:
                    # Network params: one aggregate descriptor (target_id = -1)
                    self._stream.add(UpdateDescriptor(
                        target_id=-1,
                        attr_class=AttrClass.NETWORK,
                        delta_norm=float(g.norm().item()),
                        version=self._version,
                        struct_changed=False,
                    ))
                    # PUOE for network: aggregate norm over n_params (memory-bound
                    # reduction, 1 sqrt total — not N sqrts).
                    self._cycles.add_puoe_network(g.numel())
                else:
                    # Per-Gaussian: shape [N, ...] → per-row L2 norm
                    if g.shape[0] == 0:
                        # Empty param group (e.g., zero added-Gaussians); skip.
                        continue
                    norms = g.reshape(g.shape[0], -1).norm(dim=1)
                    # PUOE: N independent norms, each with D MACs + 1 sqrt.
                    _g2d = g.reshape(g.shape[0], -1)
                    self._cycles.add_puoe(_g2d.shape[0], _g2d.shape[1])
                    for i, n in enumerate(norms.tolist()):
                        self._stream.add(UpdateDescriptor(
                            target_id=i,
                            attr_class=attr_cls,
                            delta_norm=float(n),
                            version=self._version,
                            struct_changed=(i in struct_set),
                        ))
        return self._stream

    # ── Stage 1: Consolidation → gradient scale ───────────────────────────────

    def _apply_consolidation(
        self,
        optimizer: object,
        stream_start: int = 0,
    ) -> None:
        """Scale per-Gaussian gradients according to ConsolidationUnit output.

        For network groups: apply hard NTC threshold to the whole param grad.
        For Gaussian groups: build a per-row scale vector and mul_ in-place.

        Parameters
        ----------
        stream_start : Passed to compute_scales so that only descriptors
                       added since this index are (re-)processed.  NTC
                       descriptors added by ingest_ntc_deltas earlier in the
                       same frame must not be re-scanned in Stage-2 steps.
        """
        # Compute scales only for descriptors added since stream_start;
        # also fills desc.grad_scale on those descriptors in-place.
        gaussian_scales = self.consolidation.compute_scales(
            self._stream, start_idx=stream_start
        )

        # UCN cycle count: new Gaussian descriptors only.
        _new_descs = self._stream.descriptors[stream_start:]
        _n_gs_ucn = sum(1 for d in _new_descs if d.target_id >= 0)
        if _n_gs_ucn > 0:
            self._cycles.add_ucn(_n_gs_ucn)

        # Track how many Gaussians retain grad_scale > 0 after consolidation;
        # used in pre_optimizer_step to compute mode-1 optimizer savings.
        self._n_gs_active = sum(
            1 for d in _new_descs
            if d.target_id >= 0 and gaussian_scales.get(d.target_id, 1.0) > 0.0
        )

        for group in optimizer.param_groups:
            name       = group.get("name", "")
            is_network = name in self.slot.network_attr_groups

            for param in group["params"]:
                if param.grad is None:
                    continue

                if is_network:
                    if param.grad.norm().item() < self.cfg.ntc_grad_threshold:
                        param.grad.zero_()
                    continue

                # Build per-row scale vector
                n  = param.grad.shape[0]
                if n == 0:
                    continue
                sv = torch.tensor(
                    [gaussian_scales.get(i, 1.0) for i in range(n)],
                    device=param.grad.device,
                    dtype=param.grad.dtype,
                )
                # Broadcast across attribute dimensions
                for _ in range(param.grad.dim() - 1):
                    sv = sv.unsqueeze(-1)
                param.grad.mul_(sv)

                # ICE scale-only (mode == 1): count cycles for this param group.
                # mode >= 2 uses add_ice_merged in _apply_cold_mask instead.
                if self.cfg.mode == 1:
                    _d = param.grad.reshape(n, -1).shape[1]
                    self._cycles.add_ice_scale(n, _d)

    # ── Stage 2+3: Live-State Fabric — zero cold Gaussian gradients ────────────

    def _apply_cold_mask(self, optimizer: object) -> None:
        """Zero gradients for CSS (cold) Gaussians post-backward.

        Only applied to param groups in slot.gaussian_attr_groups.
        Network groups are never cold-masked (they're global, not per-Gaussian).
        """
        n    = self.slot.n_gaussians()
        cold = self.fabric.get_cold_mask(n)

        _n_cold_total = int(cold.sum().item())
        _n_hot_total  = n - _n_cold_total

        for group in optimizer.param_groups:
            name = group.get("name", "")
            if name not in self.slot.gaussian_attr_groups:
                continue
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.shape[0] == 0:
                    continue
                # Guard against size mismatch (e.g., mid-densification)
                ng = min(param.grad.shape[0], cold.shape[0])
                param.grad[:ng][cold[:ng]] = 0.0

                # ICE merged pass (mode >= 2): scale + cold-mask in one scan.
                # Cheaper than two separate passes (saves ~33% memory bandwidth).
                _d = param.grad.reshape(param.grad.shape[0], -1).shape[1]
                self._cycles.add_ice_merged(_n_hot_total, _n_cold_total, _d)

    # ── NTC delta ingestion (called from install_ntc_delta_hook) ─────────────

    def ingest_ntc_deltas(
        self,
        norms: torch.Tensor,
        total_d: int,
        *,
        ntc_macs_per_g: int = 0,
        n_hash_params: int = 0,
        enc_dim: int = 0,
        ntc_output_dim: int = 0,
        n_ntc_iters_max: int = 0,
    ) -> None:
        """Ingest per-Gaussian NTC delta norms into the ASIC cycle model.

        Called once per frame by the update_by_ntc() hook installed in
        Op3DGStream.install_ntc_delta_hook().  Handles all ~280K base
        Gaussians whose updates flow through the NTC write path rather than
        the Adam gradient path captured by pre_optimizer_step().

        Parameters
        ----------
        norms            : float32 tensor [N] — L2 norm of (new_attr − old_attr)
                           concatenated across xyz, rotation, features_dc,
                           features_rest.  Computed from real tensor diffs.
        total_d          : int — total attribute dimensions (from tensor shapes).
        ntc_macs_per_g   : MLP MACs per Gaussian (from layer weight counts).
        n_hash_params    : HashGrid fp16 parameter count (measured at runtime).
        enc_dim          : HashGrid output feature dim (n_levels × n_feat).
        ntc_output_dim   : MLP output dimension (ntc_model.n_output_dims).
        n_ntc_iters_max  : actual NTC training iterations that ran this frame.
                           For mode 0 this equals base_iters (e.g. 150).
                           For mode >= 1 the runner pre-computes
                           max(min_ntc_iters_floor, round(base_iters*(1-ratio)))
                           and sets args.iterations accordingly before training,
                           so the GPU and the cycle model use the same count.
                           0 means NTC training cycles are not counted.
        """
        N = norms.shape[0]
        if N == 0 or total_d == 0:
            return

        # NTC training cycles on the PUOE (forward + backward + Adam per iter).
        # n_ntc_iters_max already reflects the adaptive scheduling done by the
        # runner (for mode >= 1) or the full base iteration count (mode 0).
        # No further ratio scaling here — that would double-apply suppression.
        n_iters = 0
        if ntc_macs_per_g > 0 and n_ntc_iters_max > 0:
            n_iters = n_ntc_iters_max
            self._cycles.add_ntc_training(
                N, ntc_macs_per_g, n_hash_params, enc_dim, ntc_output_dim, n_iters
            )

        # Per-iteration 3DGS render forward + photometric loss + backward.
        # Uses scene statistics measured from the PREVIOUS frame's inference render
        # (same lag-one design as _prev_suppression_ratio).  Frame 0: skipped (None).
        if n_iters > 0 and self._prev_render_stats is not None:
            rs = self._prev_render_stats
            self._cycles.add_render_training_iter(
                N,
                rs["n_visible"],
                rs["tile_touches"],
                rs["H"],
                rs["W"],
                rs["sh_degree"],
                n_iters,
            )

        # NTC forward-inference cycles on the PUOE tensor core.
        # Modelled after training because inference is the final commit pass
        # (training → optimised weights → one inference → NTC deltas captured).
        if ntc_macs_per_g > 0 and enc_dim > 0 and ntc_output_dim > 0:
            self._cycles.add_ntc_forward(
                N, ntc_macs_per_g, n_hash_params, enc_dim, ntc_output_dim
            )

        # Stage 0: build N descriptors from measured delta norms.
        # Record stream length before so we can slice only the new descriptors.
        _len_before = len(self._stream.descriptors)
        for i, dn in enumerate(norms.tolist()):
            self._stream.add(UpdateDescriptor(
                target_id=i,
                attr_class=AttrClass.XYZ,   # representative; bytes via total_d
                delta_norm=float(dn),
                version=self._version,
                struct_changed=False,
            ))
        self._cycles.add_puoe(N, total_d)

        # Stage 1 — consolidation (mode >= 1).
        # compute_scales() sets d.grad_scale in-place on each descriptor,
        # which is what makes frame_suppressed_bytes / frame_suppression_ratio
        # reflect the consolidation's suppression effect.
        if self.cfg.mode >= 1:
            gaussian_scales = self.consolidation.compute_scales(self._stream)
            self._n_gs_active = sum(
                1 for v in gaussian_scales.values() if v > 0.0
            )
            self._cycles.add_ucn(N)
            # Update suppression ratio for NEXT frame's NTC iteration scheduling.
            # Counts NTC descriptors added in this call only (sliced by _len_before).
            new_descs = self._stream.descriptors[_len_before:]
            n_suppressed = sum(1 for d in new_descs if d.grad_scale == 0.0)
            self._prev_suppression_ratio = n_suppressed / max(len(new_descs), 1)

        # ICE: scale-only (mode 1) XOR merged scale+cold-mask (mode >= 2).
        # Mutually exclusive to avoid double-counting _cyc["ice"].
        if self.cfg.mode == 1:
            self._cycles.add_ice_scale(N, total_d)
        elif self.cfg.mode >= 2:
            cold   = self.fabric.get_cold_mask(N)
            n_cold = int(cold.sum().item())
            self._cycles.add_ice_merged(N - n_cold, n_cold, total_d)

        # ILSF update: record hotness only for new NTC descriptors.
        n_updated = 0
        for d in self._stream.descriptors[_len_before:]:
            if d.target_id >= 0 and d.delta_norm > 0.0:
                self.fabric.record_update(d.target_id, d.delta_norm)
                n_updated += 1
        self._cycles.add_ilsf_update(n_updated)

        # Optimizer cycles — n_opt varies by mode (models ASIC write savings):
        #   mode 0 — all Gaussians with a non-zero NTC delta
        #   mode 1 — those not suppressed by consolidation (grad_scale > 0)
        #   mode >= 2 — only the fabric render-mask (hot tiers)
        if self.cfg.mode == 0:
            n_opt = n_updated
        elif self.cfg.mode == 1:
            n_opt = self._n_gs_active
        else:
            n_opt = int(self.fabric.get_render_mask(N).sum().item())
        if n_opt > 0:
            self._cycles.add_optimizer_update(n_opt, total_d)

        # Run-level traffic stats: accumulate only the NTC descriptors added
        # in this call (snapshot approach avoids double-counting with
        # pre_optimizer_step which runs separately for added Gaussians).
        _ntc_descs = self._stream.descriptors[_len_before:]
        self._total_bytes_all      += sum(d.delta_bytes() for d in _ntc_descs)
        self._suppressed_bytes_all += sum(
            d.delta_bytes() for d in _ntc_descs if d.grad_scale < 1.0
        )

    # ── Main training hook ────────────────────────────────────────────────────

    def pre_optimizer_step(
        self,
        optimizer: object,
        struct_changed: list[int] | None = None,
    ) -> None:
        """Called by the operator slot immediately before optimizer.step().

        Executes the full Stage 0 → 1 → 2 pipeline in sequence.
        At mode=0 only Stage 0 (descriptor capture) runs.

        Parameters
        ----------
        optimizer      : the model's optimizer
        struct_changed : Gaussian IDs that changed structurally this step
        """
        # Fast path for mode 0 (base ASIC): avoid building Python descriptor
        # objects for every (Gaussian, attr) pair; keep cycle/byte accounting
        # identical while using vectorized tensor reductions only.
        if self.cfg.mode == 0:
            self._pre_optimizer_step_mode0_fast(optimizer)
            return

        # Fast path for mode 1/2/3 (Consolidation + optional cold-mask +
        # optional salience) — same motivation as mode 0: the dict/
        # descriptor path iterates N_gaussians × num_attrs Python objects
        # per optimizer step (for QUEEN: 188K × 6 ≈ 1.1M/step, ≈ 220M/
        # frame) and becomes the CPU-bound wall-clock bottleneck.  The
        # fast path replaces those loops with per-attr tensor ops that
        # preserve every cycle-accounting and byte-accounting call made
        # by the generic path.
        #
        # Mode 2 adds: cold-mask grad zeroing + ICE-merged cycle accounting
        #              + n_opt = render_mask.sum() (instead of UCN active)
        # Mode 3 adds: salience.mark_structural, but that only consumes
        #              struct_changed_ids from the stream — the fast path
        #              never pushes per-Gaussian descriptors, matching the
        #              slow path behaviour when struct_changed is None
        #              (no structural changes → empty set).
        if (
            self.cfg.mode in (1, 2, 3)
            and not self.cfg.adaptive_ucn_threshold
            and struct_changed is None
            # The fast path's unifier between main (batched CWB) and
            # fallback (dict-loop) CWB history only has a precise
            # implementation in frame-level mode.  Iteration-level UCN
            # (3DGStream / NTC baselines) never invokes this code path —
            # they take the ingest_ntc_deltas() route — so restricting to
            # frame_level_ucn keeps the fast path narrow and exact.
            and self.cfg.frame_level_ucn
        ):
            self._pre_optimizer_step_mode1_fast(optimizer)
            return

        # Snapshot stream length before _capture_stream so we can count only
        # the descriptors added by this specific optimizer step (not NTC ones).
        _len_before = len(self._stream.descriptors)

        # Stage 0: capture descriptors
        self._capture_stream(optimizer, struct_changed)

        # Stage 1: consolidation (mode >= 1) — pass _len_before so that only
        # GS-optimizer descriptors added by this call are (re-)processed.
        if self.cfg.mode >= 1:
            self._apply_consolidation(optimizer, stream_start=_len_before)

        # Stage 2+3: cold-mask gradient zeroing (mode >= 2)
        if self.cfg.mode >= 2:
            self._apply_cold_mask(optimizer)

        # Update hotness and count ILSF cycles for GS-optimizer descriptors
        # added by THIS call only (_new_descs).  NTC descriptors were already
        # processed by ingest_ntc_deltas(); re-scanning the full stream here
        # would iterate 280K items × 100 steps/frame = 28M Python ops/frame.
        n_updated = 0
        for d in self._stream.descriptors[_len_before:]:
            if d.target_id >= 0 and d.delta_norm > 0.0:
                self.fabric.record_update(d.target_id, d.delta_norm)
                n_updated += 1
        self._cycles.add_ilsf_update(n_updated)

        # Adam optimizer cycle cost for active (non-suppressed) Gaussians.
        # n_opt varies by mode to model the savings from each optimisation:
        #   mode 0 — all GS-optimizer descriptors with a non-zero gradient
        #   mode 1 — only those not suppressed by consolidation (grad_scale > 0)
        #   mode >= 2 — only the fabric render-mask (ASB + RSB hot tiers)
        if self.cfg.mode == 0:
            _n_opt = sum(
                1 for d in self._stream.descriptors[_len_before:]
                if d.target_id >= 0 and d.delta_norm > 0.0
            )
        elif self.cfg.mode == 1:
            _n_opt = self._n_gs_active  # set in _apply_consolidation
        else:
            _n_opt = int(
                self.fabric.get_render_mask(self.slot.n_gaussians()).sum().item()
            )

        # total_d: sum of attribute dimensions across all Gaussian param groups
        # for the current optimizer step (may vary if densification changes N).
        _total_d = sum(
            param.grad.reshape(param.grad.shape[0], -1).shape[1]
            for group in optimizer.param_groups
            for param in group["params"]
            if (param.grad is not None
                and param.grad.shape[0] > 0
                and group.get("name", "") in self.slot.gaussian_attr_groups)
        )
        if _n_opt > 0 and _total_d > 0:
            self._cycles.add_optimizer_update(_n_opt, _total_d)

        # Register structural changes with salience tracker (mode >= 3)
        if self.cfg.mode >= 3:
            changed = self._stream.struct_changed_ids()
            if changed:
                self.salience.mark_structural(changed)

        # Accumulate run-level traffic stats for descriptors added by THIS call
        # only (snapshot approach: slice from _len_before to avoid counting NTC
        # descriptors added earlier by ingest_ntc_deltas in the same frame).
        _new_descs = self._stream.descriptors[_len_before:]
        self._total_bytes_all      += sum(d.delta_bytes() for d in _new_descs)
        self._suppressed_bytes_all += sum(
            d.delta_bytes() for d in _new_descs if d.grad_scale < 1.0
        )

    def _pre_optimizer_step_mode0_fast(self, optimizer: object) -> None:
        """Mode-0 fast path without per-Gaussian Python descriptors.

        Semantics match the generic path:
          - PUOE cycles: per-group (network and Gaussian)
          - ILSF update cycles: count descriptors with delta_norm > 0
          - Optimizer cycles: n_opt = same descriptor count with delta_norm > 0
          - Traffic bytes: attr_class.delta_bytes() per descriptor (no suppression)
        """
        n_updated = 0
        total_d_opt = 0
        max_n_targets = 0
        total_bytes_step = 0

        for group in optimizer.param_groups:
            name = group.get("name", "")
            is_network = name in self.slot.network_attr_groups
            attr_cls = AttrClass.from_group_name(name)
            delta_bytes_per_desc = attr_cls.delta_bytes()

            for param in group["params"]:
                if param.grad is None:
                    continue
                g = param.grad.detach()
                if g.dim() == 0:
                    continue

                if is_network:
                    # One aggregate network descriptor per tensor.
                    self._cycles.add_puoe_network(g.numel())
                    total_bytes_step += delta_bytes_per_desc
                    continue

                if g.shape[0] == 0:
                    continue

                g2d = g.reshape(g.shape[0], -1)
                n_rows, d_rows = g2d.shape
                norms = g2d.norm(dim=1)
                nz = int((norms > 0.0).sum().item())

                self._cycles.add_puoe(n_rows, d_rows)
                n_updated += nz
                max_n_targets = max(max_n_targets, n_rows)
                total_bytes_step += n_rows * delta_bytes_per_desc
                if name in self.slot.gaussian_attr_groups:
                    total_d_opt += d_rows

        self._cycles.add_ilsf_update(n_updated)
        if n_updated > 0 and total_d_opt > 0:
            self._cycles.add_optimizer_update(n_updated, total_d_opt)

        self._frame_total_bytes += total_bytes_step
        self._frame_suppressed_bytes += 0
        self._frame_unique_targets = max(self._frame_unique_targets, max_n_targets)
        self._total_bytes_all += total_bytes_step
        self._suppressed_bytes_all += 0

    def _pre_optimizer_step_mode1_fast(self, optimizer: object) -> None:
        """Mode-1 fast path (QUEEN) — tensorised Consolidation.

        Produces statistics and cycle counts identical to the descriptor
        path in ``_apply_consolidation`` + tail of ``pre_optimizer_step``
        for mode 1.  Per-(gid, attr) ``grad_scale`` values match the dict
        path up to floating-point summation order inside mean() reductions
        (covered by ``tests/test_consolidation_batched.py``).

        What is identical
        -----------------
          - ``PUOE``  cycles per per-Gaussian param group and ``PUOE_network``
            cycles per network tensor.
          - ``UCN``   cycles on total per-Gaussian descriptor count.
          - ``ICE_scale`` cycles per per-Gaussian param group.
          - ``ILSF_update`` cycles on sum-over-attrs of ``(norm > 0)``.
          - ``optimizer_update`` cycles on
            ``n_active = num_per_gaussian_attrs × (#gids with min_scale > 0)``
            — this is the per-descriptor count used in the dict path.
          - Network hard-threshold gradient zeroing
            (``param.grad.norm() < ntc_grad_threshold``).
          - Per-row gradient scaling ``param.grad.mul_(sv)`` on Gaussian groups.
          - Per-frame and run-level bytes accounting (counted per descriptor,
            suppressed entries = scale < 1.0).
          - Descriptor-free maintenance of ``_n_gs_active`` used nowhere
            else after this call returns.

        What is intentionally different
        -------------------------------
          - ``fabric.record_update`` is not called.  Fabric state is never
            read in mode 1 (cold mask / render mask are mode >= 2 only),
            so its only observable effect would be a dict write of
            ``_recent_delta`` / ``_hotness`` that nothing reads.  Skipping
            it removes the last per-gid Python loop on this critical path.
          - Structural-change salience bookkeeping is not performed (mode
            >= 3 only).
        """
        # Groups handled by the tensor batched path.  These are the six
        # canonical per-Gaussian latent groups declared by the slot
        # (e.g. "xyz"/"f_dc"/"f_rest"/"opacity"/"scaling"/"rotation" for
        # QUEEN).  Every other group (network groups and QUEEN's small
        # "ldec_*" auxiliary groups) falls back to the original dict
        # capture/consolidation logic so that CWB (gid, attr) conflation
        # semantics match the slow path exactly.
        main_group_names    = set(self.slot.gaussian_attr_groups)
        network_group_names = set(self.slot.network_attr_groups)

        fallback_groups: list[object] = []

        # Per-tensor work items for the main per-Gaussian groups.  Multiple
        # tensors may share one AttrClass (e.g. QUEEN's f_dc and f_rest both
        # map to FEATURES), so we keep them separate — each tensor owns a
        # local index space ``[0..shape[0])`` and reuses the same CWB key
        # ``attr``, exactly mirroring the dict path's per-descriptor
        # conflation within an attribute class.
        main_items: list[tuple[object, str, int, int, torch.Tensor]] = []

        total_bytes_step   = 0
        main_total_d       = 0  # sum of per-tensor D over main tensors — this
                                # is the ``total_d`` argument to
                                # ``add_optimizer_update`` (matches slow path,
                                # which sums only over gaussian_attr_groups).
        n_updated          = 0
        suppressed_bytes_step = 0
        n_gs_ucn_main      = 0
        max_n_main         = 0

        # ``gid_min_main`` is the running per-gid min over main attrs —
        # parity with ``gaussian_scales[gid] = min(…)`` in the dict path.
        # Sized to the largest N seen so far; grows on demand.
        gid_min_main: torch.Tensor | None = None

        # ── Pass 1: per-group scan ───────────────────────────────────────────
        for group in optimizer.param_groups:
            name = group.get("name", "")
            if name in network_group_names:
                attr_cls = AttrClass.from_group_name(name)
                delta_bytes_per_desc = attr_cls.delta_bytes()
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    g = param.grad.detach()
                    if g.dim() == 0:
                        continue
                    if g.norm().item() < self.cfg.ntc_grad_threshold:
                        param.grad.zero_()
                    self._cycles.add_puoe_network(g.numel())
                    total_bytes_step += delta_bytes_per_desc
                continue

            if name not in main_group_names:
                fallback_groups.append(group)
                continue

            attr_cls = AttrClass.from_group_name(name)
            delta_bytes_per_desc = attr_cls.delta_bytes()
            attr_key = attr_cls.value

            for param in group["params"]:
                if param.grad is None:
                    continue
                g = param.grad.detach()
                if g.dim() == 0 or g.shape[0] == 0:
                    continue

                g2d = g.reshape(g.shape[0], -1)
                n_rows, d_rows = g2d.shape
                norms = g2d.norm(dim=1)

                self._cycles.add_puoe(n_rows, d_rows)
                total_bytes_step += n_rows * delta_bytes_per_desc
                main_total_d     += d_rows
                n_gs_ucn_main    += n_rows
                max_n_main = max(max_n_main, n_rows)

                main_items.append((param, attr_key, n_rows, d_rows, norms))

        # ── Pass 2: per-tensor batched consolidation (scales only) ──────────
        # Each call shares the tensor CWB keyspace with any other main
        # tensor that has the same attr_key, and with any previous step —
        # the semantics match
        # ``for desc in descs: cwb.push(desc.target_id, desc.attr, ...)``
        # in the dict path, including local ``[0..N)`` target ids and
        # per-attr conflation.
        # We COMPUTE the per-(gid, attr) scale and ICE-cycle it here, but
        # defer the grad multiplication until after the fallback pass so
        # the applied scale is the UNIFIED per-gid min across all attrs
        # (matches slow path's ``gaussian_scales[i]`` application).
        main_apply_items: list[
            tuple[object, str, int, int]
        ] = []  # param, attr_key, n_rows, d_rows
        # Tensor accumulators to avoid one .item() per main tensor.
        main_n_updated_t: torch.Tensor | None = None
        main_supp_bytes_t: torch.Tensor | None = None
        for param, attr_key, n_rows, d_rows, norms in main_items:
            per_attr_scale, _gid_min = (
                self.consolidation.compute_scales_batched({attr_key: norms})
            )
            scale = per_attr_scale[attr_key]

            # ILSF update + suppressed bytes per tensor.  Suppressed bytes
            # are counted per-descriptor on the per-(gid, attr) scale (the
            # value stored on the descriptor in the slow path), NOT on the
            # unified per-gid min — parity with the slow path's
            # ``for d in new_descs: if d.grad_scale < 1.0: ...`` loop.
            db = AttrClass.from_group_name(attr_key).delta_bytes()
            nu_this = (norms > 0.0).to(torch.int64).sum()
            sb_this = (scale < 1.0).to(torch.int64).sum() * int(db)
            if main_n_updated_t is None:
                main_n_updated_t = nu_this
                main_supp_bytes_t = sb_this
            else:
                main_n_updated_t = main_n_updated_t + nu_this
                main_supp_bytes_t = main_supp_bytes_t + sb_this

            # Mode 1 counts ICE cycles here (scale-only pass).  Mode >= 2
            # defers to ``add_ice_merged`` after the cold-mask pass, which
            # fuses scale + cold-zero into a single ICE scan (saves ~33%
            # memory BW) — see ``_apply_cold_mask`` in the slow path.
            if self.cfg.mode == 1:
                self._cycles.add_ice_scale(n_rows, d_rows)
            main_apply_items.append((param, attr_key, n_rows, d_rows))

            # Update running per-gid min across main tensors (dict-path parity).
            if gid_min_main is None:
                gid_min_main = scale.clone()
            else:
                if gid_min_main.shape[0] < n_rows:
                    pad = torch.ones(
                        n_rows - gid_min_main.shape[0],
                        dtype=gid_min_main.dtype, device=gid_min_main.device,
                    )
                    gid_min_main = torch.cat([gid_min_main, pad])
                    gid_min_main = torch.minimum(gid_min_main, scale)
                elif gid_min_main.shape[0] > n_rows:
                    gid_min_main[:n_rows] = torch.minimum(
                        gid_min_main[:n_rows], scale
                    )
                else:
                    gid_min_main = torch.minimum(gid_min_main, scale)

        # ── Pass 3: fallback dict path for non-main, non-network groups ──────
        # Processes ldec_* (tiny MLP weights with ``shape[0]`` ≪ N) via
        # the original descriptor capture → compute_scales → scale-apply
        # flow, so CWB (gid, attr) conflation with main groups is preserved.
        # Network threshold was already applied above in Pass 1.
        (fb_n_updated, fb_new_descs, fb_gaussian_scales,
         fb_total_bytes, fb_supp_bytes, fb_n_gs_ucn, fb_apply_items) = (
            self._mode1_dict_fallback(fallback_groups)
        )
        # Fuse the two host syncs (main + fallback scalar totals) into one.
        if main_n_updated_t is not None:
            stacked = torch.stack([
                main_n_updated_t.to(torch.int64).reshape(()),
                main_supp_bytes_t.to(torch.int64).reshape(()),
            ]).cpu().tolist()
            n_updated += int(stacked[0])
            suppressed_bytes_step += int(stacked[1])
        n_updated += fb_n_updated
        total_bytes_step += fb_total_bytes
        suppressed_bytes_step += fb_supp_bytes

        # UCN cycles: ONE combined call over main + fallback descriptors.
        # Calling add_ucn separately per pass would give a different cycle
        # total due to the ceil() rounding inside add_ucn.  The slow path
        # always invokes add_ucn exactly once per step with the combined
        # descriptor count, so match that.
        n_gs_ucn_total = n_gs_ucn_main + fb_n_gs_ucn
        if n_gs_ucn_total > 0:
            self._cycles.add_ucn(n_gs_ucn_total)

        self._cycles.add_ilsf_update(n_updated)

        # ── Unified per-gid min scale across main (tensor) and fallback
        # (tensor) contributions.  The dict slow path builds one
        # ``gaussian_scales`` dict over ALL new descriptors in a single
        # ``compute_scales`` call, so its per-gid min covers both main and
        # fallback attributes.  Reproduce that union here on the device.
        if gid_min_main is None:
            gid_min_unified = None
        else:
            gid_min_unified = gid_min_main.clone()
            # ``fb_new_descs`` items are ``(attr_cls, n_rows, scale)``;
            # reduce each scale tensor element-wise into the unified
            # tensor.  All on-device — no host sync per fallback tensor.
            for _attr_cls, fb_n_rows, fb_scale in fb_new_descs:
                if fb_n_rows == 0 or fb_scale is None:
                    continue
                if gid_min_unified.shape[0] < fb_n_rows:
                    pad = torch.ones(
                        fb_n_rows - gid_min_unified.shape[0],
                        dtype=gid_min_unified.dtype,
                        device=gid_min_unified.device,
                    )
                    gid_min_unified = torch.cat([gid_min_unified, pad])
                fb_s = fb_scale.to(
                    device=gid_min_unified.device,
                    dtype=gid_min_unified.dtype,
                )
                gid_min_unified[:fb_n_rows] = torch.minimum(
                    gid_min_unified[:fb_n_rows], fb_s
                )

        # Optimizer cycles.  Slow path passes ``(n_active, total_d)``; the
        # meaning of ``n_active`` depends on the mode:
        #   mode 1 — count EVERY new descriptor (main + fallback) whose
        #            per-gid unified min > 0 (UCN active count).
        #   mode >= 2 — use fabric.get_render_mask(N).sum() (render mask
        #            supersedes UCN because cold-mask zeros those grads
        #            anyway).  We still record ``self._n_gs_active`` for
        #            consistency with downstream ``stream_stats`` callers.
        n_active_t: torch.Tensor | None = None
        if gid_min_unified is not None:
            device = gid_min_unified.device
            # Main contributions.
            for _param, _attr_key, n_rows, _d_rows, _norms in main_items:
                t = (gid_min_unified[:n_rows] > 0.0).to(torch.int64).sum()
                n_active_t = t if n_active_t is None else n_active_t + t
            # Fallback contributions.
            for _attr_cls, fb_n_rows, _fb_scale in fb_new_descs:
                if fb_n_rows == 0:
                    continue
                if gid_min_unified.shape[0] >= fb_n_rows:
                    gm = gid_min_unified[:fb_n_rows]
                else:
                    pad = torch.ones(
                        fb_n_rows - gid_min_unified.shape[0],
                        dtype=gid_min_unified.dtype, device=device,
                    )
                    gm = torch.cat([gid_min_unified, pad])
                t = (gm > 0.0).to(torch.int64).sum()
                n_active_t = t if n_active_t is None else n_active_t + t
            self._n_gs_active = (
                int(n_active_t.item()) if n_active_t is not None else 0
            )
        else:
            # No main contributions — fallback-only (rare): every fb row
            # is active (no per-gid min was computed).
            self._n_gs_active = sum(n for _a, n, _s in fb_new_descs)

        # Mode >= 2: override n_opt with the fabric render-mask count
        # (hot tier only — ASC + RSB).  Cold Gaussians get their grads
        # zeroed below, so they contribute nothing to the optimizer.
        if self.cfg.mode >= 2:
            _n_opt = int(
                self.fabric.get_render_mask(
                    self.slot.n_gaussians()
                ).sum().item()
            )
        else:
            _n_opt = self._n_gs_active
        if _n_opt > 0 and main_total_d > 0:
            self._cycles.add_optimizer_update(_n_opt, main_total_d)

        # ── Unified grad-scale application (slow-path parity) ────────────────
        # The slow path multiplies every per-Gaussian param.grad by
        # ``gaussian_scales[i]`` (per-gid min across all attrs).  Apply the
        # same unified per-gid min to every main and fallback param.grad.
        if gid_min_unified is not None:
            for param, _attr_key, n_rows, _d_rows in main_apply_items:
                if n_rows == 0 or param.grad is None:
                    continue
                if gid_min_unified.shape[0] >= n_rows:
                    sv = gid_min_unified[:n_rows]
                else:
                    pad = torch.ones(
                        n_rows - gid_min_unified.shape[0],
                        dtype=gid_min_unified.dtype,
                        device=gid_min_unified.device,
                    )
                    sv = torch.cat([gid_min_unified, pad])
                sv_b = sv.to(
                    device=param.grad.device, dtype=param.grad.dtype
                )
                for _ in range(param.grad.dim() - 1):
                    sv_b = sv_b.unsqueeze(-1)
                param.grad.mul_(sv_b)

            for param, _attr_key, n_rows, _d_rows in fb_apply_items:
                if n_rows == 0 or param.grad is None:
                    continue
                if gid_min_unified.shape[0] >= n_rows:
                    sv = gid_min_unified[:n_rows]
                else:
                    pad = torch.ones(
                        n_rows - gid_min_unified.shape[0],
                        dtype=gid_min_unified.dtype,
                        device=gid_min_unified.device,
                    )
                    sv = torch.cat([gid_min_unified, pad])
                sv_b = sv.to(
                    device=param.grad.device, dtype=param.grad.dtype
                )
                for _ in range(param.grad.dim() - 1):
                    sv_b = sv_b.unsqueeze(-1)
                param.grad.mul_(sv_b)
        elif fb_apply_items:
            # No main groups contributed a scale tensor but fallback did —
            # build the unified min on-device from fb_new_descs scales.
            device = fb_apply_items[0][0].grad.device
            max_n = max(n for _p, _a, n, _d in fb_apply_items)
            sv_full = torch.ones(max_n, dtype=torch.float32, device=device)
            for _attr_cls, fb_n_rows, fb_scale in fb_new_descs:
                if fb_n_rows == 0 or fb_scale is None:
                    continue
                fs = fb_scale.to(device=device, dtype=sv_full.dtype)
                sv_full[:fb_n_rows] = torch.minimum(
                    sv_full[:fb_n_rows], fs
                )
            for param, _attr_key, n_rows, _d_rows in fb_apply_items:
                if n_rows == 0 or param.grad is None:
                    continue
                sv_b = sv_full[:n_rows].to(param.grad.dtype)
                for _ in range(param.grad.dim() - 1):
                    sv_b = sv_b.unsqueeze(-1)
                param.grad.mul_(sv_b)

        # ── Mode >= 2: cold-mask gradient zeroing + ICE-merged cycles ────────
        # Slow path semantics: zero grads for CSS (cold) Gaussians and log
        # ONE ``add_ice_merged(n_hot, n_cold, d)`` per main param group.
        # Fallback (ldec_*) tensors are NOT cold-masked by the slow path
        # (they're not in ``slot.gaussian_attr_groups``), so skip them here.
        if self.cfg.mode >= 2:
            n_g  = self.slot.n_gaussians()
            cold = self.fabric.get_cold_mask(n_g)
            n_cold_total = int(cold.sum().item())
            n_hot_total  = n_g - n_cold_total
            for param, _attr_key, n_rows, d_rows in main_apply_items:
                if n_rows == 0 or param.grad is None:
                    continue
                ng = min(n_rows, cold.shape[0])
                if ng > 0:
                    cold_dev = cold[:ng].to(param.grad.device)
                    param.grad[:ng][cold_dev] = 0.0
                self._cycles.add_ice_merged(
                    n_hot_total, n_cold_total, d_rows
                )

        # ── Mode >= 3: salience structural-change marking ────────────────────
        # The fast path never enters when ``struct_changed`` is not None
        # (dispatcher guard), so the struct set is empty — matches slow
        # path's behaviour when no structural changes occurred this step.
        # No-op kept explicit for readability / future extension.

        # ── Traffic accounting ───────────────────────────────────────────────
        self._frame_total_bytes      += total_bytes_step
        self._frame_suppressed_bytes += suppressed_bytes_step
        self._frame_unique_targets = max(
            self._frame_unique_targets, max_n_main
        )
        self._total_bytes_all      += total_bytes_step
        self._suppressed_bytes_all += suppressed_bytes_step

    def _mode1_dict_fallback(
        self, groups: list
    ) -> tuple[int, list, dict, int, int, int, list]:
        """Dict-path consolidation for auxiliary (non-main) per-row groups.

        Called by ``_pre_optimizer_step_mode1_fast`` for groups that are not
        in ``slot.gaussian_attr_groups`` and not in ``slot.network_attr_groups``
        (e.g. QUEEN's ``ldec_*`` latent-decoder MLP tensors).  These tensors
        have ``shape[0]`` up to a few hundred — the descriptor-per-row cost
        is negligible — so using the original dict-based
        ``consolidation.compute_scales()`` keeps CWB ``(gid, attr)``
        conflation identical to the generic path.

        Returns
        -------
        (n_updated, new_descs, gaussian_scales, total_bytes, suppressed_bytes,
         n_gs_ucn)

        where:
          - ``n_updated``        : count of fallback descriptors with
            delta_norm > 0 (fed into ``add_ilsf_update``).
          - ``new_descs``        : list of the fallback descriptors just
            created, in the stream (needed by the caller to compute the
            unified per-gid active count).
          - ``gaussian_scales``  : per-gid min scale across fallback
            descriptors (dict).  Caller unifies it with the main path's
            per-gid min so that optimizer-cycle counting matches the slow
            path's single ``gaussian_scales`` pass.
          - ``total_bytes``      : sum of ``delta_bytes()`` for the fallback
            descriptors (added into per-frame and run-level totals).
          - ``suppressed_bytes`` : sum of ``delta_bytes()`` for fallback
            descriptors whose per-descriptor grad_scale < 1.0 (semantics
            match the slow path, which stores per-(gid, attr) scales, not
            per-gid mins, on each descriptor).
          - ``n_gs_ucn``         : number of fallback per-Gaussian
            descriptors with target_id >= 0.  The caller combines this
            with the main-path count and invokes ``add_ucn`` once so the
            cycle total matches the slow path's single ``add_ucn`` call.
            (Calling ``add_ucn`` twice per step would give a slightly
            different total due to the ``ceil()`` rounding inside.)

        Side effects: writes to ``self._stream`` (descriptors), advances
        the dict CWB / frame accumulator inside ``ConsolidationUnit``,
        scales ``param.grad`` in-place, counts PUOE + ICE cycles.  UCN
        cycles are NOT counted here (see ``n_gs_ucn`` above).
        """
        if not groups:
            return (0, [], {}, 0, 0, 0, [])

        total_bytes = 0
        suppressed_bytes_t: torch.Tensor | None = None  # accumulate as tensor
        n_updated_t: torch.Tensor | None = None         # accumulate as tensor
        n_gs_ucn = 0
        # Per-gid min (as tensor) across all fallback tensors.
        fb_gid_min_t: torch.Tensor | None = None
        # Deferred grad-apply work items so the caller can multiply by the
        # UNIFIED per-gid min scale (main + fallback) rather than the
        # per-(gid, attr) scale we compute here.  Slow path's
        # ``_apply_consolidation`` uses ``gaussian_scales[i]`` (per-gid
        # min across all attrs) for every param, so we must do the same.
        # Each item: (param, attr_key, n_rows, d_rows).
        fb_apply_items: list[tuple[object, str, int, int]] = []
        # For parity with the slow path we need a per-descriptor scale
        # tensor view kept alive in ``fb_new_descs_proxy`` so the caller
        # can use it for the n_gs_active recount.  Store only lightweight
        # ``_FBDesc`` namedtuples with target_id and scale — no stream
        # object allocation (which cost per-descriptor Python overhead in
        # the original implementation).
        fb_new_descs: list = []

        for group in groups:
            name = group.get("name", "")
            attr_cls = AttrClass.from_group_name(name)
            attr_key = attr_cls.value
            delta_bytes_per_desc = attr_cls.delta_bytes()
            for param in group["params"]:
                if param.grad is None:
                    continue
                g = param.grad.detach()
                if g.dim() == 0 or g.shape[0] == 0:
                    continue
                g2d = g.reshape(g.shape[0], -1)
                n_rows, d_rows = g2d.shape
                norms = g2d.norm(dim=1)
                self._cycles.add_puoe(n_rows, d_rows)
                total_bytes += n_rows * delta_bytes_per_desc
                n_gs_ucn += n_rows

                # Batched scale computation reading & writing the shared
                # batched CWB (semantic parity with the scalar variant).
                scale = (
                    self.consolidation.compute_scales_fallback_batched(
                        attr_key, norms
                    )
                )

                # Per-descriptor byte + n_updated accounting (all tensor-
                # only; we never force a host sync in the inner loop).
                suppressed_mask = (scale < 1.0)
                active_mask     = (norms > 0.0)
                sb_this = suppressed_mask.to(torch.int64).sum() * int(
                    delta_bytes_per_desc
                )
                if suppressed_bytes_t is None:
                    suppressed_bytes_t = sb_this
                else:
                    suppressed_bytes_t = suppressed_bytes_t + sb_this
                nu_this = active_mask.to(torch.int64).sum()
                if n_updated_t is None:
                    n_updated_t = nu_this
                else:
                    n_updated_t = n_updated_t + nu_this

                # Update fb per-gid min tensor.
                if fb_gid_min_t is None:
                    fb_gid_min_t = scale.clone()
                else:
                    if fb_gid_min_t.shape[0] < n_rows:
                        pad = torch.ones(
                            n_rows - fb_gid_min_t.shape[0],
                            dtype=fb_gid_min_t.dtype,
                            device=fb_gid_min_t.device,
                        )
                        fb_gid_min_t = torch.cat([fb_gid_min_t, pad])
                    fb_gid_min_t[:n_rows] = torch.minimum(
                        fb_gid_min_t[:n_rows], scale
                    )

                self._cycles.add_ice_scale(n_rows, d_rows)
                fb_apply_items.append((param, attr_key, n_rows, d_rows))

                # Track per-descriptor grad_scale as a tensor view so the
                # caller can recount n_active without a Python loop over
                # every descriptor.  We store ONLY the per-tensor scale
                # (not a stream.add call — that was the per-descriptor
                # Python allocation bottleneck).
                fb_new_descs.append(
                    (attr_cls, n_rows, scale)
                )

        # Single host sync for both scalar stats (fuse to one .item()).
        if suppressed_bytes_t is not None or n_updated_t is not None:
            sb = (suppressed_bytes_t if suppressed_bytes_t is not None
                  else torch.tensor(0, dtype=torch.int64))
            nu = (n_updated_t if n_updated_t is not None
                  else torch.tensor(0, dtype=torch.int64))
            stacked = torch.stack([
                sb.to(torch.int64).reshape(()),
                nu.to(torch.int64).reshape(()),
            ]).cpu().tolist()
            suppressed_bytes = int(stacked[0])
            n_updated = int(stacked[1])
        else:
            suppressed_bytes = 0
            n_updated = 0

        # ``gaussian_scales`` is kept as an empty dict — the caller now
        # uses the on-device ``fb_new_descs`` scale tensors directly for
        # the unify + apply passes.  The dict shape is preserved purely
        # for backwards-compatible tuple unpacking.
        return (n_updated, fb_new_descs, {},
                total_bytes, suppressed_bytes, n_gs_ucn, fb_apply_items)

    # ── Post-render hook ──────────────────────────────────────────────────────

    def post_render(self, render_pkg: dict) -> None:
        """Called by the operator slot after every render() call.

        Updates LiveStateFabric hotness from visibility and
        SalienceFirstIssue scores from radii × opacity (mode >= 3).

        Parameters
        ----------
        render_pkg : dict with keys from diff-gaussian-rasterization:
            "visibility_filter" (BoolTensor [N]) — required for fabric
            "radii"             (IntTensor [N])  — required for salience
        Additional keys like "opacity" are used if present.
        """
        if "visibility_filter" in render_pkg:
            vf = render_pkg["visibility_filter"]
            if vf is not None and vf.numel() > 0:
                self.fabric.record_render(vf)
                # ILSF_render runs in parallel with Salience (different memory).
                self._cycles.add_ilsf_render(int(vf.sum().item()))

        if self.cfg.mode >= 3 and "radii" in render_pkg:
            radii = render_pkg["radii"].float()
            if "opacity" in render_pkg:
                opacity = render_pkg["opacity"].float()
            else:
                try:
                    opacity = self.slot.get_gaussians().get_opacity.squeeze(-1)
                except Exception:
                    opacity = torch.ones(radii.shape[0], device=radii.device)
            self.salience.update(radii, opacity)
            # Salience cycles run in parallel with ILSF_render above.
            self._cycles.add_salience(radii.shape[0])

    # ── Stage 4 + 5: Inference render proxy ──────────────────────────────────

    def make_render_proxy(self) -> MaskedGaussianProxy:
        """Build a MaskedGaussianProxy for FPS measurement.

        Applies LiveStateFabric render mask (mode >= 2) and
        SalienceFirstIssue keep_mask (mode >= 3).

        **Never called during training** — full Gaussians are always used
        for loss computation to preserve training quality.
        """
        gaussians = self.slot.get_gaussians()
        n         = gaussians.get_xyz.shape[0]

        if n == 0:
            raise RuntimeError("No Gaussians in model; cannot make proxy.")

        if self.cfg.mode >= 2:
            mask = self.fabric.get_render_mask(n)
        else:
            mask = torch.ones(n, dtype=torch.bool)

        if self.cfg.mode >= 3:
            keep, _ = self.salience.classify(n)
            mask = mask & keep

        # Safety: ensure at least 1 Gaussian is always rendered
        if not mask.any():
            mask[0] = True

        indices = mask.nonzero(as_tuple=False).squeeze(1)
        # Move indices to the same device as the Gaussian tensors
        device  = gaussians.get_xyz.device
        return MaskedGaussianProxy(gaussians, indices.to(device))

    # ── FPS measurement ───────────────────────────────────────────────────────

    def measure_fps(
        self,
        render_fn,
        cameras: list,
        pipe,
        bg: torch.Tensor,
    ) -> float:
        """Measure actual render throughput (frames per second).

        Uses a MaskedGaussianProxy reflecting the current ablation mode so
        that the measured FPS corresponds to the ASIC's actual render load.

        Parameters
        ----------
        render_fn : callable(camera, gaussians, pipe, bg) → render_pkg
        cameras   : list of camera objects to rotate through
        pipe      : pipeline params (passed through to render_fn)
        bg        : background colour tensor

        Returns
        -------
        float — frames per second (GPU-synchronised)
        """
        proxy = self.make_render_proxy()

        # Warm up: JIT / kernel caching
        for i in range(self.cfg.fps_warmup):
            render_fn(cameras[i % len(cameras)], proxy, pipe, bg)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for i in range(self.cfg.fps_measure):
            render_fn(cameras[i % len(cameras)], proxy, pipe, bg)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return self.cfg.fps_measure / max(elapsed, 1e-9)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stream_stats(self, n_gaussians: int | None = None) -> dict:
        """Return per-frame and cumulative traffic statistics.

        Parameters
        ----------
        n_gaussians : override for the Gaussian count used by fabric.tier_counts.
            Useful when calling after slot.uninstall_hooks() has cleared the
            internal Gaussian reference (e.g. in QUEEN's runner).  If None,
            falls back to slot.n_gaussians() (may return 0 when uninstalled).
        """
        s = self._stream
        if n_gaussians is None:
            n_gaussians = self.slot.n_gaussians()
        fabric_counts  = self.fabric.tier_counts(max(n_gaussians, 1))
        # The mode-1 fast path also populates the per-frame byte caches
        # (empty descriptor stream), so reuse the cache when it is non-zero.
        use_cache = (
            self.cfg.mode == 0
            or (self.cfg.mode == 1 and self._frame_total_bytes > 0)
        )
        if use_cache:
            frame_total = self._frame_total_bytes
            frame_supp = self._frame_suppressed_bytes
            frame_active = frame_total - frame_supp
            frame_ratio = (frame_supp / frame_total) if frame_total > 0 else 0.0
            frame_unique = self._frame_unique_targets
        else:
            frame_total = s.total_bytes()
            frame_supp = s.suppressed_bytes()
            frame_active = s.active_bytes()
            frame_ratio = s.suppression_ratio()
            frame_unique = len(s.unique_targets())
        return {
            # Per-frame (last frame)
            "frame_total_bytes":      frame_total,
            "frame_suppressed_bytes": frame_supp,
            "frame_active_bytes":     frame_active,
            "frame_suppression_ratio": frame_ratio,
            "frame_unique_targets":   frame_unique,
            # Cumulative over all frames in this run
            "run_total_bytes":        self._total_bytes_all,
            "run_suppressed_bytes":   self._suppressed_bytes_all,
            # Fabric tier counts
            **fabric_counts,
        }

    # ── Cycle-accurate timing ─────────────────────────────────────────────────

    def cycle_summary(self) -> dict:
        """Return per-unit ASIC cycle counts and critical-path total.

        Includes the 1 GHz timing estimate (asic_time_s_1ghz).
        Call after the last frame and after set_render_cycles_from_fps().
        """
        return self._cycles.summary()

    def set_render_cycles_from_fps(self, fps: float) -> None:
        """Store render cycle count back-calculated from measured FPS.

        render_cycles = clock_hz / fps  (no estimation, direct from hardware spec)

        Must be called after MetricsCollector.measure_fps() returns so that
        the render cycles are included in the next call to cycle_summary().
        """
        self._cycles.set_render_cycles_from_fps(fps)

    def add_network_training(self, n_params: int, n_iters: int) -> None:
        """Account for compact MLP network (decoder/gate) training per frame.

        QUEEN trains LatentDecoder and Gate networks alongside Gaussian attrs.
        Call once per training frame (before begin_frame()) with the total
        network parameter count and the number of training iterations executed.

        Cycles are added to the ntc_train accumulator (same as 3DGStream NTC)
        so that asic_ntc_train_cycles and asic_train_time_s both capture them.

        Parameters
        ----------
        n_params : total trainable parameter count across ALL network modules
                   (sum over decoder and gate param groups).
        n_iters  : number of optimizer.step() calls executed in this frame.
        """
        if n_params <= 0 or n_iters <= 0:
            return
        self._cycles.add_mlp_training(n_params, n_iters)

    def add_gaussian_training_iters(self, n_iters: int, n_total: int = 0) -> None:
        """Account for QUEEN-style direct-Gaussian training iterations.

        QUEEN does not have NTC and never calls ingest_ntc_deltas(), so the
        per-iteration 3DGS render+loss+backward cycles would otherwise be
        missing from the cycle summary.  Call this once per training frame
        (before begin_frame()) with the number of training iterations executed
        in that frame.  Internally delegates to CycleAccumulator.
        add_render_training_iter() reusing the same ntc_train_render counter so
        that asic_ntc_train_render_cycles and asic_train_time_s work correctly.

        Parameters
        ----------
        n_iters : number of optimizer.step() calls (training iterations) in the
                  completed frame.  Pass 0 to skip (e.g. frame 0 with no history).
        n_total : total Gaussian count (gaussians.get_xyz.shape[0]).  Used for
                  Phase 1 preprocessing in add_render_training_iter.  Falls back
                  to n_visible from _prev_render_stats if 0.
        """
        if n_iters <= 0 or self._prev_render_stats is None:
            return
        rs  = self._prev_render_stats
        n   = n_total if n_total > 0 else rs["n_visible"]
        self._cycles.add_render_training_iter(
            n,
            rs["n_visible"],
            rs["tile_touches"],
            rs["H"],
            rs["W"],
            rs["sh_degree"],
            n_iters,
        )

    def set_render_stats_from_render_pkg(
        self,
        render_pkg: dict,
        sh_degree: int,
    ) -> None:
        """Capture scene render statistics for the NTC training render cycle model.

        Called from the ablation runner immediately after post_render().  The
        captured statistics are stored as _prev_render_stats and used in the
        NEXT frame's ingest_ntc_deltas() to parameterise add_render_training_iter().

        Parameters measured from the live render output (no magic numbers):
          n_visible    : (radii > 0).sum()
          avg_radii_px : mean screen-space radius of visible Gaussians (pixels)
          tile_touches : n_visible × π × (avg_radii_px / 16)²
                         16 = CUDA tile block size, diff-gaussian-rasterization
                         config.h: #define BLOCK_X 16 / #define BLOCK_Y 16
          H, W         : render_pkg["render"].shape[-2:]
          sh_degree    : gaussians.active_sh_degree (passed by caller)
        """
        radii    = render_pkg["radii"]
        n_vis    = int((radii > 0).sum().item())
        img      = render_pkg["render"]
        H        = int(img.shape[-2])
        W        = int(img.shape[-1])
        if n_vis > 0:
            avg_r = float(radii[radii > 0].float().mean().item())
        else:
            avg_r = 0.0
        # tile_size = 16: BLOCK_X = BLOCK_Y = 16 in diff-gaussian-rasterization config.h
        tile_size = 16
        ttouches = int(n_vis * math.pi * (avg_r / tile_size) ** 2) if avg_r > 0 else 0
        self._prev_render_stats = {
            "n_visible":    n_vis,
            "tile_touches": ttouches,
            "H":            H,
            "W":            W,
            "sh_degree":    sh_degree,
        }

    def reset(self) -> None:
        """Reset all stateful components. Used between consecutive ablation runs."""
        self.consolidation.flush_frame_accum()  # discard any partial frame (dict)
        self.consolidation.flush_frame_accum_batched()  # and tensor path
        self.consolidation.reset()
        self.fabric.reset()
        self.salience.reset()
        self._cycles.reset()
        self._n_gs_active          = 0
        self._version              = 0
        self._stream               = UpdateStream()
        self._total_bytes_all      = 0
        self._suppressed_bytes_all = 0
        self._frame_total_bytes = 0
        self._frame_suppressed_bytes = 0
        self._frame_unique_targets = 0
