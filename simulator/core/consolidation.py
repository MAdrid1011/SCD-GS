"""
ConsolidationUnit — Stage 1 of the GLASS ASIC pipeline.

Implements Update-Stream Consolidation: detects temporally correlated
per-Gaussian gradient updates and suppresses (folds) the redundant portion.

How it works
------------
For each Gaussian i and attribute class a, the unit maintains a ring buffer
of the last `cwb_window` gradient L2-norms (CorrelationWindowBuffer).

Each frame, the incoming delta_norm is compared to the buffer mean via the
norm-ratio similarity proxy:

    sim(cur, mean) = min(cur, mean) / max(cur, mean)   in [0, 1]

This is a conservative scalar approximation of cosine similarity. It equals
1.0 when the magnitude is unchanged, and decreases as magnitudes diverge.
It may over-suppress in edge cases (same norm, opposite direction), but the
effect is bounded and safe for training quality.

Decision table (binary hardware gate)
--------------------------------------
  delta_norm < min_delta_norm        → grad_scale = 0.0  (near-zero: always skip)
  sim > merge_cos_threshold          → grad_scale = 0.0  (stable: suppress entirely)
  sim <= merge_cos_threshold         → grad_scale = 1.0  (diverging: pass through)
  no history yet                     → grad_scale = 1.0  (first frame: pass through)

The hardware UCN unit implements a binary suppress/pass gate. Partial
scaling (1.0 − sim) would require a per-Gaussian multiplier, which has
higher area cost than a simple comparator + zero-mask. The binary model
matches the `add_ucn` docstring which describes a single threshold CMP as
the final gate step. Bandwidth suppression is reflected in `frame_suppressed_bytes`
(grad_scale < 1.0 → all suppressed Gaussians), while optimizer savings are
reflected in n_opt (only grad_scale > 0 Gaussians go through Adam).

The resulting grad_scale values are stored back into each UpdateDescriptor
and returned as a {gaussian_id → scale} dict consumed by GLASSPipeline.

This module is model-agnostic; it only operates on UpdateDescriptor fields.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from simulator.config import ASICConfig
    from simulator.core.update_descriptor import UpdateDescriptor, UpdateStream


# ── Correlation Window Buffer ─────────────────────────────────────────────────

class CorrelationWindowBuffer:
    """Per-(Gaussian, attr_class) ring buffer of recent delta_norm values.

    Indexed by (target_id, attr_class) tuples to keep history separated
    per attribute group (position history is distinct from rotation history).
    """

    def __init__(self, window: int) -> None:
        self.window = window
        self._buf: dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def push(self, gid: int, attr: str, norm: float) -> None:
        self._buf[(gid, attr)].append(norm)

    def mean_norm(self, gid: int, attr: str) -> float:
        buf = self._buf[(gid, attr)]
        if not buf:
            return 0.0
        return float(np.mean(buf))

    def has_history(self, gid: int, attr: str) -> bool:
        return len(self._buf[(gid, attr)]) > 0

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ── Batched (tensor-backed) CWB ──────────────────────────────────────────────

class BatchedCorrelationWindowBuffer:
    """Per-attr tensor-backed ring buffer.

    Mirrors CorrelationWindowBuffer semantics but stored as tensors
    ``[N, window]`` per attribute, so batched reads and writes run as
    single tensor ops instead of millions of dict operations.

    Semantics match the dict-backed version exactly:
      ``mean_norm(gid, attr)`` = arithmetic mean over the up-to-``window``
      values pushed into ``(gid, attr)``, or 0 if no history.
      ``has_history(gid, attr)`` iff at least one push happened for that key.

    N (row count) may grow across frames due to densification.  When the
    caller's N exceeds the currently allocated rows, the tensor is padded
    with zeros and the per-row fill count resets to 0 for the new rows
    (so they report ``has_history == False``, identical to a fresh key in
    the dict version).  Rows are never shrunk.
    """

    def __init__(self, window: int) -> None:
        self.window = int(window)
        # attr_str -> Tensor[N, window]
        self._buf: dict[str, torch.Tensor] = {}
        # attr_str -> Tensor[N] int32 (count of valid entries, clamped to window)
        self._filled: dict[str, torch.Tensor] = {}
        # attr_str -> Tensor[N] int32 per-row ring head.  Per-row (not
        # scalar) because push_frame_batched() only writes rows that got a
        # push this frame — rows skipped this frame must keep their head
        # where it was so the next push doesn't clobber valid history.
        self._head: dict[str, torch.Tensor] = {}

    def _ensure(self, attr: str, n: int, device: torch.device) -> None:
        if attr not in self._buf:
            self._buf[attr] = torch.zeros(
                (n, self.window), dtype=torch.float32, device=device
            )
            self._filled[attr] = torch.zeros(n, dtype=torch.int32, device=device)
            self._head[attr] = torch.zeros(n, dtype=torch.int32, device=device)
            return
        buf = self._buf[attr]
        if buf.shape[0] >= n:
            return
        # Grow: append zero rows; filled counts and heads start at 0 →
        # has_history == False for new rows, identical to fresh dict keys.
        add = n - buf.shape[0]
        pad_buf = torch.zeros(
            (add, self.window), dtype=buf.dtype, device=buf.device
        )
        pad_cnt = torch.zeros(add, dtype=torch.int32, device=buf.device)
        pad_head = torch.zeros(add, dtype=torch.int32, device=buf.device)
        self._buf[attr] = torch.cat([buf, pad_buf], dim=0)
        self._filled[attr] = torch.cat([self._filled[attr], pad_cnt], dim=0)
        self._head[attr] = torch.cat([self._head[attr], pad_head], dim=0)

    def push_batched(self, attr: str, norms: torch.Tensor) -> None:
        """Push one norm per Gaussian for ``attr`` (all rows).

        Used by the iteration-level UCN path (``frame_level_ucn=False``):
        every optimizer step writes one value per row.  For frame-level
        UCN use ``push_frame_batched`` with a per-row mask instead.
        """
        n = int(norms.shape[0])
        if n == 0:
            return
        self._ensure(attr, n, norms.device)
        mask = torch.ones(n, dtype=torch.bool, device=norms.device)
        self.push_frame_batched(attr, norms, mask)

    def push_frame_batched(
        self,
        attr: str,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Push ``values[i]`` into row ``i`` only where ``mask[i]``.

        Rows where ``mask[i]`` is False keep their existing ring head,
        filled count, and stored values — mirroring the dict path which
        leaves ``(gid, attr)`` keys untouched when they had no pushes
        this frame.

        Parameters
        ----------
        values : Tensor[n] — one value per row; ignored on masked-out rows.
        mask   : BoolTensor[n] — True for rows that got a push this frame.
        """
        n = int(values.shape[0])
        if n == 0:
            return
        assert mask.shape[0] == n, "mask must match values length"
        self._ensure(attr, n, values.device)
        buf   = self._buf[attr]
        head  = self._head[attr][:n]           # int32 per row
        mask_i = mask.to(torch.bool)
        active_rows = mask_i.nonzero(as_tuple=False).squeeze(-1)
        if active_rows.numel() == 0:
            return
        head_act = head[active_rows].to(torch.long)
        row_act  = active_rows.to(torch.long)
        buf[row_act, head_act] = values[active_rows].to(buf.dtype)
        # Advance head and increment filled ONLY on active rows.
        new_head = (head_act.to(torch.int32) + 1) % self.window
        self._head[attr][active_rows] = new_head
        filled = self._filled[attr][:n]
        filled[active_rows] = torch.clamp(
            filled[active_rows] + 1, max=self.window
        )
        self._filled[attr][:n] = filled

    def mean_norm_batched(
        self, attr: str, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean[N], has_history[N])`` for the first ``n`` rows.

        ``mean`` equals ``sum / count`` over valid entries, identical to
        ``float(np.mean(deque))`` in the dict version.  Entries with 0
        fill count return mean=0 and has_history=False.
        """
        if attr not in self._buf or n == 0:
            return (
                torch.zeros(n, dtype=torch.float32, device=device),
                torch.zeros(n, dtype=torch.bool, device=device),
            )
        self._ensure(attr, n, device)
        buf   = self._buf[attr][:n]            # [N, window]
        filled = self._filled[attr][:n]        # [N] int32
        has = filled > 0
        # Sum in float64 to match np.mean's precision (NumPy accumulates
        # the deque values in float64 even if the stored values are fp32).
        sum_rows = buf.to(torch.float64).sum(dim=1)
        denom = filled.to(torch.float64).clamp_min(1.0)
        mean = (sum_rows / denom).to(buf.dtype)
        mean = torch.where(has, mean, torch.zeros_like(mean))
        return mean, has

    def clear(self) -> None:
        self._buf.clear()
        self._filled.clear()
        self._head.clear()


# ── Consolidation Unit ────────────────────────────────────────────────────────

class ConsolidationUnit:
    """Stage 1: Update-Stream Consolidation.

    Stateful across frames (CorrelationWindowBuffer persists).
    Created once per ablation run inside GLASSPipeline.__init__().

    Two operating modes controlled by cfg.frame_level_ucn:

    Iteration-level (default)
        CWB entry = one optimizer step's norm.  Compares step N to recent
        cwb_window steps.  Detects within-frame convergence.

    Frame-level (frame_level_ucn=True)
        CWB entry = mean over all optimizer steps in one training frame.
        Norms are accumulated in _frame_accum during training; flushed to
        CWB by flush_frame_accum() at begin_frame().  Compares each step of
        frame N against the aggregate from frames N-1 … N-cwb_window.
        Correctly handles dynamic-scene baselines (QUEEN): dynamic Gaussians
        change norm across frames → not suppressed; static Gaussians stable
        across frames → suppressed.
    """

    def __init__(self, config: "ASICConfig") -> None:
        self.cfg = config
        self.cwb = CorrelationWindowBuffer(config.cwb_window)
        # Accumulator for frame-level UCN: key=(gid, attr), value=list of norms
        self._frame_accum: dict[tuple, list] = defaultdict(list)

        # ── Batched (tensor) state — used by the mode-1 fast path only ────────
        # The batched path keeps its own CWB and its own per-frame accumulator
        # so that it never races with the dict-based path; GLASSPipeline
        # selects exactly one of the two at construction time by calling
        # compute_scales_batched() or compute_scales().
        self.cwb_batched = BatchedCorrelationWindowBuffer(config.cwb_window)
        # attr_str -> Tensor[N] running sum of this frame's per-step norms
        self._batched_frame_sum: dict[str, torch.Tensor] = {}
        # attr_str -> Tensor[N] per-row number of pushes that contributed to
        # the sum this frame.  Per-row (not scalar) because different groups
        # sharing an attr_class may cover different row counts — e.g. the
        # main xyz tensor has N rows while a fallback ldec_xyz tensor may
        # have only 1 row.  The dict path's flush_frame_accum() computes
        # np.mean per (gid, attr) key, which is exactly per-gid denom.
        self._batched_frame_count: dict[str, torch.Tensor] = {}

    def flush_frame_accum(self) -> None:
        """Push per-frame mean norms into the CWB.

        Called by GLASSPipeline.begin_frame() when frame_level_ucn=True.
        Each (Gaussian, attr) pair gets exactly one CWB entry per frame
        (the mean over all optimizer steps in that frame).  The accumulator
        is then cleared for the next frame.
        """
        for key, norms in self._frame_accum.items():
            if norms:
                gid, attr = key
                self.cwb.push(gid, attr, float(np.mean(norms)))
        self._frame_accum.clear()

    def compute_scales(
        self,
        stream: "UpdateStream",
        start_idx: int = 0,
    ) -> dict[tuple[int, str], float]:
        """Compute per-(Gaussian, attr) gradient scales for this frame.

        Modifies each descriptor's grad_scale field in-place.
        Returns a dict keyed by target_id → minimum scale across all attr
        classes for use in GLASSPipeline._apply_consolidation().

        Parameters
        ----------
        start_idx : Only process descriptors from this index onward.
                    Pass the stream length captured before _capture_stream
                    so that NTC descriptors (already processed by
                    ingest_ntc_deltas) are not re-scanned in Stage-2 steps.
        """
        # (gid, attr) → scale
        fine_scales: dict[tuple[int, str], float] = {}
        # Per-Gaussian aggregate (minimum scale across attrs) for cold-mask use
        gaussian_scales: dict[int, float] = {}

        for desc in stream.descriptors[start_idx:]:
            gid  = desc.target_id
            attr = desc.attr_class.value
            key  = (gid, attr)

            scale = self._compute_one(gid, attr, desc.delta_norm)
            fine_scales[key] = scale
            desc.grad_scale  = scale

            # Track minimum scale per Gaussian (most conservative)
            if gid >= 0:
                gaussian_scales[gid] = min(gaussian_scales.get(gid, 1.0), scale)

            # Update CWB (or frame accumulator for frame-level mode)
            if gid >= 0:
                if self.cfg.frame_level_ucn:
                    # Accumulate; CWB is pushed at end of frame by flush_frame_accum()
                    self._frame_accum[(gid, attr)].append(desc.delta_norm)
                else:
                    self.cwb.push(gid, attr, desc.delta_norm)

        return gaussian_scales   # used by GLASSPipeline._apply_consolidation

    def _compute_one(self, gid: int, attr: str, delta_norm: float) -> float:
        """Compute grad_scale for a single (Gaussian, attr) descriptor."""
        # Network parameters: hard threshold only
        if gid < 0:
            return 1.0  # network threshold handled separately in pipeline

        # Near-zero gradient: always suppress
        if delta_norm < self.cfg.min_delta_norm:
            return 0.0

        # No history yet: always pass through
        if not self.cwb.has_history(gid, attr):
            return 1.0

        mean_n = self.cwb.mean_norm(gid, attr)
        if mean_n < self.cfg.min_delta_norm:
            # Historical norms were also near-zero → the Gaussian is dormant
            return 0.0

        # ── Adaptive threshold ────────────────────────────────────────────────
        # When adaptive_ucn_threshold is enabled, raise the suppression
        # threshold proportionally to the CWB's coefficient of variation (CV).
        # High CV (norm is changing a lot) → model is still learning →
        #   effective_threshold → 1.0 → nothing suppressed.
        # Low CV (norm is stable)          → model converged →
        #   effective_threshold = merge_cos_threshold → normal suppression.
        #
        # Hardware mapping: a single additional std/mean divider + comparator
        # gate.  The divider reuses the SRAM bandwidth already used for the
        # EMA read, so area overhead is < 5 % of the UCN comparator array.
        if self.cfg.adaptive_ucn_threshold:
            buf = self.cwb._buf[(gid, attr)]
            if len(buf) >= 2:
                vals = list(buf)
                mean_v = float(np.mean(vals))
                std_v  = float(np.std(vals))
                cv     = std_v / (mean_v + 1e-8)
                cv_c   = min(cv, 1.0)
                effective_threshold = (
                    self.cfg.merge_cos_threshold
                    + cv_c * (1.0 - self.cfg.merge_cos_threshold)
                )
            else:
                effective_threshold = self.cfg.merge_cos_threshold
        else:
            effective_threshold = self.cfg.merge_cos_threshold

        # Norm-ratio similarity proxy: in [0, 1]
        lo, hi = min(delta_norm, mean_n), max(delta_norm, mean_n)
        sim = lo / hi  # 1.0 → identical magnitude; 0.0 → orders-of-magnitude apart

        if sim > effective_threshold:
            # Stable update: hardware binary gate suppresses entirely.
            return 0.0

        # Dissimilar (new information): pass through unchanged.
        return 1.0

    # ── Batched (tensor) API ──────────────────────────────────────────────────
    # These methods mirror compute_scales / flush_frame_accum but operate on
    # per-attr tensors instead of per-descriptor Python objects.  They are
    # used by the GLASSPipeline mode-1 fast path (QUEEN).  Outputs are
    # bit-equivalent to the dict path up to floating-point order-of-summation
    # differences inside mean() reductions (see the self-test in
    # tests/test_consolidation_batched.py).

    def compute_scales_batched(
        self,
        attr_to_norms: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        """Vectorized equivalent of compute_scales() for one optimizer step.

        Parameters
        ----------
        attr_to_norms : {attr_str: Tensor[N]} — per-Gaussian gradient L2 norms
            for each per-Gaussian attribute group at this step.  All tensors
            must share the same N and device; the caller guarantees this by
            iterating optimizer.param_groups in the pipeline fast path.

        Returns
        -------
        per_attr_scale   : {attr_str: Tensor[N]} — grad_scale in {0, 1}.
        gaussian_min     : Tensor[N] | None — min scale across attrs per gid,
                           used by the pipeline to count _n_gs_active and to
                           avoid reading back the dict (None when attrs empty).

        Also:
          - Updates the per-frame batched accumulator (when frame_level_ucn).
          - Pushes directly into the batched CWB (when not frame_level_ucn).

        Semantic parity with compute_scales():
          - near-zero: delta_norm < min_delta_norm       → scale = 0
          - no history yet                               → scale = 1
          - mean_history < min_delta_norm                → scale = 0
          - sim = min(d, m) / max(d, m) > threshold      → scale = 0
          - else                                         → scale = 1

        The adaptive_ucn_threshold branch is NOT implemented here; callers
        that enable it must use compute_scales() (dict path).  QUEEN uses
        adaptive_ucn_threshold=False so this restriction is harmless.
        """
        if self.cfg.adaptive_ucn_threshold:
            raise NotImplementedError(
                "compute_scales_batched() does not support "
                "adaptive_ucn_threshold; use compute_scales() instead."
            )
        per_attr_scale: dict[str, torch.Tensor] = {}
        gid_min: torch.Tensor | None = None
        min_dn = float(self.cfg.min_delta_norm)
        thr    = float(self.cfg.merge_cos_threshold)

        for attr, norms in attr_to_norms.items():
            n = int(norms.shape[0])
            if n == 0:
                per_attr_scale[attr] = norms.new_empty((0,))
                continue

            mean_n, has_hist = self.cwb_batched.mean_norm_batched(
                attr, n, norms.device
            )

            # Start from pass-through (1.0), then apply the three zero
            # conditions in order.
            scale = torch.ones(n, dtype=norms.dtype, device=norms.device)

            # (a) near-zero current gradient: always suppress.
            scale = torch.where(
                norms < min_dn, torch.zeros_like(scale), scale
            )

            # (b) history exists but dormant: mean_n < min_dn → suppress.
            dormant = has_hist & (mean_n < min_dn)
            scale = torch.where(dormant, torch.zeros_like(scale), scale)

            # (c) history exists and non-dormant: compute similarity and
            #     suppress if above threshold.  Rows without history already
            #     have scale unchanged (stays at 1 unless zeroed by (a)).
            active_hist = has_hist & (mean_n >= min_dn) & (norms >= min_dn)
            lo = torch.minimum(norms, mean_n)
            hi = torch.maximum(norms, mean_n)
            # hi is >= min_dn on active_hist rows, so division is safe;
            # on other rows we mask the result out.
            sim = torch.where(
                active_hist, lo / hi.clamp_min(1e-38), torch.zeros_like(lo)
            )
            stable = active_hist & (sim > thr)
            scale = torch.where(stable, torch.zeros_like(scale), scale)

            per_attr_scale[attr] = scale

            # Per-gid min scale across attrs (parity with
            # ``gaussian_scales[gid] = min(gaussian_scales.get(gid,1.0), scale)``).
            if gid_min is None:
                gid_min = scale.clone()
            else:
                if gid_min.shape[0] < n:
                    # Pad older gid_min to current N (new gids get their
                    # per-attr scale as the first observation).
                    pad = torch.ones(
                        n - gid_min.shape[0],
                        dtype=gid_min.dtype, device=gid_min.device,
                    )
                    gid_min = torch.cat([gid_min, pad])
                elif gid_min.shape[0] > n:
                    # Current attr has fewer rows: leave the extra gid_min
                    # entries untouched (they keep min from earlier attrs).
                    gid_min[:n] = torch.minimum(gid_min[:n], scale)
                    continue
                gid_min = torch.minimum(gid_min, scale)

            # History update: frame-level accumulation vs. per-step push.
            # Frame-level sums are kept in float64 to match the dict path's
            # ``float(np.mean(list))`` reduction (NumPy accumulates in fp64);
            # dropping to fp32 here would shift a handful of scale decisions
            # at the suppression threshold and break cycle-count parity.
            if self.cfg.frame_level_ucn:
                self._frame_accum_add_batched(
                    attr, norms.detach(), start=0, n=n
                )
            else:
                self.cwb_batched.push_batched(attr, norms.detach())

        return per_attr_scale, gid_min

    def compute_scales_fallback_batched(
        self,
        attr: str,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised scale computation for a fallback (``ldec_*``) tensor.

        Produces per-row ``{0, 1}`` scales identical to calling
        ``compute_scale_single_via_batched(i, attr, norms[i], device)`` for
        every ``i`` — including pushing to the same batched frame
        accumulator so that ``flush_frame_accum_batched`` sees exactly the
        same history.  The scalar variant incurs O(N) GPU syncs through
        ``.item()`` on every descriptor; this batched variant runs the
        same logic on tensors and only returns a tensor (no host sync).

        Restricted to ``frame_level_ucn=True`` and
        ``adaptive_ucn_threshold=False`` (same as
        ``compute_scale_single_via_batched`` and the QUEEN fast path).
        """
        if not self.cfg.frame_level_ucn:
            raise NotImplementedError(
                "compute_scales_fallback_batched is only implemented "
                "for frame_level_ucn=True"
            )
        if self.cfg.adaptive_ucn_threshold:
            raise NotImplementedError(
                "compute_scales_fallback_batched does not support "
                "adaptive_ucn_threshold"
            )
        n = int(norms.shape[0])
        if n == 0:
            return norms.new_empty((0,))
        device = norms.device
        min_dn = float(self.cfg.min_delta_norm)
        thr    = float(self.cfg.merge_cos_threshold)

        # Read CURRENT CWB (history from prior frames only — not yet
        # accumulated for this step).  Matches the scalar version which
        # reads ``mean_norm_batched`` BEFORE calling ``_append_single``.
        mean_n, has_hist = self.cwb_batched.mean_norm_batched(
            attr, n, device
        )

        scale = torch.ones(n, dtype=norms.dtype, device=device)
        scale = torch.where(
            norms < min_dn, torch.zeros_like(scale), scale
        )
        dormant = has_hist & (mean_n < min_dn)
        scale = torch.where(dormant, torch.zeros_like(scale), scale)
        active_hist = has_hist & (mean_n >= min_dn) & (norms >= min_dn)
        lo = torch.minimum(norms, mean_n)
        hi = torch.maximum(norms, mean_n)
        sim = torch.where(
            active_hist, lo / hi.clamp_min(1e-38), torch.zeros_like(lo)
        )
        stable = active_hist & (sim > thr)
        scale = torch.where(stable, torch.zeros_like(scale), scale)

        # Frame accumulator push — identical row range as the scalar
        # version would visit (rows ``0..n-1``), so subsequent
        # ``flush_frame_accum_batched`` produces the same per-frame mean.
        self._frame_accum_add_batched(attr, norms.detach(), start=0, n=n)
        return scale

    def compute_scale_single_via_batched(
        self, gid: int, attr: str, delta_norm: float,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """Single-descriptor scale computation reading the BATCHED CWB.

        Used by GLASSPipeline's mode-1 fast path's fallback loop (ldec_*
        tiny tensors).  Keeps the fallback's CWB reads/writes on the same
        tensor store as the main per-Gaussian path — without this unifier
        the two paths would maintain independent histories and the slow
        path's ``gaussian_scales`` per-gid min (which is built over both
        main and fallback descriptors against a single dict CWB) would
        diverge from the fast path.

        Semantics: identical to ``_compute_one`` (binary gate, same three
        zero conditions), except the CWB is the tensor-backed one and the
        frame accumulator is the tensor-backed one.
        """
        if not self.cfg.frame_level_ucn:
            raise NotImplementedError(
                "compute_scale_single_via_batched is only implemented "
                "for frame_level_ucn=True"
            )
        if self.cfg.adaptive_ucn_threshold:
            raise NotImplementedError(
                "compute_scale_single_via_batched does not support "
                "adaptive_ucn_threshold"
            )
        if gid < 0:
            return 1.0

        if delta_norm < self.cfg.min_delta_norm:
            # Near-zero: hard-zero regardless of history, but still
            # accumulate for frame_accum so history reflects reality.
            self._append_single_batched(attr, gid, delta_norm, device)
            return 0.0

        mean, has = self.cwb_batched.mean_norm_batched(
            attr, gid + 1, device
        )
        has_here = bool(has[gid].item())
        mean_n = float(mean[gid].item())

        self._append_single_batched(attr, gid, delta_norm, device)

        if not has_here:
            return 1.0
        if mean_n < self.cfg.min_delta_norm:
            return 0.0
        thr = self.cfg.merge_cos_threshold
        lo, hi = min(delta_norm, mean_n), max(delta_norm, mean_n)
        sim = lo / hi if hi > 0 else 0.0
        return 0.0 if sim > thr else 1.0

    def _frame_accum_add_batched(
        self,
        attr: str,
        norms: torch.Tensor,
        start: int = 0,
        n: int | None = None,
    ) -> None:
        """Accumulate ``norms[:n]`` into ``_batched_frame_sum[attr][start:start+n]``.

        Also increments ``_batched_frame_count[attr][start:start+n]`` by 1
        (per-row, not scalar — mirrors the dict path which counts pushes
        per (gid, attr)).  Values are stored in float64 to match NumPy's
        ``np.mean`` accumulator precision.
        """
        if n is None:
            n = int(norms.shape[0])
        if n == 0:
            return
        device = norms.device
        end = start + n
        sum_prev = self._batched_frame_sum.get(attr)
        cnt_prev = self._batched_frame_count.get(attr)
        if sum_prev is None or sum_prev.shape[0] < end:
            new_sum = torch.zeros(end, dtype=torch.float64, device=device)
            new_cnt = torch.zeros(end, dtype=torch.int32, device=device)
            if sum_prev is not None:
                k = sum_prev.shape[0]
                new_sum[:k] = sum_prev.to(new_sum.dtype)
                if cnt_prev is not None:
                    new_cnt[:k] = cnt_prev
            sum_prev = new_sum
            cnt_prev = new_cnt
        sum_prev[start:end] = (
            sum_prev[start:end] + norms.to(sum_prev.dtype)
        )
        cnt_prev[start:end] = cnt_prev[start:end] + 1
        self._batched_frame_sum[attr] = sum_prev
        self._batched_frame_count[attr] = cnt_prev

    def _append_single_batched(
        self, attr: str, gid: int, val: float, device: torch.device
    ) -> None:
        """Frame-level accumulator: add ``val`` at row ``gid`` for ``attr``."""
        v = torch.tensor([float(val)], dtype=torch.float64, device=device)
        self._frame_accum_add_batched(attr, v, start=gid, n=1)

    def flush_frame_accum_batched(self) -> None:
        """Push per-frame mean norms (batched accumulator) into the batched CWB.

        Mirrors flush_frame_accum() for the tensor path.  Called by
        GLASSPipeline.begin_frame() when frame_level_ucn=True.  Each (attr,
        gid) gets exactly one CWB entry per frame — the arithmetic mean
        over all pushes for that (gid, attr) this frame.  The accumulator
        is then cleared.

        Rows with zero pushes (dormant this frame) are skipped by pushing
        nothing into the CWB, which keeps the dict-path semantics: those
        keys never appear in the dict accumulator either.
        """
        for attr, sum_t in self._batched_frame_sum.items():
            cnt = self._batched_frame_count.get(attr)
            if sum_t is None or cnt is None or sum_t.numel() == 0:
                continue
            pushed = cnt > 0
            if not bool(pushed.any().item()):
                continue
            denom = cnt.to(torch.float64).clamp_min(1.0)
            mean_t = (sum_t / denom).to(torch.float32)
            # Rows with zero pushes: set mean to 0 so they are not counted
            # (they also have no increment in _filled → has_history stays 0).
            mean_t = torch.where(
                pushed, mean_t, torch.zeros_like(mean_t)
            )
            self.cwb_batched.push_frame_batched(attr, mean_t, pushed)
        self._batched_frame_sum.clear()
        self._batched_frame_count.clear()

    def reset(self) -> None:
        """Clear all CWB history and frame accumulator. Called between ablation runs."""
        self.cwb.clear()
        self._frame_accum.clear()
        self.cwb_batched.clear()
        self._batched_frame_sum.clear()
        self._batched_frame_count.clear()

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    def suppression_stats(
        self, stream: "UpdateStream"
    ) -> dict[str, float]:
        """Return suppression statistics after compute_scales() has been called."""
        if not stream.descriptors:
            return {"suppressed_ratio": 0.0, "zero_ratio": 0.0,
                    "mean_scale": 1.0}
        scales = [d.grad_scale for d in stream.descriptors]
        n = len(scales)
        n_zero = sum(1 for s in scales if s == 0.0)
        n_suppressed = sum(1 for s in scales if s < 1.0)
        return {
            "suppressed_ratio": n_suppressed / n,
            "zero_ratio":       n_zero / n,
            "mean_scale":       float(np.mean(scales)),
        }
