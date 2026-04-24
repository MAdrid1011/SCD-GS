"""
ConvergenceGrader — Stage 4 of the GLASS ASIC pipeline (Convergence-Graded
Splatting, CGS).

Replaces the legacy ``SalienceFirstIssue`` (render-side opacity×radii
heuristic) with a training-driven three-way classifier that assigns each
Gaussian one of three execution grades:

  * **Grade-A (LivePrecise)**  — Full path (SH3 + fp32), rendered unchanged.
  * **Grade-B (StableCompact)** — Compact path (SH :math:`L=`
    ``cfg.compact_sh_degree``, typically SH1, plus fp16 state reads via
    ``cfg.compact_state_byte_scale=0.5``); the resulting Compact-to-Full
    per-primitive cycle ratio is derived on the fly by
    :meth:`CycleAccumulator.compact_primitive_weight` — no hardcoded
    "0.33" multiplier anywhere in the accounting chain.
  * **Grade-D (Dormant)**      — Skipped entirely (equivalent to the legacy
    SKIP classification).

Assignment rule
---------------
Let ``S_i^(W)`` be the window-mean of the squared parameter delta norm,
approximated here by the per-Gaussian gradient-norm stream already
captured by the pipeline (``Δθ ≈ lr × ∇θ`` for SGD/Adam at O(1)).  Let
``c_i`` be the contribution EMA (``opacity × radii / max_radii`` averaged
across renders).  After normalising ``S_i^(W)`` by its current max across
the live set, the rule is::

    Grade-A : S_hat_i > tau_wake  OR  c_i > tau_hot
    Grade-D : S_hat_i ≤ tau_stable  AND  c_i < tau_dormant
    Grade-B : otherwise

A ``grade_a_floor`` fraction of Gaussians (by ``max(S_hat_i, c_i)``) is
always forced to Grade-A as a quality safeguard — mirroring the legacy
``full_path_fraction`` knob.

State layout (vectorised)
-------------------------
The ingest / classify hot paths are called per training iter on models
with ~3 × 10⁵ Gaussians, so all three state containers are **contiguous
numpy arrays** — no Python-level ``dict``/``deque`` per Gaussian.  The
``S_i^(W)`` deque is replaced by a circular buffer ``S[N, W]`` plus a
per-Gaussian write count, ``c_i`` by a ``C[N]`` EMA array plus a
``seen[N]`` init flag so the first touch uses the raw signal (matching
the legacy ``_contrib_ema.get(i, raw)`` semantics).  Containers grow
in-place when a new frame brings more Gaussians than the current
capacity, so densification / pruning across frames costs one
:func:`numpy.concatenate` on growth, nothing per step.

Legacy ``SalienceFirstIssue`` is kept in :mod:`simulator.core.salience_issue`
as a thin shim forwarding to :class:`ConvergenceGrader` so existing
imports and tests continue to load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from simulator.config import ASICConfig


_MIN_INIT_CAPACITY = 1024


class ConvergenceGrader:
    """Stage 4: Convergence-Graded Splatting classifier (Grade-A/B/D).

    Stateful across frames (``S`` window + contribution EMA + structural
    invalidation flags persist).  Instantiated once per ablation run inside
    :class:`GLASSPipeline.__init__`.

    Public API consumed by :class:`GLASSPipeline`
    ---------------------------------------------
    record_update(gid, delta_norm)     — training-side stability signal
    record_updates_bulk(norms)         — bulk tensor ingest (fast path)
    update_contribution(radii, opacity) — post-render contribution EMA
    mark_structural(gids)              — invalidate history on spawn/split
    assign_grades(n) → (A, B, D) masks — per-primitive grade masks
    tier_counts(n) → dict              — {n_grade_a, n_grade_b, n_grade_d}
    classify(n)   → (keep, promote)    — back-compat shim for salience API
    reset()                            — clear all state between runs
    """

    def __init__(self, config: "ASICConfig") -> None:
        self.cfg = config

        W = max(1, int(config.grade_window))
        self._W = W

        # Circular buffer ``S[N, W]`` — column ``_s_count[i] % W`` is the
        # next write slot for Gaussian ``i``; ``_s_count[i]`` is the
        # total number of writes issued (capped at ``2*W`` to keep it in
        # int32 indefinitely while preserving the ``>= W`` readiness
        # criterion).
        self._s_buf:   np.ndarray = np.zeros((0, W), dtype=np.float32)
        self._s_count: np.ndarray = np.zeros(0,    dtype=np.int32)

        # Contribution EMA and first-touch flag (matches legacy
        # ``self._contrib_ema.get(i, raw)`` semantics).
        self._contrib_ema:  np.ndarray = np.zeros(0, dtype=np.float32)
        self._contrib_seen: np.ndarray = np.zeros(0, dtype=bool)

        # Gaussians flagged as structurally modified this frame.  Kept
        # as a ``bool[N]`` mask so the classifier never re-scans a
        # Python ``set`` for 10⁵ entries.
        self._struct_changed: np.ndarray = np.zeros(0, dtype=bool)

        self._capacity: int = 0

    # ── Capacity management ───────────────────────────────────────────────

    def _ensure_capacity(self, n_needed: int) -> None:
        """Grow all per-Gaussian buffers so indices ``[0, n_needed)`` are
        addressable.  Growth doubles up to a minimum floor to avoid a
        reallocation cascade on the first few frames."""
        if n_needed <= self._capacity:
            return
        new_cap = max(
            int(n_needed),
            int(self._capacity) * 2,
            _MIN_INIT_CAPACITY,
        )
        # ``np.resize`` fills with a wrap-around copy we don't want; use
        # explicit zero-fill allocations and copy the live prefix instead.
        new_s_buf = np.zeros((new_cap, self._W), dtype=np.float32)
        new_s_buf[: self._capacity] = self._s_buf
        self._s_buf = new_s_buf

        new_count = np.zeros(new_cap, dtype=np.int32)
        new_count[: self._capacity] = self._s_count
        self._s_count = new_count

        new_ema = np.zeros(new_cap, dtype=np.float32)
        new_ema[: self._capacity] = self._contrib_ema
        self._contrib_ema = new_ema

        new_seen = np.zeros(new_cap, dtype=bool)
        new_seen[: self._capacity] = self._contrib_seen
        self._contrib_seen = new_seen

        new_struct = np.zeros(new_cap, dtype=bool)
        new_struct[: self._capacity] = self._struct_changed
        self._struct_changed = new_struct

        self._capacity = new_cap

    # ── Training-side signal ingestion ────────────────────────────────────

    def record_update(self, gid: int, delta_norm: float) -> None:
        """Push one per-Gaussian gradient / delta-norm sample onto the
        S_i^(W) circular buffer.  Squared so the window mean is an
        energy (``||Δθ||²``), matching the paper's S_i^(W) definition.
        """
        if gid < 0 or delta_norm <= 0.0:
            return
        self._ensure_capacity(gid + 1)
        col = int(self._s_count[gid]) % self._W
        self._s_buf[gid, col] = float(delta_norm) ** 2
        # Cap count at 2*W: preserves the ``>= W`` readiness test forever
        # while keeping the int32 indexable without overflow.
        if self._s_count[gid] < 2 * self._W:
            self._s_count[gid] += 1

    def record_updates_bulk(self, norms: "torch.Tensor | np.ndarray") -> None:
        """Bulk variant for the tensorised pipeline fast paths.

        Accepts a 1-D tensor / ndarray of per-Gaussian delta-norm proxies
        indexed by local gid.  Only strictly positive entries advance the
        window — zero entries are treated as "no update this step" and
        leave the row intact (matching the legacy per-descriptor loop's
        ``delta_norm > 0`` guard).

        Fully vectorised: one host sync at the top (if a CUDA tensor is
        passed) followed by a constant number of numpy ops — no Python
        for-loop over Gaussians on the critical path.  On a 280 K model
        this replaces ~280 K Python calls per training iter with three
        numpy array operations.
        """
        if isinstance(norms, torch.Tensor):
            arr = norms.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            arr = np.asarray(norms, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return

        n = int(arr.shape[0])
        self._ensure_capacity(n)

        mask = arr > 0.0
        if not mask.any():
            return

        idxs = np.flatnonzero(mask)
        # Per-gid next-write column is ``count % W``.  Vectorised writes
        # through advanced indexing — single numpy op.
        cols = self._s_count[idxs] % self._W
        self._s_buf[idxs, cols] = arr[idxs] ** 2
        # Increment counts (clamped so int32 never overflows while still
        # exceeding the readiness threshold W).
        new_counts = self._s_count[idxs] + 1
        cap = np.int32(2 * self._W)
        np.minimum(new_counts, cap, out=new_counts)
        self._s_count[idxs] = new_counts

    # ── Render-side contribution EMA ──────────────────────────────────────

    def update_contribution(
        self, radii: torch.Tensor, opacity: torch.Tensor
    ) -> None:
        """Update per-Gaussian contribution EMA from a render_pkg.

        Vectorised: the pointwise EMA and first-touch initialisation are
        both single numpy ops — no per-Gaussian Python loop.

        radii   : FloatTensor [N] — projected screen-space radius (px).
        opacity : FloatTensor [N] or [N, 1] — activated opacity.
        """
        op = opacity.squeeze(-1).float()
        r = radii.float()
        max_r = float(r.max().item())
        if max_r < 1e-8:
            return  # All offscreen — leave EMA state untouched.

        raw = (op * r / max_r).detach().cpu().numpy().astype(np.float32)
        n = int(raw.shape[0])
        self._ensure_capacity(n)

        alpha = float(self.cfg.contrib_ema_alpha)
        ema_view  = self._contrib_ema[:n]
        seen_view = self._contrib_seen[:n]

        # First-touch: initialise directly to ``raw`` (matches the legacy
        # ``_contrib_ema.get(i, float(s))`` semantics — no ``1-alpha``
        # damping on the first observation).  Subsequent touches follow
        # the standard EMA update.  Expressed as a single ``np.where``
        # to avoid scattered writes.
        updated = alpha * ema_view + (1.0 - alpha) * raw
        self._contrib_ema[:n] = np.where(seen_view, updated, raw)
        self._contrib_seen[:n] = True

        # Gaussians that have now been rendered after a structural event
        # can re-enter the normal classification path.
        if self._capacity > 0:
            self._struct_changed.fill(False)

    # ── Structural invalidation ───────────────────────────────────────────

    def mark_structural(self, gids) -> None:
        """Flag Gaussians whose geometry changed this frame.

        Flagged entries are forced to Grade-A by :meth:`assign_grades`
        until the next :meth:`update_contribution` call re-renders them.

        Accepts any iterable of non-negative ints (list, tuple, ndarray,
        tensor).  Vectorised: a single boolean-mask write.
        """
        if gids is None:
            return
        if isinstance(gids, torch.Tensor):
            arr = gids.detach().to("cpu", dtype=torch.int64).numpy()
        else:
            arr = np.asarray(gids, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return
        self._ensure_capacity(int(arr.max()) + 1)
        # Filter out negatives cheaply.
        arr = arr[arr >= 0]
        self._struct_changed[arr] = True

    # ── Primary classifier ────────────────────────────────────────────────

    def assign_grades(
        self, n_gaussians: int
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """Return ``(grade_a_mask, grade_b_mask, grade_d_mask)``.

        Each is a ``BoolTensor`` of length ``n_gaussians`` on CPU.  The
        three masks are mutually exclusive and partition the Gaussian set.

        Distribution-adaptive normalisation
        -----------------------------------
        Both the training-side S_i^(W) and the render-side contribution
        EMA are normalised via **percentile rank** within the live
        Gaussian set, not by division by the per-frame max.  Rationale:
        on NTC-based baselines (3DGStream, 4DGC) the NTC predicts
        near-uniform per-Gaussian delta magnitudes every frame, so
        ``S_i^(W) / max(S_i^(W))`` compresses the entire population
        into a single narrow band (empirically p25 = p75 ≈ 0.88 on
        flame_steak) and every absolute threshold either catches all
        280 K Gaussians or none of them.  Percentile rank preserves the
        ordering — which is the actual physical signal (who is more /
        less active than whom) — and guarantees that a threshold of
        ``tau_wake = 0.85`` always selects the top 15 % regardless of
        whether the signal has a heavy tail (QUEEN direct training)
        or is nearly uniform (3DGStream NTC).  Thresholds therefore
        keep the same [0, 1] range and paper-level semantics, but they
        are now quantile cut-offs, which is what makes CGS effective
        on every baseline (the explicit user requirement).

        A Gaussian whose ``S`` window has fewer than ``grade_window``
        samples is forced to Grade-A — the grader does not have enough
        history to downgrade it yet.  This matches the legacy behaviour
        where ``variance = inf`` blocked SKIP classification.
        """
        if n_gaussians <= 0:
            empty = torch.zeros(0, dtype=torch.bool)
            return empty, empty.clone(), empty.clone()

        self._ensure_capacity(n_gaussians)
        W = self._W
        n = int(n_gaussians)

        # ── Assemble S_i^(W) as window-mean (energy) ─────────────────────
        # The circular buffer is zero-padded for unwritten slots, so
        # ``sum(axis=1) / min(count, W)`` is exactly the mean of valid
        # entries — no Python loop required.
        count = self._s_count[:n]
        row_sum = self._s_buf[:n].sum(axis=1)
        divisor = np.maximum(np.minimum(count, W), 1).astype(np.float32)
        s_vals = (row_sum / divisor).astype(np.float32)
        c_vals = self._contrib_ema[:n].astype(np.float32).copy()
        seen  = self._contrib_seen[:n]

        # Readiness for non-Grade-A classification.  Two disjoint cases:
        #
        #   (a) ``count >= W``                 — legitimate full training window.
        #   (b) ``count == 0 AND contrib_seen`` — the Gaussian has never been
        #       touched by an optimiser step (i.e. it is intentionally not
        #       being retrained this frame, as in HiCoM's base-Gaussian set
        #       or 4DGC's stable primitives) BUT the renderer has seen it.
        #       Semantically this is exactly what CGS calls "stable": no
        #       evolving ``Δθ`` signal AND a well-defined contribution
        #       history.  Forcing it to Grade-A would make CGS a no-op on
        #       any streaming baseline that freezes its backbone after
        #       frame 0 (the empirical failure mode before this fix: HiCoM
        #       m3 assigned 100 % Grade-A despite a 200 K-primitive set).
        #
        # Gaussians in neither bucket (spawned-but-not-yet-rendered, or
        # partially-filled window ``0 < count < W``) still default to
        # Grade-A — we genuinely do not have enough history to downgrade
        # them yet.
        never_trained_but_seen = (count == 0) & seen
        s_ready = (count >= W) | never_trained_but_seen

        # ── Percentile-rank normalisation ────────────────────────────────
        # rank ∈ [0, 1], with 0 = smallest and 1 = largest.  Stable sort
        # for reproducibility across identical runs.  Single argsort-of-
        # argsort per signal — O(n log n), fully vectorised.
        if n > 1:
            order_s = np.argsort(s_vals, kind="stable")
            rank_s = np.empty(n, dtype=np.float32)
            rank_s[order_s] = np.arange(n, dtype=np.float32) / float(n - 1)

            order_c = np.argsort(c_vals, kind="stable")
            rank_c = np.empty(n, dtype=np.float32)
            rank_c[order_c] = np.arange(n, dtype=np.float32) / float(n - 1)
        else:
            rank_s = np.zeros(n, dtype=np.float32)
            rank_c = np.zeros(n, dtype=np.float32)

        s_hat = torch.from_numpy(rank_s)
        c_hat = torch.from_numpy(rank_c)
        ready_t = torch.from_numpy(s_ready.copy())
        struct_t = torch.from_numpy(self._struct_changed[:n].copy())

        # ── Apply the assignment rule ───────────────────────────────────
        # Thresholds are quantile cut-offs in [0, 1]:
        #   tau_wake    — top (1 - tau_wake)    fraction by S-rank → Grade-A
        #   tau_hot     — top (1 - tau_hot)     fraction by C-rank → Grade-A
        #   tau_stable  — bottom tau_stable     fraction by S-rank → Grade-D cand.
        #   tau_dormant — bottom tau_dormant    fraction by C-rank → Grade-D cand.
        tau_stable  = float(self.cfg.tau_stable)
        tau_wake    = float(self.cfg.tau_wake)
        tau_hot     = float(self.cfg.tau_hot)
        tau_dormant = float(self.cfg.tau_dormant)

        # Grade-A: actively evolving OR hot visual contribution, OR
        # forced-A by structural invalidation, OR window not ready yet.
        grade_a = (
            (s_hat > tau_wake)
            | (c_hat > tau_hot)
            | struct_t
            | ~ready_t
        )

        # Grade-D: converged AND invisible — only among the non-A set.
        grade_d = (
            (s_hat <= tau_stable)
            & (c_hat <= tau_dormant)
            & ~grade_a
        )

        # ── Quality safeguard: top ``grade_a_floor`` by max(s_hat, c_hat) ─
        # Guarantees a minimum fraction of Grade-A even on scenes where
        # absolute thresholds happen to under-select (e.g. everybody
        # converges simultaneously).  Both signals are already in [0, 1]
        # rank space, so ``max`` is a well-defined priority.
        floor = float(self.cfg.grade_a_floor)
        if 0.0 < floor < 1.0:
            k = max(1, int(n * floor))
            if k < n:
                priority = torch.maximum(s_hat, c_hat)
                top_vals = priority.topk(k).values
                top_thresh = float(top_vals[-1].item())
                force_a = priority >= top_thresh
                grade_a = grade_a | force_a
                grade_d = grade_d & ~force_a

        grade_b = ~(grade_a | grade_d)

        return grade_a, grade_b, grade_d

    # ── Backwards-compat shim (old SalienceFirstIssue.classify API) ───────

    def classify(
        self, n_gaussians: int
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Emulate the legacy ``(keep_mask, promote_mask)`` tuple.

        ``keep_mask`` = Grade-A ∪ Grade-B (everything except Dormant).
        ``promote_mask`` = Grade-A (the former PROMOTE set).

        Kept so that helpers still calling ``salience.classify`` in legacy
        code paths (e.g. ``_train_render_active_counts``) keep working
        without a second code path.
        """
        A, B, _D = self.assign_grades(n_gaussians)
        keep = A | B
        return keep, A

    # ── Statistics ────────────────────────────────────────────────────────

    def tier_counts(self, n_gaussians: int) -> dict[str, int]:
        """Return Gaussian counts for each grade.

        Keys: ``n_grade_a``, ``n_grade_b``, ``n_grade_d``.  Legacy aliases
        ``n_promote`` / ``n_full`` / ``n_skip`` are included so
        ``stream_stats`` dict consumers keep working unchanged.
        """
        A, B, D = self.assign_grades(n_gaussians)
        n_a = int(A.sum().item())
        n_b = int(B.sum().item())
        n_d = int(D.sum().item())
        return {
            "n_grade_a": n_a,
            "n_grade_b": n_b,
            "n_grade_d": n_d,
            # Legacy aliases — consumed by older ablation result JSON
            # schemas and cross-baseline plots.  n_promote == Grade-A,
            # n_full == Grade-B, n_skip == Grade-D.
            "n_promote": n_a,
            "n_full":    n_b,
            "n_skip":    n_d,
        }

    def reset(self) -> None:
        """Clear all state. Called between ablation runs."""
        self._s_buf        = np.zeros((0, self._W), dtype=np.float32)
        self._s_count      = np.zeros(0, dtype=np.int32)
        self._contrib_ema  = np.zeros(0, dtype=np.float32)
        self._contrib_seen = np.zeros(0, dtype=bool)
        self._struct_changed = np.zeros(0, dtype=bool)
        self._capacity = 0


# ── Backwards-compat aliases ─────────────────────────────────────────────
# ``SalienceFirstIssue`` is re-exported from ``salience_issue.py`` via a
# thin subclass so ``isinstance`` checks in old code keep working.
from simulator.core.support_domain_grade import ConvergenceGrader as ConvergenceGrader
