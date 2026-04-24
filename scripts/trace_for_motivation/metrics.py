"""Pure-function implementations of the 11 GLASS.md motivation metrics.

Each function takes :class:`~scripts.trace_for_motivation.schema.FrameRecord`
instances (plus, for time-axis metrics, a small history buffer) and returns a
scalar / small dict.  All functions are **defensive**: if a baseline could not
populate the required field they return ``None`` rather than raising, and the
caller appends the metric name to ``PerFrameMetrics.degraded``.

Correctness philosophy
----------------------
These are the implementations the paper-figure code will ultimately consume,
so they should be arithmetically right on small records — but they are not
optimized.  The expensive baselines (QUEEN at 2 M gaussians) should still be
fine because every metric is either O(n_gaussians) or O(history_window).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .schema import FrameRecord


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _attr_concat(attrs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate per-attribute (n,d) arrays into a single (n, D) matrix.

    Returns ``None`` if the dict is empty or arrays disagree on row count.
    """
    if not attrs:
        return None
    rows = {a.shape[0] for a in attrs.values()}
    if len(rows) != 1:
        return None
    parts = []
    for name in sorted(attrs):
        a = attrs[name]
        if a.ndim == 1:
            a = a[:, None]
        parts.append(a.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1)


def _row_bytes(attrs: Dict[str, np.ndarray]) -> int:
    """Total bytes per gaussian across all attributes (used for traffic math)."""
    if not attrs:
        return 0
    total = 0
    for a in attrs.values():
        if a.ndim == 1:
            total += a.dtype.itemsize
        else:
            total += int(a.shape[1]) * a.dtype.itemsize
    return total


def _align_ids(prev: FrameRecord, curr: FrameRecord) -> int:
    """Min n_gaussians the two frames share (handles growth via densify)."""
    return min(prev.n_gaussians, curr.n_gaussians)


# ---------------------------------------------------------------------------
# Challenge 1 — update-stream redundancy
# ---------------------------------------------------------------------------


def duplicated_inter_frame_bytes(
    prev: Optional[FrameRecord],
    curr: FrameRecord,
    sim_threshold: float = 0.95,
) -> Optional[float]:
    """Bytes of update objects in frame t whose payload is ~= the same id's
    update object in frame t-1 (cosine similarity >= sim_threshold)."""
    if prev is None:
        return 0.0  # first frame has nothing to duplicate
    a_prev = _attr_concat(prev.attr_after)
    b_prev = _attr_concat(prev.attr_before)
    a_curr = _attr_concat(curr.attr_after)
    b_curr = _attr_concat(curr.attr_before)
    if any(x is None for x in (a_prev, b_prev, a_curr, b_curr)):
        return None

    # Must intersect across *all four* arrays: a baseline that prunes between
    # snapshots can leave ``attr_before`` bigger than ``attr_after`` (QUEEN).
    n = min(
        _align_ids(prev, curr),
        a_prev.shape[0], b_prev.shape[0],
        a_curr.shape[0], b_curr.shape[0],
    )
    # Some baselines (3DGStream's loaded SH tail) leave NaN entries in
    # unused attribute slots; treat missing/NaN values as zero so we don't
    # accidentally mark those ids as "updated".
    delta_prev = np.nan_to_num(a_prev[:n] - b_prev[:n], nan=0.0, posinf=0.0, neginf=0.0)
    delta_curr = np.nan_to_num(a_curr[:n] - b_curr[:n], nan=0.0, posinf=0.0, neginf=0.0)

    # Only consider ids that actually had a nonzero update this frame
    curr_active = np.linalg.norm(delta_curr, axis=1) > 0.0
    prev_active = np.linalg.norm(delta_prev, axis=1) > 0.0
    both = curr_active & prev_active
    if not np.any(both):
        return 0.0

    dp = np.einsum("nd,nd->n", delta_prev[both], delta_curr[both])
    np_curr = np.linalg.norm(delta_curr[both], axis=1)
    np_prev = np.linalg.norm(delta_prev[both], axis=1)
    denom = np.maximum(np_curr * np_prev, 1e-12)
    cos = dp / denom
    dup_mask = cos >= sim_threshold

    row_bytes = _row_bytes(curr.attr_after)
    return float(dup_mask.sum() * row_bytes)


def duplicated_inter_stage_bytes(curr: FrameRecord) -> Optional[float]:
    """Bytes passed from update-gen to commit that are subset-equivalent to
    the update-gen output.  Measured as: bytes_commit that mention an id
    already in update_gen_ids with the same payload footprint."""
    if curr.update_gen_ids is None or curr.commit_ids is None:
        # Fall back to: baselines without a clean stage split have
        # bytes_commit >= bytes_update_gen almost tautologically, so report
        # the overlap of those two counts.
        if curr.bytes_commit > 0 and curr.bytes_update_gen > 0:
            return float(min(curr.bytes_commit, curr.bytes_update_gen))
        return None
    overlap_ids = np.logical_and(curr.update_gen_ids, curr.commit_ids)
    n_overlap = int(overlap_ids.sum())
    if curr.update_gen_ids.sum() == 0:
        return 0.0
    # Payload bytes / id is bytes_update_gen / |update_gen_ids|.
    per_id = curr.bytes_update_gen / max(int(curr.update_gen_ids.sum()), 1)
    return float(n_overlap * per_id)


def iter_grad_cosine_stable_fraction(curr: FrameRecord) -> Optional[float]:
    """Frame-mean of ``n_stable_cos_ge_0.9 / n_top_both_iters``.

    Drivers fill ``iter_grad_cos_counts`` per iteration from
    :class:`StateDeltaCache`.  The numerator is rows with
    ``cos(Δp_t, Δp_{t-1}) ≥ 0.9``; the denominator is rows that lie in
    the *top* (non-tail) Δ-energy subset both this iter and last iter.
    Restricting to the top subset removes tail rows whose direction is
    numerically meaningless, and directly answers "of the Gaussians
    whose update energy actually matters, what fraction keeps the same
    direction iter-to-iter".
    """
    counts = curr.iter_grad_cos_counts
    if not counts:
        return None
    fracs = []
    for stable, active in counts:
        if active <= 0:
            continue
        fracs.append(stable / active)
    if not fracs:
        return None
    return float(np.mean(fracs))


def low_grad_fraction(curr: FrameRecord) -> Optional[float]:
    """Frame-mean of ``n_tail / n_total`` across iterations.

    "Tail" = the smallest set of Gaussians that together hold only
    ``TAIL_EPS`` (default 1%) of the iter's total Δ L2² energy — i.e.,
    Gaussians whose update, if skipped, would cost <1% of the aggregate
    update signal.  This is the universal "skippable" fraction: for
    HiCoM the tail is dominated by exact-zero rows, while for dense
    NTC / Fourier baselines the tail is the low-energy bulk left over
    after scene-motion concentration on a few loud Gaussians.  Iterations
    are averaged uniformly.
    """
    counts = curr.iter_low_grad_counts
    if not counts:
        return None
    fracs = []
    for low, active in counts:
        if active <= 0:
            continue
        fracs.append(low / active)
    if not fracs:
        return None
    return float(np.mean(fracs))


def stage_time_breakdown(curr: FrameRecord) -> Optional[Dict[str, float]]:
    """Three-bucket wall-clock breakdown: render / backward / optimizer.

    Drivers record one ``{"render_ms", "backward_ms", "optimizer_ms"}`` dict
    per iter via CUDA events.  Returns frame-mean per bucket plus the
    render+backward / total ratio that the narrative needs directly.
    """
    times = curr.iter_stage_times_ms
    if not times:
        return None
    r_ms = [float(t.get("render_ms", 0.0)) for t in times if t]
    b_ms = [float(t.get("backward_ms", 0.0)) for t in times if t]
    o_ms = [float(t.get("optimizer_ms", 0.0)) for t in times if t]
    # Drop iters where all three are zero (stage skipped for that iter).
    keep = [(r, b, o) for r, b, o in zip(r_ms, b_ms, o_ms) if (r + b + o) > 0.0]
    if not keep:
        return None
    r_mean = float(np.mean([r for r, _, _ in keep]))
    b_mean = float(np.mean([b for _, b, _ in keep]))
    o_mean = float(np.mean([o for _, _, o in keep]))
    total = r_mean + b_mean + o_mean
    rb_frac = (r_mean + b_mean) / total if total > 0.0 else 0.0
    return {
        "render_ms": r_mean,
        "backward_ms": b_mean,
        "optimizer_ms": o_mean,
        "render_backward_fraction": float(rb_frac),
    }


def stable_render_backward_compute_fraction(curr: FrameRecord) -> Optional[float]:
    """Fraction of render+backward compute time spent on redundant Gaussians.

    Combines the two per-iter signals populated by :class:`StateDeltaCache`
    into a single "compute-wasted-on-stable" share:

    * ``tail_frac``  — Gaussians in the bottom-ε-energy tail; their Δ is
      effectively zero this iter so the R+B cycles they consume are pure
      waste.
    * ``cos_stable_frac`` — fraction of the remaining top-energy rows
      that still keep the same Δ direction as last iter; their R+B
      cycles produce an update that is a small extrapolation of the
      previous iter's, so they are also redundant.

    Combined per iter as
    ``skip = tail_frac + (1 − tail_frac) × cos_stable_frac``, averaged
    across iters, and scaled by the iter's render+backward time
    fraction — a multiplicative estimate of the share of total
    iteration time spent on Gaussians that could be consolidated or
    skipped without meaningfully altering the update.
    """
    cos_counts = curr.iter_grad_cos_counts
    low_counts = curr.iter_low_grad_counts
    if not cos_counts or not low_counts:
        return None
    # Align iter counts (both lists are appended in lockstep by
    # StateDeltaCache, so their lengths should match — but skip any iter
    # that is degenerate in either channel).
    fracs: List[float] = []
    for (n_stable, n_top), (n_tail, n_total) in zip(cos_counts, low_counts):
        if n_total <= 0:
            continue
        tail = n_tail / n_total
        cos = (n_stable / n_top) if n_top > 0 else 0.0
        fracs.append(tail + (1.0 - tail) * cos)
    if not fracs:
        return None
    skip = float(np.mean(fracs))
    stages = stage_time_breakdown(curr)
    if stages is None:
        return None
    rb = stages.get("render_backward_fraction")
    if rb is None:
        return None
    return float(skip * rb)


def propagation_amplification(
    prev: Optional[FrameRecord],
    curr: FrameRecord,
) -> Optional[Dict[str, float]]:
    """Compound propagation amplification = downstream byte-cycle triggered
    per unit of semantically-independent update byte-cycle.

    Decomposed:
      - ``frame_factor``   : duplicated_inter_frame_bytes / unique_update_bytes
      - ``stage_factor``   : (commit + state_write + render_expand_bytes)
                             / bytes_update_gen
      - ``total``          : frame_factor * stage_factor
                             (multiplicative, not additive: a duplicated
                              update object pays the stage-amplification
                              cost every time it gets re-generated)
    """
    dup_frame = duplicated_inter_frame_bytes(prev, curr)
    if curr.bytes_update_gen <= 0:
        return None
    unique_update_bytes = max(
        curr.bytes_update_gen - (dup_frame or 0.0),
        1.0,  # avoid div-by-zero when everything is dup
    )
    frame_factor = 1.0 + (dup_frame or 0.0) / unique_update_bytes

    render_expand_bytes = 0.0
    if curr.tile_touches_per_id is not None and curr.attr_after:
        row_bytes = _row_bytes(curr.attr_after)
        render_expand_bytes = float(curr.tile_touches_per_id.sum() * row_bytes)
    stage_num = curr.bytes_commit + curr.bytes_state_write + render_expand_bytes
    if stage_num <= 0:
        return None
    stage_factor = stage_num / curr.bytes_update_gen

    return {
        "frame_factor": float(frame_factor),
        "stage_factor": float(stage_factor),
        "total": float(frame_factor * stage_factor),
    }


# ---------------------------------------------------------------------------
# Challenge 2 — state-movement redundancy
# ---------------------------------------------------------------------------


def state_movement_bytes(curr: FrameRecord) -> Optional[float]:
    if curr.bytes_state_read == 0 and curr.bytes_state_write == 0:
        return None
    return float(curr.bytes_state_read + curr.bytes_state_write)


def changed_state_fraction(
    prev: Optional[FrameRecord],
    curr: FrameRecord,
    eps: float = 0.0,
) -> Optional[float]:
    if prev is None:
        return None
    a_prev = _attr_concat(prev.attr_after)
    a_curr = _attr_concat(curr.attr_after)
    if a_prev is None or a_curr is None:
        return None
    n = min(a_prev.shape[0], a_curr.shape[0])
    delta = np.nan_to_num(a_curr[:n] - a_prev[:n], nan=0.0, posinf=0.0, neginf=0.0)
    changed = np.any(np.abs(delta) > eps, axis=1)
    return float(changed.sum() / max(curr.n_gaussians, 1))


def hot_readmask_overlap(curr: FrameRecord, topk: float = 0.20) -> Optional[float]:
    """Fraction of render reads covered by top-``topk`` hottest gaussians.

    Hotness proxy = tile_touches_per_id (render visit count).  We report the
    coverage ratio sum(top_k tile_touches) / sum(tile_touches) — the closer
    to 1 the more obviously hot-skewed the workload is.
    """
    if curr.tile_touches_per_id is None:
        return None
    tt = curr.tile_touches_per_id
    total = float(tt.sum())
    if total <= 0.0:
        return None
    k = max(int(tt.size * topk), 1)
    top_vals = np.partition(tt, -k)[-k:]
    return float(top_vals.sum() / total)


def _jaccard(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a is None or b is None:
        return None
    n = min(a.size, b.size)
    a = a[:n].astype(bool)
    b = b[:n].astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def update_render_jaccard(curr: FrameRecord) -> Optional[float]:
    return _jaccard(curr.update_ids, curr.render_ids)


def redundant_shared_state_ratio(curr: FrameRecord) -> Optional[float]:
    """Fraction of per-iter state-movement bytes that are redundant
    round-trips of ids touched by both update and render in the same frame.

    Model: if update and render enumerated their compute sets independently
    (as current baselines do), every id in ``update_ids ∩ render_ids`` is
    pulled off HBM twice per iter — once by update, once by render — where
    a shared-path machine would pay ``|U ∪ R|`` row-fetches instead of
    ``|U| + |R|``.  The ratio is iter-invariant:

        (|U| + |R|) × row_bytes × n_iter  − |U ∪ R| × row_bytes × n_iter
        ───────────────────────────────────────────────────────────────
                 (|U| + |R|) × row_bytes × n_iter

    which simplifies to ``|U ∩ R| / (|U| + |R|)``.  The byte counters
    therefore cancel, so this metric needs only the frame-aggregate update
    and render id masks.
    """
    if curr.update_ids is None or curr.render_ids is None:
        return None
    n = min(curr.update_ids.size, curr.render_ids.size)
    u = curr.update_ids[:n].astype(bool)
    r = curr.render_ids[:n].astype(bool)
    u_total = int(u.sum())
    r_total = int(r.sum())
    denom = u_total + r_total
    if denom <= 0:
        return None
    shared = int((u & r).sum())
    return float(shared / denom)


def compute_mask_overlap(curr: FrameRecord) -> Optional[float]:
    """Cost-weighted version of ``update_render_jaccard``.

    Answers: *of the work the machine actually spent (tile-touches on render
    side + update-cost on update side), how much was on ids that sit in the
    intersection of both masks?*
    """
    if (
        curr.update_ids is None
        or curr.render_ids is None
        or curr.tile_touches_per_id is None
    ):
        return None
    n = min(
        curr.update_ids.size,
        curr.render_ids.size,
        curr.tile_touches_per_id.size,
    )
    u = curr.update_ids[:n].astype(bool)
    r = curr.render_ids[:n].astype(bool)
    tt = curr.tile_touches_per_id[:n].astype(np.float64)
    uc = (
        curr.update_cost_per_id[:n].astype(np.float64)
        if curr.update_cost_per_id is not None
        else u.astype(np.float64)  # unit cost per updated id
    )
    total_cost = tt.sum() + uc.sum()
    if total_cost <= 0.0:
        return None
    inter = u & r
    inter_cost = tt[inter].sum() + uc[inter].sum()
    return float(inter_cost / total_cost)


# ---------------------------------------------------------------------------
# Challenge 3 — late-discarded low-value work
# ---------------------------------------------------------------------------


def contribution_CDF(curr: FrameRecord, bins: int = 50) -> Optional[List[float]]:
    if curr.contribution_per_id is None or curr.contribution_per_id.size == 0:
        return None
    c = np.sort(curr.contribution_per_id.astype(np.float64))
    total = c.sum()
    if total <= 0.0:
        return [0.0] * bins
    cdf = np.cumsum(c) / total
    xs = np.linspace(0.0, 1.0, bins)
    idx = np.clip((xs * (c.size - 1)).astype(int), 0, c.size - 1)
    return cdf[idx].tolist()


def long_term_low_contribution(
    history: Sequence[FrameRecord],
    window: int = 30,
    low_pct: float = 5.0,
) -> Optional[float]:
    """Fraction of gaussians whose contribution stayed below the p5 threshold
    for at least ``window`` consecutive frames in the provided history."""
    contribs = [
        r.contribution_per_id for r in history if r.contribution_per_id is not None
    ]
    if len(contribs) < window:
        return None
    n = min(c.size for c in contribs)
    stack = np.stack([c[:n] for c in contribs[-window:]], axis=0)  # (W, n)
    threshold = np.percentile(stack, low_pct)
    low = stack < threshold
    always_low = np.all(low, axis=0)
    return float(always_low.sum() / max(n, 1))


# ---------------------------------------------------------------------------
# Challenge 3 — five-frame stability bucket (GLASS.md L140–L142 C3 table)
# ---------------------------------------------------------------------------


# Phase-3 alpha-composite state footprint in bytes per tile-touch:
# 9 float32 (μ.xy (2) + ΔΣ (3) + color (3) + opacity (1)) × 4 B = 36 B.
# We use this to weight the Phase-3 (tile-touch dominated) side of the
# stable-bandwidth share against the Phase-1 (state-read) side.
_PHASE3_BYTES_PER_TILE_TOUCH = 9 * 4
# ``tile_touches_per_id`` is recorded by the baseline hooks as a pixel-area
# proxy (``radii²``) rather than as an actual tile count, because stock
# diff-gaussian-rasterization does not expose per-id tile-range histograms.
# 3DGS rasterisers use 16×16-pixel tiles, so one tile-touch ≈ 256 pixels;
# to convert the pixel-area proxy into a tile-touch count the Phase-3 term
# must divide by ``_PIXELS_PER_TILE``.  Without this conversion the Phase-3
# DRAM is over-counted by 256× and swamps the preprocess side of every
# render-DRAM ratio we emit.
_PIXELS_PER_TILE = 16 * 16
_SH_COEFF_FP32_BYTES = 4


def _stable_mask(curr: FrameRecord) -> Optional[np.ndarray]:
    m = curr.extras.get("stable_mask_w5") if curr.extras else None
    if m is None:
        return None
    return np.asarray(m, dtype=bool)


def _render_calls(curr: FrameRecord) -> int:
    """Number of ``render()`` invocations captured this frame.

    ``bytes_state_read`` accumulates one ``row_bytes × n`` block per
    render call, while ``tile_touches_per_id`` / ``render_ids`` are
    snapshotted from the *latest* render only.  Metrics that mix these
    two quantities must normalise both sides to the same per-call scale
    or the counts-with-calls side dominates by the render-call count
    (≈250 for 3DGStream Stage-1 training).  The profiler wrapper in
    hooks.py writes the call count to ``render_kernel_n_calls``; when
    the profiler is disabled we fall back to ``1`` (the metric will
    still be self-consistent as long as only one render ran).
    """
    if not curr.extras:
        return 1
    n = curr.extras.get("render_kernel_n_calls")
    try:
        n_int = int(n) if n is not None else 0
    except Exception:
        n_int = 0
    return max(1, n_int)


def stable_fraction_w5(curr: FrameRecord) -> Optional[float]:
    """``|stable| / (n_gaussians − windowless_count)`` (XX%).

    Windowless ids (Gaussians that joined inside the 5-frame window and
    therefore could not be scored) are excluded from the denominator.
    Returns ``None`` until the window first fills.
    """
    mask = _stable_mask(curr)
    if mask is None:
        return None
    windowless = curr.extras.get("stable_windowless_count_w5", 0) if curr.extras else 0
    denom = int(mask.size) - int(windowless or 0)
    if denom <= 0:
        return None
    return float(int(mask.sum()) / denom)


def stable_bandwidth_share_w5(curr: FrameRecord) -> Optional[float]:
    """YY% — off-chip bandwidth share spent on stable primitives.

    Combines two proxies already populated by the GPU trace:

    * **Phase-1 (preprocess)** traffic is dominated by ``bytes_state_read``
      per rendered id; the stable share is weighted by the count ratio
      ``|stable ∩ render| / |render|``.
    * **Phase-3 (α-composite)** traffic scales with tile-touches per id
      (``9 fp32 × 4 B`` per tile-touch); the stable share is weighted by
      ``Σ_{g∈stable∩render} tt_g / Σ_{g∈render} tt_g``.

    The two shares are combined by their absolute byte weights
    (``B_p1 = bytes_state_read`` and ``B_p3 = Σ tt_g × 36 B``), giving a
    single "fraction of render-side DRAM bytes spent on stable primitives"
    number.
    """
    mask = _stable_mask(curr)
    if mask is None or curr.render_ids is None:
        return None
    render = np.asarray(curr.render_ids, dtype=bool)
    n = min(mask.size, render.size)
    stable = mask[:n] & render[:n]
    n_render = int(render[:n].sum())
    if n_render > 0:
        p1_ratio = float(int(stable.sum()) / n_render)
    else:
        p1_ratio = 0.0
    n_calls = _render_calls(curr)
    b_p1 = float(curr.bytes_state_read) / n_calls
    tt = curr.tile_touches_per_id
    if tt is not None:
        m = min(n, tt.size)
        tt64 = tt[:m].astype(np.float64)
        stable_m = mask[:m] & render[:m]
        render_m = render[:m]
        total_tt = float(tt64[render_m].sum())
        if total_tt > 0.0:
            p3_ratio = float(tt64[stable_m].sum() / total_tt)
            b_p3 = (total_tt / _PIXELS_PER_TILE) * _PHASE3_BYTES_PER_TILE_TOUCH
        else:
            p3_ratio = 0.0
            b_p3 = 0.0
    else:
        p3_ratio = 0.0
        b_p3 = 0.0
    denom = b_p1 + b_p3
    if denom <= 0.0:
        return None
    return float((p1_ratio * b_p1 + p3_ratio * b_p3) / denom)


def stable_compute_share_w5(curr: FrameRecord) -> Optional[float]:
    """ZZ% — render-side compute-time share spent on stable primitives.

    Uses the same two per-id cost proxies as :func:`stable_bandwidth_share_w5`
    (count for preprocess, tile-touches for α-composite) but weights them by
    the two phases' *measured* wall-clock times instead of their byte
    footprints.  When ``render_kernel_breakdown_ms`` (``torch.profiler``
    output) is absent, falls back to the bandwidth-weighted ratio as a
    best-effort proxy; metric consumers should treat that fallback as an
    under-specified approximation (per-primitive cost assumed uniform).
    """
    mask = _stable_mask(curr)
    if mask is None or curr.render_ids is None:
        return None
    render = np.asarray(curr.render_ids, dtype=bool)
    n = min(mask.size, render.size)
    stable_render = mask[:n] & render[:n]
    n_render = int(render[:n].sum())
    p1_ratio = float(int(stable_render.sum()) / n_render) if n_render > 0 else 0.0
    tt = curr.tile_touches_per_id
    if tt is not None:
        m = min(n, tt.size)
        tt64 = tt[:m].astype(np.float64)
        stable_m = mask[:m] & render[:m]
        render_m = render[:m]
        total_tt = float(tt64[render_m].sum())
        p3_ratio = float(tt64[stable_m].sum() / total_tt) if total_tt > 0.0 else 0.0
    else:
        p3_ratio = 0.0

    breakdown = None
    if curr.extras:
        breakdown = curr.extras.get("render_kernel_breakdown_ms")
    if isinstance(breakdown, dict) and breakdown:
        t_p1 = float(breakdown.get("preprocess_ms", 0.0) or 0.0)
        t_p3 = float(breakdown.get("render_ms", 0.0) or 0.0)
        # ``misc`` (sort, dedup, tile-range build) scales with render size;
        # we fold it into P3 since tile-touch ratio is the closest proxy for
        # how sort/scan work distributes across primitives.
        t_p3 += float(breakdown.get("misc_ms", 0.0) or 0.0)
        denom = t_p1 + t_p3
        if denom > 0.0:
            return float((p1_ratio * t_p1 + p3_ratio * t_p3) / denom)
    n_calls = _render_calls(curr)
    b_p1 = float(curr.bytes_state_read) / n_calls
    if tt is not None:
        m = min(n, tt.size)
        total_tt = float(tt[:m].astype(np.float64)[render[:m]].sum())
        b_p3 = (total_tt / _PIXELS_PER_TILE) * _PHASE3_BYTES_PER_TILE_TOUCH
    else:
        b_p3 = 0.0
    denom = b_p1 + b_p3
    if denom <= 0.0:
        return None
    return float((p1_ratio * b_p1 + p3_ratio * b_p3) / denom)


def sh_load_bandwidth_share(
    curr: FrameRecord,
    sh_degree: Optional[int] = None,
) -> Optional[float]:
    """AA% — fraction of total render DRAM bytes spent loading SH fp32 coeffs.

    ``n_rendered × 3(d+1)² × 4 B`` is the SH-only Phase-1 payload; the
    denominator is the full render-side DRAM footprint modelled as
    ``bytes_state_read + Σ tt_g × 36 B`` (preprocess state read plus
    α-composite per-tile-touch state).  ``sh_degree`` is provided by each
    baseline hook via ``tracer.meta['sh_degree']``.
    """
    if sh_degree is None:
        return None
    if curr.render_ids is None:
        return None
    n_rendered = int(np.asarray(curr.render_ids, dtype=bool).sum())
    if n_rendered <= 0:
        return None
    sh_coeffs = 3 * (int(sh_degree) + 1) ** 2
    sh_bytes = float(n_rendered * sh_coeffs * _SH_COEFF_FP32_BYTES)
    n_calls = _render_calls(curr)
    total = float(curr.bytes_state_read) / n_calls
    if curr.tile_touches_per_id is not None:
        total += (
            float(curr.tile_touches_per_id.astype(np.float64).sum()) / _PIXELS_PER_TILE
        ) * _PHASE3_BYTES_PER_TILE_TOUCH
    if total <= 0.0:
        return None
    return float(sh_bytes / total)


def alpha_blend_compute_share(curr: FrameRecord) -> Optional[float]:
    """BB% — share of render wall-clock spent inside the α-blend kernel.

    Reads ``render_kernel_breakdown_ms`` (populated by the ``torch.profiler``
    wrapper in hooks.py) and returns ``render_ms / (preprocess + render +
    misc)`` where ``render_ms`` is the ``renderCUDA``/``alphaBlend`` kernel
    bucket.  Returns ``None`` when the profiler breakdown is absent.
    """
    if not curr.extras:
        return None
    breakdown = curr.extras.get("render_kernel_breakdown_ms")
    if not isinstance(breakdown, dict) or not breakdown:
        return None
    preprocess = float(breakdown.get("preprocess_ms", 0.0) or 0.0)
    render = float(breakdown.get("render_ms", 0.0) or 0.0)
    misc = float(breakdown.get("misc_ms", 0.0) or 0.0)
    total = preprocess + render + misc
    if total <= 0.0:
        return None
    return float(render / total)


def late_discarded_work(curr: FrameRecord) -> Optional[float]:
    """Fraction of tile-expansion work that was expanded and later discarded.

    If ``discard_mask`` is not available we approximate with a contribution
    floor (ids with contribution < 1% of the median).  The core tracer
    records the approximation so the aggregator can flag it.
    """
    if curr.tile_touches_per_id is None:
        return None
    tt = curr.tile_touches_per_id.astype(np.float64)
    total = tt.sum()
    if total <= 0.0:
        return None
    if curr.discard_mask is not None:
        n = min(curr.discard_mask.size, tt.size)
        return float(tt[: n][curr.discard_mask[: n].astype(bool)].sum() / total)
    if curr.contribution_per_id is None:
        return None
    c = curr.contribution_per_id
    n = min(c.size, tt.size)
    floor = 0.01 * np.median(c[:n][c[:n] > 0] if np.any(c[:n] > 0) else c[:n])
    discarded = c[:n] < floor
    return float(tt[:n][discarded].sum() / total)


# ---------------------------------------------------------------------------
# Registry — metric name -> callable (used by MotivationTracer.end_frame)
# ---------------------------------------------------------------------------


METRICS_CHALLENGE1 = {
    "duplicated_inter_frame_bytes": "pair",
    "duplicated_inter_stage_bytes": "single",
    "propagation_amplification": "pair",
    "iter_grad_cosine_stable_fraction": "single",
    "low_grad_fraction": "single",
    "stage_time_breakdown": "single",
    "stable_render_backward_compute_fraction": "single",
}

METRICS_CHALLENGE2 = {
    "state_movement_bytes": "single",
    "changed_state_fraction": "pair",
    "hot_readmask_overlap": "single",
    "update_render_jaccard": "single",
    "compute_mask_overlap": "single",
    "redundant_shared_state_ratio": "single",
}

METRICS_CHALLENGE3 = {
    "contribution_CDF": "single",
    "long_term_low_contribution": "history",
    "late_discarded_work": "single",
    "stable_fraction_w5": "single",
    "stable_bandwidth_share_w5": "single",
    "stable_compute_share_w5": "single",
    "sh_load_bandwidth_share": "single",
    "alpha_blend_compute_share": "single",
}
