"""
ASICConfig — all tunable parameters for the GLASS ASIC simulator.
AblationMode — the four ablation configurations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum

from simulator.core.cycle_model import ASICSpec


class AblationMode(IntEnum):
    """The four progressive ablation configurations.

    Each mode adds one more ASIC optimisation on top of the previous:
      BASE_ASIC    : descriptor capture overhead only; no optimisation active
      CONSOLIDATION: + Update-Stream Consolidation (gradient fold/suppress)
      LIVE_FABRIC  : + Incremental Live-State Fabric (hot/cold Gaussian tiers)
      SALIENCE     : + Salience-First Render Issue (PROMOTE/FULL/SKIP filter)
    """
    BASE_ASIC     = 0
    CONSOLIDATION = 1
    LIVE_FABRIC   = 2
    SALIENCE      = 3


@dataclass
class ASICConfig:
    """All simulator knobs in one place.

    Passed to GLASSPipeline, ConsolidationUnit, LiveStateFabric, and
    SalienceFirstIssue unchanged — each component reads only its own fields.
    """

    # ── Ablation mode ─────────────────────────────────────────────────────────
    mode: int = AblationMode.BASE_ASIC
    """0=base, 1=+consolidation, 2=+live_fabric, 3=+salience. May also be
    passed as an int; compared with >= so intermediate modes work correctly."""

    # ── Stage 1: Update-Stream Consolidation ─────────────────────────────────
    cwb_window: int = 5
    """Number of recent frames kept in the Correlation Window Buffer per
    Gaussian. Larger → more history → more stable similarity estimates."""

    merge_cos_threshold: float = 0.90
    """Norm-ratio proxy threshold above which a Gaussian's gradient is
    considered temporally stable and gets folded (scaled down).
    Range [0, 1]; higher = more aggressive suppression."""

    adaptive_ucn_threshold: bool = True
    """If True, enables the **feedback-based threshold controller**.  The
    effective UCN threshold is updated once per frame based on the observed
    per-frame suppression ratio:

        err = observed_ratio − adaptive_ucn_target_ratio
        effective_threshold ← clip(
            effective_threshold + adaptive_ucn_kp · err,
            adaptive_ucn_threshold_min, adaptive_ucn_threshold_max
        )

    Rationale
    ---------
    A static merge_cos_threshold cannot work well for every baseline:
      * 3DGStream (NTC path)      → wants high suppression to skip redundant
        NTC iters.
      * QUEEN (frame-level UCN)   → wants moderate suppression; dynamic
        Gaussians must survive.
      * HiCoM (iteration-level,   → at 0.90 the default over-suppresses and
        dense per-iter Adam)         kills 99.5 % of updates → quality crash.

    A feedback loop automatically finds the right threshold for each
    baseline's gradient landscape: when too many Gaussians are being
    suppressed, the threshold rises (harder to suppress); when too few,
    it falls.

    Hardware cost
    -------------
    One frame-level 16-bit register + 1 mul + 1 add + 1 clamp per frame —
    area / cycle overhead is negligible versus the UCN comparator array
    itself.  The per-step UCN gate is unchanged — only the scalar threshold
    it reads moves slowly across frames.

    Back-compat: set to ``False`` to freeze the threshold at
    ``merge_cos_threshold`` for every frame (original fixed-threshold
    behaviour).  The legacy per-Gaussian coefficient-of-variation formula
    has been retired — it never triggered in practice because Adam's
    post-warmup gradient norms are stable and CV stays ≈ 0."""

    adaptive_ucn_target_ratio: float = 0.70
    """Target per-frame suppression ratio used by the feedback threshold
    controller.  Default 0.70 keeps 30 % of updates live on average — a
    good trade-off between compute savings and training quality across
    all supported baselines.  Out of range [0, 1] → disabled check."""

    adaptive_ucn_kp: float = 0.30
    """Proportional gain for the feedback loop.  Smaller → slower
    convergence but smoother threshold trajectory (less frame-to-frame
    jitter).  0.30 yields ≈ 3-5 frames to converge from the initial
    threshold to steady state for typical error magnitudes — matched to
    the 10-frame ablation budget so the feedback loop has enough headroom
    to settle before the metrics window closes.  The static clamps
    ``adaptive_ucn_threshold_min`` / ``_max`` cap any overshoot."""

    adaptive_ucn_threshold_min: float = 0.80
    """Lower clamp on the feedback-controlled effective threshold.  Below
    this value UCN becomes too permissive and stops saving cycles; at
    0.80, ≥ 20 % magnitude mismatch is required to treat a gradient as
    novel."""

    adaptive_ucn_threshold_max: float = 1.0
    """Upper clamp on the feedback-controlled effective threshold.

    At ``threshold = 1.0`` the comparator ``sim > threshold`` is always
    false (sim ∈ [0, 1]), which means UCN suppresses nothing — an
    effective "UCN off" state that the feedback loop can reach when the
    baseline's gradient-norm distribution has sim values clustered
    close to 1.0 (HiCoM frame-level: Adam convergence keeps frame-mean
    norms almost identical across frames, so sim routinely > 0.99).
    Clamping lower (e.g. 0.995) would trap the threshold below the
    cluster and leave suppression pinned at the ceiling — the bug that
    motivated this controller."""

    auto_structural_ucn: bool = True
    """If True, GLASSPipeline auto-detects the UCN operating regime from
    the observed training cadence and switches ``frame_level_ucn`` and
    ``reset_cwb_each_frame`` appropriately at the start of frame 2.

    Detection heuristics
    --------------------
    * ``pre_optimizer_step`` count per frame ≥ ``auto_structural_step_threshold``
      → dense per-iteration Adam training (HiCoM stage-2, QUEEN temporal
      residual) → switch to ``frame_level_ucn=True`` so the CWB doesn't
      over-suppress within a single frame's Adam warm-up.
    * The first-parameter id in ``optimizer.param_groups`` changes across
      frames → the baseline re-creates its optimizer each frame (QUEEN,
      HiCoM gaussians_incr) → set ``reset_cwb_each_frame=True`` so stale
      history from the retired optimizer doesn't bleed into the next
      frame's decisions.

    Both toggles are one-shot: the decision is frozen after frame 2.
    Explicit user overrides (passing ``frame_level_ucn=True`` in the
    runner) are respected and never overwritten."""

    auto_structural_step_threshold: int = 20
    """Minimum number of ``pre_optimizer_step`` calls in frame 1 required
    to switch into frame-level UCN.  20 comfortably separates dense Adam
    training (100+ steps/frame) from NTC-style single-step frames."""

    reset_cwb_each_frame: bool = False
    """If True, the Correlation Window Buffer is cleared at the start of every
    new training frame (begin_frame()).  Use for baselines whose optimizer is
    re-initialised each frame (e.g. QUEEN) so that stale cross-frame history
    does not cause over-suppression of intra-frame gradient updates.
    3DGStream and NTC-based models keep this False (cross-frame stability is
    the intended UCN signal)."""

    frame_level_ucn: bool = False
    """If True, the CWB operates at FRAME granularity instead of iteration
    granularity.

    Iteration-level (default, frame_level_ucn=False)
    -------------------------------------------------
    CWB is pushed once per optimizer step; compares step N's norm against
    the mean of the last cwb_window steps.  Detects within-frame convergence
    (stable across ~5 consecutive gradient steps).  Works well for NTC-based
    models (3DGStream) where the raw-Gaussian optimizer converges quickly and
    the NTC inference path handles frame-to-frame adaptation.

    Frame-level (frame_level_ucn=True)
    -----------------------------------
    Per-iteration norms are accumulated in a temporary buffer during each
    frame; at begin_frame() the per-Gaussian mean over those iterations is
    computed and pushed as a single CWB entry.  The CWB then contains
    frame-level mean norms from the last cwb_window frames.  At every
    optimizer step within the CURRENT frame, the step's norm is compared
    against that cross-frame mean.

    Effect on each model type
    -------------------------
    * QUEEN (dynamic, compact, per-frame optimizer): each frame corresponds
      to a different part of the dynamic scene (e.g. different flame
      pattern).  Dynamic Gaussians have noticeably different mean norms
      across frames → sim is LOW → NOT suppressed → quality preserved.
      Static background Gaussians are similar across frames → suppressed →
      some cycle savings (expected ~1.2–1.8x).
    * 3DGStream (NTC-based, 280 K Gaussians): raw Gaussian attribute norms
      are moderately stable across frames (NTC absorbs adaptation).
      Frame-level UCN gives similar suppression to iteration-level but at
      coarser granularity; speedup may be marginally smaller (~1.1x vs
      1.3x) while quality remains equally stable."""

    min_ntc_iters_floor: int = 10
    """Minimum NTC training iterations per frame for mode >= 1 runs.
    The adaptive scheduler computes round(base_iters * (1 - suppression_ratio))
    but clamps the result to this floor so training never fully collapses.
    At base_iters=150 this caps maximum suppression at (150-10)/150 ≈ 93%.
    Quality metrics are then measured under the actual reduced iteration count."""

    min_delta_norm: float = 1e-5
    """Per-Gaussian gradient L2-norm below which the gradient is zeroed
    entirely (near-zero delta → nothing meaningful to propagate)."""

    ntc_grad_threshold: float = 5e-5
    """Gradient norm threshold for network-level param groups (NTC, decoder,
    gate, hash). Applied as a hard zero if the whole-param norm is below this."""

    # ── Stage 3: Incremental Live-State Fabric ───────────────────────────────
    asb_capacity: int = 3000
    """Active State Cache (ASC) size: top-N hottest Gaussians; always rendered
    and always receive gradients."""

    rsb_capacity: int = 10000
    """Resident State Bank (RSB) size: next-N Gaussians; rendered and updated
    only when their recent delta_norm exceeds rsb_delta_thresh."""

    rsb_delta_thresh: float = 1e-4
    """Minimum recent delta_norm for a RSB-tier Gaussian to be included in
    the render mask. Gaussians below this become effectively cold."""

    hotness_alpha: float = 0.85
    """EMA decay for hotness scores. Closer to 1 → slower decay → longer
    memory. Closer to 0 → rapid forgetting."""

    hotness_update_reward: float = 1.0
    """Hotness score increment added when a Gaussian receives a gradient
    update this frame."""

    hotness_render_reward: float = 0.4
    """Hotness score increment added when a Gaussian is visible (in
    visibility_filter) this frame."""

    ucn_drive_fabric_demotion: bool = True
    """Enable UCN→Fabric demotion signal (Path 1 of UCN generalisation).

    When enabled, every ``pre_optimizer_step`` / ``ingest_ntc_deltas`` call
    reports per-Gaussian UCN suppression events to the fabric.  A Gaussian
    that has been UCN-suppressed for ``ucn_demotion_streak`` consecutive
    frames is considered "temporally stable" and stops receiving the
    hotness reward from new gradient events — its hotness then decays
    naturally via the EMA, and it eventually falls below both the ASC and
    RSB hotness cuts and enters the CSS tier.  CSS Gaussians are excluded
    from the Mode ≥ 2 render mask, so this signal turns UCN's per-frame
    stability judgement into **physical render+backward savings** for
    baselines whose optimizer itself is cheap (e.g. HiCoM hard-freezing).
    Streaks are reset the moment a Gaussian again receives a
    non-suppressed gradient, so dynamic Gaussians never get stranded in
    CSS — the quality floor is the usual ``asb_capacity + rsb_capacity``
    hot-tier guarantee.  Set to False to restore legacy behaviour where
    fabric hotness depends only on gradient-event presence.
    """

    ucn_demotion_streak: int = 3
    """Consecutive UCN-suppressed frames required before the fabric stops
    rewarding a Gaussian's hotness on the next gradient event.  3 is a
    safe default: below the 10-frame ablation window so demotion has
    room to kick in, but high enough that a one-off suppression on a
    noisy frame does not evict a Gaussian from the hot tier."""

    # ── Stage 4: Salience-First Render Issue ─────────────────────────────────
    salience_ema_alpha: float = 0.80
    """EMA decay for per-Gaussian salience scores."""

    promote_thresh: float = 0.55
    """Salience score above which a Gaussian is PROMOTE (always rendered,
    prioritised in tile queue)."""

    skip_thresh: float = 0.04
    """Salience score below which a stable Gaussian is SKIP (excluded from
    render call entirely)."""

    stability_window: int = 6
    """Number of recent frames used to estimate salience variance for the
    stability gate."""

    stability_var_max: float = 0.01
    """Maximum score variance for a Gaussian to be classified as 'stable'
    (prerequisite for SKIP classification)."""

    full_path_fraction: float = 0.10
    """Minimum fraction of Gaussians that are always classified FULL regardless
    of score. Quality safeguard: prevents over-pruning when score history is
    stale or all scores are low."""

    # ── FPS measurement ───────────────────────────────────────────────────────
    fps_warmup: int = 5
    """Number of render passes to execute before starting the FPS timer
    (GPU warm-up, JIT compilation, etc.)."""

    fps_measure: int = 50
    """Number of render passes timed for the FPS measurement."""

    # ── Hardware specification for cycle-accurate timing model ────────────────
    asic_spec: ASICSpec = field(default_factory=ASICSpec)
    """Hardware parameters passed to CycleAccumulator in GLASSPipeline.
    Override individual fields (e.g. mac_units, clock_hz) to model different
    ASIC configurations without changing any other code."""

    # ── Ablation scope (overridden from DatasetConfig at runtime) ─────────────
    n_frames_ablation: int = 30
    """Number of incremental frames to train per ablation run. Overridden by
    DatasetConfig.n_frames_ablation when building ASICConfig in a runner."""

    def __post_init__(self):
        if self.mode not in range(4):
            raise ValueError(f"mode must be 0–3, got {self.mode}")
        if not 0.0 <= self.merge_cos_threshold <= 1.0:
            raise ValueError("merge_cos_threshold must be in [0, 1]")
        if not 0.0 < self.hotness_alpha < 1.0:
            raise ValueError("hotness_alpha must be in (0, 1)")
        if not 0.0 < self.salience_ema_alpha < 1.0:
            raise ValueError("salience_ema_alpha must be in (0, 1)")
        if not 0.0 < self.full_path_fraction <= 1.0:
            raise ValueError("full_path_fraction must be in (0, 1]")
