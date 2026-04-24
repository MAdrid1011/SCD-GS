"""Canonical ablation result schema.

All four ablation runners (run_3dgstream, run_queen, run_hicom, run_4dgc)
emit a single JSON per (baseline, mode).  Historically each runner assembled
this dict in its own order and with subtly different keys — some wrote
``avg_train_time_s`` / ``all_frame_times_s`` (GPU wall-clock), some only wrote
``asic_train_time_s``; the NTC baselines included per-frame iteration
schedules while the Gaussian-only baselines did not; and ``cycle_summary``
still emitted a redundant ``asic_time_s_1ghz`` that was easy to confuse
with ``asic_train_time_s``.

This module centralises the schema so every runner produces an identical set
of keys in an identical order.  Fields that are inapplicable to a specific
baseline (e.g. ``per_frame_train_iters`` for QUEEN's monolithic schedule) are
emitted as ``None`` rather than being silently omitted.

Design principles
-----------------
1. **Every key must relate to one of the four evaluation axes**:
     - ASIC training time (headline ``asic_train_time_s``)
     - Render FPS
     - Reconstruction quality (PSNR / SSIM / LPIPS)
     - Storage (total checkpoint + incremental stream)
   Fields that only provide architectural decomposition of these axes
   (ASIC cycle buckets, UCN suppression bytes, Live-State Fabric tier
   counts, per-frame training iteration schedule) are retained because
   they directly explain *why* one of the four headline numbers moves
   between modes.  GPU wall-clock fields were dropped — the simulator
   hosts the baseline code on a desktop GPU for quality/delta capture
   only, so its per-frame time is a simulator-host artefact, not an
   ASIC property.

2. **No redundant keys.**  ``asic_time_s_1ghz`` was
   ``asic_total_cycles / clock_hz`` and easy to misread as the ASIC
   training time (it excludes NTC training, which is by far the
   dominant bucket for 3DGStream / 4DGC).  We drop it and leave
   ``asic_total_cycles`` as the single source of truth in that bucket.

3. **Uniform iter-schedule keys.**  Renamed from ``*_ntc_iters`` to
   ``*_train_iters`` so HiCoM (which trains Gaussians, not an NTC) and
   4DGC (which trains a Motion Estimation Module, not strictly an NTC
   either) can share the key with 3DGStream's tcnn-backed NTC path.
   QUEEN emits ``None`` here because its ``opt.iterations`` is monolithic
   and not driven by the adaptive UCN feedback scheduler.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["SCHEMA_KEYS", "MODE_LABELS", "build_result"]


# ── Mode labels (single source of truth; previously duplicated across
#    collect_ablation.py and collect_summary.py).
MODE_LABELS: dict[int, str] = {
    0: "Base ASIC",
    1: "+UCN",
    2: "+ILSF",
    3: "+CGS",  # Convergence-Graded Splatting (legacy "+Salience" label)
}


# ── Canonical key order.  Every ``mode_*.json`` emitted by any runner
# must present these keys, in this order.  Runners that don't measure
# a given field should pass ``None``; they must never omit the key.
SCHEMA_KEYS: tuple[str, ...] = (
    # ── Identity ─────────────────────────────────────────────────────
    "baseline",
    "dataset",
    "mode",
    "mode_label",
    "n_frames_trained",
    "incremental",

    # ── Quality ──────────────────────────────────────────────────────
    "psnr",
    "ssim",
    "lpips",

    # ── Inference throughput ─────────────────────────────────────────
    # render_fps               — HEADLINE. ASIC modelled FPS from
    #                            add_render_inference(n_active, n_visible,
    #                            tile_touches, H, W, sh_degree) at
    #                            ASICSpec.clock_hz.  Directly comparable
    #                            across baselines because it depends only
    #                            on the masked-proxy geometry the ASIC
    #                            would see, not on simulator-host kernel
    #                            launch overhead, SH-degree overhead or
    #                            AMP flags.
    # render_fps_gpu_measured  — DIAGNOSTIC.  Wall-clock FPS the host GPU
    #                            achieves on the same MaskedGaussianProxy.
    #                            Kept for sanity checks; per-baseline M0
    #                            values differ by up to 3-4× for reasons
    #                            unrelated to the GLASS architecture.
    # render_n_active          — proxy.n_gaussians() at FPS-capture time
    # render_n_visible         — (radii>0).sum() on the single captured render
    # render_tile_touches      — n_visible × π × (avg_r/16)²
    # render_H, render_W       — image resolution
    "render_fps",
    "render_fps_gpu_measured",
    "render_n_active",
    "render_n_visible",
    "render_tile_touches",
    "render_H",
    "render_W",

    # ── Convergence-Graded Splatting diagnostics (mode 3 only) ───────
    # n_grade_a / n_grade_b / n_grade_d   — CGS three-way grade counts
    # compact_path_ratio                  — Grade-B per-primitive cost
    #                                       relative to Grade-A
    # render_full_cycles / render_compact_cycles
    #                                     — per-path cycle breakdown
    #                                       that sums (via max) to
    #                                       asic_render_cycles
    # Legacy aliases still present in asic_cycle breakdown below:
    #   asic_salience_cycles  — stays populated as the Stage-4
    #                           support-domain evaluator bucket.
    "n_grade_a",
    "n_grade_b",
    "n_grade_d",
    "compact_path_ratio",
    "cgs_rho_mean",
    "cgs_kappa_mean",
    "cgs_potential_mean",
    "render_full_cycles",
    "render_compact_cycles",

    # ── Storage ──────────────────────────────────────────────────────
    "total_storage_mb",
    "delta_storage_mb",

    # ── Headline ASIC timing ─────────────────────────────────────────
    # asic_train_time_s = per-frame incremental training time at
    #                     ASICSpec.clock_hz, including render+loss+bwd,
    #                     NTC / MEM / MLP network training, commit pipe
    #                     and optimizer.  The main metric the paper
    #                     reports against GPU / Jetson baselines.
    # asic_commit_time_s = subset: gradient pipeline + optimizer only.
    #                     Useful for isolating the Mode 0→1→2 progression
    #                     without NTC-training cycles masking it.
    "asic_train_time_s",
    "asic_commit_time_s",

    # ── ASIC cycle breakdown (explains asic_train_time_s) ────────────
    "asic_ntc_train_cycles",
    "asic_ntc_train_render_cycles",
    "asic_ntc_cycles",
    "asic_puoe_cycles",
    "asic_ucn_cycles",
    "asic_ice_cycles",
    "asic_ilsf_cycles",
    "asic_salience_cycles",
    "asic_optimizer_cycles",
    "asic_render_cycles",
    # CGS per-path cycle breakdown (zero when mode < 3).  Their
    # ``max`` equals ``asic_render_cycles`` under CGS because the
    # hardware dispatcher runs the Full / Compact paths on
    # independent warp / tile queues.
    "asic_render_full_cycles",
    "asic_render_compact_cycles",
    "asic_gradient_pipeline_cycles",
    "asic_post_render_cycles",
    "asic_total_cycles",

    # ── Update-path traffic — Challenge 1 / UCN evidence ─────────────
    "frame_total_bytes",
    "frame_suppressed_bytes",
    "frame_active_bytes",
    "frame_suppression_ratio",
    "frame_unique_targets",
    "run_total_bytes",
    "run_suppressed_bytes",
    "run_suppression_ratio",

    # ── Live-state residency — Challenge 2 / ILSF evidence ───────────
    "n_asc_rsb",
    "n_css",
    "n_known",
    "n_total",

    # ── Per-frame training iteration schedule (Path 2 UCN feedback) ──
    # All three keys are ``None`` for baselines without adaptive
    # scheduling (QUEEN's opt.iterations is monolithic).  For the
    # adaptive baselines these explain a share of asic_train_time_s
    # savings between Mode 0 and Mode 1.
    "base_train_iters",
    "avg_train_iters",
    "per_frame_train_iters",
)


def build_result(
    *,
    baseline: str,
    dataset: str,
    mode: int,
    n_frames_trained: int,
    quality: dict,
    render_fps: float,
    total_storage_mb: float,
    delta_storage_mb: float,
    asic_train_time_s: float,
    cycle_summary: dict,
    stream_stats: dict,
    fps_diagnostic:        Optional[dict]       = None,
    base_train_iters:      Optional[int]        = None,
    avg_train_iters:       Optional[float]      = None,
    per_frame_train_iters: Optional[list[int]]  = None,
    incremental: bool = True,
) -> dict[str, Any]:
    """Assemble a ``mode_*.json`` result dict in canonical key order.

    The helper is deliberately strict: it builds the return dict by
    iterating :data:`SCHEMA_KEYS`, so any runner that tries to smuggle
    in an extra key sees the key silently dropped.  Conversely, every
    caller gets exactly the same key order in every emitted JSON.

    Parameters
    ----------
    baseline, dataset, mode, n_frames_trained, incremental
        Identity columns.
    quality
        Dict returned by :meth:`MetricsCollector.measure_quality`,
        must contain ``psnr`` / ``ssim`` / ``lpips``.
    render_fps
        Output of :meth:`MetricsCollector.measure_fps`, already
        reflecting the pipeline's ablation mode via the
        ``MaskedGaussianProxy`` (Mode ≥ 2) and salience
        classifier (Mode ≥ 3).
    total_storage_mb, delta_storage_mb
        Final checkpoint footprint and summed per-frame incremental
        stream, respectively.
    asic_train_time_s
        Pre-computed by the caller — every runner uses slightly
        different bucket combinations (NTC / MEM / MLP presence), so
        the sum is supplied rather than inferred here.
    cycle_summary
        Output of :meth:`GLASSPipeline.cycle_summary`.  Keys missing
        from the summary default to 0.
    stream_stats
        Output of :meth:`GLASSPipeline.stream_stats`.  Missing keys
        default to 0; ``run_suppression_ratio`` is derived on the fly.
    base_train_iters, avg_train_iters, per_frame_train_iters
        Per-frame iteration schedule exposed by 3DGStream / HiCoM /
        4DGC runners.  QUEEN passes ``None`` for all three.

    Returns
    -------
    dict
        A plain ``dict`` preserving insertion order (the JSON writer
        will therefore emit keys in :data:`SCHEMA_KEYS` order).
    """
    run_total = int(stream_stats.get("run_total_bytes", 0) or 0)
    run_supp  = int(stream_stats.get("run_suppressed_bytes", 0) or 0)
    run_ratio = (run_supp / run_total) if run_total > 0 else 0.0

    diag = fps_diagnostic or {}

    assembled = {
        # Identity
        "baseline":                    baseline,
        "dataset":                     dataset,
        "mode":                        int(mode),
        "mode_label":                  MODE_LABELS.get(int(mode), str(mode)),
        "n_frames_trained":            int(n_frames_trained),
        "incremental":                 bool(incremental),

        # Quality
        "psnr":                        quality.get("psnr"),
        "ssim":                        quality.get("ssim"),
        "lpips":                       quality.get("lpips"),

        # Inference throughput
        "render_fps":                  render_fps,
        "render_fps_gpu_measured":     diag.get("gpu_fps"),
        "render_n_active":             diag.get("n_active"),
        "render_n_visible":            diag.get("n_visible"),
        "render_tile_touches":         diag.get("tile_touches"),
        "render_H":                    diag.get("H"),
        "render_W":                    diag.get("W"),

        # Convergence-Graded Splatting diagnostics (mode 3 only; older
        # ``n_promote / n_full / n_skip`` field names are still
        # produced by :meth:`ConvergenceGrader.tier_counts` under
        # ``stream_stats`` so downstream analysis scripts keep working).
        "n_grade_a":                   diag.get("n_grade_a", 0),
        "n_grade_b":                   diag.get("n_grade_b", 0),
        "n_grade_d":                   diag.get("n_grade_d", 0),
        "compact_path_ratio":          diag.get("compact_path_ratio", 0.0),
        "cgs_rho_mean":                diag.get("cgs_rho_mean", 1.0),
        "cgs_kappa_mean":              diag.get("cgs_kappa_mean", 0.0),
        "cgs_potential_mean":          diag.get("cgs_potential_mean", 1.0),
        "render_full_cycles":          diag.get("render_full_cycles", 0),
        "render_compact_cycles":       diag.get("render_compact_cycles", 0),

        # Storage
        "total_storage_mb":            total_storage_mb,
        "delta_storage_mb":            delta_storage_mb,

        # Headline ASIC timing
        "asic_train_time_s":           asic_train_time_s,
        "asic_commit_time_s":          cycle_summary.get("asic_commit_time_s", 0.0),

        # ASIC cycle breakdown
        "asic_ntc_train_cycles":       cycle_summary.get("asic_ntc_train_cycles", 0),
        "asic_ntc_train_render_cycles":cycle_summary.get("asic_ntc_train_render_cycles", 0),
        "asic_ntc_cycles":             cycle_summary.get("asic_ntc_cycles", 0),
        "asic_puoe_cycles":            cycle_summary.get("asic_puoe_cycles", 0),
        "asic_ucn_cycles":             cycle_summary.get("asic_ucn_cycles", 0),
        "asic_ice_cycles":             cycle_summary.get("asic_ice_cycles", 0),
        "asic_ilsf_cycles":            cycle_summary.get("asic_ilsf_cycles", 0),
        "asic_salience_cycles":        cycle_summary.get("asic_salience_cycles", 0),
        "asic_optimizer_cycles":       cycle_summary.get("asic_optimizer_cycles", 0),
        "asic_render_cycles":          cycle_summary.get("asic_render_cycles", 0),
        "asic_render_full_cycles":     cycle_summary.get("asic_render_full_cycles", 0),
        "asic_render_compact_cycles":  cycle_summary.get("asic_render_compact_cycles", 0),
        "asic_gradient_pipeline_cycles":cycle_summary.get("asic_gradient_pipeline_cycles", 0),
        "asic_post_render_cycles":     cycle_summary.get("asic_post_render_cycles", 0),
        "asic_total_cycles":           cycle_summary.get("asic_total_cycles", 0),

        # Update-path traffic
        "frame_total_bytes":           stream_stats.get("frame_total_bytes", 0),
        "frame_suppressed_bytes":      stream_stats.get("frame_suppressed_bytes", 0),
        "frame_active_bytes":          stream_stats.get("frame_active_bytes", 0),
        "frame_suppression_ratio":     stream_stats.get("frame_suppression_ratio", 0.0),
        "frame_unique_targets":        stream_stats.get("frame_unique_targets", 0),
        "run_total_bytes":             run_total,
        "run_suppressed_bytes":        run_supp,
        "run_suppression_ratio":       run_ratio,

        # Live-state residency
        "n_asc_rsb":                   stream_stats.get("n_asc_rsb", 0),
        "n_css":                       stream_stats.get("n_css", 0),
        "n_known":                     stream_stats.get("n_known", 0),
        "n_total":                     stream_stats.get("n_total", 0),

        # Per-frame training iteration schedule
        "base_train_iters":            base_train_iters,
        "avg_train_iters":             avg_train_iters,
        "per_frame_train_iters":       per_frame_train_iters,
    }

    # Enforce canonical key order; silently drops any extras a caller
    # may have passed through a legacy path (there should be none, but
    # this guarantees a stable on-disk schema).
    return {k: assembled[k] for k in SCHEMA_KEYS}
