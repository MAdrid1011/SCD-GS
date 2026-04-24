"""Dataclasses that describe what every baseline driver must feed the tracer.

A :class:`FrameRecord` is produced per (baseline, frame) by the baseline's
``patch_*`` context manager; :class:`PerFrameMetrics` is what the tracer emits
per frame after running the 11 metric functions; :class:`TraceOutput` is the
full JSON schema written to ``results/motivation_trace/<BASELINE>/<SCENE>.json``.

The fields on :class:`FrameRecord` are intentionally wide so the core tracer
can stay baseline-agnostic.  Each baseline only fills in what it can observe
cheaply; the metric functions in :mod:`metrics` are expected to be defensive
about missing fields and return ``None`` for any metric they cannot compute
from the data they got, rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Raw per-frame record (populated by hooks.*)
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """Per-frame observable state captured by a baseline driver.

    Every array here is a **CPU numpy array** of length ``n_gaussians`` (or a
    scalar) — the hooks are responsible for ``.detach().cpu().numpy()`` before
    handing tensors to the tracer, so the core stays GPU-free and the metric
    code can be run offline on a replayed record.
    """

    baseline: str
    scene: str
    frame_idx: int
    n_gaussians: int

    # --- identity masks -----------------------------------------------------
    update_ids: Optional[np.ndarray] = None   # bool mask, shape (n_gaussians,)
    render_ids: Optional[np.ndarray] = None   # bool mask, shape (n_gaussians,)

    # --- per-id cost proxies ------------------------------------------------
    tile_touches_per_id: Optional[np.ndarray] = None     # int, shape (n_gaussians,)
    contribution_per_id: Optional[np.ndarray] = None     # float32, shape (n_gaussians,)
    update_cost_per_id: Optional[np.ndarray] = None      # float32, shape (n_gaussians,)
    discard_mask: Optional[np.ndarray] = None            # bool, "expanded but discarded"

    # --- attribute snapshots (before / after the frame's updates) -----------
    # dict keyed by attr name ("xyz"/"feat"/"scale"/"rot"/"opacity"/...);
    # values are float32 arrays of shape (n_gaussians, attr_dim).  The tracer
    # only ever needs these for Challenge 2 metrics, so drivers are allowed to
    # populate a subset.
    attr_before: Dict[str, np.ndarray] = field(default_factory=dict)
    attr_after: Dict[str, np.ndarray] = field(default_factory=dict)

    # --- byte counters ------------------------------------------------------
    # Drivers count these by walking the buffers they already snapshot; we
    # keep them explicit so baselines with exotic update paths (HiCoM's
    # add/update two-stage, 4DGC's EntropyBottleneck) can override.
    bytes_update_gen: int = 0         # bytes produced by the update kernel
    bytes_commit: int = 0             # bytes handed to optimizer.step / writeback
    bytes_state_read: int = 0         # bytes pulled by the render path
    bytes_state_write: int = 0        # bytes written back by the commit path

    # --- IDs observable at the stage boundary (HiCoM splits this cleanly) ---
    update_gen_ids: Optional[np.ndarray] = None
    commit_ids: Optional[np.ndarray] = None

    # --- per-iteration grad / timing accumulators (Challenge 1 YY/ZZ slots) -
    # Every training iteration inside a single frame appends one entry.  The
    # hooks populate these from inside ``optimizer.step`` (grad stats) and
    # from CUDA event markers (stage timings); metric functions then
    # aggregate them into per-frame scalars.
    iter_grad_cos_counts: List[Tuple[int, int]] = field(default_factory=list)
    # (n_stable_cos_ge_0.9, n_active_rows)
    iter_low_grad_counts: List[Tuple[int, int]] = field(default_factory=list)
    # (n_below_eps, n_active_rows)
    iter_stage_times_ms: List[Dict[str, float]] = field(default_factory=list)
    # {"render_ms": .., "backward_ms": .., "optimizer_ms": ..}

    # --- provenance ---------------------------------------------------------
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output metrics (per-frame; what the paper figures ultimately consume)
# ---------------------------------------------------------------------------


@dataclass
class Challenge1Metrics:
    duplicated_inter_frame_bytes: Optional[float] = None
    duplicated_inter_stage_bytes: Optional[float] = None
    propagation_amplification: Optional[Dict[str, float]] = None
    # new slots populated by hooks + derived aggregates (GLASS.md L113/L115)
    iter_grad_cosine_stable_fraction: Optional[float] = None
    low_grad_fraction: Optional[float] = None
    stage_time_breakdown_ms: Optional[Dict[str, float]] = None
    render_backward_time_fraction: Optional[float] = None
    stable_render_backward_compute_fraction: Optional[float] = None


@dataclass
class Challenge2Metrics:
    state_movement_bytes: Optional[float] = None
    changed_state_fraction: Optional[float] = None
    hot_readmask_overlap: Optional[float] = None
    update_render_jaccard: Optional[float] = None
    compute_mask_overlap: Optional[float] = None
    # new slot populated by derived metric (GLASS.md L122)
    redundant_shared_state_ratio: Optional[float] = None


@dataclass
class Challenge3Metrics:
    contribution_CDF: Optional[List[float]] = None
    long_term_low_contribution: Optional[float] = None
    late_discarded_work: Optional[float] = None
    # new slots populated by the C3 five-column table (GLASS.md L140–L142)
    stable_fraction_w5: Optional[float] = None            # XX%
    stable_bandwidth_share_w5: Optional[float] = None     # YY%
    stable_compute_share_w5: Optional[float] = None       # ZZ%
    sh_load_bandwidth_share: Optional[float] = None       # AA%
    alpha_blend_compute_share: Optional[float] = None     # BB%
    # Optional S^(W) CDF dump (50 points) — only populated when the tracer
    # is started with ``emit_stability_cdf=True`` (used by the calibration
    # pilot to pick ``τ_stable``; off by default to keep JSON compact).
    stable_s_w5_cdf: Optional[List[float]] = None


@dataclass
class PerFrameMetrics:
    frame_idx: int
    n_gaussians: int
    challenge1: Challenge1Metrics = field(default_factory=Challenge1Metrics)
    challenge2: Challenge2Metrics = field(default_factory=Challenge2Metrics)
    challenge3: Challenge3Metrics = field(default_factory=Challenge3Metrics)
    degraded: List[str] = field(default_factory=list)  # names of metrics we could not compute

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceOutput:
    baseline: str
    scene: str
    n_frames: int
    per_frame: List[PerFrameMetrics] = field(default_factory=list)
    aggregates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    degraded_metrics: List[str] = field(default_factory=list)  # union across frames
    schema_version: str = "1.1"

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        # dataclass' asdict handles nested dataclasses fine
        return out
