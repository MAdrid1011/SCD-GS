"""MotivationTracer: the baseline-agnostic core that owns history, metric
evaluation, and JSON serialization.

Lifecycle per frame::

    tracer.start_frame(frame_idx)
    # ... baseline runs, hooks call tracer.record(field=..., ...) ...
    tracer.end_frame()

After the whole run::

    tracer.close()   # writes the JSON to out_path

The tracer is deliberately small — all scientific content lives in
:mod:`metrics` and :mod:`hooks`.  The core only stores the last full record
(for pair metrics) plus a rolling window of contribution arrays (for
:func:`metrics.long_term_low_contribution`).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from . import metrics as M
from .schema import (
    Challenge1Metrics,
    Challenge2Metrics,
    Challenge3Metrics,
    FrameRecord,
    PerFrameMetrics,
    TraceOutput,
)


# ---------------------------------------------------------------------------
# Cross-frame state snapshot (Challenge 3 — five-frame stability window)
# ---------------------------------------------------------------------------


class CrossFrameStateSnapshot:
    """Fixed-window deque of per-Gaussian 11-D state snapshots.

    Each snapshot is a float32 ``(n, 11)`` ndarray built from ``xyz`` (3) +
    ``scale`` (3) + ``rot`` (4) + ``opacity`` (1); SH coefficients are
    intentionally omitted to keep residency bounded on QUEEN-scale models.
    ``compute_stable_mask(threshold)`` returns the bool mask of ids whose
    windowed mean squared state-delta ``S^(W)_i = (1/W) Σ ||Δθ_τ||²`` sits
    below the absolute threshold, plus the count of ids that entered the
    scene mid-window and therefore could not be assessed.
    """

    _STATE_KEYS = ("xyz", "scale", "rot", "opacity")

    def __init__(self, maxlen: int = 5) -> None:
        self.maxlen = int(maxlen)
        self._deque: Deque[np.ndarray] = deque(maxlen=self.maxlen)

    def reset(self) -> None:
        self._deque.clear()

    @classmethod
    def build_state_vector(cls, attrs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Pack xyz/scale/rot/opacity into a single (n, 11) float32 matrix.

        Returns ``None`` when any required key is missing or row counts
        disagree across attributes.
        """
        if not attrs:
            return None
        missing = [k for k in cls._STATE_KEYS if k not in attrs]
        if missing:
            return None
        parts: List[np.ndarray] = []
        n: Optional[int] = None
        for key in cls._STATE_KEYS:
            a = attrs[key]
            if a is None:
                return None
            if a.ndim == 1:
                a = a[:, None]
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                return None
            a = a.astype(np.float32, copy=False)
            if key == "rot":
                # Quaternions ``q`` and ``-q`` encode the same rotation but
                # differ by L2² = 4 componentwise; left unsigned, random sign
                # flips frame-to-frame alone swamp the state-delta signal
                # (observed ~3.2 ≈ 4·(W-1)/W per id on 3DGStream).  Canonicalise
                # to the hemisphere with ``w ≥ 0`` so the geodesic distance
                # is reflected in the raw L2 we concatenate below.
                if a.shape[1] == 4:
                    sign = np.where(a[:, :1] >= 0.0, 1.0, -1.0).astype(np.float32)
                    a = a * sign
            # Defensive: some baselines leave NaN/Inf in unused attribute
            # slots (e.g. 3DGStream's SH tail); replace with 0 so stability
            # metrics are not contaminated.
            if not np.all(np.isfinite(a)):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(a)
        if n is None or n <= 0:
            return None
        return np.concatenate(parts, axis=1)

    def push(self, attrs: Dict[str, np.ndarray]) -> bool:
        """Append a fresh state snapshot; returns True on success."""
        vec = self.build_state_vector(attrs)
        if vec is None:
            return False
        self._deque.append(vec)
        return True

    def compute_stable_mask(
        self, threshold: float
    ) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """Return ``(stable_mask, s_values, windowless_count)``.

        The per-id score ``S^(W)_i = (1/W) Σ_τ ||Δθ̃_τ||²`` is evaluated on a
        **per-column z-scored state** θ̃ — each of the 11 attribute channels
        (``xyz``/``scale``/``rot``/``opacity``) is scaled by the standard
        deviation of its latest snapshot across the current id prefix before
        the frame-to-frame squared delta is taken.  Without this
        normalisation the score is dominated by whichever attribute happens
        to live in the largest numerical range (e.g. world-unit ``xyz`` on
        QUEEN yields S ≈ 400 while entropy-quantised 4DGC gives S ≈ 10⁻⁶),
        and no single absolute threshold calibrates across baselines.  Under
        z-scoring a threshold of ``1`` means "per-attribute movement equal
        to one σ of the scene-wide attribute scale"; sensible ``τ`` values
        sit in ``[1e-4, 1e-1]``.

        * ``stable_mask``: bool ndarray of length = current n_gaussians, True
          iff the id has been present across the whole window *and* its
          windowed state-delta mean ``S^(W)_i`` sits at or below ``threshold``.
        * ``s_values``: float32 ndarray of the per-id ``S^(W)_i`` score
          (length = min common prefix across the window); ``None``-like ids
          that entered mid-window do not appear.
        * ``windowless_count``: number of current ids that were added
          inside the window and therefore have no full-window score.

        Returns ``None`` if the window is not yet full (first W-1 frames).
        """
        if len(self._deque) < self.maxlen:
            return None
        sizes = [s.shape[0] for s in self._deque]
        n_min = min(sizes)
        n_curr = sizes[-1]
        if n_min <= 0:
            mask = np.zeros(n_curr, dtype=bool)
            return mask, np.zeros(0, dtype=np.float32), n_curr
        stack = np.stack([s[:n_min] for s in self._deque], axis=0)  # (W, n_min, 11)
        # Per-column scale taken from the most recent (post-frame) snapshot
        # — using the *latest* frame's std keeps the reference consistent
        # across runs that grow the id set via densify (the older snapshots
        # have different spatial coverage and would wobble σ frame-to-frame).
        latest = stack[-1]  # (n_min, 11)
        sigma = latest.std(axis=0)  # (11,)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1.0)
        scaled = stack / sigma[None, None, :]
        deltas = np.diff(scaled, axis=0)  # (W-1, n_min, 11)
        sq = np.einsum("wnd,wnd->wn", deltas, deltas)  # (W-1, n_min)
        s_w = (sq.sum(axis=0) / float(self.maxlen)).astype(np.float32, copy=False)
        stable_prefix = s_w <= float(threshold)
        mask = np.zeros(n_curr, dtype=bool)
        mask[:n_min] = stable_prefix
        windowless = int(n_curr - n_min)
        return mask, s_w, windowless

logger = logging.getLogger("motivation_trace")


class MotivationTracer:
    def __init__(
        self,
        baseline: str,
        scene: str,
        out_path: Path,
        *,
        history_window: int = 30,
        stable_threshold: float = 1e-4,
        stable_window: int = 5,
        emit_stability_cdf: bool = False,
    ) -> None:
        self.baseline = baseline
        self.scene = scene
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_window = history_window
        self.stable_threshold = float(stable_threshold)
        self.stable_window = int(stable_window)
        self.emit_stability_cdf = bool(emit_stability_cdf)

        self._output = TraceOutput(baseline=baseline, scene=scene, n_frames=0)
        self._curr: Optional[FrameRecord] = None
        self._prev: Optional[FrameRecord] = None
        # We only retain the contribution arrays + frame_idx, not full records,
        # to bound memory.
        self._contrib_history: Deque[FrameRecord] = deque(maxlen=history_window)
        # CUDA-event queue for per-iter stage timings.  Hooks record events
        # asynchronously and push ``(render_start, render_end, step_start,
        # step_end)`` tuples; we batch-sync + flush them to
        # ``FrameRecord.iter_stage_times_ms`` at ``end_frame`` to avoid
        # per-iter ``cudaEventQuery`` stalls.
        self._events_queue: List[Any] = []
        # Challenge 3 five-frame stability window — snapshots the post-frame
        # 11-D θ at every ``end_frame`` call, yielding ``stable_mask_w5``
        # once the window fills.  Hooks may observe ``self.snapshot_w5`` and
        # push earlier (e.g. mid-frame) if they prefer, but the tracer pushes
        # automatically from ``attr_after`` / ``attr_before`` below.
        self.snapshot_w5 = CrossFrameStateSnapshot(maxlen=self.stable_window)
        # Shared baseline-provided metadata (e.g. ``sh_degree``) published by
        # hooks so metric code can read it without importing the baseline.
        self.meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def start_frame(self, frame_idx: int, n_gaussians: int = 0) -> FrameRecord:
        self._curr = FrameRecord(
            baseline=self.baseline,
            scene=self.scene,
            frame_idx=frame_idx,
            n_gaussians=n_gaussians,
        )
        self._events_queue = []
        return self._curr

    def push_iter_events(
        self,
        render_start: Any,
        render_end: Any,
        step_start: Any,
        step_end: Any,
    ) -> None:
        """Queue a set of CUDA events marking one training iter's boundaries.

        Events are held un-queried (no sync) until :meth:`end_frame` drains
        them in one ``torch.cuda.synchronize`` pass.  Hooks may pass ``None``
        for any boundary they cannot observe (e.g. pure-eval frames); the
        flush logic drops such iters from the timing aggregate.
        """
        if self._curr is None:
            return
        self._events_queue.append(
            {
                "render_start": render_start,
                "render_end": render_end,
                "step_start": step_start,
                "step_end": step_end,
            }
        )

    def record(self, **fields: Any) -> None:
        """Update fields on the in-flight :class:`FrameRecord`.

        Hooks call this any number of times during a frame.  Missing fields
        are fine — the metric code is defensive about ``None`` entries.
        """
        if self._curr is None:
            raise RuntimeError("record() called outside a start_frame/end_frame scope")
        for k, v in fields.items():
            if not hasattr(self._curr, k):
                raise KeyError(f"FrameRecord has no field {k!r}")
            setattr(self._curr, k, v)

    def add_bytes(self, **counters: int) -> None:
        """Incremental byte-counter update (``bytes_update_gen`` et al.)."""
        if self._curr is None:
            raise RuntimeError("add_bytes() called outside a start_frame/end_frame scope")
        for k, v in counters.items():
            if not hasattr(self._curr, k):
                raise KeyError(f"FrameRecord has no field {k!r}")
            setattr(self._curr, k, getattr(self._curr, k) + int(v))

    def end_frame(self) -> PerFrameMetrics:
        if self._curr is None:
            raise RuntimeError("end_frame() called without a matching start_frame()")
        curr = self._curr
        prev = self._prev
        # Drain the CUDA-event queue into iter_stage_times_ms before running
        # metric code — one sync per frame keeps overhead bounded.
        self._flush_events_queue(curr)
        # Push this frame's post-commit θ into the W-frame stability deque
        # and, once the window is full, stash the ``stable_mask_w5`` on the
        # frame record so Challenge-3 metrics can consume it.  We prefer
        # ``attr_after`` (state rendered at the end of the frame) and fall
        # back to ``attr_before`` (some hooks only snapshot once).
        self._push_stability_snapshot(curr)
        pfm = PerFrameMetrics(frame_idx=curr.frame_idx, n_gaussians=curr.n_gaussians)

        stage_breakdown = M.stage_time_breakdown(curr)
        c1 = Challenge1Metrics(
            duplicated_inter_frame_bytes=M.duplicated_inter_frame_bytes(prev, curr),
            duplicated_inter_stage_bytes=M.duplicated_inter_stage_bytes(curr),
            propagation_amplification=M.propagation_amplification(prev, curr),
            iter_grad_cosine_stable_fraction=M.iter_grad_cosine_stable_fraction(curr),
            low_grad_fraction=M.low_grad_fraction(curr),
            stage_time_breakdown_ms=stage_breakdown,
            render_backward_time_fraction=(
                stage_breakdown.get("render_backward_fraction")
                if stage_breakdown is not None
                else None
            ),
            stable_render_backward_compute_fraction=(
                M.stable_render_backward_compute_fraction(curr)
            ),
        )
        c2 = Challenge2Metrics(
            state_movement_bytes=M.state_movement_bytes(curr),
            changed_state_fraction=M.changed_state_fraction(prev, curr),
            hot_readmask_overlap=M.hot_readmask_overlap(curr),
            update_render_jaccard=M.update_render_jaccard(curr),
            compute_mask_overlap=M.compute_mask_overlap(curr),
            redundant_shared_state_ratio=M.redundant_shared_state_ratio(curr),
        )
        # Keep a lightweight snapshot for ``long_term_low_contribution``: only
        # the fields that metric reads, copied out before :func:`_slim` mutates
        # ``curr`` below.  Otherwise the history ends up holding references to
        # slimmed records (contribution_per_id=None) and the metric degrades
        # forever.
        self._contrib_history.append(
            FrameRecord(
                baseline=curr.baseline,
                scene=curr.scene,
                frame_idx=curr.frame_idx,
                n_gaussians=curr.n_gaussians,
                contribution_per_id=(
                    curr.contribution_per_id.copy()
                    if curr.contribution_per_id is not None
                    else None
                ),
            )
        )
        c3 = Challenge3Metrics(
            contribution_CDF=M.contribution_CDF(curr),
            long_term_low_contribution=M.long_term_low_contribution(
                list(self._contrib_history), window=self.history_window
            ),
            late_discarded_work=M.late_discarded_work(curr),
            stable_fraction_w5=M.stable_fraction_w5(curr),
            stable_bandwidth_share_w5=M.stable_bandwidth_share_w5(curr),
            stable_compute_share_w5=M.stable_compute_share_w5(curr),
            sh_load_bandwidth_share=M.sh_load_bandwidth_share(
                curr, sh_degree=self.meta.get("sh_degree")
            ),
            alpha_blend_compute_share=M.alpha_blend_compute_share(curr),
            stable_s_w5_cdf=(
                curr.extras.get("stable_s_w5_cdf") if self.emit_stability_cdf else None
            ),
        )

        pfm.challenge1 = c1
        pfm.challenge2 = c2
        pfm.challenge3 = c3
        pfm.degraded = self._collect_degraded(c1, c2, c3)
        self._output.per_frame.append(pfm)
        self._output.n_frames = len(self._output.per_frame)

        # Roll prev; drop large tensors from prev to bound memory
        self._prev = self._slim(curr)
        self._curr = None
        return pfm

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def close(self) -> Path:
        self._output.aggregates = self._aggregate()
        self._output.degraded_metrics = sorted(
            {name for pfm in self._output.per_frame for name in pfm.degraded}
        )
        with self.out_path.open("w") as f:
            json.dump(self._output.to_dict(), f, indent=2, default=_json_default)
        logger.info("Wrote motivation trace to %s", self.out_path)
        return self.out_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push_stability_snapshot(self, rec: FrameRecord) -> None:
        """Feed the W-frame snapshot deque and stash ``stable_mask_w5``.

        Picks ``attr_after`` when available (post-commit θ is what the next
        render call consumes) and falls back to ``attr_before`` for baselines
        that only take one snapshot per frame.  Silent on failure — this is
        a Challenge-3-only accessory, metric functions degrade gracefully.
        """
        try:
            source = rec.attr_after if rec.attr_after else rec.attr_before
            ok = self.snapshot_w5.push(source)
            if not ok:
                return
            result = self.snapshot_w5.compute_stable_mask(self.stable_threshold)
            if result is None:
                # Window not full yet — explicitly mark so C3 metrics report
                # ``None`` instead of guessing.
                rec.extras["stable_mask_w5"] = None
                rec.extras["stable_s_w5_values"] = None
                rec.extras["stable_windowless_count_w5"] = None
                return
            stable_mask, s_values, windowless = result
            rec.extras["stable_mask_w5"] = stable_mask
            rec.extras["stable_s_w5_values"] = s_values
            rec.extras["stable_windowless_count_w5"] = windowless
            if self.emit_stability_cdf and s_values is not None and s_values.size > 0:
                rec.extras["stable_s_w5_cdf"] = _build_cdf(s_values, bins=50)
        except Exception as exc:
            logger.debug("stability snapshot failed: %s", exc)

    def _flush_events_queue(self, rec: FrameRecord) -> None:
        """Synchronize CUDA once, then translate queued events to ms."""
        if not self._events_queue:
            return
        try:
            import torch  # local import: tracer core must stay GPU-free
        except Exception as exc:
            logger.debug("torch unavailable, skipping event flush: %s", exc)
            self._events_queue = []
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            logger.debug("cuda.synchronize failed: %s", exc)
            self._events_queue = []
            return
        for ev in self._events_queue:
            try:
                rs, re = ev.get("render_start"), ev.get("render_end")
                ss, se = ev.get("step_start"), ev.get("step_end")
                if rs is None or re is None or ss is None or se is None:
                    continue
                render_ms = float(rs.elapsed_time(re))
                # render_end -> step_start is the backward window (autograd
                # runs between the render callsite returning and
                # optimizer.step being entered).
                backward_ms = float(re.elapsed_time(ss))
                optimizer_ms = float(ss.elapsed_time(se))
                # Negative elapsed_time means the events were recorded on
                # streams with unresolvable ordering; drop the iter instead
                # of propagating NaNs downstream.
                if render_ms < 0 or backward_ms < 0 or optimizer_ms < 0:
                    continue
                rec.iter_stage_times_ms.append(
                    {
                        "render_ms": render_ms,
                        "backward_ms": backward_ms,
                        "optimizer_ms": optimizer_ms,
                    }
                )
            except Exception as exc:
                logger.debug("event elapsed_time failed: %s", exc)
                continue
        self._events_queue = []

    @staticmethod
    def _collect_degraded(*blocks) -> list[str]:
        degraded: list[str] = []
        for block in blocks:
            for name, val in block.__dict__.items():
                if val is None:
                    degraded.append(name)
        return degraded

    @staticmethod
    def _slim(rec: FrameRecord) -> FrameRecord:
        """Drop per-id arrays we do not need for pair metrics on the next frame.

        For ``changed_state_fraction`` we need ``attr_after``; for
        ``duplicated_inter_frame_bytes`` we need ``attr_before`` and
        ``attr_after``.  Everything else can be released.
        """
        rec.tile_touches_per_id = None
        rec.contribution_per_id = None
        rec.update_ids = None
        rec.render_ids = None
        rec.discard_mask = None
        rec.update_cost_per_id = None
        rec.update_gen_ids = None
        rec.commit_ids = None
        rec.extras = {}
        # iter-level accumulators are already aggregated into
        # Challenge1Metrics; they are not consumed by any pair metric.
        rec.iter_grad_cos_counts = []
        rec.iter_low_grad_counts = []
        rec.iter_stage_times_ms = []
        return rec

    def _aggregate(self) -> Dict[str, Dict[str, float]]:
        """Mean / median per metric; CDF-type metrics are skipped."""
        agg: Dict[str, Dict[str, float]] = {}
        for challenge in ("challenge1", "challenge2", "challenge3"):
            agg[challenge] = {}
            for pfm in self._output.per_frame:
                block = getattr(pfm, challenge)
                for name, val in block.__dict__.items():
                    if val is None:
                        continue
                    if isinstance(val, (int, float)):
                        agg[challenge].setdefault(name, []).append(float(val))
                    elif isinstance(val, dict):
                        for sub, sv in val.items():
                            key = f"{name}.{sub}"
                            agg[challenge].setdefault(key, []).append(float(sv))
            agg[challenge] = {
                k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
                for k, v in agg[challenge].items()
            }
        return agg


def _build_cdf(values: np.ndarray, bins: int = 50) -> List[float]:
    """Return a 50-point empirical CDF of the given (already per-id) values.

    Values are log-spaced indexed so the left tail stays visible on a
    stability-sensitive CDF (most S^(W) mass piles near zero).  Fall back to
    linear indexing when all values are zero.
    """
    if values is None or values.size == 0:
        return [0.0] * bins
    v = np.sort(values.astype(np.float64))
    n = v.size
    idx = np.clip((np.linspace(0.0, 1.0, bins) * (n - 1)).astype(int), 0, n - 1)
    return v[idx].tolist()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(f"cannot serialize {type(o)!r}")
