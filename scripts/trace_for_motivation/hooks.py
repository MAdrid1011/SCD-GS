"""Per-baseline monkey-patch context managers.

Every ``patch_*`` returns a ``contextlib.contextmanager`` that, while active,
installs hooks on the baseline's training / rendering entry points and feeds
the enclosing :class:`MotivationTracer` with per-frame records.

Design rules (hard):

* No baseline source file is edited.  If a hook needs invasive changes we
  degrade the metric for that baseline instead.
* Every patch is symmetric: the context manager *always* restores the
  original function on exit (even on exception).
* Hooks catch and log any exception from their own code — a tracer bug must
  not take the baseline training with it.

The metric math lives in :mod:`metrics`; hooks are pure data collectors.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np

from .core import CrossFrameStateSnapshot, MotivationTracer

logger = logging.getLogger("motivation_trace.hooks")


# ---------------------------------------------------------------------------
# Render-kernel profiler (Challenge 3 BB% / alpha-blend compute share)
# ---------------------------------------------------------------------------


class RenderKernelProfiler:
    """Wraps a rasterizer ``render()`` call with ``torch.profiler`` and
    accumulates per-frame kernel wall-clock into three buckets
    (``preprocess_ms`` / ``render_ms`` / ``misc_ms``).  The breakdown is
    written on every render call to ``tracer._curr.extras``, so downstream
    metric functions pick up the latest cumulative split.
    """

    _BUCKETS = ("preprocess_ms", "render_ms", "misc_ms")

    def __init__(self, tracer: MotivationTracer) -> None:
        self.tracer = tracer
        self._accum: Dict[str, float] = {k: 0.0 for k in self._BUCKETS}
        self._n_calls = 0
        self._disabled = False

    def reset(self) -> None:
        self._accum = {k: 0.0 for k in self._BUCKETS}
        self._n_calls = 0

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        # Always bump the call counter so downstream per-call scaling
        # (``metrics._render_calls``) stays correct even when profiling is
        # disabled mid-frame (the call count and the byte/state counters
        # must agree or sh_load_bandwidth_share under-weights by ~n_calls).
        self._n_calls += 1
        self._publish_counts()
        if self._disabled:
            return fn(*args, **kwargs)
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
        except Exception as exc:
            logger.debug("torch.profiler unavailable; disabling: %s", exc)
            self._disabled = True
            return fn(*args, **kwargs)
        try:
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                out = fn(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as exc:
            logger.debug("torch.profiler wrap failed: %s; disabling", exc)
            self._disabled = True
            return fn(*args, **kwargs)
        try:
            self._accumulate(prof)
        except Exception as exc:
            logger.debug("profiler aggregation failed: %s", exc)
        return out

    def _publish_counts(self) -> None:
        if self.tracer._curr is not None:
            self.tracer._curr.extras["render_kernel_n_calls"] = self._n_calls

    def _accumulate(self, prof: Any) -> None:
        events = prof.key_averages()
        for ev in events:
            name = (getattr(ev, "key", None) or "").lower()
            us = float(getattr(ev, "device_time_total", 0.0) or 0.0)
            if us <= 0.0:
                continue
            ms = us / 1000.0
            if "preprocesscuda" in name:
                self._accum["preprocess_ms"] += ms
            elif "rendercuda" in name or "alphablend" in name:
                self._accum["render_ms"] += ms
            else:
                # Sort, dedup, tile-range build, cub kernels, anything else
                # that ran on the GPU during the render window.
                self._accum["misc_ms"] += ms
        if self.tracer._curr is not None:
            self.tracer._curr.extras["render_kernel_breakdown_ms"] = dict(self._accum)
            self.tracer._curr.extras["render_kernel_n_calls"] = self._n_calls


def _publish_sh_degree(tracer: MotivationTracer, pc: Any) -> None:
    """Stash ``pc.max_sh_degree`` on ``tracer.meta`` the first time we see it.

    ``sh_load_bandwidth_share`` reads this value at ``end_frame`` time to
    convert ``n_rendered`` into SH fp32 bytes.  No-op if already published.
    """
    if "sh_degree" in tracer.meta:
        return
    sh = getattr(pc, "max_sh_degree", None)
    if sh is None:
        sh = getattr(pc, "active_sh_degree", None)
    if sh is None:
        return
    try:
        tracer.meta["sh_degree"] = int(sh)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared helpers for iter-level grad stats + CUDA event timing
# ---------------------------------------------------------------------------

# Cosine-similarity threshold for classifying a Gaussian's iter-N grad as
# directionally stable w.r.t. its iter-(N-1) grad (GLASS.md C1 narrative).
GRAD_COS_STABLE_THRESHOLD = 0.90
# Low-grad classifier: L2 norm below this fraction of the frame's per-iter
# max L2 is treated as "grad below noise floor".
GRAD_LOW_EPS = 1e-3


def _cuda_event():
    """Make a timing-enabled CUDA event, or ``None`` if CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.Event(enable_timing=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cuda event allocation failed: %s", exc)
        return None


def _record(ev: Any) -> None:
    if ev is None:
        return
    try:
        ev.record()
    except Exception as exc:  # pragma: no cover
        logger.debug("event.record() failed: %s", exc)


class IterEventPairer:
    """Tracks one in-flight iter's (render, step) CUDA events.

    Hooks call :meth:`on_render_start/end` around the render callsite and
    :meth:`on_step_start/end` around ``optimizer.step``; once all four are
    present the pairer flushes a single iter onto
    :meth:`MotivationTracer.push_iter_events`.  Events are kept un-queried
    so the core can batch-sync them at ``end_frame`` time.
    """

    def __init__(self, tracer: MotivationTracer) -> None:
        self.tracer = tracer
        self._pending: Dict[str, Any] = {}

    def reset(self) -> None:
        self._pending = {}

    def on_render_start(self) -> None:
        ev = _cuda_event()
        _record(ev)
        self._pending["render_start"] = ev

    def on_render_end(self) -> None:
        ev = _cuda_event()
        _record(ev)
        self._pending["render_end"] = ev

    def on_step_start(self) -> None:
        ev = _cuda_event()
        _record(ev)
        self._pending["step_start"] = ev

    def on_step_end(self) -> None:
        ev = _cuda_event()
        _record(ev)
        self._pending["step_end"] = ev
        self._flush_if_complete()

    def _flush_if_complete(self) -> None:
        p = self._pending
        if all(k in p and p[k] is not None for k in (
            "render_start", "render_end", "step_start", "step_end",
        )):
            self.tracer.push_iter_events(
                render_start=p["render_start"],
                render_end=p["render_end"],
                step_start=p["step_start"],
                step_end=p["step_end"],
            )
        self._pending = {}


class GradStatsCache:
    """Per-optimizer cache of the previous iter's grad matrix (on GPU).

    Produces per-iter ``(n_stable_cos, n_active, n_low_grad, n_active_low)``
    counts for :func:`metrics.iter_grad_cosine_stable_fraction` and
    :func:`metrics.low_grad_fraction`.  Cache is keyed by ``id(optimizer)``
    so S1 -> S2 stage changes (which create fresh optimizer objects with
    different param shapes) naturally discard the stale prev-iter snapshot.
    """

    def __init__(self) -> None:
        self._prev: Dict[int, Any] = {}

    def reset(self) -> None:
        self._prev = {}

    def collect(self, optimizer: Any) -> Optional[Tuple[int, int, int, int]]:
        try:
            import torch
        except Exception:
            return None
        rows = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.numel() == 0:
                    # ``_added_*`` tensors can legitimately be shape (0, *)
                    # before densification fires; a ``reshape`` with an
                    # unresolvable -1 would raise otherwise.
                    continue
                if g.ndim > 1:
                    g = g.reshape(g.shape[0], -1)
                else:
                    g = g.reshape(-1, 1)
                rows.append(g.to(dtype=torch.float32))
        if not rows:
            return None
        n = min(r.shape[0] for r in rows)
        if n <= 0:
            return None
        mat = torch.cat([r[:n] for r in rows], dim=1).contiguous()
        l2 = torch.linalg.norm(mat, dim=1)
        active = l2 > 0.0
        n_active = int(active.sum().item())
        if n_active <= 0:
            # Nothing to compare; still refresh cache so the next iter sees
            # an empty prev and reports zero stable.
            self._prev[id(optimizer)] = mat.detach()
            return (0, 0, 0, 0)
        max_l2 = float(l2.max().item())
        threshold = GRAD_LOW_EPS * max_l2 if max_l2 > 0.0 else 0.0
        if threshold > 0.0:
            n_low = int(((l2 < threshold) & active).sum().item())
        else:
            n_low = 0
        prev = self._prev.get(id(optimizer))
        n_stable = 0
        if prev is not None and prev.shape == mat.shape:
            prev_l2 = torch.linalg.norm(prev, dim=1)
            both = active & (prev_l2 > 0.0)
            if bool(both.any().item()):
                dp = (prev * mat).sum(dim=1)
                denom = torch.clamp(prev_l2 * l2, min=1e-12)
                cos = dp / denom
                n_stable = int(((cos >= GRAD_COS_STABLE_THRESHOLD) & both).sum().item())
        self._prev[id(optimizer)] = mat.detach()
        return (n_stable, n_active, n_low, n_active)


def _append_grad_stats(tracer: MotivationTracer, stats: Optional[Tuple[int, int, int, int]]) -> None:
    if stats is None or tracer._curr is None:
        return
    n_stable, n_active_cos, n_low, n_active_low = stats
    tracer._curr.iter_grad_cos_counts.append((int(n_stable), int(n_active_cos)))
    tracer._curr.iter_low_grad_counts.append((int(n_low), int(n_active_low)))


class StateDeltaCache:
    """Per-Gaussian Δstate stability cache (replaces the grad-based cache).

    Each call to :meth:`collect` reads the render-facing live state
    (``get_xyz / get_rotation / get_scaling / get_opacity``; falls back to
    the raw ``_xyz / _rotation / ...`` parameters) and compares with the
    previous iter's snapshot.  Features (SH) are intentionally skipped to
    keep per-iter GPU residency under 100 MB even at QUEEN's 2 M Gaussians.

    Returns ``(n_stable_cos, n_cos_denom, n_tail, n_total)`` suitable for
    :func:`_append_grad_stats`:

    * ``low`` counts (``n_tail / n_total``): *energy-tail* fraction — rows
      that together hold only ``TAIL_EPS`` (default 1%) of the iter's total
      Δ L2² energy.  This is the universal "skippable" signal: for
      HiCoM most rows have Δ exactly 0 and trivially land in the tail;
      for NTC / Fourier baselines where the prediction is dense, scene
      motion concentration still leaves a long low-energy tail that
      contributes negligibly to training progress.  Denominator is all
      rows.
    * ``cos`` counts (``n_stable / n_cos_denom``): cosine stability
      *restricted to the top-energy subset* — rows that are **outside**
      the tail both this iter and last iter.  This directly answers "of
      the Gaussians whose updates are large enough to matter, what
      fraction keeps the same direction iter-to-iter".

    Rationale: hardware-side UCN v2 can consolidate a Gaussian update iff
    the *resulting live state change* for that Gaussian is either already
    negligible or aligned with the previous iter.  Measuring ``.grad``
    directly misses two important cases:

    1. optimizers with per-param LR scaling or on-the-fly quantization
       (QUEEN) that mask gradients stochastically across iters, producing
       iter-to-iter orthogonal ``.grad`` supports while the commited Δstate
       is still temporally coherent;
    2. stage-1 update paths that are parameterized *outside* the Gaussian
       model (3DGStream NTC, 4DGC Fourier motion field), so no per-Gaussian
       ``.grad`` exists at all, but the NTC's per-Gaussian *output*
       (Δxyz / Δrot) is exactly the signal UCN cares about.

    Reading the render-facing state at each iter captures both cases
    uniformly.
    """

    _ATTR_GETTERS = ("get_xyz", "get_rotation", "get_scaling", "get_opacity")
    _ATTR_FALLBACK = ("_xyz", "_rotation", "_scaling", "_opacity")
    # keep 99% of Δ-energy in the "top" subset; the complementary tail is
    # the bottom-1%-energy subset considered effectively zero-contribution.
    TAIL_EPS = 0.01

    def __init__(self) -> None:
        self._prev_state = None  # torch.Tensor (N, D) or None
        self._prev_delta = None  # torch.Tensor (N, D) or None
        self._prev_top_mask = None  # torch.BoolTensor (N,) or None

    def reset(self) -> None:
        self._prev_state = None
        self._prev_delta = None
        self._prev_top_mask = None

    def _snapshot(self, gaussians: Any):
        try:
            import torch
        except Exception:
            return None
        parts = []
        for getter, fallback in zip(self._ATTR_GETTERS, self._ATTR_FALLBACK):
            val = getattr(gaussians, getter, None)
            if val is None:
                val = getattr(gaussians, fallback, None)
            if val is None or not hasattr(val, "detach"):
                continue
            t = val.detach()
            if t.numel() == 0:
                continue
            if t.ndim > 1:
                t = t.reshape(t.shape[0], -1)
            else:
                t = t.reshape(-1, 1)
            parts.append(t.to(dtype=torch.float32))
        if not parts:
            return None
        n = min(p.shape[0] for p in parts)
        if n <= 0:
            return None
        return torch.cat([p[:n] for p in parts], dim=1).contiguous()

    def collect(self, gaussians: Any) -> Optional[Tuple[int, int, int, int]]:
        try:
            import torch
        except Exception:
            return None
        mat = self._snapshot(gaussians)
        if mat is None:
            return None

        prev_state = self._prev_state
        prev_delta = self._prev_delta
        prev_top_mask = self._prev_top_mask

        if prev_state is None or prev_state.shape != mat.shape:
            # Shape mismatch = densification / first iter = seed and skip.
            self._prev_state = mat.detach()
            self._prev_delta = None
            self._prev_top_mask = None
            return None

        delta = mat - prev_state
        l2_sq = (delta * delta).sum(dim=1)
        l2 = torch.sqrt(l2_sq.clamp_min(0.0))
        n_total = int(mat.shape[0])
        total_energy = float(l2_sq.sum().item())

        # Energy-tail partition: sort rows ascending by ||Δ||², then the
        # largest prefix whose cumsum stays below ``TAIL_EPS × total_energy``
        # is the "bottom-ε-energy tail" — Gaussians whose joint contribution
        # to this iter's update energy is <ε.  The complement is the
        # "top" subset that carries ≥(1−ε) of the energy.
        if total_energy > 0.0:
            sorted_sq, sort_idx = torch.sort(l2_sq)
            cumsum = torch.cumsum(sorted_sq, dim=0)
            cutoff = self.TAIL_EPS * total_energy
            n_tail = int((cumsum <= cutoff).sum().item())
            top_mask = torch.ones(n_total, dtype=torch.bool, device=l2.device)
            if n_tail > 0:
                top_mask[sort_idx[:n_tail]] = False
        else:
            # Zero motion this iter → every row is "in the tail".
            n_tail = n_total
            top_mask = torch.zeros(n_total, dtype=torch.bool, device=l2.device)

        # Cos stability restricted to rows that are in the top subset
        # *both* this iter and last iter.  Skipping the tail here is
        # critical: cos on a tail row is numerically unstable (tiny
        # dot-product noise) and semantically irrelevant (the row's
        # update is effectively zero and its direction doesn't matter).
        n_stable = 0
        cos_denom = 0
        if (
            prev_delta is not None
            and prev_top_mask is not None
            and prev_delta.shape == delta.shape
            and prev_top_mask.shape[0] == n_total
        ):
            both_top = top_mask & prev_top_mask
            cos_denom = int(both_top.sum().item())
            if cos_denom > 0:
                a = prev_delta[both_top]
                b = delta[both_top]
                a_l2 = torch.linalg.norm(a, dim=1)
                b_l2 = torch.linalg.norm(b, dim=1)
                dp = (a * b).sum(dim=1)
                denom = torch.clamp(a_l2 * b_l2, min=1e-12)
                cos = dp / denom
                n_stable = int((cos >= GRAD_COS_STABLE_THRESHOLD).sum().item())

        self._prev_state = mat.detach()
        self._prev_delta = delta.detach()
        self._prev_top_mask = top_mask.detach()
        return (n_stable, cos_denom, n_tail, n_total)


# ---------------------------------------------------------------------------
# Tensor -> numpy helpers (lazy imports: torch is only available inside
# baseline envs, which is fine because these helpers are only called from
# within the patched callables, which necessarily run inside torch)
# ---------------------------------------------------------------------------


def _np(t: Any) -> Optional[np.ndarray]:
    try:
        import torch  # noqa: F401

        if t is None:
            return None
        if hasattr(t, "detach"):
            return t.detach().to("cpu", copy=False).numpy()
        if isinstance(t, np.ndarray):
            return t
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("tensor -> numpy coercion failed: %s", exc)
    return None


def _bytes_of(tensors: dict) -> int:
    total = 0
    for v in tensors.values():
        if v is None:
            continue
        if hasattr(v, "element_size") and hasattr(v, "numel"):
            total += int(v.element_size() * v.numel())
        elif isinstance(v, np.ndarray):
            total += int(v.nbytes)
    return total


def _snapshot_gaussian_attrs(gaussians: Any) -> dict[str, np.ndarray]:
    """Pull ``_xyz / _features_dc / _features_rest / _scaling / _rotation /
    _opacity`` off a standard GS GaussianModel, as numpy arrays.  Any field
    the model doesn't expose is simply skipped.
    """
    mapping = {
        "xyz": "_xyz",
        "feat_dc": "_features_dc",
        "feat_rest": "_features_rest",
        "scale": "_scaling",
        "rot": "_rotation",
        "opacity": "_opacity",
    }
    out: dict[str, np.ndarray] = {}
    for key, attr in mapping.items():
        if hasattr(gaussians, attr):
            a = _np(getattr(gaussians, attr))
            if a is not None:
                if a.ndim > 2:
                    a = a.reshape(a.shape[0], -1)
                out[key] = a.astype(np.float32, copy=False)
    return out


def _render_ids_from_output(render_out: dict, n_gaussians: int) -> tuple[
    Optional[np.ndarray], Optional[np.ndarray]
]:
    """Return (render_ids_mask, tile_touches_per_id_proxy).

    ``render_ids`` = ``visibility_filter`` (standard GS key = radii > 0).
    Tile-touch count is not exposed by stock diff-gaussian-rasterization, so
    we fall back to ``radii**2`` as a proxy (touches scale with screen area).
    """
    vis = render_out.get("visibility_filter")
    radii = render_out.get("radii")
    vis_np = _np(vis)
    radii_np = _np(radii)
    tt = None
    if radii_np is not None:
        r = radii_np.astype(np.float32)
        tt = np.clip(r, 0, None) ** 2
    if vis_np is not None and vis_np.size < n_gaussians:
        pad = np.zeros(n_gaussians, dtype=bool)
        pad[: vis_np.size] = vis_np.astype(bool)
        vis_np = pad
    return vis_np, tt


def _contribution_from_render(render_out: dict, n_gaussians: int) -> Optional[np.ndarray]:
    """Best-effort per-id contribution approximation.

    Stock diff-gaussian-rasterization does not return per-id alpha · T
    integrals, so we approximate contribution by ``radii**2 * opacity`` — the
    same two signals salience-first ranking uses at RTL level.  The tracer
    flags this as approximate via the :attr:`FrameRecord.extras`.
    """
    radii = _np(render_out.get("radii"))
    opac = _np(render_out.get("opacity")) if "opacity" in render_out else None
    if radii is None:
        return None
    contrib = np.clip(radii.astype(np.float32), 0, None) ** 2
    if opac is not None and opac.size == contrib.size:
        contrib = contrib * opac.astype(np.float32).reshape(-1)
    if contrib.size < n_gaussians:
        pad = np.zeros(n_gaussians, dtype=np.float32)
        pad[: contrib.size] = contrib
        contrib = pad
    return contrib


# ---------------------------------------------------------------------------
# 3DGStream
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patch_3dgstream(tracer: MotivationTracer) -> Iterator[None]:
    """Wrap 3DGStream's per-frame train loop + GS optimizer + render entry.

    Patches (and restores on exit):

    * ``train_frames.training_one_frame``                       → frame start/end
    * ``train_frames.render`` and ``gaussian_renderer.render``  → render-side
      metrics (render_ids, tile_touches proxy, contribution proxy).  Both names
      must be patched because ``train_frames`` does
      ``from gaussian_renderer import render`` at module-load time, which
      snapshots the reference and would otherwise miss a
      ``renderer_mod.render`` replacement.
    * ``GaussianModel.training_one_frame_setup`` (Stage 1) and
      ``GaussianModel.training_one_frame_s2_setup`` (Stage 2) → install an
      optimizer.step wrapper on the *new* optimizer that each stage creates.
      (``training_setup`` only fires on the very first frame; per-frame setup
      is via these two methods.)
    """
    import importlib

    train_mod = importlib.import_module("train_frames")
    renderer_mod = importlib.import_module("gaussian_renderer")
    gaussian_model_mod = importlib.import_module("scene.gaussian_model")
    GaussianModel = gaussian_model_mod.GaussianModel

    orig_training_one_frame = train_mod.training_one_frame
    orig_render = renderer_mod.render
    orig_query_ntc = getattr(GaussianModel, "query_ntc", None)
    orig_update_by_ntc = getattr(GaussianModel, "update_by_ntc", None)

    installed_step_ids: set[int] = set()
    # Shared per-frame state (reset at training_one_frame entry).  ``row_bytes``
    # caches per-gaussian state footprint so we do not re-walk the attribute
    # list on every S1 iter; ``n_full`` records how many Gaussians exist at
    # the moment ``update_by_ntc`` finished committing (= end-of-S1 size).
    stage_state: dict[str, int] = {"row_bytes": 0, "n_full": 0}
    # C1 delta/timing instrumentation.
    event_pairer = IterEventPairer(tracer)
    state_cache = StateDeltaCache()
    kernel_profiler = RenderKernelProfiler(tracer)

    def _row_bytes_of(g: Any) -> int:
        total = 0
        for name in (
            "_xyz", "_features_dc", "_features_rest",
            "_scaling", "_rotation", "_opacity",
        ):
            a = getattr(g, name, None)
            if a is None or not hasattr(a, "element_size") or not hasattr(a, "shape"):
                continue
            per = a.element_size()
            cols = 1
            if a.ndim > 1:
                for d in a.shape[1:]:
                    cols *= int(d)
            total += per * cols
        return total

    def _install_step_hook(gaussians: Any) -> None:
        if not hasattr(gaussians, "optimizer") or gaussians.optimizer is None:
            return
        # Re-install on every new optimizer identity (S1/S2 create distinct
        # optimizer objects, so identity changes mid-frame).
        opt_id = id(gaussians.optimizer)
        if opt_id in installed_step_ids:
            return
        orig_step = gaussians.optimizer.step

        def wrapped_step(closure=None):
            try:
                update_mask_accum: Optional[np.ndarray] = None
                upd_bytes = 0
                for group in gaussians.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        upd_bytes += int(p.grad.element_size() * p.grad.numel())
                        g_np = _np(p.grad)
                        if g_np is None:
                            continue
                        if g_np.ndim > 1:
                            mask = np.any(g_np.reshape(g_np.shape[0], -1) != 0.0, axis=1)
                        else:
                            mask = g_np != 0.0
                        if update_mask_accum is None:
                            update_mask_accum = mask
                        elif mask.size == update_mask_accum.size:
                            update_mask_accum = update_mask_accum | mask
                if update_mask_accum is not None:
                    # Stage 2 optimizer param groups hold only the ``_added_*``
                    # tensors (size = n_added), while the frame-level
                    # ``update_ids`` accumulator already covers the 280 K base
                    # gaussians that ``update_by_ntc`` committed.  Combine the
                    # two so the final update mask covers *both* the bulk-NTC
                    # commit and the densified-added tail with the correct
                    # sizes.
                    rec = tracer._curr
                    existing: Optional[np.ndarray] = None
                    if rec is not None and rec.update_ids is not None:
                        existing = rec.update_ids.astype(bool)
                    m = update_mask_accum
                    if existing is None:
                        merged = m
                    elif existing.size == m.size:
                        merged = existing | m
                    elif existing.size > m.size:
                        # existing covers full 280 K, m is smaller = added only;
                        # concatenate so merged covers 280 K + n_added.
                        merged = np.concatenate([existing, m], axis=0)
                    else:
                        # rare: existing smaller than incoming full-size mask
                        padded = np.zeros(m.size, dtype=bool)
                        padded[: existing.size] = existing
                        merged = padded | m
                    tracer.record(update_ids=merged)
                tracer.add_bytes(
                    bytes_update_gen=upd_bytes,
                    bytes_commit=upd_bytes,
                    bytes_state_write=upd_bytes,
                )
            except Exception as exc:
                logger.debug("optimizer.step hook failed: %s", exc)
            event_pairer.on_step_start()
            try:
                return orig_step(closure) if closure is not None else orig_step()
            finally:
                event_pairer.on_step_end()

        gaussians.optimizer.step = wrapped_step  # type: ignore[method-assign]
        installed_step_ids.add(opt_id)

    def _install_timing_step_hook(optimizer: Any) -> None:
        """Wrap ``optimizer.step`` for CUDA event timing only.

        Used for optimizers (e.g. 3DGStream/4DGC's ``ntc_optimizer``) whose
        gradients are *not* per-Gaussian and therefore do not feed the
        cosine / low-grad metrics; we still want the iter to contribute to
        the stage-time breakdown so stage-1 shows up in render_backward_ms.
        """
        if optimizer is None:
            return
        opt_id = id(optimizer)
        if opt_id in installed_step_ids:
            return
        orig_step = optimizer.step

        def wrapped(closure=None):
            event_pairer.on_step_start()
            try:
                return orig_step(closure) if closure is not None else orig_step()
            finally:
                event_pairer.on_step_end()

        optimizer.step = wrapped  # type: ignore[method-assign]
        installed_step_ids.add(opt_id)

    # NTC "update-gen" hook — runs once per S1 iter and touches all N base
    # Gaussians (the NTC predicts (Δxyz, Δrot) for each).  Bytes = 7 floats ×
    # N per call.  ``update_gen_ids`` accumulates to "this Gaussian had at
    # least one update generated this frame" (= all-ones after iter 1).
    def wrapped_query_ntc(self):
        ret = orig_query_ntc(self)
        try:
            n = int(self._xyz.shape[0])
            if stage_state["row_bytes"] == 0:
                stage_state["row_bytes"] = _row_bytes_of(self)
            # Per-iter update-gen payload: Δxyz (3f32) + Δrot (4f32) = 28 B/id.
            gen_bytes = 7 * 4 * n
            # NTC input read: _xyz (12 B/id) per iter.
            read_bytes = 12 * n
            tracer.add_bytes(
                bytes_update_gen=gen_bytes,
                bytes_state_read=read_bytes,
            )
            rec = tracer._curr
            ones = np.ones(n, dtype=bool)
            if rec is not None and rec.update_gen_ids is not None and \
               rec.update_gen_ids.size == n:
                ones = ones | rec.update_gen_ids.astype(bool)
            tracer.record(update_gen_ids=ones)
        except Exception as exc:
            logger.debug("query_ntc hook failed: %s", exc)
        return ret

    # NTC "commit" hook — runs once per frame, bulk-writes the NTC prediction
    # back into _xyz / _rotation / etc. for all N base Gaussians.  This is
    # also the point at which the baseline *does* modify every id, so we
    # register ``update_ids`` = ones(N) here (S2's optimizer.step hook later
    # merges in the per-added mask).
    def wrapped_update_by_ntc(self):
        try:
            n = int(self._xyz.shape[0])
            if stage_state["row_bytes"] == 0:
                stage_state["row_bytes"] = _row_bytes_of(self)
            row_bytes = stage_state["row_bytes"]
            stage_state["n_full"] = n
        except Exception:
            n = 0
            row_bytes = 0
        ret = orig_update_by_ntc(self)
        try:
            if n > 0:
                commit_mask = np.ones(n, dtype=bool)
                tracer.record(commit_ids=commit_mask, update_ids=commit_mask.copy())
                tracer.add_bytes(
                    bytes_commit=row_bytes * n,
                    bytes_state_write=row_bytes * n,
                )
        except Exception as exc:
            logger.debug("update_by_ntc hook failed: %s", exc)
        return ret

    def wrapped_render(viewpoint_camera, pc, pipe, bg_color, *args, **kwargs):
        # Snapshot render-facing live state *before* starting the render
        # timer: this is the state the rasterizer is about to consume and
        # whose Δ-vs-last-iter drives UCN's skip decision.
        try:
            _append_grad_stats(tracer, state_cache.collect(pc))
        except Exception as exc:
            logger.debug("3dgs state-delta snapshot failed: %s", exc)
        _publish_sh_degree(tracer, pc)
        event_pairer.on_render_start()
        render_out = kernel_profiler.run(
            orig_render, viewpoint_camera, pc, pipe, bg_color, *args, **kwargs
        )
        event_pairer.on_render_end()
        try:
            # ``get_xyz`` returns _new_xyz (size N) during S1 and
            # cat(_xyz, _added_xyz) (size N + n_added) during S2; the
            # rasterizer's radii / visibility tensors match that size, so we
            # use it as the canonical "n" for render-side masks.
            n_base = int(getattr(pc, "_xyz").shape[0])
            n = n_base
            added = getattr(pc, "_added_xyz", None)
            if added is not None and hasattr(added, "shape"):
                n += int(added.shape[0])
            vis_mask, tt = _render_ids_from_output(render_out, n)
            contrib_input = {"radii": render_out.get("radii")}
            if hasattr(pc, "get_opacity"):
                contrib_input["opacity"] = pc.get_opacity
            contrib = _contribution_from_render(contrib_input, n)
            tracer.record(
                n_gaussians=n,
                render_ids=vis_mask,
                tile_touches_per_id=tt,
                contribution_per_id=contrib,
            )
            if stage_state["row_bytes"] == 0:
                stage_state["row_bytes"] = _row_bytes_of(pc)
            tracer.add_bytes(bytes_state_read=stage_state["row_bytes"] * n)
        except Exception as exc:
            logger.debug("render hook failed: %s", exc)
        return render_out

    frame_counter = {"i": 0}
    # Stash the gaussians instance that S1 setup saw, so we can snapshot
    # ``attr_after`` on the same object at frame-end (S2 setup creates its own
    # ``_added_*`` tensors but the full state still lives on the same model).
    live_gaussians: dict[str, Any] = {"ref": None}

    def wrapped_training_one_frame(dataset, opt, pipe, *a, **kw):
        frame_counter["i"] += 1
        live_gaussians["ref"] = None
        stage_state["row_bytes"] = 0
        stage_state["n_full"] = 0
        event_pairer.reset()
        state_cache.reset()
        kernel_profiler.reset()
        tracer.start_frame(frame_counter["i"], n_gaussians=0)
        try:
            return orig_training_one_frame(dataset, opt, pipe, *a, **kw)
        finally:
            g = live_gaussians["ref"]
            if g is not None:
                try:
                    attrs = _snapshot_gaussian_attrs(g)
                    if attrs:
                        tracer.record(
                            attr_after=attrs,
                            n_gaussians=int(attrs[next(iter(attrs))].shape[0]),
                        )
                except Exception as exc:
                    logger.debug("attr_after snapshot failed: %s", exc)
            tracer.end_frame()

    orig_s1 = getattr(GaussianModel, "training_one_frame_setup", None)
    orig_s2 = getattr(GaussianModel, "training_one_frame_s2_setup", None)

    def wrapped_s1(self, *a, **kw):
        ret = orig_s1(self, *a, **kw)
        try:
            live_gaussians["ref"] = self
            _install_step_hook(self)
            _install_timing_step_hook(getattr(self, "ntc_optimizer", None))
            attrs = _snapshot_gaussian_attrs(self)
            if attrs:
                tracer.record(
                    attr_before=attrs,
                    n_gaussians=int(attrs[next(iter(attrs))].shape[0]),
                )
        except Exception as exc:
            logger.debug("s1 setup hook failed: %s", exc)
        return ret

    def wrapped_s2(self, *a, **kw):
        ret = orig_s2(self, *a, **kw)
        try:
            _install_step_hook(self)
        except Exception as exc:
            logger.debug("s2 setup hook failed: %s", exc)
        return ret

    if orig_s1 is not None:
        GaussianModel.training_one_frame_setup = wrapped_s1  # type: ignore[method-assign]
    if orig_s2 is not None:
        GaussianModel.training_one_frame_s2_setup = wrapped_s2  # type: ignore[method-assign]
    if orig_query_ntc is not None:
        GaussianModel.query_ntc = wrapped_query_ntc  # type: ignore[method-assign]
    if orig_update_by_ntc is not None:
        GaussianModel.update_by_ntc = wrapped_update_by_ntc  # type: ignore[method-assign]

    train_mod.training_one_frame = wrapped_training_one_frame
    renderer_mod.render = wrapped_render
    if hasattr(train_mod, "render"):
        train_mod.render = wrapped_render  # rebind the name resolved at import

    try:
        yield
    finally:
        train_mod.training_one_frame = orig_training_one_frame
        renderer_mod.render = orig_render
        if hasattr(train_mod, "render"):
            train_mod.render = orig_render
        if orig_s1 is not None:
            GaussianModel.training_one_frame_setup = orig_s1  # type: ignore[method-assign]
        if orig_s2 is not None:
            GaussianModel.training_one_frame_s2_setup = orig_s2  # type: ignore[method-assign]
        if orig_query_ntc is not None:
            GaussianModel.query_ntc = orig_query_ntc  # type: ignore[method-assign]
        if orig_update_by_ntc is not None:
            GaussianModel.update_by_ntc = orig_update_by_ntc  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# HiCoM — the cleanest inter-stage split (add_stage / update_stage)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patch_hicom(tracer: MotivationTracer) -> Iterator[None]:
    """Wrap HiCoM's per-frame ``Trainer.frame_training_loop`` plus enough
    inner hooks that all 11 motivation metrics populate.

    HiCoM's stage split is cleaner than 3DGStream's:

    * **Stage 1** (`deform` MLP): every step calls ``self.deform(get_xyz_ori)``
      which produces ``delta_xyz, delta_rot`` for every base gaussian.  That
      is the "update-gen" work.
    * **Commit**: one call to ``GaussianModel.apply_deformation`` at the end of
      Stage 1 bulk-writes ``_xyz`` and ``_rotation`` for every base gaussian.
    * **Stage 2** (``gaussians_incr``): new add-only gaussians are trained by
      ``gaussians_incr.optimizer.step`` — same shape as 3DGStream's S2.

    We hook all three plus the training_step's ``render_pkg`` return so the
    render-side metrics get their (render_ids, tile_touches_per_id,
    contribution_per_id) data.
    """
    import importlib

    trainer_mod = importlib.import_module("pipeline.hicom.trainer")
    module_mod = importlib.import_module("pipeline.hicom.module")
    gaussian_mod = importlib.import_module("pipeline.hicom.model.gaussian")
    try:
        libgs_renderer_mod = importlib.import_module("libgs.renderer")
        LibgsRenderer = getattr(libgs_renderer_mod, "Renderer", None)
    except Exception:
        LibgsRenderer = None

    Trainer = trainer_mod.Trainer
    Module = module_mod.Module
    GaussianModel = gaussian_mod.GaussianModel

    orig_frame_loop = Trainer.frame_training_loop
    orig_training_step = Module.training_step
    orig_post = getattr(Module, "post_training_step", None)
    orig_apply_deformation = getattr(GaussianModel, "apply_deformation", None)
    orig_renderer_forward = (
        getattr(LibgsRenderer, "forward", None) if LibgsRenderer is not None else None
    )

    stage_state: dict[str, int] = {"row_bytes": 0}
    installed_step_ids: set[int] = set()
    event_pairer = IterEventPairer(tracer)
    state_cache = StateDeltaCache()
    kernel_profiler = RenderKernelProfiler(tracer)

    def _row_bytes_of(g: Any) -> int:
        total = 0
        for name in (
            "_xyz", "_features_dc", "_features_rest",
            "_scaling", "_rotation", "_opacity",
        ):
            a = getattr(g, name, None)
            if a is None or not hasattr(a, "element_size") or not hasattr(a, "shape"):
                continue
            per = a.element_size()
            cols = 1
            if a.ndim > 1:
                for d in a.shape[1:]:
                    cols *= int(d)
            total += per * cols
        return total

    def _install_step_hook(gaussians: Any) -> None:
        if not hasattr(gaussians, "optimizer") or gaussians.optimizer is None:
            return
        opt_id = id(gaussians.optimizer)
        if opt_id in installed_step_ids:
            return
        orig_step = gaussians.optimizer.step

        def wrapped_step(closure=None):
            try:
                mask_accum: Optional[np.ndarray] = None
                upd_bytes = 0
                for group in gaussians.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        upd_bytes += int(p.grad.element_size() * p.grad.numel())
                        g_np = _np(p.grad)
                        if g_np is None:
                            continue
                        if g_np.ndim > 1:
                            m = np.any(g_np.reshape(g_np.shape[0], -1) != 0.0, axis=1)
                        else:
                            m = g_np != 0.0
                        if mask_accum is None:
                            mask_accum = m
                        elif m.size == mask_accum.size:
                            mask_accum = mask_accum | m
                if mask_accum is not None:
                    rec = tracer._curr
                    existing: Optional[np.ndarray] = None
                    if rec is not None and rec.update_ids is not None:
                        existing = rec.update_ids.astype(bool)
                    if existing is None:
                        merged = mask_accum
                    elif existing.size == mask_accum.size:
                        merged = existing | mask_accum
                    elif existing.size > mask_accum.size:
                        merged = np.concatenate([existing, mask_accum], axis=0)
                    else:
                        padded = np.zeros(mask_accum.size, dtype=bool)
                        padded[: existing.size] = existing
                        merged = padded | mask_accum
                    tracer.record(update_ids=merged)
                tracer.add_bytes(
                    bytes_update_gen=upd_bytes,
                    bytes_commit=upd_bytes,
                    bytes_state_write=upd_bytes,
                )
            except Exception as exc:
                logger.debug("hicom optimizer.step hook failed: %s", exc)
            return orig_step(closure) if closure is not None else orig_step()

        gaussians.optimizer.step = wrapped_step  # type: ignore[method-assign]
        installed_step_ids.add(opt_id)

    def wrapped_frame_loop(self, *a, **kw):
        module = getattr(self, "module", None)
        base_gaussians = getattr(module, "gaussians", None) if module is not None else None
        stage_state["row_bytes"] = 0
        event_pairer.reset()
        state_cache.reset()
        kernel_profiler.reset()
        # HiCoM frames are 0-indexed internally; report 1-based to match
        # other baselines.
        tracer.start_frame(
            frame_idx=int(getattr(self, "current_frame", 0)) + 1,
            n_gaussians=int(base_gaussians._xyz.shape[0]) if base_gaussians is not None and hasattr(base_gaussians, "_xyz") else 0,
        )
        try:
            if base_gaussians is not None:
                attrs_before = _snapshot_gaussian_attrs(base_gaussians)
                if attrs_before:
                    tracer.record(attr_before=attrs_before)
                _install_step_hook(base_gaussians)
            # The incr optimizer is created mid-frame by module.pre_training_step
            # (gaussians_incr.training_setup); we'll re-check after the loop runs
            # as well via the optimizer identity guard inside _install_step_hook.
        except Exception as exc:
            logger.debug("hicom attr_before snapshot failed: %s", exc)

        try:
            return orig_frame_loop(self, *a, **kw)
        finally:
            try:
                # Also try to install the hook on the incr optimizer if it
                # ended up being created during the frame.
                incr = getattr(module, "gaussians_incr", None) if module is not None else None
                if incr is not None:
                    _install_step_hook(incr)
            except Exception:
                pass
            try:
                if base_gaussians is not None:
                    attrs_after = _snapshot_gaussian_attrs(base_gaussians)
                    if attrs_after:
                        tracer.record(
                            attr_after=attrs_after,
                            n_gaussians=int(attrs_after[next(iter(attrs_after))].shape[0]),
                        )
            except Exception as exc:
                logger.debug("hicom attr_after snapshot failed: %s", exc)
            tracer.end_frame()

    def wrapped_training_step(self, *a, **kw):
        # Snapshot render-facing live state *before* render kicks off:
        # HiCoM's active render gaussians live on ``self.renderer.gaussians``
        # (or fall back to ``self.gaussians`` in stage 1).
        try:
            renderer = getattr(self, "renderer", None)
            snap_g = getattr(renderer, "gaussians", None) if renderer is not None else None
            if snap_g is None:
                snap_g = getattr(self, "gaussians", None)
            if snap_g is not None:
                _append_grad_stats(tracer, state_cache.collect(snap_g))
        except Exception as exc:
            logger.debug("hicom state-delta snapshot failed: %s", exc)
        event_pairer.on_render_start()
        try:
            out = orig_training_step(self, *a, **kw)
        finally:
            event_pairer.on_render_end()
        try:
            # HiCoM's training_step returns (loss, metrics, render_pkg).
            render_pkg = None
            if isinstance(out, tuple) and len(out) >= 3 and isinstance(out[2], dict):
                render_pkg = out[2]
            elif isinstance(out, dict):
                render_pkg = out
            if render_pkg is not None:
                renderer = getattr(self, "renderer", None)
                rgauss = getattr(renderer, "gaussians", None) if renderer is not None else None
                n_render = 0
                if rgauss is not None and hasattr(rgauss, "get_xyz"):
                    try:
                        n_render = int(rgauss.get_xyz.shape[0])
                    except Exception:
                        pass
                if n_render == 0 and "radii" in render_pkg and hasattr(render_pkg["radii"], "shape"):
                    n_render = int(render_pkg["radii"].shape[0])
                vis, tt = _render_ids_from_output(render_pkg, n_render)
                contrib_in = {"radii": render_pkg.get("radii")}
                if rgauss is not None and hasattr(rgauss, "get_opacity"):
                    contrib_in["opacity"] = rgauss.get_opacity
                contrib = _contribution_from_render(contrib_in, n_render)
                tracer.record(
                    n_gaussians=n_render,
                    render_ids=vis,
                    tile_touches_per_id=tt,
                    contribution_per_id=contrib,
                )
                # State-read bytes — the render path pulls a full row_bytes × N
                # block off HBM (approximation: ignore cache reuse).
                base_g = getattr(getattr(self, "gaussians", None), "_xyz", None)
                if stage_state["row_bytes"] == 0 and hasattr(self, "gaussians"):
                    stage_state["row_bytes"] = _row_bytes_of(self.gaussians)
                if stage_state["row_bytes"] and n_render:
                    tracer.add_bytes(bytes_state_read=stage_state["row_bytes"] * n_render)

            # Deform-side "update-gen" bytes: HiCoM's deform MLP outputs
            # delta_xyz (3f) + delta_rot (4f) per base gaussian every iter
            # before step = num_stage1_steps.  render_pkg carries them when
            # deform was active this iter.
            if render_pkg is not None and "delta_xyz" in render_pkg:
                dxyz = render_pkg.get("delta_xyz")
                drot = render_pkg.get("delta_rotation")
                n_gen = 0
                gen_bytes = 0
                for t in (dxyz, drot):
                    if t is not None and hasattr(t, "element_size") and hasattr(t, "numel"):
                        gen_bytes += int(t.element_size() * t.numel())
                        if hasattr(t, "shape"):
                            n_gen = max(n_gen, int(t.shape[0]))
                if gen_bytes > 0:
                    tracer.add_bytes(bytes_update_gen=gen_bytes)
                if n_gen > 0:
                    ones = np.ones(n_gen, dtype=bool)
                    rec = tracer._curr
                    if rec is not None and rec.update_gen_ids is not None and rec.update_gen_ids.size == n_gen:
                        ones = ones | rec.update_gen_ids.astype(bool)
                    tracer.record(update_gen_ids=ones)
        except Exception as exc:
            logger.debug("hicom training_step hook failed: %s", exc)
        return out

    def wrapped_post_training_step(self, render_pkg):
        # Δstate snapshotting now happens at render_start in
        # wrapped_training_step; this wrapper only carries step timing.
        event_pairer.on_step_start()
        try:
            return orig_post(self, render_pkg)
        finally:
            event_pairer.on_step_end()

    def wrapped_apply_deformation(self, delta_xyz, delta_rotation):
        # Bulk commit point: writes _xyz and _rotation for all N base gaussians.
        try:
            n = int(self._xyz.shape[0]) if hasattr(self, "_xyz") else 0
            if stage_state["row_bytes"] == 0:
                stage_state["row_bytes"] = _row_bytes_of(self)
        except Exception:
            n = 0
        ret = orig_apply_deformation(self, delta_xyz, delta_rotation)
        try:
            if n > 0:
                commit = np.ones(n, dtype=bool)
                tracer.record(commit_ids=commit, update_ids=commit.copy())
                per_id = 7 * 4  # 3 floats xyz + 4 floats rot
                tracer.add_bytes(
                    bytes_commit=per_id * n,
                    bytes_state_write=per_id * n,
                )
        except Exception as exc:
            logger.debug("hicom apply_deformation hook failed: %s", exc)
        return ret

    def wrapped_renderer_forward(self, *a, **kw):
        try:
            _publish_sh_degree(tracer, getattr(self, "gaussians", None))
        except Exception:
            pass
        return kernel_profiler.run(orig_renderer_forward, self, *a, **kw)

    Trainer.frame_training_loop = wrapped_frame_loop  # type: ignore[method-assign]
    Module.training_step = wrapped_training_step  # type: ignore[method-assign]
    if orig_post is not None:
        Module.post_training_step = wrapped_post_training_step  # type: ignore[method-assign]
    if orig_apply_deformation is not None:
        GaussianModel.apply_deformation = wrapped_apply_deformation  # type: ignore[method-assign]
    if LibgsRenderer is not None and orig_renderer_forward is not None:
        LibgsRenderer.forward = wrapped_renderer_forward  # type: ignore[method-assign]

    try:
        yield
    finally:
        Trainer.frame_training_loop = orig_frame_loop  # type: ignore[method-assign]
        Module.training_step = orig_training_step  # type: ignore[method-assign]
        if orig_post is not None:
            Module.post_training_step = orig_post  # type: ignore[method-assign]
        if orig_apply_deformation is not None:
            GaussianModel.apply_deformation = orig_apply_deformation  # type: ignore[method-assign]
        if LibgsRenderer is not None and orig_renderer_forward is not None:
            LibgsRenderer.forward = orig_renderer_forward  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 4DGC — same as 3DGStream plus the EntropyBottleneck path
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patch_4dgc(tracer: MotivationTracer) -> Iterator[None]:
    """Wrap 4DGC's per-frame train loop + GS optimizer + render entry.

    Same structure as :func:`patch_3dgstream` (4DGC inherits 3DGStream's
    two-stage per-frame setup), plus an optional EntropyBottleneck hook for
    the MEM compression forward bytes.
    """
    import importlib

    train_mod = importlib.import_module("train_frames")
    renderer_mod = importlib.import_module("gaussian_renderer")
    gaussian_model_mod = importlib.import_module("scene.gaussian_model")
    GaussianModel = gaussian_model_mod.GaussianModel

    orig_training_one_frame = train_mod.training_one_frame
    orig_render = renderer_mod.render

    installed_step_ids: set[int] = set()
    event_pairer = IterEventPairer(tracer)
    state_cache = StateDeltaCache()
    kernel_profiler = RenderKernelProfiler(tracer)

    def _install_step_hook(gaussians: Any) -> None:
        if not hasattr(gaussians, "optimizer") or gaussians.optimizer is None:
            return
        opt_id = id(gaussians.optimizer)
        if opt_id in installed_step_ids:
            return
        orig_step = gaussians.optimizer.step

        def wrapped_step(closure=None):
            try:
                update_mask_accum: Optional[np.ndarray] = None
                upd_bytes = 0
                for group in gaussians.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        upd_bytes += int(p.grad.element_size() * p.grad.numel())
                        g_np = _np(p.grad)
                        if g_np is None:
                            continue
                        if g_np.ndim > 1:
                            mask = np.any(g_np.reshape(g_np.shape[0], -1) != 0.0, axis=1)
                        else:
                            mask = g_np != 0.0
                        if update_mask_accum is None:
                            update_mask_accum = mask
                        elif mask.size == update_mask_accum.size:
                            update_mask_accum = update_mask_accum | mask
                if update_mask_accum is not None:
                    rec = tracer._curr
                    if rec is not None and rec.update_ids is not None and \
                       rec.update_ids.size == update_mask_accum.size:
                        update_mask_accum = update_mask_accum | rec.update_ids.astype(bool)
                    tracer.record(update_ids=update_mask_accum)
                tracer.add_bytes(
                    bytes_update_gen=upd_bytes,
                    bytes_commit=upd_bytes,
                    bytes_state_write=upd_bytes,
                )
            except Exception as exc:
                logger.debug("4dgc optimizer.step hook failed: %s", exc)
            event_pairer.on_step_start()
            try:
                return orig_step(closure) if closure is not None else orig_step()
            finally:
                event_pairer.on_step_end()

        gaussians.optimizer.step = wrapped_step  # type: ignore[method-assign]
        installed_step_ids.add(opt_id)

    def _install_timing_step_hook(optimizer: Any) -> None:
        if optimizer is None:
            return
        opt_id = id(optimizer)
        if opt_id in installed_step_ids:
            return
        orig_step = optimizer.step

        def wrapped(closure=None):
            event_pairer.on_step_start()
            try:
                return orig_step(closure) if closure is not None else orig_step()
            finally:
                event_pairer.on_step_end()

        optimizer.step = wrapped  # type: ignore[method-assign]
        installed_step_ids.add(opt_id)

    def wrapped_render(viewpoint_camera, pc, pipe, bg_color, *a, **kw):
        try:
            _append_grad_stats(tracer, state_cache.collect(pc))
        except Exception as exc:
            logger.debug("4dgc state-delta snapshot failed: %s", exc)
        _publish_sh_degree(tracer, pc)
        event_pairer.on_render_start()
        out = kernel_profiler.run(
            orig_render, viewpoint_camera, pc, pipe, bg_color, *a, **kw
        )
        event_pairer.on_render_end()
        try:
            if pc is not None and hasattr(pc, "_xyz"):
                n = int(pc._xyz.shape[0])
                vis, tt = _render_ids_from_output(out, n)
                contrib_in = {"radii": out.get("radii")}
                if hasattr(pc, "get_opacity"):
                    contrib_in["opacity"] = pc.get_opacity
                contrib = _contribution_from_render(contrib_in, n)
                tracer.record(
                    n_gaussians=n,
                    render_ids=vis,
                    tile_touches_per_id=tt,
                    contribution_per_id=contrib,
                )
                row_bytes = 0
                for name in ("_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"):
                    a_ = getattr(pc, name, None)
                    if a_ is not None and hasattr(a_, "element_size") and hasattr(a_, "shape"):
                        row_bytes += int(a_.element_size() * (a_.shape[-1] if a_.ndim > 1 else 1))
                tracer.add_bytes(bytes_state_read=row_bytes * n)
        except Exception as exc:
            logger.debug("4dgc render hook failed: %s", exc)
        return out

    frame_counter = {"i": 0}
    live_gaussians: dict[str, Any] = {"ref": None}

    def wrapped_training(*a, **kw):
        frame_counter["i"] += 1
        live_gaussians["ref"] = None
        event_pairer.reset()
        state_cache.reset()
        kernel_profiler.reset()
        tracer.start_frame(frame_counter["i"])
        try:
            return orig_training_one_frame(*a, **kw)
        finally:
            g = live_gaussians["ref"]
            if g is not None:
                try:
                    attrs = _snapshot_gaussian_attrs(g)
                    if attrs:
                        tracer.record(
                            attr_after=attrs,
                            n_gaussians=int(attrs[next(iter(attrs))].shape[0]),
                        )
                except Exception as exc:
                    logger.debug("4dgc attr_after snapshot failed: %s", exc)
            tracer.end_frame()

    orig_s1 = getattr(GaussianModel, "training_one_frame_setup", None)
    orig_s2 = getattr(GaussianModel, "training_one_frame_s2_setup", None)

    def wrapped_s1(self, *a, **kw):
        ret = orig_s1(self, *a, **kw)
        try:
            live_gaussians["ref"] = self
            _install_step_hook(self)
            _install_timing_step_hook(getattr(self, "mem_optimizer", None))
            attrs = _snapshot_gaussian_attrs(self)
            if attrs:
                tracer.record(
                    attr_before=attrs,
                    n_gaussians=int(attrs[next(iter(attrs))].shape[0]),
                )
        except Exception as exc:
            logger.debug("4dgc s1 setup hook failed: %s", exc)
        return ret

    def wrapped_s2(self, *a, **kw):
        ret = orig_s2(self, *a, **kw)
        try:
            _install_step_hook(self)
        except Exception as exc:
            logger.debug("4dgc s2 setup hook failed: %s", exc)
        return ret

    if orig_s1 is not None:
        GaussianModel.training_one_frame_setup = wrapped_s1  # type: ignore[method-assign]
    if orig_s2 is not None:
        GaussianModel.training_one_frame_s2_setup = wrapped_s2  # type: ignore[method-assign]

    train_mod.training_one_frame = wrapped_training
    renderer_mod.render = wrapped_render
    if hasattr(train_mod, "render"):
        train_mod.render = wrapped_render

    try:
        eb_mod = importlib.import_module("utils.entropy_models")
        EntropyBottleneck = getattr(eb_mod, "EntropyBottleneck", None)
    except Exception:
        EntropyBottleneck = None

    orig_eb_forward = None
    if EntropyBottleneck is not None and hasattr(EntropyBottleneck, "forward"):
        orig_eb_forward = EntropyBottleneck.forward

        def wrapped_eb(self, x, *a, **kw):
            try:
                if hasattr(x, "element_size"):
                    tracer.add_bytes(bytes_update_gen=int(x.element_size() * x.numel()))
            except Exception:
                pass
            return orig_eb_forward(self, x, *a, **kw)

        EntropyBottleneck.forward = wrapped_eb  # type: ignore[method-assign]

    try:
        yield
    finally:
        train_mod.training_one_frame = orig_training_one_frame
        renderer_mod.render = orig_render
        if hasattr(train_mod, "render"):
            train_mod.render = orig_render
        if orig_s1 is not None:
            GaussianModel.training_one_frame_setup = orig_s1  # type: ignore[method-assign]
        if orig_s2 is not None:
            GaussianModel.training_one_frame_s2_setup = orig_s2  # type: ignore[method-assign]
        if EntropyBottleneck is not None and orig_eb_forward is not None:
            EntropyBottleneck.forward = orig_eb_forward  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# QUEEN — bundled training; hook optimizer.step inside train.py loop
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patch_queen(tracer: MotivationTracer) -> Iterator[None]:
    """Hook QUEEN's bundled training loop.

    QUEEN has one long-running ``training(...)`` call that iterates all frames
    internally.  Frame transitions are detected via ``gaussians.frame_idx``,
    which ``train.py`` sets at the top of every outer-loop iteration.

    Hooks installed:

    * ``gaussian_renderer.render_mask`` (primary training renderer) and
      ``gaussian_renderer.render``: capture ``render_ids``, ``tile_touches``,
      ``contribution_per_id``, ``n_gaussians``, plus open/close frame
      boundaries on the first call of each new ``gaussians.frame_idx``.
    * ``GaussianModel.training_setup``: install ``optimizer.step`` wrapper
      for update-mask / bytes_* accumulation.
    * ``train.training_report``: fall-back frame-tick in case a frame has
      zero render calls (shouldn't happen in practice).
    """
    import importlib

    train_mod = importlib.import_module("train")
    renderer_mod = importlib.import_module("gaussian_renderer")

    try:
        gaussian_model_mod = importlib.import_module("scene.gaussian_model")
        GaussianModel = gaussian_model_mod.GaussianModel
    except Exception as exc:
        logger.warning("patch_queen could not import GaussianModel: %s", exc)
        yield
        return

    orig_training_setup = getattr(GaussianModel, "training_setup", None)
    orig_render = getattr(renderer_mod, "render", None)
    orig_render_mask = getattr(renderer_mod, "render_mask", None)
    orig_render_mask_shift = getattr(renderer_mod, "render_mask_shift", None)
    orig_report = getattr(train_mod, "training_report", None)

    state: dict[str, Any] = {"last_frame": -1, "gaussians_ref": None}
    event_pairer = IterEventPairer(tracer)
    state_cache = StateDeltaCache()
    kernel_profiler = RenderKernelProfiler(tracer)

    def _row_bytes_of(g: Any) -> int:
        total = 0
        for name in (
            "_xyz", "_features_dc", "_features_rest",
            "_scaling", "_rotation", "_opacity",
        ):
            a = getattr(g, name, None)
            if a is None or not hasattr(a, "element_size") or not hasattr(a, "shape"):
                continue
            per = a.element_size()
            cols = 1
            if a.ndim > 1:
                for d in a.shape[1:]:
                    cols *= int(d)
            total += per * cols
        return total

    def _tick_frame(new_idx: int, gaussians: Any) -> None:
        if new_idx == state["last_frame"]:
            return
        # Always advance last_frame first — this way a metric failure inside
        # end_frame() does not cause every subsequent render_mask call to
        # retry the same transition and spam the log.
        state["last_frame"] = new_idx
        state["gaussians_ref"] = gaussians
        event_pairer.reset()
        state_cache.reset()
        kernel_profiler.reset()
        if tracer._curr is not None:
            try:
                attrs_after = _snapshot_gaussian_attrs(gaussians)
                if attrs_after:
                    tracer.record(
                        attr_after=attrs_after,
                        n_gaussians=int(attrs_after[next(iter(attrs_after))].shape[0]),
                    )
            except Exception as exc:
                logger.debug("queen attr_after snapshot failed: %s", exc)
            try:
                tracer.end_frame()
            except Exception as exc:
                logger.warning("queen end_frame failed: %s", exc)
                tracer._curr = None  # force-reset so start_frame below works
        tracer.start_frame(new_idx)
        try:
            attrs_before = _snapshot_gaussian_attrs(gaussians)
            if attrs_before:
                tracer.record(
                    attr_before=attrs_before,
                    n_gaussians=int(attrs_before[next(iter(attrs_before))].shape[0]),
                )
        except Exception as exc:
            logger.debug("queen attr_before snapshot failed: %s", exc)

    def _wrap_step(gaussians: Any) -> None:
        if not hasattr(gaussians, "optimizer") or gaussians.optimizer is None:
            return
        orig_step = gaussians.optimizer.step

        def wrapped(closure=None):
            try:
                state["gaussians_ref"] = gaussians
                upd_bytes = 0
                mask_accum: Optional[np.ndarray] = None
                for group in gaussians.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        upd_bytes += int(p.grad.element_size() * p.grad.numel())
                        g_np = _np(p.grad)
                        if g_np is None:
                            continue
                        if g_np.ndim > 1:
                            mask = np.any(g_np.reshape(g_np.shape[0], -1) != 0.0, axis=1)
                        else:
                            mask = g_np != 0.0
                        if mask_accum is None:
                            mask_accum = mask
                        elif mask.size == mask_accum.size:
                            mask_accum = mask_accum | mask
                if mask_accum is not None:
                    rec = tracer._curr
                    if rec is not None and rec.update_ids is not None and \
                       rec.update_ids.size == mask_accum.size:
                        mask_accum = mask_accum | rec.update_ids.astype(bool)
                    tracer.record(update_ids=mask_accum)
                tracer.add_bytes(
                    bytes_update_gen=upd_bytes,
                    bytes_commit=upd_bytes,
                    bytes_state_write=upd_bytes,
                )
            except Exception as exc:
                logger.debug("queen optimizer.step hook failed: %s", exc)
            event_pairer.on_step_start()
            try:
                return orig_step(closure) if closure is not None else orig_step()
            finally:
                event_pairer.on_step_end()

        gaussians.optimizer.step = wrapped  # type: ignore[method-assign]

    def wrapped_training_setup(self, *a, **kw):
        ret = orig_training_setup(self, *a, **kw)
        _wrap_step(self)
        return ret

    def _wrap_renderer(orig_fn):
        fn_name = getattr(orig_fn, "__name__", "?")

        def wrapped(viewpoint_camera, pc, pipe, bg_color, *a, **kw):
            try:
                fidx = int(getattr(pc, "frame_idx", -1))
                if fidx > 0 and fidx != state["last_frame"]:
                    _tick_frame(fidx, pc)
            except Exception as exc:
                logger.debug("queen frame-tick via render failed: %s", exc)
            try:
                _append_grad_stats(tracer, state_cache.collect(pc))
            except Exception as exc:
                logger.debug("queen state-delta snapshot failed: %s", exc)
            _publish_sh_degree(tracer, pc)
            event_pairer.on_render_start()
            out = kernel_profiler.run(
                orig_fn, viewpoint_camera, pc, pipe, bg_color, *a, **kw
            )
            event_pairer.on_render_end()
            try:
                if pc is not None and hasattr(pc, "_xyz"):
                    n = int(pc._xyz.shape[0])
                    vis, tt = _render_ids_from_output(out, n)
                    contrib_in = {"radii": out.get("radii")}
                    if hasattr(pc, "get_opacity"):
                        contrib_in["opacity"] = pc.get_opacity
                    contrib = _contribution_from_render(contrib_in, n)
                    tracer.record(
                        n_gaussians=n,
                        render_ids=vis,
                        tile_touches_per_id=tt,
                        contribution_per_id=contrib,
                    )
                    tracer.add_bytes(bytes_state_read=_row_bytes_of(pc) * n)
            except Exception as exc:
                logger.debug("queen render hook failed: %s", exc)
            return out
        return wrapped

    if orig_training_setup is not None:
        GaussianModel.training_setup = wrapped_training_setup  # type: ignore[method-assign]
    if orig_render is not None:
        renderer_mod.render = _wrap_renderer(orig_render)
        if hasattr(train_mod, "render"):
            train_mod.render = renderer_mod.render  # rebind resolved-at-import name
    if orig_render_mask is not None:
        renderer_mod.render_mask = _wrap_renderer(orig_render_mask)
        if hasattr(train_mod, "render_mask"):
            train_mod.render_mask = renderer_mod.render_mask
    if orig_render_mask_shift is not None:
        renderer_mod.render_mask_shift = _wrap_renderer(orig_render_mask_shift)
        if hasattr(train_mod, "render_mask_shift"):
            train_mod.render_mask_shift = renderer_mod.render_mask_shift

    if orig_report is not None:
        def wrapped_report(tb_writer, wandb_enabled, model_args, frame_idx, *a, **kw):
            try:
                g = state.get("gaussians_ref")
                if g is not None:
                    _tick_frame(int(frame_idx), g)
            except Exception as exc:
                logger.debug("queen training_report hook failed: %s", exc)
            return orig_report(tb_writer, wandb_enabled, model_args, frame_idx, *a, **kw)

        train_mod.training_report = wrapped_report  # type: ignore[assignment]

    try:
        yield
    finally:
        if orig_training_setup is not None:
            GaussianModel.training_setup = orig_training_setup  # type: ignore[method-assign]
        if orig_render is not None:
            renderer_mod.render = orig_render
            if hasattr(train_mod, "render"):
                train_mod.render = orig_render
        if orig_render_mask is not None:
            renderer_mod.render_mask = orig_render_mask
            if hasattr(train_mod, "render_mask"):
                train_mod.render_mask = orig_render_mask
        if orig_render_mask_shift is not None:
            renderer_mod.render_mask_shift = orig_render_mask_shift
            if hasattr(train_mod, "render_mask_shift"):
                train_mod.render_mask_shift = orig_render_mask_shift
        if orig_report is not None:
            train_mod.training_report = orig_report  # type: ignore[assignment]
        try:
            if tracer._curr is not None and state["gaussians_ref"] is not None:
                attrs_after = _snapshot_gaussian_attrs(state["gaussians_ref"])
                if attrs_after:
                    tracer.record(
                        attr_after=attrs_after,
                        n_gaussians=int(attrs_after[next(iter(attrs_after))].shape[0]),
                    )
                tracer.end_frame()
        except Exception as exc:
            logger.debug("queen final flush failed: %s", exc)
