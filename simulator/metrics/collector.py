"""
MetricsCollector — standardised measurement utilities for ablation runners.

Provides:
  frame_timer()       — context manager for GPU-synchronised per-frame timing
  avg_frame_time()    — trimmed mean of all frame times (drop frame 0 + top 10%)
  measure_fps()       — delegate to GLASSPipeline.measure_fps()
  measure_quality()   — PSNR / SSIM / LPIPS via torchmetrics
  measure_storage()   — scan output dir for .ply / .pth / .pkl checkpoints
  measure_delta_storage() — sum UpdateStream byte counts across frames

Usage in ablation runners
-------------------------
    collector = MetricsCollector()

    for frame in frames:
        with collector.frame_timer():
            train_one_frame(...)

    fps     = collector.measure_fps(pipeline, render_fn, cameras, pipe, bg)
    quality = collector.measure_quality(renders, gts)
    storage = collector.measure_storage(output_path)
    delta   = collector.measure_delta_storage(per_frame_bytes)

LPIPS requires a pretrained AlexNet; it downloads automatically on first use.
Pass lpips_net="vgg" to use VGG backbone instead (larger, slower).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from simulator.pipeline import GLASSPipeline


class MetricsCollector:
    """Collects and aggregates metrics for one ablation run."""

    def __init__(self) -> None:
        self._frame_times: list[float] = []

    # ── Training time ─────────────────────────────────────────────────────────

    @contextmanager
    def frame_timer(self):
        """Context manager for GPU-synchronised wall-clock frame timing.

        Synchronises the CUDA device before and after the enclosed block so
        that kernel execution time is fully included in the measurement.

        Usage::
            with collector.frame_timer():
                train_one_frame(...)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._frame_times.append(time.perf_counter() - t0)

    def avg_frame_time(self) -> float:
        """Return trimmed mean of per-frame training times (seconds).

        - Drops frame 0 (static/initial reconstruction step — unusually slow)
        - Drops top 10% outliers (stalls, background garbage collection, etc.)
        Returns NaN when fewer than 2 frames were measured.
        """
        ts = self._frame_times[1:]   # exclude frame 0
        if not ts:
            return float("nan")
        clip = max(1, int(len(ts) * 0.9))
        return float(sum(sorted(ts)[:clip]) / clip)

    def all_frame_times(self) -> list[float]:
        """Raw per-frame wall-clock times including frame 0."""
        return list(self._frame_times)

    def reset_frame_times(self) -> None:
        self._frame_times.clear()

    # ── FPS measurement ───────────────────────────────────────────────────────

    def measure_fps(
        self,
        pipeline: "GLASSPipeline",
        render_fn,
        cameras: list,
        pipe,
        bg: torch.Tensor,
    ) -> float:
        """Measure render FPS by delegating to the pipeline.

        All masking logic (LiveStateFabric + SalienceFirstIssue) is applied
        inside pipeline.measure_fps() via make_render_proxy(), so the measured
        FPS correctly reflects the active ablation mode.

        Parameters
        ----------
        pipeline  : GLASSPipeline (carries mode config + fabric/salience state)
        render_fn : callable(camera, gaussians, pipe, bg) → render_pkg dict
        cameras   : list of camera objects to cycle through
        pipe      : pipeline params passed to render_fn
        bg        : background colour tensor
        """
        return pipeline.measure_fps(render_fn, cameras, pipe, bg)

    # ── Image quality ─────────────────────────────────────────────────────────

    def measure_quality(
        self,
        renders: list[torch.Tensor],
        gts:     list[torch.Tensor],
        lpips_net: str = "alex",
    ) -> dict[str, float]:
        """Compute PSNR, SSIM, LPIPS over a list of rendered / GT image pairs.

        Parameters
        ----------
        renders   : list of Float[C, H, W] tensors in [0, 1]
        gts       : list of Float[C, H, W] tensors in [0, 1]
        lpips_net : "alex" (default) or "vgg" backbone for LPIPS

        Returns
        -------
        dict with keys "psnr", "ssim", "lpips"
        """
        try:
            from torchmetrics.image import (
                PeakSignalNoiseRatio         as PSNR,
                StructuralSimilarityIndexMeasure as SSIM,
                LearnedPerceptualImagePatchSimilarity as LPIPS,
            )
        except ImportError as exc:
            raise ImportError(
                "torchmetrics is required for quality measurement. "
                "Install with: pip install torchmetrics[image]"
            ) from exc

        psnr_fn  = PSNR(data_range=1.0)
        ssim_fn  = SSIM(data_range=1.0)
        lpips_fn = LPIPS(net_type=lpips_net)

        ps, ss, ls = [], [], []
        for r, g in zip(renders, gts):
            r_ = r.cpu().unsqueeze(0).clamp(0.0, 1.0)   # [1, C, H, W]
            g_ = g.cpu().unsqueeze(0).clamp(0.0, 1.0)

            ps.append(float(psnr_fn(r_, g_).item()))
            ss.append(float(ssim_fn(r_, g_).item()))
            ls.append(float(lpips_fn(r_, g_).item()))

        if not ps:
            return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}

        return {
            "psnr":  sum(ps) / len(ps),
            "ssim":  sum(ss) / len(ss),
            "lpips": sum(ls) / len(ls),
        }

    # ── Checkpoint storage ────────────────────────────────────────────────────

    def measure_storage(self, model_path: str | Path) -> float:
        """Measure total size of checkpoint files in model_path (MiB).

        Scans recursively for .ply (Gaussian PLY), .pth (PyTorch checkpoint),
        and .pkl (pickle) files.

        Returns 0.0 if the path does not exist or contains no such files.
        """
        p = Path(model_path)
        if not p.exists():
            return 0.0
        total = sum(
            f.stat().st_size
            for f in p.rglob("*")
            if f.is_file() and f.suffix in (".ply", ".pth", ".pkl")
        )
        return total / (1024 ** 2)   # bytes → MiB

    # ── Delta (incremental) storage ───────────────────────────────────────────

    def measure_delta_storage(self, per_frame_bytes: list[int]) -> float:
        """Compute total incremental storage traffic from UpdateStream sizes.

        Parameters
        ----------
        per_frame_bytes : list of per-frame active_bytes() values from the stream.
            Typically obtained via pipeline.stream_stats()["frame_active_bytes"].

        Returns
        -------
        Total MiB of delta traffic (suppressed updates excluded).
        """
        return sum(per_frame_bytes) / (1024 ** 2)

    # ── Convenience: render a list of test cameras and collect images ─────────

    @staticmethod
    def render_test_cameras(
        gaussians,
        cameras: list,
        render_fn,
        pipe,
        bg: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Render all test cameras and return (renders, gts) image lists.

        Parameters
        ----------
        gaussians  : Gaussian model (or MaskedGaussianProxy for evaluation)
        cameras    : list of camera objects with .original_image attribute
        render_fn  : callable(camera, gaussians, pipe, bg) → render_pkg
        pipe       : pipeline params
        bg         : background colour tensor

        Returns
        -------
        (renders, gts) — parallel lists of [C, H, W] float32 tensors
        """
        renders, gts = [], []
        for cam in cameras:
            with torch.no_grad():
                pkg = render_fn(cam, gaussians, pipe, bg)
            renders.append(pkg["render"].detach().cpu())
            gt = getattr(cam, "original_image", None)
            if gt is not None:
                gts.append(gt.float().cpu())
        return renders, gts
