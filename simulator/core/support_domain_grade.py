"""
Support-domain implementation of Convergence-Graded Splatting (CGS).

This module is the simulator implementation of the paper-side CGS formula:
training-side update energy becomes convergence confidence, current-view
graphics support becomes render potential, and both signals jointly shrink
the effective support domain before Grade-A/B/D dispatch.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from simulator.config import ASICConfig


_MIN_INIT_CAPACITY = 1024
_EPS = 1e-12


class ConvergenceGrader:
    """Stage-4 CGS classifier based on training-aware support domains.

    ``grade_window`` is a frame/sample window.  Multiple optimizer steps in
    the same streaming frame accumulate into one per-Gaussian frame-energy
    sample, so dense-iteration baselines no longer fill the CGS history inside
    a single frame.
    """

    def __init__(self, config: "ASICConfig") -> None:
        self.cfg = config
        self._W = max(1, int(config.grade_window))

        self._s_buf = np.zeros((0, self._W), dtype=np.float32)
        self._s_count = np.zeros(0, dtype=np.int32)
        self._frame_energy = np.zeros(0, dtype=np.float32)
        self._frame_touched = np.zeros(0, dtype=bool)

        self._support_area = np.zeros(0, dtype=np.float32)
        self._render_potential = np.ones(0, dtype=np.float32)
        self._support_seen = np.zeros(0, dtype=bool)
        self._color_bound = np.ones(0, dtype=np.float32)
        self._struct_changed = np.zeros(0, dtype=bool)

        self._last_kappa = np.zeros(0, dtype=np.float32)
        self._last_rho = np.ones(0, dtype=np.float32)
        self._last_omega_cgs = np.zeros(0, dtype=np.float32)
        self._last_grade_a = np.zeros(0, dtype=bool)
        self._last_grade_b = np.zeros(0, dtype=bool)
        self._last_grade_d = np.zeros(0, dtype=bool)
        self._last_p_ref = 1.0
        self._last_s_ref = 1.0

        self._capacity = 0

    def _ensure_capacity(self, n_needed: int) -> None:
        if n_needed <= self._capacity:
            return
        new_cap = max(int(n_needed), self._capacity * 2, _MIN_INIT_CAPACITY)

        new_s = np.zeros((new_cap, self._W), dtype=np.float32)
        new_s[: self._capacity] = self._s_buf
        self._s_buf = new_s

        def grow(old: np.ndarray, fill, dtype) -> np.ndarray:
            new = np.full(new_cap, fill, dtype=dtype)
            new[: self._capacity] = old
            return new

        self._s_count = grow(self._s_count, 0, np.int32)
        self._frame_energy = grow(self._frame_energy, 0.0, np.float32)
        self._frame_touched = grow(self._frame_touched, False, bool)
        self._support_area = grow(self._support_area, 0.0, np.float32)
        self._render_potential = grow(self._render_potential, 1.0, np.float32)
        self._support_seen = grow(self._support_seen, False, bool)
        self._color_bound = grow(self._color_bound, 1.0, np.float32)
        self._struct_changed = grow(self._struct_changed, False, bool)
        self._last_kappa = grow(self._last_kappa, 0.0, np.float32)
        self._last_rho = grow(self._last_rho, 1.0, np.float32)
        self._last_omega_cgs = grow(self._last_omega_cgs, 0.0, np.float32)
        self._last_grade_a = grow(self._last_grade_a, False, bool)
        self._last_grade_b = grow(self._last_grade_b, False, bool)
        self._last_grade_d = grow(self._last_grade_d, False, bool)
        self._capacity = new_cap

    # Training-side signal -------------------------------------------------

    def record_update(self, gid: int, delta_norm: float) -> None:
        if gid < 0:
            return
        self._ensure_capacity(gid + 1)
        self._frame_touched[gid] = True
        if delta_norm > 0.0:
            self._frame_energy[gid] += float(delta_norm) ** 2

    def record_updates_bulk(self, norms: "torch.Tensor | np.ndarray") -> None:
        if isinstance(norms, torch.Tensor):
            arr = norms.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            arr = np.asarray(norms, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return
        n = int(arr.shape[0])
        self._ensure_capacity(n)
        self._frame_touched[:n] = True
        mask = arr > 0.0
        if mask.any():
            idx = np.flatnonzero(mask)
            self._frame_energy[idx] += arr[idx] ** 2

    def flush_frame_updates(self) -> None:
        """Commit one per-frame update-energy sample into the circular window."""
        if self._capacity == 0:
            return
        mask = self._frame_touched
        if not mask.any():
            return
        idx = np.flatnonzero(mask)
        cols = self._s_count[idx] % self._W
        self._s_buf[idx, cols] = self._frame_energy[idx]
        new_counts = self._s_count[idx] + 1
        np.minimum(new_counts, np.int32(2 * self._W), out=new_counts)
        self._s_count[idx] = new_counts
        self._frame_energy[idx] = 0.0
        self._frame_touched[idx] = False

    # Render-side support state -------------------------------------------

    def update_support_domain(
        self,
        radii: torch.Tensor,
        opacity: torch.Tensor,
        color_bound: "torch.Tensor | np.ndarray | None" = None,
    ) -> None:
        """Update current-view support area and graphics render potential.

        The paper formula uses the conic determinant to compute support area.
        Current baseline render packages expose ``radii`` instead, so the
        simulator uses the equivalent circular support proxy
        ``Omega ~= pi * radii^2``.
        """
        r = radii.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
        if r.size == 0:
            return
        n = int(r.shape[0])
        self._ensure_capacity(n)

        op = opacity.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
        if op.shape[0] != n:
            op = np.ones(n, dtype=np.float32)
        op = np.clip(np.nan_to_num(op, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if color_bound is None:
            cb = np.ones(n, dtype=np.float32)
        elif isinstance(color_bound, torch.Tensor):
            cb = color_bound.detach().to("cpu", dtype=torch.float32).reshape(-1).numpy()
        else:
            cb = np.asarray(color_bound, dtype=np.float32).reshape(-1)
        if cb.shape[0] != n:
            cb = np.ones(n, dtype=np.float32)
        cb = np.maximum(np.nan_to_num(cb, nan=1.0, posinf=1.0, neginf=0.0), 0.0)

        r = np.maximum(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        area = (math.pi * r * r).astype(np.float32)
        raw_p = (op.astype(np.float32) * area * cb.astype(np.float32)).astype(np.float32)

        explicit_ref = float(getattr(self.cfg, "cgs_p_ref", 0.0))
        positive = raw_p[raw_p > 0.0]
        if explicit_ref > 0.0:
            p_ref = explicit_ref
        elif positive.size > 0:
            q = float(getattr(self.cfg, "cgs_p_ref_quantile", 0.95))
            p_ref = float(np.quantile(positive, min(max(q, 0.0), 1.0)))
        else:
            p_ref = 1.0
        p_ref = max(p_ref, _EPS)

        self._support_area[:n] = area
        self._render_potential[:n] = np.clip(raw_p / p_ref, 0.0, 1.0).astype(np.float32)
        self._support_seen[:n] = True
        self._color_bound[:n] = cb.astype(np.float32)
        self._last_p_ref = p_ref
        self._struct_changed[:n] = False

    def update_contribution(
        self,
        radii: torch.Tensor,
        opacity: torch.Tensor,
        color_bound: "torch.Tensor | np.ndarray | None" = None,
    ) -> None:
        """Compatibility alias for the old salience API."""
        self.update_support_domain(radii, opacity, color_bound)

    def mark_structural(self, gids) -> None:
        if gids is None:
            return
        if isinstance(gids, torch.Tensor):
            arr = gids.detach().to("cpu", dtype=torch.int64).numpy()
        else:
            arr = np.asarray(gids, dtype=np.int64).reshape(-1)
        arr = arr[arr >= 0]
        if arr.size == 0:
            return
        self._ensure_capacity(int(arr.max()) + 1)
        self._struct_changed[arr] = True

    # Classification -------------------------------------------------------

    def _window_energy(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        count = self._s_count[:n]
        row_sum = self._s_buf[:n].sum(axis=1)
        divisor = np.maximum(np.minimum(count, self._W), 1).astype(np.float32)
        return (row_sum / divisor).astype(np.float32), count

    def _kappa(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        s_vals, count = self._window_energy(n)
        support_seen = self._support_seen[:n]
        never_trained_but_seen = (count == 0) & support_seen
        min_ready = int(getattr(self.cfg, "cgs_min_ready_samples", self._W))
        ready_samples = max(1, min(self._W, min_ready))
        ready = (count >= ready_samples) | never_trained_but_seen

        candidates = s_vals[count > 0]
        min_ref = float(getattr(self.cfg, "cgs_update_ref_min", 1e-12))
        if candidates.size > 0:
            q = float(getattr(self.cfg, "cgs_update_ref_quantile", 0.90))
            s_ref = float(np.quantile(candidates, min(max(q, 0.0), 1.0)))
        else:
            s_ref = min_ref
        s_ref = max(s_ref, min_ref, _EPS)
        self._last_s_ref = s_ref

        norm_s = np.clip(s_vals / s_ref, 0.0, 1.0).astype(np.float32)
        kappa = (1.0 - norm_s).astype(np.float32)
        kappa[never_trained_but_seen] = 1.0
        kappa[~ready] = 0.0
        return kappa, ready

    def assign_grades(
        self,
        n_gaussians: int,
        *,
        flush_pending: bool = True,
        convergence_hint: "torch.Tensor | np.ndarray | None" = None,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        if n_gaussians <= 0:
            empty = torch.zeros(0, dtype=torch.bool)
            return empty, empty.clone(), empty.clone()

        if flush_pending:
            self.flush_frame_updates()

        n = int(n_gaussians)
        self._ensure_capacity(n)

        kappa, ready = self._kappa(n)
        if convergence_hint is not None:
            if isinstance(convergence_hint, torch.Tensor):
                hint = convergence_hint.detach().to("cpu").reshape(-1).numpy()
            else:
                hint = np.asarray(convergence_hint).reshape(-1)
            if hint.shape[0] >= n:
                hint = hint[:n]
                if hint.dtype == bool:
                    hint_mask = hint
                    kappa[hint_mask] = np.maximum(kappa[hint_mask], 1.0)
                    ready[hint_mask] = True
                else:
                    hint_f = np.clip(hint.astype(np.float32), 0.0, 1.0)
                    kappa = np.maximum(kappa, hint_f)
                    ready = ready | (hint_f > 0.0)
        support_seen = self._support_seen[:n].copy()
        potential = self._render_potential[:n].astype(np.float32).copy()
        area = self._support_area[:n].astype(np.float32).copy()
        struct = self._struct_changed[:n].copy()

        lam = min(max(float(getattr(self.cfg, "cgs_lambda", 0.75)), 0.0), 1.0)
        rho = 1.0 - lam * kappa * (1.0 - potential)
        rho = np.clip(rho, 1.0 - lam, 1.0).astype(np.float32)
        omega_cgs = (rho * area).astype(np.float32)

        tau_a = float(getattr(self.cfg, "cgs_tau_a", 0.90))
        tau_d = float(getattr(self.cfg, "cgs_tau_d_area", 4.0))

        grade_a = (rho >= tau_a) | (~ready) | (~support_seen) | struct
        grade_d = (omega_cgs < tau_d) & ready & support_seen & (~grade_a)

        floor = float(getattr(self.cfg, "grade_a_floor", 0.10))
        if 0.0 < floor < 1.0 and n > 1:
            k = max(1, int(n * floor))
            if k < n:
                priority = np.maximum(1.0 - kappa, potential)
                top_idx = np.argpartition(priority, n - k)[n - k :]
                grade_a[top_idx] = True
                grade_d[top_idx] = False

        grade_b = ~(grade_a | grade_d)

        self._last_kappa[:n] = kappa
        self._last_rho[:n] = rho
        self._last_omega_cgs[:n] = omega_cgs
        self._last_grade_a[:n] = grade_a
        self._last_grade_b[:n] = grade_b
        self._last_grade_d[:n] = grade_d

        return (
            torch.from_numpy(grade_a.copy()),
            torch.from_numpy(grade_b.copy()),
            torch.from_numpy(grade_d.copy()),
        )

    def render_path_stats(
        self,
        n_gaussians: int,
        tile_size: int = 16,
        tile_touches_total: int | None = None,
    ) -> dict[str, float]:
        """Per-grade visibility/support stats for graded render accounting."""
        if n_gaussians <= 0:
            return {
                "n_visible_a": 0,
                "n_visible_b": 0,
                "tile_touches_a": 0,
                "tile_touches_b": 0,
                "support_area_a": 0.0,
                "support_area_b": 0.0,
                "support_area_cgs_b": 0.0,
                "rho_mean": 1.0,
                "kappa_mean": 0.0,
                "render_potential_mean": 1.0,
                "p_ref": self._last_p_ref,
                "s_ref": self._last_s_ref,
            }

        n = int(n_gaussians)
        self._ensure_capacity(n)

        A = self._last_grade_a[:n]
        B = self._last_grade_b[:n]
        area = self._support_area[:n]
        omega = self._last_omega_cgs[:n]
        visible = area > 0.0
        tile_area = float(tile_size * tile_size)

        area_a = float(area[A & visible].sum())
        area_b_orig = float(area[B & visible].sum())
        area_b_eff = float(omega[B & visible].sum())
        if tile_touches_total is not None and tile_touches_total >= 0:
            denom = max(area_a + area_b_orig, _EPS)
            touch_scale = float(tile_touches_total) / denom if denom > _EPS else 0.0
            tile_touches_a = int(round(area_a * touch_scale))
            tile_touches_b = int(round(area_b_eff * touch_scale))
        else:
            tile_area = float(tile_size * tile_size)
            tile_touches_a = int(area_a / tile_area)
            tile_touches_b = int(area_b_eff / tile_area)

        return {
            "n_visible_a": int((A & visible).sum()),
            "n_visible_b": int((B & visible).sum()),
            "tile_touches_a": tile_touches_a,
            "tile_touches_b": tile_touches_b,
            "support_area_a": area_a,
            "support_area_b": area_b_orig,
            "support_area_cgs_b": area_b_eff,
            "rho_mean": float(self._last_rho[:n].mean()) if n > 0 else 1.0,
            "kappa_mean": float(self._last_kappa[:n].mean()) if n > 0 else 0.0,
            "render_potential_mean": float(self._render_potential[:n].mean()) if n > 0 else 1.0,
            "p_ref": float(self._last_p_ref),
            "s_ref": float(self._last_s_ref),
        }

    def classify(
        self, n_gaussians: int
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        A, B, _D = self.assign_grades(n_gaussians)
        return A | B, A

    def tier_counts(self, n_gaussians: int) -> dict[str, int]:
        A, B, D = self.assign_grades(n_gaussians)
        n_a = int(A.sum().item())
        n_b = int(B.sum().item())
        n_d = int(D.sum().item())
        return {
            "n_grade_a": n_a,
            "n_grade_b": n_b,
            "n_grade_d": n_d,
            "n_promote": n_a,
            "n_full": n_b,
            "n_skip": n_d,
        }

    def reset(self) -> None:
        self._s_buf = np.zeros((0, self._W), dtype=np.float32)
        self._s_count = np.zeros(0, dtype=np.int32)
        self._frame_energy = np.zeros(0, dtype=np.float32)
        self._frame_touched = np.zeros(0, dtype=bool)
        self._support_area = np.zeros(0, dtype=np.float32)
        self._render_potential = np.ones(0, dtype=np.float32)
        self._support_seen = np.zeros(0, dtype=bool)
        self._color_bound = np.ones(0, dtype=np.float32)
        self._struct_changed = np.zeros(0, dtype=bool)
        self._last_kappa = np.zeros(0, dtype=np.float32)
        self._last_rho = np.ones(0, dtype=np.float32)
        self._last_omega_cgs = np.zeros(0, dtype=np.float32)
        self._last_grade_a = np.zeros(0, dtype=bool)
        self._last_grade_b = np.zeros(0, dtype=bool)
        self._last_grade_d = np.zeros(0, dtype=bool)
        self._last_p_ref = 1.0
        self._last_s_ref = 1.0
        self._capacity = 0
