"""
LiveStateFabric — Stage 3 of the GLASS ASIC pipeline.

Implements the Incremental Live-State Fabric: a three-tier hotness-based
Gaussian residency model that decides which Gaussians are "active" and
which are "cold" (excluded from gradient updates and rendering).

Tiers
-----
ASC (Active State Cache) — top asb_capacity Gaussians by hotness.
    Always included in render. Always receive gradient updates.
    Hardware analogy: small fast on-chip SRAM.

RSB (Resident State Bank) — next rsb_capacity Gaussians by hotness,
    additionally filtered by rsb_delta_thresh.
    Included in render only if their recent delta_norm > rsb_delta_thresh.
    Hardware analogy: larger on-chip state bank.

CSS (Cold State Store) — all remaining Gaussians.
    Excluded from render mask. Gradients zeroed post-backward.
    Hardware analogy: off-chip DRAM; accessed only on miss/spill.

Hotness score update (EMA)
--------------------------
    hotness_new = alpha * hotness_old + (1 - alpha) * reward
where reward is hotness_update_reward for gradient events and
hotness_render_reward for render-visibility events.

This module is model-agnostic; it operates only on Gaussian indices (ints)
and boolean tensors. All model-specific access is handled by the pipeline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from simulator.config import ASICConfig


class LiveStateFabric:
    """Stage 3: Incremental Live-State Fabric.

    Stateful across frames (hotness scores persist).
    Created once per ablation run inside GLASSPipeline.__init__().

    Public API used by GLASSPipeline
    ---------------------------------
    record_update(gid, delta_norm)  — called after each gradient update event
    record_render(visibility_filter) — called after each render() call
    get_render_mask(n) → BoolTensor  — True for ASC ∪ qualified-RSB
    get_cold_mask(n)   → BoolTensor  — True for CSS
    """

    def __init__(self, config: "ASICConfig") -> None:
        self.cfg = config
        # Hotness scores: dict[int, float] persists across frames
        self._hotness: dict[int, float] = {}
        # Most recent per-Gaussian delta_norm (set during record_update)
        self._recent_delta: dict[int, float] = {}
        # Cached mask from last get_render_mask() call
        self._cached_mask: torch.BoolTensor | None = None
        self._cached_n: int = 0

    # ── Score updates ─────────────────────────────────────────────────────────

    def record_update(self, gid: int, delta_norm: float) -> None:
        """Update hotness for a Gaussian that received a gradient this frame.

        Also records the delta_norm for the RSB threshold check.
        """
        alpha = self.cfg.hotness_alpha
        reward = self.cfg.hotness_update_reward
        old = self._hotness.get(gid, 0.0)
        self._hotness[gid] = alpha * old + (1.0 - alpha) * reward
        self._recent_delta[gid] = max(
            self._recent_delta.get(gid, 0.0), delta_norm
        )

    def record_render(self, visibility_filter: torch.BoolTensor) -> None:
        """Update hotness for all Gaussians visible in the last render.

        Parameters
        ----------
        visibility_filter : BoolTensor, shape [N]
            True for Gaussians that contributed to the rendered image.
        """
        alpha  = self.cfg.hotness_alpha
        reward = self.cfg.hotness_render_reward
        # Iterate only visible indices to keep this O(visible) not O(N)
        visible_ids = visibility_filter.nonzero(as_tuple=False).squeeze(1)
        for gid in visible_ids.tolist():
            old = self._hotness.get(gid, 0.0)
            self._hotness[gid] = alpha * old + (1.0 - alpha) * reward
        # Invalidate cached mask (scene has changed)
        self._cached_mask = None

    # ── Tier classification ───────────────────────────────────────────────────

    def _sorted_ids(self) -> list[int]:
        """Return all known Gaussian IDs sorted by descending hotness."""
        return sorted(self._hotness, key=lambda k: -self._hotness[k])

    def get_render_mask(self, n_gaussians: int) -> torch.BoolTensor:
        """Boolean mask: True for Gaussians that should be rendered.

        ASC tier  : top-asb_capacity by hotness — always True
        RSB tier  : next-rsb_capacity by hotness, AND recent_delta > thresh
        CSS tier  : all others — False

        Gaussians not yet seen (no hotness entry) are treated as CSS until
        they accumulate score via gradient or render events.

        The result is cached until record_render() invalidates it.
        """
        if (self._cached_mask is not None and
                self._cached_n == n_gaussians):
            return self._cached_mask

        mask = torch.zeros(n_gaussians, dtype=torch.bool)

        sorted_ids = self._sorted_ids()

        # ASC tier: top asb_capacity
        asc_end = min(self.cfg.asb_capacity, len(sorted_ids))
        for gid in sorted_ids[:asc_end]:
            if gid < n_gaussians:
                mask[gid] = True

        # RSB tier: next rsb_capacity, filtered by delta threshold
        rsb_end = min(asc_end + self.cfg.rsb_capacity, len(sorted_ids))
        for gid in sorted_ids[asc_end:rsb_end]:
            if gid < n_gaussians:
                delta = self._recent_delta.get(gid, 0.0)
                if delta > self.cfg.rsb_delta_thresh:
                    mask[gid] = True

        self._cached_mask = mask
        self._cached_n    = n_gaussians
        return mask

    def get_cold_mask(self, n_gaussians: int) -> torch.BoolTensor:
        """Boolean mask: True for CSS (cold) Gaussians.

        These Gaussians' gradients are zeroed by the pipeline after backward.
        """
        return ~self.get_render_mask(n_gaussians)

    # ── Per-frame delta reset ─────────────────────────────────────────────────

    def reset_frame_deltas(self) -> None:
        """Clear per-frame delta_norm cache at the start of each new frame.

        This ensures the RSB threshold is evaluated on current-frame activity
        rather than cumulative history.
        """
        self._recent_delta.clear()
        self._cached_mask = None

    # ── Statistics ────────────────────────────────────────────────────────────

    def tier_counts(self, n_gaussians: int) -> dict[str, int]:
        """Return counts for each tier given current hotness scores."""
        mask = self.get_render_mask(n_gaussians)
        n_active = int(mask.sum().item())
        return {
            "n_asc_rsb": n_active,
            "n_css":     n_gaussians - n_active,
            "n_known":   min(len(self._hotness), n_gaussians),
            "n_total":   n_gaussians,
        }

    def delta_storage_bytes(self) -> int:
        """Estimated incremental checkpoint size for the current frame.

        Only ASC + RSB Gaussians store their delta. Each Gaussian's full
        attribute set is approximated as 6 attrs × 4 bytes (xyz, rot, scale,
        opacity, features_dc, features_rest_sum). This is a rough lower bound.
        """
        n_active = min(
            self.cfg.asb_capacity + self.cfg.rsb_capacity,
            len(self._hotness),
        )
        # 6 float32 attribute scalars per Gaussian (rough approximation)
        return n_active * 6 * 4

    def reset(self) -> None:
        """Clear all state. Called between ablation runs."""
        self._hotness.clear()
        self._recent_delta.clear()
        self._cached_mask = None
        self._cached_n = 0
