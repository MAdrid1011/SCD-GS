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
        # Tensor-backed per-Gaussian state.  Earlier simulator revisions kept
        # these as Python dicts and sorted lists of IDs, which became a host
        # bottleneck for NTC baselines with hundreds of thousands of Gaussians.
        self._hotness: torch.Tensor | None = None
        self._recent_delta: torch.Tensor | None = None
        self._known: torch.Tensor | None = None
        self._ucn_streak: torch.Tensor | None = None
        # Cached mask from last get_render_mask() call
        self._cached_mask: torch.BoolTensor | None = None
        self._cached_n: int = 0

    def _ensure_capacity(self, n_gaussians: int) -> None:
        if n_gaussians <= 0:
            return
        if self._hotness is None:
            self._hotness = torch.zeros(n_gaussians, dtype=torch.float32)
            self._recent_delta = torch.zeros(n_gaussians, dtype=torch.float32)
            self._known = torch.zeros(n_gaussians, dtype=torch.bool)
            self._ucn_streak = torch.zeros(n_gaussians, dtype=torch.int32)
            return
        if self._hotness.shape[0] >= n_gaussians:
            return
        add = n_gaussians - self._hotness.shape[0]
        self._hotness = torch.cat([
            self._hotness,
            torch.zeros(add, dtype=self._hotness.dtype),
        ])
        self._recent_delta = torch.cat([
            self._recent_delta,
            torch.zeros(add, dtype=self._recent_delta.dtype),
        ])
        self._known = torch.cat([
            self._known,
            torch.zeros(add, dtype=self._known.dtype),
        ])
        self._ucn_streak = torch.cat([
            self._ucn_streak,
            torch.zeros(add, dtype=self._ucn_streak.dtype),
        ])

    # ── Score updates ─────────────────────────────────────────────────────────

    def record_update(self, gid: int, delta_norm: float) -> None:
        """Update hotness for a Gaussian that received a gradient this frame.

        Also records the delta_norm for the RSB threshold check.

        If ``ucn_drive_fabric_demotion`` is enabled and the Gaussian's UCN
        suppression streak has reached ``ucn_demotion_streak``, the
        hotness reward is skipped (hotness decays by the EMA alpha only).
        Either way — reward applied or not — reaching this function means
        a non-suppressed gradient arrived, so the streak is reset to 0.
        """
        if gid < 0:
            return
        self._ensure_capacity(gid + 1)
        assert self._hotness is not None
        assert self._recent_delta is not None
        assert self._known is not None
        assert self._ucn_streak is not None

        alpha = self.cfg.hotness_alpha
        old = float(self._hotness[gid].item())
        if (
            self.cfg.ucn_drive_fabric_demotion
            and int(self._ucn_streak[gid].item()) >= self.cfg.ucn_demotion_streak
        ):
            # Long-suppressed Gaussian just came back; decay-only this step
            # (no reward injection) so it has to earn its way back into the
            # hot tier over a few frames of sustained activity.  This damps
            # one-off noise and keeps hotness ordering stable.
            self._hotness[gid] = alpha * old
        else:
            reward = self.cfg.hotness_update_reward
            self._hotness[gid] = alpha * old + (1.0 - alpha) * reward
        self._recent_delta[gid] = max(float(self._recent_delta[gid].item()), delta_norm)
        self._known[gid] = True
        if self.cfg.ucn_drive_fabric_demotion:
            self._ucn_streak[gid] = 0
        self._cached_mask = None

    def record_ucn_suppression(self, gid: int) -> None:
        """Report that UCN suppressed this Gaussian's update this frame.

        Increments the per-Gaussian streak counter consumed by
        :meth:`record_update`.  No-op when ``ucn_drive_fabric_demotion``
        is disabled so this method can be called unconditionally from the
        pipeline without a per-call config check on the hot path.

        Does NOT touch ``_recent_delta`` or ``_hotness`` — the Gaussian's
        existing scores simply keep decaying via the EMA each time any
        other Gaussian triggers a ``record_render`` invalidation, and the
        sorted-by-hotness order eventually pushes it out of ASC/RSB.
        """
        if not self.cfg.ucn_drive_fabric_demotion:
            return
        if gid < 0:
            return
        self._ensure_capacity(gid + 1)
        assert self._ucn_streak is not None
        self._ucn_streak[gid] = self._ucn_streak[gid] + 1

    def record_updates_bulk(
        self,
        delta_norms: torch.Tensor,
        grad_scale: torch.Tensor | None = None,
    ) -> None:
        """Vectorized update/suppression feed for dense per-Gaussian paths.

        ``grad_scale`` follows the descriptor semantics: rows with
        ``grad_scale <= 0`` are UCN-suppressed and therefore only advance the
        demotion streak; rows with positive scale receive the normal hotness
        reward and update ``recent_delta``.
        """
        if delta_norms.numel() == 0:
            return
        norms = delta_norms.detach().reshape(-1).to("cpu", dtype=torch.float32)
        n = int(norms.shape[0])
        self._ensure_capacity(n)
        assert self._hotness is not None
        assert self._recent_delta is not None
        assert self._known is not None
        assert self._ucn_streak is not None

        active = norms > 0.0
        if not bool(active.any().item()):
            return

        if grad_scale is None:
            pass_mask = active
            suppressed = torch.zeros_like(active)
        else:
            scale = grad_scale.detach().reshape(-1).to("cpu", dtype=torch.float32)
            if scale.shape[0] != n:
                scale = torch.ones(n, dtype=torch.float32)
            pass_mask = active & (scale > 0.0)
            suppressed = active & (scale <= 0.0)

        if self.cfg.ucn_drive_fabric_demotion and bool(suppressed.any().item()):
            self._ucn_streak[:n][suppressed] += 1

        if bool(pass_mask.any().item()):
            alpha = float(self.cfg.hotness_alpha)
            old = self._hotness[:n][pass_mask]
            if self.cfg.ucn_drive_fabric_demotion:
                demoted = (
                    self._ucn_streak[:n][pass_mask]
                    >= int(self.cfg.ucn_demotion_streak)
                )
                rewarded = alpha * old + (
                    1.0 - alpha
                ) * float(self.cfg.hotness_update_reward)
                decayed = alpha * old
                self._hotness[:n][pass_mask] = torch.where(
                    demoted, decayed, rewarded
                )
                self._ucn_streak[:n][pass_mask] = 0
            else:
                self._hotness[:n][pass_mask] = alpha * old + (
                    1.0 - alpha
                ) * float(self.cfg.hotness_update_reward)

            self._recent_delta[:n][pass_mask] = torch.maximum(
                self._recent_delta[:n][pass_mask],
                norms[pass_mask],
            )
            self._known[:n][pass_mask] = True
            self._cached_mask = None

    def record_render(self, visibility_filter: torch.BoolTensor) -> None:
        """Update hotness for all Gaussians visible in the last render.

        Parameters
        ----------
        visibility_filter : BoolTensor, shape [N]
            True for Gaussians that contributed to the rendered image.
        """
        if visibility_filter.numel() == 0:
            return
        vf = visibility_filter.detach().reshape(-1).to("cpu", dtype=torch.bool)
        n = int(vf.shape[0])
        self._ensure_capacity(n)
        assert self._hotness is not None
        assert self._known is not None
        if not bool(vf.any().item()):
            return
        alpha  = self.cfg.hotness_alpha
        reward = self.cfg.hotness_render_reward
        old = self._hotness[:n][vf]
        self._hotness[:n][vf] = alpha * old + (1.0 - alpha) * reward
        self._known[:n][vf] = True
        # Invalidate cached mask (scene has changed)
        self._cached_mask = None

    # ── Tier classification ───────────────────────────────────────────────────

    def _sorted_ids(self, n_gaussians: int) -> torch.Tensor:
        """Return all known Gaussian IDs sorted by descending hotness."""
        if n_gaussians <= 0:
            return torch.empty(0, dtype=torch.long)
        self._ensure_capacity(n_gaussians)
        assert self._hotness is not None
        assert self._known is not None
        known = self._known[:n_gaussians]
        if not bool(known.any().item()):
            return torch.empty(0, dtype=torch.long)
        ids = known.nonzero(as_tuple=False).squeeze(1)
        order = torch.argsort(self._hotness[ids], descending=True)
        return ids[order]

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
        if n_gaussians <= 0:
            return mask
        sorted_ids = self._sorted_ids(n_gaussians)
        assert self._recent_delta is not None

        # ASC tier: top asb_capacity
        asc_end = min(self.cfg.asb_capacity, len(sorted_ids))
        if asc_end > 0:
            mask[sorted_ids[:asc_end]] = True

        # RSB tier: next rsb_capacity, filtered by delta threshold
        rsb_end = min(asc_end + self.cfg.rsb_capacity, len(sorted_ids))
        if rsb_end > asc_end:
            rsb_ids = sorted_ids[asc_end:rsb_end]
            keep = self._recent_delta[rsb_ids] > float(self.cfg.rsb_delta_thresh)
            if bool(keep.any().item()):
                mask[rsb_ids[keep]] = True

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
        if self._recent_delta is not None:
            self._recent_delta.zero_()
        self._cached_mask = None

    # ── Statistics ────────────────────────────────────────────────────────────

    def tier_counts(self, n_gaussians: int) -> dict[str, int]:
        """Return counts for each tier given current hotness scores."""
        mask = self.get_render_mask(n_gaussians)
        n_active = int(mask.sum().item())
        if self._known is None:
            n_known = 0
        else:
            n_known = int(self._known[:n_gaussians].sum().item())
        return {
            "n_asc_rsb": n_active,
            "n_css":     n_gaussians - n_active,
            "n_known":   n_known,
            "n_total":   n_gaussians,
        }

    def delta_storage_bytes(self) -> int:
        """Estimated incremental checkpoint size for the current frame.

        Only ASC + RSB Gaussians store their delta. Each Gaussian's full
        attribute set is approximated as 6 attrs × 4 bytes (xyz, rot, scale,
        opacity, features_dc, features_rest_sum). This is a rough lower bound.
        """
        if self._known is None:
            n_known = 0
        else:
            n_known = int(self._known.sum().item())
        n_active = min(self.cfg.asb_capacity + self.cfg.rsb_capacity, n_known)
        # 6 float32 attribute scalars per Gaussian (rough approximation)
        return n_active * 6 * 4

    def reset(self) -> None:
        """Clear all state. Called between ablation runs."""
        self._hotness = None
        self._recent_delta = None
        self._known = None
        self._ucn_streak = None
        self._cached_mask = None
        self._cached_n = 0
