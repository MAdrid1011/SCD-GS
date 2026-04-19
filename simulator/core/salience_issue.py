"""
SalienceFirstIssue — Stage 4 of the GLASS ASIC pipeline.

Implements Salience-First Render Issue: classifies each Gaussian into
PROMOTE, FULL, or SKIP before tile expansion, based on historical
contribution scores and structural stability.

Salience score proxy
--------------------
After each render(), the score for Gaussian i is updated via EMA:

    score_i ← alpha * score_i + (1 - alpha) * (opacity_i * radii_i / max_radii)

Both opacity (activated) and radii are available from render_pkg without
any additional CUDA kernel. The product is a proxy for visual contribution:
high opacity × large projected radius = large screen-space footprint.

Validity / Stability Tracker (VST)
-----------------------------------
Tracks per-Gaussian salience variance over the last stability_window frames.
A Gaussian is "stable" if variance < stability_var_max. Stability is a
prerequisite for SKIP classification.

Structural changes (spawn/prune) immediately invalidate a Gaussian's
historical score by setting a flag; the Gaussian is classified FULL until
it accumulates new reliable history.

Classifier
----------
  struct_changed                           → FULL  (history invalid)
  score > promote_thresh AND stable        → PROMOTE
  score < skip_thresh AND stable           → SKIP
  otherwise                                → FULL

Quality safeguard
-----------------
top full_path_fraction of Gaussians by current score are always FULL,
preventing over-pruning when most scores are similar (e.g., early frames).

This module is model-agnostic; it operates only on tensors and Python dicts.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from simulator.config import ASICConfig


class SalienceFirstIssue:
    """Stage 4: Salience-First Render Issue.

    Stateful across frames (EMA scores + history persist).
    Created once per ablation run inside GLASSPipeline.__init__().

    Public API used by GLASSPipeline
    ---------------------------------
    update(radii, opacity)       — called after each render(), updates scores
    mark_structural(gids)        — flag Gaussians whose geometry changed
    classify(n) → (keep, promote)— generate render masks for FPS measurement
    """

    def __init__(self, config: "ASICConfig") -> None:
        self.cfg = config
        self._scores: dict[int, float] = {}         # EMA salience scores
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.stability_window)
        )
        self._struct_changed: set[int] = set()      # Gaussians with fresh structural change

    # ── Score update (called post-render) ─────────────────────────────────────

    def update(self, radii: torch.Tensor, opacity: torch.Tensor) -> None:
        """Update per-Gaussian salience scores from render output.

        Parameters
        ----------
        radii   : FloatTensor shape [N] — projected 2D radius in screen pixels
        opacity : FloatTensor shape [N] or [N, 1] — activated opacity
        """
        op = opacity.squeeze(-1).float()
        r  = radii.float()
        max_r = r.max().item()
        if max_r < 1e-8:
            return  # degenerate render (all offscreen)

        raw_scores = (op * r / max_r).detach().cpu().numpy()
        alpha = self.cfg.salience_ema_alpha

        for i, s in enumerate(raw_scores):
            old = self._scores.get(i, float(s))
            new = alpha * old + (1.0 - alpha) * float(s)
            self._scores[i] = new
            self._history[i].append(new)

        # Clear structural flags: Gaussian has now been rendered post-change
        self._struct_changed.clear()

    # ── Structural change notification ────────────────────────────────────────

    def mark_structural(self, gids: list[int]) -> None:
        """Flag Gaussians whose geometry changed this frame (spawn/prune/split).

        Flagged Gaussians are classified FULL until update() is called at
        least once after the change.
        """
        self._struct_changed.update(gids)

    # ── Classification ────────────────────────────────────────────────────────

    def classify(
        self, n_gaussians: int
    ) -> tuple[torch.BoolTensor, torch.BoolTensor]:
        """Classify all Gaussians and return (keep_mask, promote_mask).

        keep_mask    : True for PROMOTE + FULL (included in render call)
        promote_mask : True for PROMOTE only (render first in tile queue)
        SKIP         : ~keep_mask

        Parameters
        ----------
        n_gaussians : int — total Gaussian count in the current model

        Returns
        -------
        (keep_mask, promote_mask) — both BoolTensors of shape [n_gaussians]
        """
        scores = torch.tensor(
            [self._scores.get(i, 1.0) for i in range(n_gaussians)],
            dtype=torch.float32,
        )
        variances = torch.tensor(
            [
                float(np.var(list(self._history[i])))
                if len(self._history[i]) > 1
                else float("inf")
                for i in range(n_gaussians)
            ],
            dtype=torch.float32,
        )

        stable_mask = variances < self.cfg.stability_var_max

        struct_mask = torch.zeros(n_gaussians, dtype=torch.bool)
        for gid in self._struct_changed:
            if gid < n_gaussians:
                struct_mask[gid] = True

        # SKIP: low score, stable history, no structural change
        skip_mask = (
            (scores < self.cfg.skip_thresh)
            & stable_mask
            & ~struct_mask
        )

        # Quality safeguard: always keep top full_path_fraction by score
        n_floor = max(1, int(n_gaussians * self.cfg.full_path_fraction))
        if n_floor < n_gaussians:
            floor_thresh = scores.topk(n_floor).values[-1].item()
            skip_mask[scores >= floor_thresh] = False

        keep_mask = ~skip_mask

        # PROMOTE: high score, stable, no structural change, and kept
        promote_mask = (
            (scores > self.cfg.promote_thresh)
            & stable_mask
            & ~struct_mask
            & keep_mask
        )

        return keep_mask, promote_mask

    # ── Per-run state management ──────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state. Called between ablation runs."""
        self._scores.clear()
        self._history.clear()
        self._struct_changed.clear()

    # ── Statistics ────────────────────────────────────────────────────────────

    def tier_counts(self, n_gaussians: int) -> dict[str, int]:
        """Return Gaussian counts for each classification tier."""
        keep, promote = self.classify(n_gaussians)
        n_keep    = int(keep.sum().item())
        n_promote = int(promote.sum().item())
        n_skip    = n_gaussians - n_keep
        n_full    = n_keep - n_promote
        return {
            "n_promote": n_promote,
            "n_full":    n_full,
            "n_skip":    n_skip,
        }
