"""
Legacy module — Salience-First Render Issue.

The GLASS ASIC pipeline's Stage 4 has been replaced by
:class:`~simulator.core.support_domain_grade.ConvergenceGrader` (Convergence-
Graded Splatting, CGS).  This file is kept as a thin backwards-
compatibility layer so existing imports (``from simulator.core.salience_issue
import SalienceFirstIssue``), RTL docs, and tests continue to resolve.

The exported :class:`SalienceFirstIssue` is a trivial subclass of
:class:`ConvergenceGrader` that preserves the legacy method names:

  * :meth:`update`            — forwards to ``update_contribution``
  * :meth:`classify(n)`       — returns ``(keep_mask, promote_mask)``
    where ``keep = Grade-A ∪ Grade-B`` and ``promote = Grade-A``
  * :meth:`tier_counts(n)`    — returns both ``n_grade_*`` and legacy
    ``n_promote / n_full / n_skip`` keys (see
    :meth:`ConvergenceGrader.tier_counts`).

New code should depend directly on ``ConvergenceGrader``.
"""
from __future__ import annotations

from simulator.core.support_domain_grade import ConvergenceGrader


class SalienceFirstIssue(ConvergenceGrader):
    """Deprecated alias — use :class:`ConvergenceGrader` directly.

    Adds a single legacy method (``update``) that mirrors the old API so
    historical call sites keep compiling.  All other behaviour is
    inherited unchanged from :class:`ConvergenceGrader`.
    """

    def update(self, radii, opacity) -> None:  # type: ignore[override]
        """Legacy alias for :meth:`ConvergenceGrader.update_contribution`."""
        self.update_contribution(radii, opacity)
