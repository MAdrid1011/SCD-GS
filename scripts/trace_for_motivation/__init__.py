"""GPU-side instrumentation for GLASS.md motivation metrics.

This package is the shared core; each of
``scripts/trace_{3dgstream,hicom,4dgc,queen}.py`` imports from here and wires
baseline-specific monkey-patches. See
``docs/rtl_spec/`` for where the 11 metrics feed into the paper (Challenges
1/2/3).
"""
from __future__ import annotations

from .core import CrossFrameStateSnapshot, MotivationTracer
from .hooks import (
    patch_3dgstream,
    patch_hicom,
    patch_4dgc,
    patch_queen,
)
from .schema import FrameRecord, PerFrameMetrics, TraceOutput

__all__ = [
    "MotivationTracer",
    "CrossFrameStateSnapshot",
    "FrameRecord",
    "PerFrameMetrics",
    "TraceOutput",
    "patch_3dgstream",
    "patch_hicom",
    "patch_4dgc",
    "patch_queen",
]
