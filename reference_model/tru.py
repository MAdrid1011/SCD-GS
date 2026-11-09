"""Stage-IV global compositing order, early stop, and trajectory oracle."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from struct import pack, unpack

from .contracts import PUBLIC_MAX_FRAME_ID, IssuePath, IssueRecord, Trajectory


def _fp16_roundtrip(value: float) -> float:
    """Use IEEE-754 binary16 round-to-nearest-even for light-path color."""

    return unpack(">e", pack(">e", value))[0]


def _shade(issue: IssueRecord, path: IssuePath) -> tuple[float, float, float]:
    if path is IssuePath.FULL:
        basis_count = len(issue.projection.sh_basis)
        return tuple(
            sum(issue.state.sh[channel * basis_count + basis] * issue.projection.sh_basis[basis] for basis in range(basis_count))
            for channel in range(3)
        )
    return tuple(_fp16_roundtrip(value) for value in issue.color_estimate)


def _activation(value: float) -> float:
    """Public Act Unit realization used by the TRU blend path."""

    return max(0.0, min(1.0, 0.5 + 0.25 * value))


class TieredRenderModel:
    """Drains a closed frame in canonical `(tile, depth, gid)` order."""

    def __init__(self, transmittance_stop: float = 0.05) -> None:
        self.transmittance_stop = transmittance_stop

    def process(self, issues: list[IssueRecord]) -> tuple[list[Trajectory], int]:
        if any(not isfinite(issue.projection.depth) for issue in issues):
            raise ValueError("TRU requires finite numerical depth for global compositing order")
        terminal_training = [
            issue
            for issue in issues
            if issue.request_kind.value == "training" and issue.frame_id >= PUBLIC_MAX_FRAME_ID
        ]
        if terminal_training:
            raise ValueError("TRU training work must use a frame below the public terminal frame")
        queued_issues = [
            issue
            for issue in issues
            if not (issue.path is IssuePath.FILTER and issue.request_kind.value == "user")
        ]
        if len({issue.frame_id for issue in queued_issues}) > 1:
            raise ValueError("TRU accepts a closed batch from one frame")

        transmittance: dict[int, float] = defaultdict(lambda: 1.0)
        trajectories: list[Trajectory] = []
        early_stops = 0
        blend_order = 0
        for issue in sorted(queued_issues, key=lambda value: (value.projection.tile, value.projection.depth, value.gid)):
            tile = issue.projection.tile
            effective_path = (
                IssuePath.LIGHT
                if issue.path is IssuePath.FILTER and issue.request_kind.value == "training"
                else issue.path
            )
            if transmittance[tile] < self.transmittance_stop:
                early_stops += 1
                continue
            coverage = max(0.0, min(1.0, issue.support_area))
            alpha = max(0.0, min(1.0, _activation(issue.state.opacity) * coverage))
            transmittance[tile] *= 1.0 - alpha
            if issue.request_kind.value == "training":
                trajectories.append(
                    Trajectory(
                        gid=issue.gid,
                        frame_id=issue.frame_id,
                        pixel=tile,
                        blend_order=blend_order,
                        coverage=coverage,
                        transmittance=transmittance[tile],
                        color=_shade(issue, effective_path),
                        path=effective_path,
                        projection=issue.projection,
                    )
                )
            blend_order += 1
        return trajectories, early_stops
