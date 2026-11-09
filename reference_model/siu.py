"""Stage-I splat-impact calculation and five-observation confirmation."""

from __future__ import annotations

import math
from collections import defaultdict, deque

from .contracts import CanonicalUpdate, ProjectionSummary, Propagation, PropagationRecord, RequestKind, Structural


def _activation(value: float) -> float:
    """Public Act Unit realization shared with CIU and RTL."""

    return max(0.0, min(1.0, 0.5 + 0.25 * value))


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _matvec(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(_dot(row, vector) for row in matrix)  # type: ignore[return-value]


def _symmetric_covariance(values: tuple[float, float, float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    xx, xy, xz, yy, yz, zz = values
    return ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))


def _projected_covariance_norm(update: CanonicalUpdate, projection: ProjectionSummary) -> float:
    sigma = _symmetric_covariance(update.delta_sigma)
    j_sigma = tuple(
        tuple(sum(projection.j_cov[row][inner] * sigma[inner][column] for inner in range(3)) for column in range(3))
        for row in range(2)
    )
    projected = tuple(
        tuple(sum(j_sigma[row][inner] * projection.j_cov[column][inner] for inner in range(3)) for column in range(2))
        for row in range(2)
    )
    return math.sqrt(sum(value * value for row in projected for value in row))


class SplatImpactModel:
    """Implements the public SIU equations with one observation per gid/frame."""

    def __init__(self, tau_phi: float = 0.16, tile_area: float = 256.0) -> None:
        self.tau_phi = tau_phi
        self.tile_area = tile_area
        self._history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=5))
        self._last_frame: dict[int, int] = {}

    def process(self, update: CanonicalUpdate, projection: ProjectionSummary) -> PropagationRecord:
        if update.gid != projection.gid:
            raise ValueError("SIU update and projection must share gid")
        if projection.frame_id >= update.frame_id:
            raise ValueError("SIU projection must come from a preceding training-render frame")
        if projection.request_kind is not RequestKind.TRAINING:
            raise ValueError("SIU prior projection must come from a training render")
        previous = self._last_frame.get(update.gid)
        if previous is not None and update.frame_id <= previous:
            raise ValueError("SIU accepts at most one monotonic source-frame observation per Gaussian")

        rotated_delta_mu = _matvec(projection.r_cam, update.delta_mu)
        delta_u = _dot(projection.j_pi[0], rotated_delta_mu)
        delta_v = _dot(projection.j_pi[1], rotated_delta_mu)
        delta_r = _projected_covariance_norm(update, projection)
        delta_alpha = abs(_activation(projection.opacity_logit + update.delta_opacity) - _activation(projection.opacity_logit))
        rgb_delta = tuple(
            _dot(tuple(update.delta_sh[channel * len(projection.sh_basis) + basis] for basis in range(len(projection.sh_basis))), projection.sh_basis)
            for channel in range(3)
        )
        delta_q = abs(0.2126 * rgb_delta[0] + 0.7152 * rgb_delta[1] + 0.0722 * rgb_delta[2])
        coverage = min(max(projection.area / self.tile_area, 0.0), 1.0)
        geometry = (abs(delta_u) + abs(delta_v) + abs(delta_r)) / projection.radius
        psi = coverage * (geometry + delta_alpha + delta_q)

        history = self._history[update.gid]
        history.append(psi)
        self._last_frame[update.gid] = update.frame_id
        phi = sum(history) / len(history)
        protected = update.structural is not Structural.NONE or len(history) < 5
        commit = protected or phi >= self.tau_phi
        return PropagationRecord(
            update=update,
            projection=projection,
            psi=psi,
            phi=phi,
            effective_strength=update.strength if commit else 0.0,
            propagation=Propagation.COMMIT if commit else Propagation.CONFIRMATION,
        )
