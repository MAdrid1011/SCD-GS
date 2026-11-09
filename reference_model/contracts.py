"""Shared full-record contracts for the public SCD-GS semantic oracle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from struct import pack


SH_DEGREE = 3
SH_BASES = (SH_DEGREE + 1) ** 2
SH_COEFFICIENTS = 3 * SH_BASES
ATTRIBUTE_COUNT = 3 + 6 + 1 + SH_COEFFICIENTS
PUBLIC_FRAME_WIDTH = 16
PUBLIC_MAX_FRAME_ID = (1 << PUBLIC_FRAME_WIDTH) - 1

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Mat2x3 = tuple[Vec3, Vec3]
Mat3x3 = tuple[Vec3, Vec3, Vec3]
Covariance6 = tuple[float, float, float, float, float, float]


class Structural(str, Enum):
    NONE = "none"
    INSERT = "insert"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    RESET = "reset"


class Tier(str, Enum):
    ACTIVE = "active"
    RESIDENT = "resident"
    CANDIDATE = "candidate"


class RequestKind(str, Enum):
    USER = "user"
    TRAINING = "training"


class IssuePath(str, Enum):
    FULL = "full"
    LIGHT = "light"
    FILTER = "filter"


class Propagation(str, Enum):
    COMMIT = "commit"
    CONFIRMATION = "confirmation"


def _require_width(values: tuple[float, ...], width: int, name: str) -> None:
    if len(values) != width:
        raise ValueError(f"{name} must contain {width} values, got {len(values)}")


def _positive_normal_binary32(value: float) -> bool:
    """Return whether a scalar has the valid SIU radius binary32 encoding."""

    if not isfinite(value) or value <= 0.0:
        return False
    try:
        word = int.from_bytes(pack(">f", value), "big")
    except OverflowError:
        return False
    exponent = (word >> 23) & 0xFF
    return exponent not in {0, 0xFF} and (word >> 31) == 0


@dataclass(frozen=True)
class GaussianState:
    """Full StateSPM Gaussian record consumed by CIU, TRU, and TUU."""

    gid: int
    mean: Vec3
    covariance: Covariance6
    opacity: float
    sh: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_width(self.sh, SH_COEFFICIENTS, "GaussianState.sh")


@dataclass(frozen=True)
class CanonicalUpdate:
    """The full Stage-V to Stage-I update record."""

    gid: int
    frame_id: int
    delta_mu: Vec3
    delta_sigma: Covariance6
    delta_opacity: float
    delta_sh: tuple[float, ...]
    strength: float
    structural: Structural = Structural.NONE

    def __post_init__(self) -> None:
        _require_width(self.delta_sh, SH_COEFFICIENTS, "CanonicalUpdate.delta_sh")


@dataclass(frozen=True)
class ProjectionSummary:
    """The full prior-render summary retained in SplatSPM."""

    gid: int
    frame_id: int
    r_cam: Mat3x3
    j_pi: Mat2x3
    j_cov: Mat2x3
    view_direction: Vec3
    sh_basis: tuple[float, ...]
    center: Vec2
    conic: tuple[float, float, float]
    radius: float
    area: float
    opacity_logit: float
    view_color: Vec3
    request_kind: RequestKind
    tile: int
    depth: float

    def __post_init__(self) -> None:
        _require_width(self.sh_basis, SH_BASES, "ProjectionSummary.sh_basis")
        if not _positive_normal_binary32(self.radius):
            raise ValueError("ProjectionSummary.radius must be a positive normal binary32 value")


@dataclass(frozen=True)
class PropagationRecord:
    update: CanonicalUpdate
    projection: ProjectionSummary
    psi: float
    phi: float
    effective_strength: float
    propagation: Propagation


@dataclass
class TierEntry:
    gid: int
    tier: Tier
    row: int
    h: int = 0
    d: int = 0
    energy_sum: float = 0.0
    samples: int = 0
    touched: bool = False

    @property
    def energy(self) -> float:
        return self.energy_sum / self.samples if self.samples else 0.0


@dataclass(frozen=True)
class ResidencyRecord:
    propagation: PropagationRecord
    state: GaussianState
    tier: Tier
    row: int
    candidate_refill: bool
    state_ready: bool
    energy: float
    sample_count: int


@dataclass(frozen=True)
class IssueRecord:
    gid: int
    frame_id: int
    tier: Tier
    request_kind: RequestKind
    path: IssuePath
    support_area: float
    color_estimate: Vec3
    p: float
    kappa: float
    rho: float
    omega: float
    structural: Structural
    state: GaussianState
    projection: ProjectionSummary


@dataclass(frozen=True)
class Trajectory:
    gid: int
    frame_id: int
    pixel: int
    blend_order: int
    coverage: float
    transmittance: float
    color: Vec3
    path: IssuePath
    projection: ProjectionSummary


@dataclass(frozen=True)
class TargetPixel:
    pixel: int
    frame_id: int
    rgb: Vec3
