"""Stage-V trace consumption, full-state Adam update, and canonical feedback."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .contracts import (
    ATTRIBUTE_COUNT,
    CanonicalUpdate,
    GaussianState,
    IssuePath,
    PUBLIC_MAX_FRAME_ID,
    RequestKind,
    SH_COEFFICIENTS,
    Structural,
    TargetPixel,
    Trajectory,
)


@dataclass
class OptimizerState:
    """Per-Gaussian StateSPM record plus Adam vectors."""

    state: GaussianState
    moments: tuple[float, ...]
    variances: tuple[float, ...]


class TrainingUpdateModel:
    """Applies one matched scalar-surrogate update at a time."""

    def __init__(
        self,
        training_lanes: int = 16,
        beta1: float = 0.9,
        beta2: float = 0.999,
        learning_rate: float = 0.001,
        epsilon: float = 1e-8,
    ) -> None:
        if training_lanes <= 0 or training_lanes & (training_lanes - 1):
            raise ValueError("training_lanes must be a positive power of two")
        self.training_lanes = training_lanes
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._state: dict[int, OptimizerState] = {}
        self._target: TargetPixel | None = None
        self.invalid_pairs = 0

    def seed_state(self, state: GaussianState) -> None:
        self._state[state.gid] = OptimizerState(state, (0.0,) * ATTRIBUTE_COUNT, (0.0,) * ATTRIBUTE_COUNT)

    def optimizer_state(self, gid: int) -> OptimizerState:
        if gid not in self._state:
            raise KeyError(f"missing optimizer state for gid {gid}")
        return self._state[gid]

    def buffer_target(self, target: TargetPixel) -> None:
        """Accept one target when the single public target buffer is empty."""

        if self._target is not None:
            raise ValueError("target buffer is occupied")
        self._target = target

    def process_trace(self, trace: Trajectory) -> CanonicalUpdate | None:
        """Consume one transferred trace only when it matches the buffered target and state."""

        if self._target is None:
            return None
        target = self._target
        if (
            trace.path not in {IssuePath.FULL, IssuePath.LIGHT}
            or trace.projection.request_kind is not RequestKind.TRAINING
            or trace.pixel != target.pixel
            or trace.frame_id != target.frame_id
            or trace.frame_id >= PUBLIC_MAX_FRAME_ID
            or trace.gid not in self._state
        ):
            self.invalid_pairs += 1
            return None

        gradient = (
            sum(abs(trace.color[channel] - target.rgb[channel]) for channel in range(3))
            / 3.0
            * trace.coverage
            * trace.transmittance
        )
        optimizer = self._state[trace.gid]
        moments = tuple(self.beta1 * value + (1.0 - self.beta1) * gradient for value in optimizer.moments)
        variances = tuple(self.beta2 * value + (1.0 - self.beta2) * gradient * gradient for value in optimizer.variances)
        step = self.learning_rate * moments[0] / (sqrt(variances[0]) + self.epsilon)
        delta = -step
        old = optimizer.state
        covariance = tuple(
            value + (delta if index in (0, 3, 5) else 0.0) for index, value in enumerate(old.covariance)
        )
        next_state = GaussianState(
            gid=trace.gid,
            mean=(old.mean[0] + delta, old.mean[1] + 0.5 * delta, old.mean[2]),
            covariance=covariance,  # type: ignore[arg-type]
            opacity=old.opacity + 0.25 * delta,
            sh=tuple(value + 0.5 * delta for value in old.sh),
        )
        self._state[trace.gid] = OptimizerState(next_state, moments, variances)
        self._target = None
        return CanonicalUpdate(
            gid=trace.gid,
            frame_id=trace.frame_id + 1,
            delta_mu=(delta, 0.5 * delta, 0.0),
            delta_sigma=(delta, 0.0, 0.0, delta, 0.0, delta),
            delta_opacity=0.25 * delta,
            delta_sh=(0.5 * delta,) * SH_COEFFICIENTS,
            strength=min(1.0, abs(step)),
            structural=Structural.NONE,
        )

    def process(self, trajectories: list[Trajectory], targets: list[TargetPixel]) -> list[CanonicalUpdate]:
        """Convenience wrapper for one buffered target and transferred traces."""

        if len(targets) > 1:
            raise ValueError("single target buffer accepts at most one target per call")
        if targets:
            self.buffer_target(targets[0])
        updates: list[CanonicalUpdate] = []
        for trace in trajectories:
            if self._target is None:
                break
            update = self.process_trace(trace)
            if update is not None:
                updates.append(update)
        return updates
