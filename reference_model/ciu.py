"""Stage-III issue equations and request-sensitive path selection."""

from __future__ import annotations

from dataclasses import replace

from .contracts import IssuePath, IssueRecord, ProjectionSummary, ResidencyRecord, Structural


def _activation(value: float) -> float:
    """Public binary32 Act Unit realization shared with SIU and RTL."""

    return max(0.0, min(1.0, 0.5 + 0.25 * value))


class ConditionedIssueModel:
    """Implements CIU projection, P, kappa, rho, and omega."""

    def __init__(self, tau_rho: float = 0.90, tau_d: float = 0.15) -> None:
        self.tau_rho = tau_rho
        self.tau_d = tau_d

    def process(self, residency: ResidencyRecord, current_view: ProjectionSummary) -> IssueRecord:
        """Pair a complete residency response with its current render view."""

        update = residency.propagation.update
        if current_view.gid != update.gid or current_view.frame_id != update.frame_id:
            raise ValueError("CIU current view must match the residency gid and update frame")
        if not residency.state_ready:
            raise ValueError("CIU requires a complete StateSPM response")
        projection = current_view
        state = residency.state
        xx, xy, xz, yy, yz, zz = state.covariance
        covariance = ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))
        projected = tuple(
            tuple(
                sum(
                    projection.j_cov[row][inner] * covariance[inner][column]
                    for inner in range(3)
                )
                for column in range(3)
            )
            for row in range(2)
        )
        screen_covariance = tuple(
            tuple(
                sum(projected[row][inner] * projection.j_cov[column][inner] for inner in range(3))
                for column in range(2)
            )
            for row in range(2)
        )
        covariance_det = (
            screen_covariance[0][0] * screen_covariance[1][1]
            - screen_covariance[0][1] * screen_covariance[1][0]
        )
        support = max(0.0, min(1.0, abs(covariance_det) * 9.0))
        color = tuple(
            sum(state.sh[channel * len(projection.sh_basis) + basis] * projection.sh_basis[basis] for basis in range(4))
            for channel in range(3)
        )
        opacity = _activation(state.opacity)
        potential = max(0.0, min(1.0, sum(opacity * support * channel for channel in color) / 3.0))
        kappa = 1.0 - max(0.0, min(1.0, residency.energy))
        rho = 1.0 - kappa * (1.0 - potential)
        omega = rho * support
        structural = residency.propagation.update.structural
        forced_full = structural is not Structural.NONE or (
            residency.sample_count < 5 and residency.tier.value != "candidate"
        )
        if forced_full or rho > self.tau_rho:
            path = IssuePath.FULL
        elif omega < self.tau_d and projection.request_kind.value == "user":
            path = IssuePath.FILTER
        else:
            path = IssuePath.LIGHT
        projected_summary = replace(
            projection,
            center=tuple(
                sum(projection.j_pi[row][column] * state.mean[column] for column in range(3))
                for row in range(2)
            ),
            opacity_logit=state.opacity,
            view_color=color,
            depth=state.mean[2],
        )
        return IssueRecord(
            gid=state.gid,
            frame_id=update.frame_id,
            tier=residency.tier,
            request_kind=projection.request_kind,
            path=path,
            support_area=support,
            color_estimate=color,
            p=potential,
            kappa=kappa,
            rho=rho,
            omega=omega,
            structural=structural,
            state=state,
            projection=projected_summary,
        )
