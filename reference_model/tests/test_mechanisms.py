"""Deterministic semantic tests for every public SCD-GS mechanism stage."""

from __future__ import annotations

import unittest
from dataclasses import replace

from reference_model.ciu import ConditionedIssueModel
from reference_model.contracts import (
    CanonicalUpdate,
    GaussianState,
    IssuePath,
    PUBLIC_MAX_FRAME_ID,
    ProjectionSummary,
    Propagation,
    PropagationRecord,
    RequestKind,
    ResidencyRecord,
    SH_BASES,
    SH_COEFFICIENTS,
    Structural,
    TargetPixel,
    Tier,
)
from reference_model.fru import FusedResidencyModel
from reference_model.siu import SplatImpactModel
from reference_model.tru import TieredRenderModel
from reference_model.tuu import TrainingUpdateModel


def state(gid: int = 7, covariance: float = 0.1) -> GaussianState:
    return GaussianState(
        gid=gid,
        mean=(0.1, 0.1, 0.1),
        covariance=(covariance, 0.0, 0.0, covariance, 0.0, covariance),
        opacity=0.5,
        sh=tuple(0.5 if index % SH_BASES == 0 else 0.1 for index in range(SH_COEFFICIENTS)),
    )


def update(frame: int, strength: float = 0.1, structural: Structural = Structural.NONE) -> CanonicalUpdate:
    return CanonicalUpdate(
        7,
        frame,
        (0.1, 0.2, 0.0),
        (0.1, 0.0, 0.0, 0.1, 0.0, 0.1),
        0.1,
        tuple(0.1 if index % SH_BASES == 0 else 0.0 for index in range(SH_COEFFICIENTS)),
        strength,
        structural,
    )


def projection(
    frame: int,
    request: RequestKind = RequestKind.TRAINING,
    area: float = 0.01,
    tile: int = 3,
    depth: float = 0.1,
    radius: float = 1.0,
) -> ProjectionSummary:
    return ProjectionSummary(
        gid=7,
        frame_id=frame,
        r_cam=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        j_pi=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        j_cov=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        view_direction=(1.0, 0.0, 0.0),
        sh_basis=(1.0,) + (0.0,) * (SH_BASES - 1),
        center=(0.0, 0.0),
        conic=(1.0, 0.0, 1.0),
        radius=radius,
        area=area,
        opacity_logit=0.0,
        view_color=(0.1, 0.1, 0.1),
        request_kind=request,
        tile=tile,
        depth=depth,
    )


def residency(
    frame: int,
    *,
    tier: Tier = Tier.ACTIVE,
    request: RequestKind = RequestKind.TRAINING,
    area: float = 0.01,
    energy: float = 0.0,
    samples: int = 5,
    structural: Structural = Structural.NONE,
    covariance: float = 0.1,
) -> ResidencyRecord:
    record_update = update(frame, structural=structural)
    record_projection = projection(frame - 1, request=RequestKind.TRAINING, area=area)
    propagation = PropagationRecord(record_update, record_projection, 0.0, 0.0, record_update.strength, Propagation.COMMIT)
    return ResidencyRecord(propagation, state(covariance=covariance), tier, row=7, candidate_refill=tier is Tier.CANDIDATE, state_ready=True, energy=energy, sample_count=samples)


def current_view(record: ResidencyRecord, request: RequestKind = RequestKind.TRAINING, **changes: object) -> ProjectionSummary:
    """Build a CIU view in the residency record's current update frame."""

    return replace(
        record.propagation.projection,
        frame_id=record.propagation.update.frame_id,
        request_kind=request,
        **changes,
    )


def issue_position(record, *, gid: int, tile: int, depth: float):
    """Assign the single paper-defined tile/depth/gid source for a TRU issue."""

    return replace(
        record,
        gid=gid,
        state=replace(record.state, gid=gid),
        projection=replace(record.projection, gid=gid, tile=tile, depth=depth),
    )


class MechanismOracleTest(unittest.TestCase):
    """Each test maps directly to one row in docs/traceability.md."""

    def test_siu_warmup_confirmation_and_structural_protection(self) -> None:
        model = SplatImpactModel(tau_phi=0.16)
        for frame in range(1, 5):
            record = model.process(update(frame), projection(frame - 1))
            self.assertEqual(record.propagation, Propagation.COMMIT)
            self.assertEqual(record.effective_strength, 0.1)
        fifth = model.process(update(5), projection(4))
        self.assertEqual(fifth.propagation, Propagation.CONFIRMATION)
        self.assertEqual(fifth.effective_strength, 0.0)
        structural = model.process(update(6, structural=Structural.INSERT), projection(5))
        self.assertEqual(structural.propagation, Propagation.COMMIT)
        self.assertGreater(structural.psi, 0.0)

    def test_siu_reuses_a_retained_prior_projection_but_rejects_future_metadata(self) -> None:
        model = SplatImpactModel(tau_phi=0.16)
        feedback = model.process(update(1), projection(0))
        self.assertEqual(feedback.update.frame_id, 1)
        self.assertEqual(feedback.projection.frame_id, 0)
        with self.assertRaisesRegex(ValueError, "preceding training-render frame"):
            model.process(update(2), projection(3))
        with self.assertRaisesRegex(ValueError, "training render"):
            model.process(update(2), projection(1, request=RequestKind.USER))
        with self.assertRaisesRegex(ValueError, "positive normal"):
            projection(3, radius=0.0)
        with self.assertRaisesRegex(ValueError, "preceding training-render frame"):
            model.process(update(2), projection(2))

    def test_fru_fuses_events_allocates_ages_and_averages_energy(self) -> None:
        model = FusedResidencyModel(active_capacity=1, resident_capacity=1)
        render_only = FusedResidencyModel(active_capacity=1, resident_capacity=1)
        self.assertEqual(render_only.render_read(13).row, 13)
        self.assertEqual(model._quantize(-0.25), 0)
        self.assertEqual(model._quantize(0.5), 127)
        self.assertEqual(model._quantize(1.0), 255)
        self.assertEqual(model._quantize(1.25), 255)
        model.commit(7, 1.0)
        model.confirmation(7)
        self.assertEqual(model.entries[7].h, 1)
        self.assertEqual(model.entries[7].energy, 0.5)
        for _ in range(14):
            model.render_read(7)
        self.assertEqual(model.entries[7].h, 15)
        model.commit(8, 0.5)
        self.assertEqual(model.refills, 0)
        model.candidate_request_accepted(8)
        self.assertEqual(model.refills, 1)
        for _ in range(5):
            model.finish_frame()
        self.assertEqual(model.entries[7].tier, Tier.ACTIVE)
        self.assertEqual(model.entries[8].tier, Tier.RESIDENT)
        self.assertEqual(model.entries[7].row, 0)
        self.assertEqual(model.entries[8].row, 0)
        for _ in range(5):
            model.finish_frame()
        self.assertEqual(model.entries[7].h, 7)
        self.assertEqual(model.entries[7].d, 0)
        model.commit(11, 0.25)
        for _ in range(5):
            model.finish_frame()
        self.assertEqual(model.entries[8].tier, Tier.CANDIDATE)
        self.assertEqual(model.entries[8].row, 8)
        self.assertGreater(model.candidate_hits, 0)

    def test_fru_breaks_equal_priority_with_gid_and_uses_partition_local_rows(self) -> None:
        model = FusedResidencyModel(active_capacity=1, resident_capacity=1)
        model.commit(9, 0.5)
        model.commit(3, 0.5)
        for _ in range(5):
            model.finish_frame()
        self.assertEqual(model.entries[3].tier, Tier.ACTIVE)
        self.assertEqual(model.entries[3].row, 0)
        self.assertEqual(model.entries[9].tier, Tier.RESIDENT)
        self.assertEqual(model.entries[9].row, 0)

    def test_ciu_equations_and_training_filter_fallback(self) -> None:
        model = ConditionedIssueModel(tau_rho=0.90, tau_d=0.15)
        user_source = residency(1, tier=Tier.CANDIDATE, area=0.01, energy=0.0)
        training_source = residency(1, tier=Tier.CANDIDATE, area=0.01, energy=0.0)
        forced_source = residency(1, structural=Structural.RESET)
        user = model.process(user_source, current_view(user_source, RequestKind.USER))
        training = model.process(training_source, current_view(training_source, RequestKind.TRAINING))
        forced = model.process(forced_source, current_view(forced_source, RequestKind.USER))
        self.assertEqual(user.path, IssuePath.FILTER)
        self.assertEqual(training.path, IssuePath.LIGHT)
        self.assertEqual(forced.path, IssuePath.FULL)
        self.assertAlmostEqual(user.rho, 1.0 - user.kappa * (1.0 - user.p))
        self.assertAlmostEqual(user.omega, user.rho * user.support_area)
        with self.assertRaisesRegex(ValueError, "complete StateSPM response"):
            model.process(replace(user_source, state_ready=False), current_view(user_source, RequestKind.USER))

    def test_ciu_saturates_p_after_the_raw_low_order_color_product(self) -> None:
        model = ConditionedIssueModel(tau_rho=0.90, tau_d=0.15)
        source = residency(1, tier=Tier.CANDIDATE, energy=0.0)
        sh = [0.0] * SH_COEFFICIENTS
        for channel in range(3):
            sh[channel * SH_BASES] = 2.0
        record = model.process(
            replace(
                source,
                state=replace(source.state, covariance=(1.0, 0.0, 0.0, 1.0, 0.0, 1.0), sh=tuple(sh)),
            ),
            current_view(source),
        )
        self.assertEqual(record.color_estimate, (2.0, 2.0, 2.0))
        self.assertEqual(record.p, 1.0)

    def test_ciu_nontrivial_jcov_sigma_projection_and_decision_fields(self) -> None:
        model = ConditionedIssueModel(tau_rho=0.90, tau_d=0.15)
        source = residency(1, tier=Tier.CANDIDATE, energy=0.25)
        sh = [0.0] * SH_COEFFICIENTS
        coefficients = (
            (0.5, 0.25, -0.25, 0.125),
            (0.25, 0.5, 0.125, -0.25),
            (0.75, -0.25, 0.5, 0.25),
        )
        for channel, values in enumerate(coefficients):
            sh[channel * SH_BASES : channel * SH_BASES + 4] = values
        nontrivial_state = replace(
            source.state,
            covariance=(0.25, 0.0625, -0.03125, 0.5, 0.125, 0.75),
            sh=tuple(sh),
        )
        nontrivial_projection = replace(
            source.propagation.projection,
            j_cov=((0.5, -0.25, 0.125), (0.25, 0.5, -0.5)),
            sh_basis=(1.0, 0.5, -0.25, 0.125) + (0.0,) * (SH_BASES - 4),
            radius=0.375,
            area=0.75,
        )
        record = model.process(
            replace(
                source,
                state=nontrivial_state,
                propagation=replace(source.propagation, projection=nontrivial_projection),
            ),
            replace(nontrivial_projection, frame_id=1),
        )
        self.assertEqual(record.path, IssuePath.LIGHT)
        self.assertEqual(record.color_estimate, (45.0 / 64.0, 7.0 / 16.0, 17.0 / 32.0))
        self.assertAlmostEqual(record.support_area, 200799.0 / 1048576.0, places=12)
        self.assertAlmostEqual(record.p, 35809155.0 / 536870912.0, places=12)
        self.assertEqual(record.kappa, 0.75)
        self.assertAlmostEqual(record.rho, 644298377.0 / 2147483648.0, places=12)
        self.assertAlmostEqual(record.omega, 129374469803223.0 / 2251799813685248.0, places=12)
        self.assertEqual(record.projection.center, (0.1, 0.1))
        self.assertEqual(record.projection.depth, 0.1)
        self.assertEqual(record.projection.conic, nontrivial_projection.conic)
        self.assertEqual(record.projection.opacity_logit, 0.5)
        self.assertEqual(record.projection.view_color, record.color_estimate)
        self.assertEqual(record.projection.radius, 0.375)
        self.assertEqual(record.projection.area, 0.75)

    def test_tru_global_depth_order_early_stop_and_training_trace(self) -> None:
        issue = ConditionedIssueModel()
        active_source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0, covariance=1.0)
        resident_source = residency(1, tier=Tier.RESIDENT, area=1.0, energy=1.0)
        active = issue.process(active_source, current_view(active_source))
        resident = issue.process(resident_source, current_view(resident_source))
        active = replace(active, state=replace(active.state, opacity=4.0))
        later = issue_position(resident, gid=8, tile=3, depth=0.9)
        first = issue_position(active, gid=7, tile=3, depth=0.1)
        trajectories, early_stops = TieredRenderModel().process([later, first])
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].gid, 7)
        self.assertEqual(trajectories[0].projection, first.projection)
        self.assertEqual(early_stops, 1)

    def test_tru_merges_three_tiers_by_tile_depth_and_gid(self) -> None:
        issue = ConditionedIssueModel()
        active_source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        resident_source = residency(1, tier=Tier.RESIDENT, area=1.0, energy=1.0)
        candidate_source = residency(1, tier=Tier.CANDIDATE, area=1.0, energy=1.0)
        active = issue.process(active_source, current_view(active_source))
        resident = issue.process(resident_source, current_view(resident_source))
        candidate = issue.process(candidate_source, current_view(candidate_source))
        items = [
            issue_position(active, gid=7, tile=3, depth=0.5),
            issue_position(resident, gid=8, tile=3, depth=0.1),
            issue_position(candidate, gid=9, tile=3, depth=0.2),
        ]
        trajectories, early_stops = TieredRenderModel().process(items)
        self.assertEqual([record.gid for record in trajectories], [8, 9, 7])
        self.assertEqual([record.blend_order for record in trajectories], [0, 1, 2])
        self.assertEqual(early_stops, 0)

    def test_tru_rejects_a_closed_batch_with_multiple_frame_ids(self) -> None:
        issue = ConditionedIssueModel()
        frame_zero_source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        frame_one_source = residency(2, tier=Tier.RESIDENT, area=1.0, energy=1.0)
        frame_zero = issue.process(frame_zero_source, current_view(frame_zero_source))
        frame_one = issue.process(frame_one_source, current_view(frame_one_source))
        with self.assertRaisesRegex(ValueError, "one frame"):
            TieredRenderModel().process([frame_zero, frame_one])

    def test_tru_rejects_nonfinite_depth_before_global_compositing(self) -> None:
        issue = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        record = issue.process(source, current_view(source))
        nonfinite = replace(record, projection=replace(record.projection, depth=float("nan")))
        with self.assertRaisesRegex(ValueError, "finite numerical depth"):
            TieredRenderModel().process([nonfinite])

    def test_tru_rejects_terminal_frame_training_work_before_trajectory_generation(self) -> None:
        issue = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        record = issue.process(source, current_view(source))
        terminal = replace(
            record,
            frame_id=PUBLIC_MAX_FRAME_ID,
            projection=replace(record.projection, frame_id=PUBLIC_MAX_FRAME_ID),
        )
        with self.assertRaisesRegex(ValueError, "frame below the public terminal frame"):
            TieredRenderModel().process([terminal])

    def test_tru_uses_full_degree_three_sh_and_fp16_light_rounding(self) -> None:
        issue_model = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, energy=1.0)
        coefficients = (0.0625,) * SH_BASES + (-0.03125,) * SH_BASES + (0.03125,) * SH_BASES
        full_projection = replace(source.propagation.projection, sh_basis=(0.25,) * SH_BASES)
        full = issue_model.process(
            replace(
                source,
                state=replace(source.state, sh=coefficients),
                propagation=replace(source.propagation, projection=full_projection),
            ),
            replace(full_projection, frame_id=1),
        )
        self.assertEqual(full.path, IssuePath.FULL)
        light = replace(
            full,
            gid=8,
            path=IssuePath.LIGHT,
            color_estimate=(0.5 + 2**-12, 0.5 + 2**-12 + 2**-14, 2**-14 - 2**-25),
            state=replace(full.state, gid=8),
            projection=replace(full.projection, gid=8),
        )
        trajectories, early_stops = TieredRenderModel().process([
            issue_position(light, gid=8, tile=1, depth=0.9),
            issue_position(full, gid=7, tile=1, depth=0.1),
        ])
        self.assertEqual(early_stops, 0)
        self.assertEqual(trajectories[0].path, IssuePath.FULL)
        self.assertEqual(trajectories[0].color, (0.25, -0.125, 0.125))
        self.assertEqual(trajectories[1].path, IssuePath.LIGHT)
        self.assertEqual(trajectories[1].color, (0.5, 0.50048828125, 2**-14))

    def test_tuu_forms_a_surrogate_update_and_closes_the_feedback_transition(self) -> None:
        issue = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        record = issue.process(source, current_view(source))
        trajectories, _ = TieredRenderModel().process([record])
        model = TrainingUpdateModel(training_lanes=16)
        model.seed_state(state())
        updates = model.process(trajectories, [TargetPixel(3, 1, (0.3, 0.2, 0.1))])
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].gid, 7)
        self.assertEqual(updates[0].frame_id, 2)
        self.assertEqual(len(updates[0].delta_sh), SH_COEFFICIENTS)
        self.assertGreater(updates[0].strength, 0.0)
        optimizer = model.optimizer_state(7)
        self.assertEqual(len(optimizer.moments), 58)
        self.assertGreater(optimizer.moments[0], 0.0)
        self.assertGreater(optimizer.variances[0], 0.0)

    def test_tuu_keeps_one_target_until_a_matching_seeded_trace_consumes_it(self) -> None:
        issue = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        record = issue.process(source, current_view(source))
        traces, _ = TieredRenderModel().process([record])
        trace = traces[0]
        target = TargetPixel(3, 1, (0.3, 0.2, 0.1))

        model = TrainingUpdateModel()
        model.seed_state(state())
        model.buffer_target(target)
        with self.assertRaisesRegex(ValueError, "occupied"):
            model.buffer_target(target)
        self.assertIsNone(model.process_trace(replace(trace, path=IssuePath.FILTER)))
        self.assertEqual(model.invalid_pairs, 1)
        self.assertIsNone(
            model.process_trace(
                replace(trace, projection=replace(trace.projection, request_kind=RequestKind.USER))
            )
        )
        self.assertEqual(model.invalid_pairs, 2)
        self.assertIsNone(model.process_trace(replace(trace, pixel=4)))
        self.assertEqual(model.invalid_pairs, 3)
        update_record = model.process_trace(trace)
        self.assertIsNotNone(update_record)
        self.assertEqual(update_record.gid, 7)
        self.assertIsNone(model.process_trace(trace))

        unseeded = TrainingUpdateModel()
        unseeded.buffer_target(target)
        self.assertIsNone(unseeded.process_trace(trace))
        self.assertEqual(unseeded.invalid_pairs, 1)
        with self.assertRaisesRegex(KeyError, "missing optimizer state"):
            unseeded.optimizer_state(7)

    def test_tuu_rejects_a_terminal_frame_trajectory_without_feedback_wraparound(self) -> None:
        issue = ConditionedIssueModel()
        source = residency(1, tier=Tier.ACTIVE, area=1.0, energy=1.0)
        record = issue.process(source, current_view(source))
        trace = TieredRenderModel().process([record])[0][0]
        terminal_trace = replace(
            trace,
            frame_id=PUBLIC_MAX_FRAME_ID,
            projection=replace(trace.projection, frame_id=PUBLIC_MAX_FRAME_ID),
        )
        model = TrainingUpdateModel()
        model.seed_state(state())
        terminal_target = TargetPixel(terminal_trace.pixel, PUBLIC_MAX_FRAME_ID, (0.3, 0.2, 0.1))
        model.buffer_target(terminal_target)

        self.assertIsNone(model.process_trace(terminal_trace))
        self.assertEqual(model.invalid_pairs, 1)
        with self.assertRaisesRegex(ValueError, "occupied"):
            model.buffer_target(terminal_target)


if __name__ == "__main__":
    unittest.main()
