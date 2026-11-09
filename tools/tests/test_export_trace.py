"""Contract checks for the public JSON-to-trace exporter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORTER_PATH = ROOT / "tools" / "export_trace.py"
FIXTURE_PATH = ROOT / "docs" / "fixtures" / "six_frame.scdgs-trace"


def load_exporter() -> object:
    """Load the standalone exporter without requiring a package installation."""

    spec = importlib.util.spec_from_file_location("scdgs_export_trace", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the public trace exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_records() -> list[dict[str, str]]:
    """Convert the public text fixture into the exporter's JSON record shape."""

    records: list[dict[str, str]] = []
    for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split("|")
        kind = tokens[0]
        record: dict[str, str] = {"record_type" if kind == "REQUEST" else "kind": kind}
        record.update(dict(token.split("=", 1) for token in tokens[1:]))
        records.append(record)
    return records


def terminal_training_frame_records() -> list[dict[str, str]]:
    """Keep the bootstrap and first training frame, then move that frame to 65535."""

    records: list[dict[str, str]] = []
    for record in fixture_records():
        frame = record["id"] if record.get("kind") == "FRAME" else record["frame"]
        if frame not in {"0", "1"}:
            continue
        copied = record.copy()
        if copied.get("kind") == "FRAME" and copied["id"] == "1":
            copied["id"] = "65535"
        elif copied.get("frame") == "1":
            copied["frame"] = "65535"
        records.append(copied)
    return records


class ExportTraceContractTest(unittest.TestCase):
    """Keep exporter validation aligned with the public Scala trace validator."""

    def test_accepts_public_fixture(self) -> None:
        exporter = load_exporter()
        normalized = exporter.validate_records(fixture_records(), max_gaussians=16)
        self.assertEqual(len(normalized), 29)

    def test_rejects_repeated_update_source_frame(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        first_update = next(index for index, record in enumerate(records) if record.get("kind") == "UPDATE")
        records.insert(first_update + 1, records[first_update].copy())
        with self.assertRaisesRegex(ValueError, "not strictly monotonic"):
            exporter.validate_records(records, max_gaussians=16)

    def test_rejects_out_of_width_frames_and_terminal_training_feedback(self) -> None:
        exporter = load_exporter()
        wide_boundary = fixture_records()
        next(record for record in wide_boundary if record.get("kind") == "FRAME")["id"] = "65536"
        with self.assertRaisesRegex(ValueError, "exceeds the 16-bit public frame range"):
            exporter.validate_records(wide_boundary, max_gaussians=16)

        wide_target = fixture_records()
        next(record for record in wide_target if record.get("kind") == "TARGET")["frame"] = "65536"
        with self.assertRaisesRegex(ValueError, "exceeds the 16-bit public frame range"):
            exporter.validate_records(wide_target, max_gaussians=16)

        with self.assertRaisesRegex(ValueError, "TRAJECTORY must use a frame below 65535"):
            exporter.validate_records(terminal_training_frame_records(), max_gaussians=16)

    def test_rejects_out_of_width_tile_and_blend_fields(self) -> None:
        exporter = load_exporter()
        wide_tile = fixture_records()
        next(record for record in wide_tile if record.get("kind") == "PROJECTION")["tile"] = "4096"
        with self.assertRaisesRegex(ValueError, "exceeds the 12-bit public tile range"):
            exporter.validate_records(wide_tile, max_gaussians=16)

        wide_blend = fixture_records()
        next(record for record in wide_blend if record.get("kind") == "TRAJECTORY")["blend"] = "64"
        with self.assertRaisesRegex(ValueError, "exceeds the public TRU blend-order range \\[0, 63\\]"):
            exporter.validate_records(wide_blend, max_gaussians=16)

    def test_rejects_a_nonfinite_projection_depth(self) -> None:
        exporter = load_exporter()
        for depth in ("0x7fc00000", "0x7f800000"):
            records = fixture_records()
            next(record for record in records if record.get("kind") == "PROJECTION")["depth"] = depth
            with self.assertRaisesRegex(ValueError, "depth must be a finite binary32 value"):
                exporter.validate_records(records, max_gaussians=16)

    def test_rejects_a_nonpositive_projection_radius(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        projection = next(record for record in records if record.get("kind") == "PROJECTION")
        projection["radius"] = "0x00000000"
        with self.assertRaisesRegex(ValueError, "positive normal"):
            exporter.validate_records(records, max_gaussians=16)

    def test_rejects_a_commit_without_its_matching_canonical_update(self) -> None:
        exporter = load_exporter()
        records = [
            record
            for record in fixture_records()
            if not (record.get("kind") == "UPDATE" and record["frame"] == "1")
        ]
        with self.assertRaisesRegex(ValueError, "commit for gid 7, frame 1 has no preceding UPDATE"):
            exporter.validate_records(records, max_gaussians=16)

    def test_requires_one_siu_propagation_and_preserves_mandatory_commits(self) -> None:
        exporter = load_exporter()
        missing_propagation = [
            record
            for record in fixture_records()
            if not (
                record.get("kind") == "EVENT"
                and record["source"] == "commit"
                and record["frame"] == "2"
            )
        ]
        with self.assertRaisesRegex(ValueError, "trace omits SIU propagation EVENT for gid 7, frame 2"):
            exporter.validate_records(missing_propagation, max_gaussians=16)

        warmup_confirmation = fixture_records()
        next(
            record for record in warmup_confirmation
            if record.get("kind") == "EVENT" and record["source"] == "commit" and record["frame"] == "2"
        )["source"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "confirmation for gid 7, frame 2 violates SIU's required commit"):
            exporter.validate_records(warmup_confirmation, max_gaussians=16)

        structural_confirmation = fixture_records()
        next(
            record for record in structural_confirmation
            if record.get("kind") == "EVENT" and record["source"] == "commit" and record["frame"] == "5"
        )["source"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "confirmation for gid 7, frame 5 violates SIU's required commit"):
            exporter.validate_records(structural_confirmation, max_gaussians=16)

        allowed_confirmation = fixture_records()
        next(
            record for record in allowed_confirmation
            if record.get("kind") == "UPDATE" and record["frame"] == "5"
        )["struct"] = "none"
        next(
            record for record in allowed_confirmation
            if record.get("kind") == "EVENT" and record["source"] == "commit" and record["frame"] == "5"
        )["source"] = "confirmation"
        exporter.validate_records(allowed_confirmation, max_gaussians=16)

    def test_requires_current_views_to_follow_a_commit(self) -> None:
        exporter = load_exporter()
        no_commit = [
            record
            for record in fixture_records()
            if not (record.get("kind") == "EVENT" and record["source"] == "commit")
        ]
        with self.assertRaisesRegex(ValueError, "current PROJECTION for gid 7, frame 1 has no preceding commit"):
            exporter.validate_records(no_commit, max_gaussians=16)

    def test_rejects_duplicate_stage_handoff_records_for_one_source_frame(self) -> None:
        exporter = load_exporter()
        cases = (
            (lambda record: record.get("kind") == "STATE", "duplicate STATE"),
            (
                lambda record: record.get("kind") == "PROJECTION" and record["role"] == "current",
                "duplicate current PROJECTION",
            ),
            (
                lambda record: record.get("kind") == "EVENT" and record["source"] == "commit",
                "duplicate commit",
            ),
            (lambda record: record.get("record_type") == "REQUEST", "duplicate REQUEST"),
            (lambda record: record.get("kind") == "TRAJECTORY", "duplicate TRAJECTORY"),
            (
                lambda record: record.get("kind") == "EVENT" and record["source"] == "render-read",
                "duplicate render-read",
            ),
        )
        for matches, expected in cases:
            records = fixture_records()
            index = next(index for index, record in enumerate(records) if matches(record))
            records.insert(index + 1, records[index].copy())
            with self.assertRaisesRegex(ValueError, expected):
                exporter.validate_records(records, max_gaussians=16)

    def test_requires_updates_to_use_a_retained_training_projection(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        for record in records:
            if record.get("kind") == "PROJECTION":
                record["request"] = "user"
            elif record.get("record_type") == "REQUEST":
                record["kind"] = "user"
        with self.assertRaisesRegex(ValueError, "prior PROJECTION must request training"):
            exporter.validate_records(records, max_gaussians=16)

    def test_requires_one_consumed_bootstrap_prior_and_one_request_per_current_view(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        bootstrap_index = next(
            index
            for index, record in enumerate(records)
            if record.get("kind") == "PROJECTION" and record["role"] == "prior"
        )
        duplicate = records.copy()
        duplicate.insert(bootstrap_index + 1, records[bootstrap_index].copy())
        with self.assertRaisesRegex(ValueError, "duplicate bootstrap prior PROJECTION"):
            exporter.validate_records(duplicate, max_gaussians=16)

        orphan = records.copy()
        orphan_prior = records[bootstrap_index].copy()
        orphan_prior["gid"] = "8"
        orphan.insert(bootstrap_index + 1, orphan_prior)
        with self.assertRaisesRegex(ValueError, "bootstrap prior PROJECTION for gid 8 has no matching first UPDATE"):
            exporter.validate_records(orphan, max_gaussians=16)

        missing_request = [record for record in records if record.get("record_type") != "REQUEST"]
        with self.assertRaisesRegex(ValueError, "current PROJECTION for gid 7, frame 1 has no matching REQUEST"):
            exporter.validate_records(missing_request, max_gaussians=16)

    def test_requires_trajectories_to_use_the_latest_retained_training_projection(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        template_projection = next(
            record for record in records if record.get("kind") == "PROJECTION" and record["role"] == "current"
        )
        template_trajectory = next(record for record in records if record.get("kind") == "TRAJECTORY")
        user_projection = template_projection.copy()
        user_projection.update(frame="1", request="user", role="current")
        user_request = {"record_type": "REQUEST", "gid": "7", "frame": "1", "kind": "user"}
        user_trajectory = template_trajectory.copy()
        user_trajectory["frame"] = "1"
        commit_index = next(
            index
            for index, record in enumerate(records)
            if record.get("kind") == "EVENT" and record["source"] == "commit" and record["frame"] == "1"
        )
        records[commit_index + 1:commit_index + 1] = [user_projection, user_request, user_trajectory]
        with self.assertRaisesRegex(ValueError, "TRAJECTORY for gid 7 requires a same-frame training current PROJECTION"):
            exporter.validate_records(records, max_gaussians=16)

    def test_requires_trajectory_pixel_to_match_its_current_projection_tile(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        trajectory = next(record for record in records if record.get("kind") == "TRAJECTORY")
        target = next(record for record in records if record.get("kind") == "TARGET")
        trajectory["pixel"] = "4"
        target["pixel"] = "4"
        with self.assertRaisesRegex(ValueError, "TRAJECTORY pixel must equal its current PROJECTION tile"):
            exporter.validate_records(records, max_gaussians=16)

    def test_rejects_a_filter_trajectory(self) -> None:
        exporter = load_exporter()
        records = fixture_records()
        trajectory = next(record for record in records if record.get("kind") == "TRAJECTORY")
        trajectory["path"] = "filter"
        with self.assertRaisesRegex(ValueError, "TRAJECTORY path must be full or light"):
            exporter.validate_records(records, max_gaussians=16)

    def test_requires_render_reads_to_match_the_training_trajectory_source_frame(self) -> None:
        exporter = load_exporter()
        records = [record for record in fixture_records() if record.get("kind") != "TRAJECTORY"]
        with self.assertRaisesRegex(
            ValueError, "render-read for gid 7, frame 1 has no matching training TRAJECTORY"
        ):
            exporter.validate_records(records, max_gaussians=16)

        missing_read = [
            record
            for record in fixture_records()
            if not (record.get("kind") == "EVENT" and record["source"] == "render-read")
        ]
        with self.assertRaisesRegex(ValueError, "trace omits render-read for training TRAJECTORY gid 7, frame 1"):
            exporter.validate_records(missing_read, max_gaussians=16)

        delayed_read = [
            record
            for record in fixture_records()
            if not (record.get("kind") == "EVENT" and record["source"] == "render-read")
        ]
        frame_two = next(
            index
            for index, record in enumerate(delayed_read)
            if record.get("kind") == "FRAME" and record["id"] == "2" and record["edge"] == "begin"
        )
        delayed_read.insert(frame_two + 1, {"kind": "EVENT", "gid": "7", "frame": "2", "source": "render-read"})
        with self.assertRaisesRegex(
            ValueError, "render-read for gid 7, frame 2 has no matching training TRAJECTORY"
        ):
            exporter.validate_records(delayed_read, max_gaussians=16)

    def test_requires_one_target_for_each_training_trajectory(self) -> None:
        exporter = load_exporter()
        missing_target = [record for record in fixture_records() if record.get("kind") != "TARGET"]
        with self.assertRaisesRegex(
            ValueError, "trace omits TARGET for training TRAJECTORY pixel 3, frame 1"
        ):
            exporter.validate_records(missing_target, max_gaussians=16)

        records = fixture_records()
        target_index = next(index for index, record in enumerate(records) if record.get("kind") == "TARGET")
        records.insert(target_index + 1, records[target_index].copy())
        with self.assertRaisesRegex(
            ValueError, "trace has TARGET for pixel 3, frame 1 without a matching training TRAJECTORY"
        ):
            exporter.validate_records(records, max_gaussians=16)
