"""Convert local records into the public SCD-GS trace grammar.

The input is a JSON list of already-exported mechanism records. This tool does
not import a model package, inspect datasets, or emit experimental metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SH_BASES = 16
SH_COEFFICIENTS = 3 * SH_BASES
REQUIRED = {
    "FRAME": {"id", "edge"},
    "STATE": {"gid", "frame", "mean", "covariance", "opacity", "sh", "tier"},
    "UPDATE": {"gid", "frame", "dmu", "dsigma", "do", "dsh", "n", "struct"},
    "PROJECTION": {
        "gid",
        "frame",
        "r_cam",
        "j_pi",
        "j_cov",
        "view_direction",
        "sh_basis",
        "center",
        "conic",
        "radius",
        "area",
        "opacity",
        "view_color",
        "request",
        "role",
        "tile",
        "depth",
    },
    "EVENT": {"gid", "frame", "source"},
    "REQUEST": {"gid", "frame", "kind"},
    "TRAJECTORY": {
        "gid",
        "frame",
        "pixel",
        "blend",
        "coverage",
        "transmittance",
        "color",
        "path",
    },
    "TARGET": {"pixel", "frame", "rgb"},
}

BINARY32_FIELDS = {
    "STATE": {"mean": 3, "covariance": 6, "opacity": 1, "sh": SH_COEFFICIENTS},
    "UPDATE": {"dmu": 3, "dsigma": 6, "do": 1, "dsh": SH_COEFFICIENTS, "n": 1},
    "PROJECTION": {
        "r_cam": 9,
        "j_pi": 6,
        "j_cov": 6,
        "view_direction": 3,
        "sh_basis": SH_BASES,
        "center": 2,
        "conic": 3,
        "radius": 1,
        "area": 1,
        "opacity": 1,
        "view_color": 3,
        "depth": 1,
    },
    "TRAJECTORY": {"coverage": 1, "transmittance": 1, "color": 3},
    "TARGET": {"rgb": 3},
}
STRUCTURAL_MARKERS = {"none", "insert", "delete", "invalidate", "reset"}
TIERS = {"active", "resident", "candidate"}
REQUEST_KINDS = {"user", "training"}
PROJECTION_ROLES = {"prior", "current"}
EVENT_SOURCES = {"commit", "confirmation", "render-read"}
TRAJECTORY_PATHS = {"full", "light"}
HEX_WORD = re.compile(r"0x[0-9a-fA-F]{1,8}")
NATURAL = re.compile(r"[0-9]+")
MAX_TRACE_INTEGER = (1 << 31) - 1
# Pinned to ScdGsParams().frameWidth. The exporter has no Scala dependency, so
# this mirrors the public default elaboration shape explicitly.
PUBLIC_FRAME_WIDTH = 16
MAX_FRAME_ID = (1 << PUBLIC_FRAME_WIDTH) - 1
PUBLIC_TILE_WIDTH = 12
MAX_TILE_ID = (1 << PUBLIC_TILE_WIDTH) - 1
PUBLIC_RENDER_QUEUE_DEPTH = 64
MAX_BLEND_ORDER = PUBLIC_RENDER_QUEUE_DEPTH - 1


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _as_text(value: object, description: str) -> str:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        raise ValueError(f"{description} must be a scalar string or number")
    return str(value)


def _natural(value: str, description: str) -> int:
    if not NATURAL.fullmatch(value):
        raise ValueError(f"{description} must be a non-negative decimal integer")
    parsed = int(value)
    if parsed > MAX_TRACE_INTEGER:
        raise ValueError(f"{description} exceeds the public trace integer range")
    return parsed


def _frame_identifier(value: str, description: str) -> int:
    parsed = _natural(value, description)
    if parsed > MAX_FRAME_ID:
        raise ValueError(f"{description} exceeds the {PUBLIC_FRAME_WIDTH}-bit public frame range")
    return parsed


def _tile_identifier(value: str, description: str) -> int:
    parsed = _natural(value, description)
    if parsed > MAX_TILE_ID:
        raise ValueError(f"{description} exceeds the {PUBLIC_TILE_WIDTH}-bit public tile range")
    return parsed


def _blend_order(value: str, description: str) -> int:
    parsed = _natural(value, description)
    if parsed > MAX_BLEND_ORDER:
        raise ValueError(
            f"{description} exceeds the public TRU blend-order range [0, {MAX_BLEND_ORDER}]"
        )
    return parsed


def _binary32_list(value: str, width: int, description: str) -> None:
    words = value.split(",")
    if len(words) != width or not all(HEX_WORD.fullmatch(word) for word in words):
        raise ValueError(f"{description} must contain {width} binary32 hex words")


def _positive_normal_binary32(value: str, description: str) -> None:
    if not HEX_WORD.fullmatch(value):
        raise ValueError(f"{description} must be a binary32 hex word")
    word = int(value, 16)
    exponent = (word >> 23) & 0xFF
    if word >> 31 or exponent in {0, 0xFF}:
        raise ValueError(f"{description} must be a positive normal binary32 value")


def _finite_binary32(value: str, description: str) -> None:
    if not HEX_WORD.fullmatch(value):
        raise ValueError(f"{description} must be a binary32 hex word")
    if (int(value, 16) >> 23) & 0xFF == 0xFF:
        raise ValueError(f"{description} must be a finite binary32 value")


def _record_fields(record: object, number: int) -> tuple[str, dict[str, str]]:
    if not isinstance(record, dict):
        raise ValueError(f"trace record {number} must be a JSON object")
    if "record_type" in record:
        kind = record["record_type"]
        field_items = ((key, value) for key, value in record.items() if key != "record_type")
    else:
        kind = record.get("kind")
        field_items = ((key, value) for key, value in record.items() if key != "kind")
    if not isinstance(kind, str) or kind not in REQUIRED:
        raise ValueError(f"trace record {number} has unsupported kind {kind!r}")
    if kind == "REQUEST" and "record_type" not in record:
        raise ValueError(
            f"trace record {number} REQUEST records must use record_type to preserve their kind field"
        )
    fields = {
        key: _as_text(value, f"trace record {number} field {key}")
        for key, value in field_items
    }
    missing = REQUIRED[kind] - fields.keys()
    unknown = fields.keys() - REQUIRED[kind]
    if missing or unknown:
        raise ValueError(f"trace record {number} {kind}: missing={sorted(missing)} unknown={sorted(unknown)}")
    return kind, fields


def _validate_record_fields(kind: str, fields: dict[str, str], number: int, max_gaussians: int) -> None:
    def natural(field: str) -> int:
        return _natural(fields[field], f"trace record {number} field {field}")

    if kind in {"STATE", "UPDATE", "PROJECTION", "EVENT", "REQUEST", "TRAJECTORY"}:
        gid = natural("gid")
        if gid >= max_gaussians:
            raise ValueError(f"trace record {number} gid {gid} is outside [0, {max_gaussians})")
        _frame_identifier(fields["frame"], f"trace record {number} frame identifier")

    if kind == "FRAME":
        _frame_identifier(fields["id"], f"trace record {number} frame identifier")
        if fields["edge"] not in {"begin", "end"}:
            raise ValueError(f"trace record {number} has invalid frame edge {fields['edge']!r}")
    elif kind == "STATE":
        if fields["tier"] not in TIERS:
            raise ValueError(f"trace record {number} has invalid tier {fields['tier']!r}")
    elif kind == "UPDATE":
        if fields["struct"] not in STRUCTURAL_MARKERS:
            raise ValueError(f"trace record {number} has invalid structural marker {fields['struct']!r}")
    elif kind == "PROJECTION":
        _tile_identifier(fields["tile"], f"trace record {number} tile")
        _positive_normal_binary32(fields["radius"], f"trace record {number} radius")
        _finite_binary32(fields["depth"], f"trace record {number} depth")
        if fields["request"] not in REQUEST_KINDS:
            raise ValueError(f"trace record {number} has invalid request kind {fields['request']!r}")
        if fields["role"] not in PROJECTION_ROLES:
            raise ValueError(f"trace record {number} has invalid projection role {fields['role']!r}")
    elif kind == "EVENT":
        if fields["source"] not in EVENT_SOURCES:
            raise ValueError(f"trace record {number} has invalid event source {fields['source']!r}")
    elif kind == "REQUEST":
        if fields["kind"] not in REQUEST_KINDS:
            raise ValueError(f"trace record {number} has invalid request kind {fields['kind']!r}")
    elif kind == "TRAJECTORY":
        natural("pixel")
        _blend_order(fields["blend"], f"trace record {number} blend")
        if fields["path"] not in TRAJECTORY_PATHS:
            raise ValueError(f"trace record {number} TRAJECTORY path must be full or light")
        if int(fields["frame"]) >= MAX_FRAME_ID:
            raise ValueError(
                f"trace record {number} TRAJECTORY must use a frame below {MAX_FRAME_ID} "
                "to preserve TUU feedback causality"
            )
    elif kind == "TARGET":
        natural("pixel")
        _frame_identifier(fields["frame"], f"trace record {number} frame identifier")

    for field, width in BINARY32_FIELDS.get(kind, {}).items():
        _binary32_list(fields[field], width, f"trace record {number} field {field}")


def validate_records(records: list[object], max_gaussians: int) -> list[tuple[str, dict[str, str]]]:
    """Validate the public trace grammar and frame protocol before writing output."""

    if max_gaussians <= 0:
        raise ValueError("max_gaussians must be positive")

    normalized: list[tuple[str, dict[str, str]]] = []
    open_frame: int | None = None
    previous_begin = -1
    seeded_gids: set[int] = set()
    bootstrap_prior_frames: dict[int, int] = {}
    consumed_bootstrap_priors: set[int] = set()
    prior_summary_frames: dict[int, int] = {}
    current_view_requests: dict[tuple[int, int], str] = {}
    current_view_tiles: dict[tuple[int, int], int] = {}
    update_frames: dict[int, int] = {}
    validated_updates: set[tuple[int, int]] = set()
    required_commit_updates: set[tuple[int, int]] = set()
    accepted_update_counts: dict[int, int] = {}
    committed_updates: set[tuple[int, int]] = set()
    propagation_events: dict[tuple[int, int], str] = {}
    requested_updates: set[tuple[int, int]] = set()
    training_trajectories: set[tuple[int, int]] = set()
    training_target_counts: dict[tuple[int, int], int] = {}
    target_counts: dict[tuple[int, int], int] = {}
    render_read_events: set[tuple[int, int]] = set()

    for number, raw_record in enumerate(records, start=1):
        kind, fields = _record_fields(raw_record, number)
        _validate_record_fields(kind, fields, number, max_gaussians)
        if kind == "FRAME":
            frame = int(fields["id"])
            if fields["edge"] == "begin":
                if open_frame is not None:
                    raise ValueError(f"trace record {number} opens frame {frame} before closing frame {open_frame}")
                if frame <= previous_begin:
                    raise ValueError(f"trace record {number} has nonmonotonic frame identifier")
                open_frame = frame
                previous_begin = frame
            else:
                if open_frame != frame:
                    raise ValueError(f"trace record {number} closes frame {frame} without a matching begin")
                open_frame = None
        else:
            frame = int(fields["frame"])
            if open_frame != frame:
                raise ValueError(f"trace record {number} belongs to frame {frame} outside its open frame")

            gid = int(fields["gid"]) if "gid" in fields else None
            if kind == "STATE":
                if gid in seeded_gids:
                    raise ValueError(f"trace record {number} has duplicate STATE for gid {gid}")
                if gid in update_frames:
                    raise ValueError(f"trace record {number} STATE for gid {gid} is not initial state")
                seeded_gids.add(gid)
            elif kind == "PROJECTION":
                if fields["role"] == "prior":
                    if fields["request"] != "training":
                        raise ValueError(f"trace record {number} prior PROJECTION must request training")
                    if gid in update_frames:
                        raise ValueError(f"trace record {number} prior PROJECTION for gid {gid} is not bootstrap state")
                    if gid in bootstrap_prior_frames:
                        raise ValueError(
                            f"trace record {number} has duplicate bootstrap prior PROJECTION for gid {gid}"
                        )
                    bootstrap_prior_frames[gid] = frame
                    prior_summary_frames[gid] = frame
                else:
                    if (gid, frame) not in validated_updates:
                        raise ValueError(
                            f"trace record {number} current PROJECTION for gid {gid}, frame {frame} has no preceding UPDATE"
                        )
                    if (gid, frame) not in committed_updates:
                        raise ValueError(
                            f"trace record {number} current PROJECTION for gid {gid}, frame {frame} has no preceding commit"
                        )
                    if (gid, frame) in current_view_requests:
                        raise ValueError(
                            f"trace record {number} has duplicate current PROJECTION for gid {gid}, frame {frame}"
                        )
                    current_view_requests[(gid, frame)] = fields["request"]
                    current_view_tiles[(gid, frame)] = int(fields["tile"])
            elif kind == "UPDATE":
                if gid not in seeded_gids:
                    raise ValueError(f"trace record {number} UPDATE for gid {gid} has no preceding STATE")
                if gid not in prior_summary_frames:
                    raise ValueError(f"trace record {number} UPDATE for gid {gid} has no retained prior PROJECTION")
                if prior_summary_frames[gid] >= frame:
                    raise ValueError(
                        f"trace record {number} UPDATE for gid {gid} requires a preceding training PROJECTION"
                    )
                if gid in update_frames and update_frames[gid] >= frame:
                    raise ValueError(f"trace record {number} UPDATE for gid {gid} is not strictly monotonic")
                if gid not in update_frames:
                    consumed_bootstrap_priors.add(gid)
                accepted_count = accepted_update_counts.get(gid, 0) + 1
                accepted_update_counts[gid] = accepted_count
                if accepted_count < 5 or fields["struct"] != "none":
                    required_commit_updates.add((gid, frame))
                update_frames[gid] = frame
                validated_updates.add((gid, frame))
            elif kind == "EVENT":
                source = fields["source"]
                if source in {"commit", "confirmation"}:
                    if (gid, frame) not in validated_updates:
                        raise ValueError(
                            f"trace record {number} {source} for gid {gid}, frame {frame} has no preceding UPDATE"
                        )
                    if (gid, frame) in propagation_events:
                        raise ValueError(f"trace record {number} has duplicate {source} for gid {gid}, frame {frame}")
                    if source == "confirmation" and (gid, frame) in required_commit_updates:
                        raise ValueError(
                            f"trace record {number} confirmation for gid {gid}, frame {frame} violates SIU's required commit"
                        )
                    propagation_events[(gid, frame)] = source
                    if source == "commit":
                        committed_updates.add((gid, frame))
                else:
                    if (gid, frame) not in training_trajectories:
                        raise ValueError(
                            f"trace record {number} render-read for gid {gid}, frame {frame} has no matching training TRAJECTORY"
                        )
                    if (gid, frame) in render_read_events:
                        raise ValueError(
                            f"trace record {number} has duplicate render-read for gid {gid}, frame {frame}"
                        )
                    render_read_events.add((gid, frame))
            elif kind == "REQUEST":
                expected = current_view_requests.get((gid, frame))
                if expected != fields["kind"]:
                    raise ValueError(
                        f"trace record {number} REQUEST for gid {gid}, frame {frame} "
                        "must match a same-frame current PROJECTION request"
                    )
                if (gid, frame) not in committed_updates:
                    raise ValueError(
                        f"trace record {number} REQUEST for gid {gid}, frame {frame} has no preceding commit"
                    )
                if (gid, frame) in requested_updates:
                    raise ValueError(f"trace record {number} has duplicate REQUEST for gid {gid}, frame {frame}")
                requested_updates.add((gid, frame))
            elif kind == "TRAJECTORY":
                if current_view_requests.get((gid, frame)) != "training":
                    raise ValueError(
                        f"trace record {number} TRAJECTORY for gid {gid} requires a same-frame training current PROJECTION"
                    )
                if current_view_tiles.get((gid, frame)) != int(fields["pixel"]):
                    raise ValueError(
                        f"trace record {number} TRAJECTORY pixel must equal its current PROJECTION tile"
                    )
                if (gid, frame) not in committed_updates:
                    raise ValueError(
                        f"trace record {number} TRAJECTORY for gid {gid}, frame {frame} has no preceding commit"
                    )
                if (gid, frame) in training_trajectories:
                    raise ValueError(f"trace record {number} has duplicate TRAJECTORY for gid {gid}, frame {frame}")
                training_trajectories.add((gid, frame))
                target_key = (int(fields["pixel"]), frame)
                training_target_counts[target_key] = training_target_counts.get(target_key, 0) + 1
                prior_summary_frames[gid] = frame
            elif kind == "TARGET":
                target_key = (int(fields["pixel"]), frame)
                target_counts[target_key] = target_counts.get(target_key, 0) + 1
        normalized.append((kind, fields))

    if open_frame is not None:
        raise ValueError(f"trace ends with unclosed frame {open_frame}")
    missing_bootstrap_update = sorted(set(bootstrap_prior_frames) - consumed_bootstrap_priors)
    if missing_bootstrap_update:
        gid = missing_bootstrap_update[0]
        raise ValueError(f"trace bootstrap prior PROJECTION for gid {gid} has no matching first UPDATE")
    missing_propagations = sorted(set(validated_updates) - set(propagation_events), key=lambda item: (item[1], item[0]))
    if missing_propagations:
        gid, frame = missing_propagations[0]
        raise ValueError(f"trace omits SIU propagation EVENT for gid {gid}, frame {frame}")
    missing_requests = sorted(set(current_view_requests) - requested_updates, key=lambda item: (item[1], item[0]))
    if missing_requests:
        gid, frame = missing_requests[0]
        raise ValueError(f"trace current PROJECTION for gid {gid}, frame {frame} has no matching REQUEST")
    missing_render_reads = sorted(training_trajectories - render_read_events, key=lambda item: (item[1], item[0]))
    if missing_render_reads:
        gid, frame = missing_render_reads[0]
        raise ValueError(f"trace omits render-read for training TRAJECTORY gid {gid}, frame {frame}")
    for (pixel, frame), expected in sorted(training_target_counts.items(), key=lambda item: (item[0][1], item[0][0])):
        if target_counts.get((pixel, frame), 0) < expected:
            raise ValueError(f"trace omits TARGET for training TRAJECTORY pixel {pixel}, frame {frame}")
    for (pixel, frame), actual in sorted(target_counts.items(), key=lambda item: (item[0][1], item[0][0])):
        if training_target_counts.get((pixel, frame), 0) < actual:
            raise ValueError(
                f"trace has TARGET for pixel {pixel}, frame {frame} without a matching training TRAJECTORY"
            )
    return normalized


def format_record(record: dict[str, object]) -> str:
    """Serialize one already schema-valid record in a deterministic field order."""

    kind, fields = _record_fields(record, 1)
    _validate_record_fields(kind, fields, 1, max_gaussians=4096)
    return "|".join([kind, *[f"{key}={fields[key]}" for key in sorted(fields)]])


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local records into SCD-GS mechanism trace lines.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gaussians", type=positive_int, default=4096)
    args = parser.parse_args()
    records = json.loads(args.input_json.read_text(encoding="utf-8"), object_pairs_hook=_without_duplicate_keys)
    if not isinstance(records, list):
        raise ValueError("input JSON must be a list of mechanism records")
    lines = [
        "|".join([kind, *[f"{key}={fields[key]}" for key in sorted(fields)]])
        for kind, fields in validate_records(records, args.max_gaussians)
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
