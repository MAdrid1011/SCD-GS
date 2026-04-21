"""One-off migration for existing ``results/GLASS_ablation`` JSONs.

Rewrites every ``results/GLASS_ablation/<dataset>/<baseline>/mode_*.json``
in place so it conforms to the canonical schema in
:mod:`simulator.ablation.result_schema`:

  * Drops misleading / simulator-host-only fields:
      - ``avg_train_time_s``      (GPU wall-clock per frame)
      - ``all_frame_times_s``     (GPU wall-clock list)
      - ``asic_time_s_1ghz``      (== ``asic_total_cycles / 1e9``; omits
                                   NTC-training cycles, easy to misread)
  * Renames ``*_ntc_iters`` → ``*_train_iters`` for consistency across
    baselines (HiCoM trains Gaussians, not an NTC; 4DGC trains a MEM).
  * Injects any newly-introduced keys (``mode_label``,
    ``run_suppression_ratio``) and fills in ``None`` / 0 for keys the
    old file never produced.
  * Re-serialises with :func:`result_schema.build_result` so the on-disk
    field order matches :data:`SCHEMA_KEYS`.

Usage::

    python -m simulator.ablation.migrate_results         # rewrite in place
    python -m simulator.ablation.migrate_results --dry-run

The script is idempotent: running it twice is a no-op because the
second invocation reads back canonical JSON and emits the same bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulator.ablation.result_schema import SCHEMA_KEYS, build_result

GLASS_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = GLASS_DIR / "results" / "GLASS_ablation"


def _legacy_to_canonical(old: dict) -> dict:
    """Translate a legacy JSON dict into kwargs for :func:`build_result`.

    Handles three generations of field names:

      - ``base_ntc_iters`` / ``avg_ntc_iters`` / ``per_frame_ntc_iters``
        (3DGStream, 4DGC)  — renamed to ``*_train_iters``.
      - HiCoM legacy output omitted the iter schedule entirely — we
        reconstruct a constant-budget schedule of length
        ``n_frames_trained`` using the default ``num_incr_steps=200``
        so the new key is populated.
      - QUEEN legacy output never produced any iter schedule.  We leave
        the unified keys as ``None``.
    """
    quality = {
        "psnr":  old.get("psnr"),
        "ssim":  old.get("ssim"),
        "lpips": old.get("lpips"),
    }

    # stream_stats-style keys lived at the top level in the old schema.
    stream_stats = {
        k: old.get(k, 0) for k in (
            "frame_total_bytes", "frame_suppressed_bytes",
            "frame_active_bytes", "frame_suppression_ratio",
            "frame_unique_targets",
            "run_total_bytes", "run_suppressed_bytes",
            "n_asc_rsb", "n_css", "n_known", "n_total",
        )
    }

    cycle_summary = {
        k: old.get(k, 0) for k in (
            "asic_ntc_train_render_cycles",
            "asic_ntc_train_cycles",
            "asic_ntc_cycles",
            "asic_puoe_cycles",
            "asic_ucn_cycles",
            "asic_ice_cycles",
            "asic_ilsf_cycles",
            "asic_salience_cycles",
            "asic_optimizer_cycles",
            "asic_render_cycles",
            "asic_gradient_pipeline_cycles",
            "asic_post_render_cycles",
            "asic_total_cycles",
            "asic_commit_time_s",
        )
    }

    baseline        = old.get("baseline", "?")
    n_frames        = int(old.get("n_frames_trained", 0))
    per_frame_iters = old.get("per_frame_ntc_iters")
    base_iters      = old.get("base_ntc_iters")
    avg_iters       = old.get("avg_ntc_iters")

    # HiCoM legacy files never wrote the schedule.  Inject a constant
    # budget of 200 × N frames so the unified schema has non-null
    # values matching what the runner actually executed.
    if baseline == "HiCoM" and per_frame_iters is None:
        base_iters      = 200
        per_frame_iters = [200] * n_frames
        avg_iters       = 200.0

    return dict(
        baseline=baseline,
        dataset=old.get("dataset", "?"),
        mode=int(old.get("mode", 0)),
        n_frames_trained=n_frames,
        quality=quality,
        render_fps=old.get("render_fps"),
        total_storage_mb=old.get("total_storage_mb"),
        delta_storage_mb=old.get("delta_storage_mb"),
        asic_train_time_s=old.get("asic_train_time_s"),
        cycle_summary=cycle_summary,
        stream_stats=stream_stats,
        base_train_iters=base_iters,
        avg_train_iters=avg_iters,
        per_frame_train_iters=per_frame_iters,
        incremental=bool(old.get("incremental", True)),
    )


def migrate_file(path: Path, *, dry_run: bool = False) -> bool:
    with path.open() as f:
        old = json.load(f)

    if list(old.keys()) == list(SCHEMA_KEYS):
        # Already canonical; short-circuit without touching mtime.
        return False

    new = build_result(**_legacy_to_canonical(old))
    if dry_run:
        print(f"  would rewrite {path.relative_to(GLASS_DIR)}")
        return True
    with path.open("w") as f:
        json.dump(new, f, indent=2)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=Path, default=RESULTS_ROOT)
    args = ap.parse_args()

    if not args.root.exists():
        print(f"No results at {args.root}", file=sys.stderr)
        return

    files = sorted(args.root.glob("*/*/mode_*.json"))
    rewritten = 0
    for f in files:
        if migrate_file(f, dry_run=args.dry_run):
            rewritten += 1
    tag = "would rewrite" if args.dry_run else "rewrote"
    print(f"{tag} {rewritten}/{len(files)} files under {args.root}")


if __name__ == "__main__":
    main()
