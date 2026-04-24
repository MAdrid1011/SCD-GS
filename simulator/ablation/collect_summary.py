#!/usr/bin/env python
"""Collect GLASS ablation JSONs into a single comparison table.

Produces:
  - results/GLASS_ablation/flame_steak/ablation_summary.md   (human-readable)
  - results/GLASS_ablation/flame_steak/ablation_summary.json (machine-readable)
"""
from __future__ import annotations

import json
from pathlib import Path

from simulator.ablation.result_schema import MODE_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]
ABL_ROOT  = REPO_ROOT / "results" / "GLASS_ablation" / "flame_steak"
BASELINES = ["3DGStream", "QUEEN", "HiCoM", "4DGC"]
MODES     = [0, 1, 2, 3]

# Columns intentionally omit GPU wall-clock (simulator-host artefact)
# and ``asic_commit_time_s`` (a sub-component of the headline
# ``asic_train_time_s``; exposing both in the summary invites readers
# to add them, and the delta between modes is already visible in the
# headline column).  ``asic_commit_time_s`` is still emitted in every
# ``mode_*.json`` for diagnostic decomposition.
COLUMNS = [
    ("mode",               "Mode"),
    ("n_frames_trained",   "Frames"),
    ("asic_train_time_s",  "ASIC train/frame (s)"),
    ("render_fps",         "Render FPS"),
    ("psnr",               "PSNR"),
    ("ssim",               "SSIM"),
    ("lpips",              "LPIPS"),
    ("total_storage_mb",   "Total storage (MB)"),
    ("delta_storage_mb",   "Δ stream (MB)"),
]

def fmt(val, key):
    if val is None:
        return "-"
    if isinstance(val, float):
        if key in ("psnr",):
            return f"{val:6.2f}"
        if key in ("ssim",):
            return f"{val:6.4f}"
        if key in ("lpips",):
            return f"{val:6.4f}"
        if key in ("render_fps",):
            return f"{val:7.1f}"
        if key in ("asic_train_time_s",):
            return f"{val:7.3f}"
        if key in ("total_storage_mb", "delta_storage_mb"):
            return f"{val:9.2f}"
        return f"{val:.3f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def build_table() -> dict:
    out: dict[str, dict[int, dict]] = {}
    for baseline in BASELINES:
        out[baseline] = {}
        for m in MODES:
            fp = ABL_ROOT / baseline / f"mode_{m}.json"
            if not fp.exists():
                continue
            with open(fp) as f:
                d = json.load(f)
            entry = {k: d.get(k) for k, _ in COLUMNS}
            out[baseline][m] = entry
    return out


def as_markdown(table: dict) -> str:
    lines: list[str] = []
    lines.append("# GLASS ASIC Ablation — flame_steak\n")
    lines.append(
        "> **Note**: The `Frames` column reflects the *actual* `n_frames_trained` "
        "recorded in each `mode_*.json`. Modes may have been collected across "
        "different simulator revisions; any row whose Frames ≠ 10 is legacy "
        "baseline data kept for reference and should not be compared "
        "apples-to-apples with the 10-frame rows (see GLASS.md §A.6).\n"
    )
    lines.append(
        "| Baseline | Mode | Frames | ASIC train/frame (s) "
        "| Render FPS | PSNR | SSIM | LPIPS | Total storage (MB) | Δ stream (MB) |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for baseline in BASELINES:
        for m in MODES:
            entry = table.get(baseline, {}).get(m)
            if entry is None:
                continue
            row = [
                baseline,
                MODE_LABELS[m],
                fmt(entry.get("n_frames_trained"), "n_frames_trained"),
                fmt(entry.get("asic_train_time_s"), "asic_train_time_s"),
                fmt(entry.get("render_fps"), "render_fps"),
                fmt(entry.get("psnr"), "psnr"),
                fmt(entry.get("ssim"), "ssim"),
                fmt(entry.get("lpips"), "lpips"),
                fmt(entry.get("total_storage_mb"), "total_storage_mb"),
                fmt(entry.get("delta_storage_mb"), "delta_storage_mb"),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    table = build_table()
    md    = as_markdown(table)
    out_md = ABL_ROOT / "ablation_summary.md"
    out_json = ABL_ROOT / "ablation_summary.json"
    out_md.write_text(md)
    out_json.write_text(json.dumps(table, indent=2))
    print(md)
    print(f"\nWrote: {out_md}\n       {out_json}")


if __name__ == "__main__":
    main()
