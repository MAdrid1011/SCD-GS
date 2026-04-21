"""
collect_ablation.py — aggregate GLASS ASIC ablation results.

Reads all 16 (4 baselines × 4 modes) JSON result files from:
    results/GLASS_ablation/<dataset>/<baseline>/mode_<mode>.json

Produces:
    results/GLASS_ablation/<dataset>/summary.csv   — full numeric table
    results/GLASS_ablation/<dataset>/summary.txt   — pretty ASCII table
    (also pretty-prints the table to stdout)

Columns
-------
  baseline, mode, mode_label,
  asic_train_time_s, Δtrain_time_s,
  render_fps,       Δrender_fps,
  psnr,             Δpsnr,
  ssim,             Δssim,
  lpips,            Δlpips,
  total_storage_mb, Δtotal_storage_mb,
  delta_storage_mb, Δdelta_storage_mb,
  run_suppressed_bytes, suppression_ratio_pct

Δ columns are the absolute difference relative to mode 0 for that baseline.
Positive Δ = improvement vs baseline ASIC (mode 0).

The headline time is the ASIC per-frame training time; GPU wall-clock
was removed from the unified schema (it's a simulator-host artefact
and is not representative of the ASIC target).

Usage
-----
    python ablation/collect_ablation.py --dataset flame_steak
    python ablation/collect_ablation.py --dataset custom \\
        --dataset-config /path/to/custom.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

GLASS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GLASS_DIR))

from simulator.ablation.result_schema import MODE_LABELS
from simulator.datasets import load_dataset_config

# ── Constants ─────────────────────────────────────────────────────────────────

BASELINES = ["3DGStream", "QUEEN", "HiCoM", "4DGC"]
MODES     = [0, 1, 2, 3]

# Metrics where higher is better (Δ positive = good)
HIGHER_IS_BETTER = {"render_fps", "psnr", "ssim"}
# Metrics where lower is better (Δ negative = good)
LOWER_IS_BETTER  = {
    "asic_train_time_s",
    "lpips", "total_storage_mb", "delta_storage_mb",
}

NUMERIC_COLS = [
    "asic_train_time_s",
    "render_fps",
    "psnr",
    "ssim",
    "lpips",
    "total_storage_mb",
    "delta_storage_mb",
    "run_total_bytes",
    "run_suppressed_bytes",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_result(results_dir: Path, baseline: str, mode: int) -> dict | None:
    p = results_dir / baseline / f"mode_{mode}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def compute_delta(val: Any, ref: Any) -> float | None:
    """Absolute delta: val - ref.  Returns None if either is NaN/None."""
    if val is None or ref is None:
        return None
    try:
        v, r = float(val), float(ref)
        if math.isnan(v) or math.isnan(r):
            return None
        return v - r
    except (TypeError, ValueError):
        return None


# ── ASCII table helper ────────────────────────────────────────────────────────

def _fmt(v: Any, width: int = 9, decimals: int = 3) -> str:
    if v is None:
        return " " * (width - 1) + "–"
    try:
        f = float(v)
        if math.isnan(f):
            return " " * (width - 1) + "–"
        s = f"{f:{width}.{decimals}f}"
        return s
    except (TypeError, ValueError):
        return str(v)[:width].rjust(width)


def _sign(v: float | None) -> str:
    if v is None:
        return ""
    return "+" if v > 0 else ""


def build_table(rows: list[dict]) -> str:
    cols = [
        ("Baseline",         12, None),
        ("Mode",              2, None),
        ("Label",            16, None),
        ("ASICTrain(s)",      9, "asic_train_time_s"),
        ("Δ",                 7, "d_asic_train_time_s"),
        ("FPS",               8, "render_fps"),
        ("Δ",                 7, "d_render_fps"),
        ("PSNR",              7, "psnr"),
        ("Δ",                 6, "d_psnr"),
        ("SSIM",              6, "ssim"),
        ("Δ",                 6, "d_ssim"),
        ("LPIPS",             6, "lpips"),
        ("Δ",                 6, "d_lpips"),
        ("Storage(MB)",      10, "total_storage_mb"),
        ("Δ",                 8, "d_total_storage_mb"),
        ("ΔStorage(MB)",     10, "delta_storage_mb"),
        ("SuppRatio%",       10, "suppression_ratio_pct"),
    ]

    header = " | ".join(
        name.ljust(w) if key is None else name.center(w)
        for name, w, key in cols
    )
    sep = "-+-".join("-" * w for _, w, _ in cols)

    lines = [header, sep]
    prev_baseline = None
    for row in rows:
        if row["baseline"] != prev_baseline:
            if prev_baseline is not None:
                lines.append(sep)
            prev_baseline = row["baseline"]

        cells = []
        for name, w, key in cols:
            if key is None:
                # Label column
                if name == "Baseline":
                    cells.append(row.get("baseline", "")[:w].ljust(w))
                elif name == "Mode":
                    cells.append(str(row.get("mode", "")).rjust(w))
                else:  # Label
                    cells.append(row.get("mode_label", "")[:w].ljust(w))
            elif key.startswith("d_"):
                raw_key = key[2:]
                v = row.get(key)
                # Determine sign convention for display
                s = _sign(v)
                cells.append(f"{s}{_fmt(v, w, 3)}"[:w].rjust(w))
            else:
                cells.append(_fmt(row.get(key), w, 3).rjust(w))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


# ── Main aggregation ──────────────────────────────────────────────────────────

def aggregate(dataset: str, dataset_config: str | None = None) -> list[dict]:
    ds_cfg = load_dataset_config(dataset, dataset_config)
    results_dir = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name

    rows: list[dict] = []
    missing: list[str] = []

    for baseline in BASELINES:
        # Load mode-0 result as reference for Δ columns
        ref = load_result(results_dir, baseline, mode=0)
        if ref is None:
            missing.append(f"{baseline}/mode_0")

        for mode in MODES:
            data = load_result(results_dir, baseline, mode)
            if data is None:
                missing.append(f"{baseline}/mode_{mode}")
                # Insert placeholder row
                row: dict = {
                    "baseline":   baseline,
                    "mode":       mode,
                    "mode_label": MODE_LABELS[mode],
                }
                for c in NUMERIC_COLS:
                    row[c]         = None
                    row[f"d_{c}"]  = None
                row["suppression_ratio_pct"] = None
                rows.append(row)
                continue

            row = {
                "baseline":   baseline,
                "mode":       mode,
                "mode_label": MODE_LABELS[mode],
                "n_frames_trained": data.get("n_frames_trained"),
                "incremental":      data.get("incremental", True),
            }

            for c in NUMERIC_COLS:
                row[c] = data.get(c)
                row[f"d_{c}"] = compute_delta(
                    data.get(c), ref.get(c) if ref else None
                )

            # Suppression ratio in %
            total = data.get("run_total_bytes", 0) or 0
            supp  = data.get("run_suppressed_bytes", 0) or 0
            row["suppression_ratio_pct"] = (
                100.0 * supp / total if total > 0 else 0.0
            )

            rows.append(row)

    if missing:
        print(f"\n⚠  Missing result files:", file=sys.stderr)
        for m in missing:
            print(f"     results/GLASS_ablation/{ds_cfg.name}/{m}.json",
                  file=sys.stderr)

    return rows, ds_cfg


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys()]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GLASS ASIC ablation results"
    )
    parser.add_argument("--dataset",        type=str, default="flame_steak")
    parser.add_argument("--dataset-config", type=str, default=None,
                        dest="dataset_config")
    args = parser.parse_args()

    rows, ds_cfg = aggregate(args.dataset, args.dataset_config)

    out_dir = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    write_csv(rows, csv_path)
    print(f"CSV  saved → {csv_path}")

    # ASCII table
    table = build_table(rows)
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"GLASS ASIC Ablation — dataset: {ds_cfg.name}\n")
        f.write("=" * 120 + "\n\n")
        f.write(table)
        f.write("\n\nΔ = absolute difference vs Mode 0 (Base ASIC) for each baseline\n")
        f.write("+ = improvement for FPS/PSNR/SSIM (higher is better)\n")
        f.write("- = improvement for TrainTime/LPIPS/Storage (lower is better)\n")

    print(f"Table saved → {txt_path}\n")
    print(f"\nGLASS ASIC Ablation — dataset: {ds_cfg.name}")
    print("=" * 120)
    print(table)
    print("\nΔ = absolute difference vs Mode 0 (Base ASIC) for each baseline")


if __name__ == "__main__":
    main()
