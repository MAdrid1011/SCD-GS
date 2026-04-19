"""
Aggregate benchmark results from all four baselines into a single table.

Every number printed here comes from either:
  (a) a per-baseline run output JSON
      (bench_queen.py / eval_3dgstream.py / Swift4D metrics.py / HiCoM log), or
  (b) the minimal QUEEN prior-run subset committed under
      data/assets/QUEEN/flame_steak_prior/  (see benchmarks/README.md).

No baseline number is hard-coded in this file.
"""
import json
import os
import re
from pathlib import Path
import numpy as np

# Repo-relative paths — `benchmarks/scripts/*.py` → parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS   = REPO_ROOT / "results"
# Prior-run subset used to source QUEEN's reference metrics.
QUEEN_PRIOR = REPO_ROOT / "data" / "assets" / "QUEEN" / "flame_steak_prior"


def safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return {}


# ── QUEEN (metrics loaded from the committed prior-run subset) ───────────────

def load_queen():
    """Parse QUEEN numbers from the prior-run JSON files.

    Source files (all under data/assets/QUEEN/flame_steak_prior/):
      - results.json         → PSNR / SSIM / LPIPS (final)
      - avg_metrics.json     → Size (MB) / Frame time / Elapsed time
      - training_metrics.json → per-frame timing + per-frame storage
    """
    d = {"method": "QUEEN", "dataset": "flame_steak",
         "source": str(QUEEN_PRIOR.relative_to(REPO_ROOT))}

    results_path  = QUEEN_PRIOR / "results.json"
    avg_path      = QUEEN_PRIOR / "avg_metrics.json"
    training_path = QUEEN_PRIOR / "training_metrics.json"

    if results_path.exists():
        r = safe_load(results_path)
        d["avg_psnr"]  = r.get("PSNR")
        d["avg_ssim"]  = r.get("SSIM")
        d["avg_lpips"] = r.get("LPIPS")

    if avg_path.exists():
        a = safe_load(avg_path)
        d["avg_train_time_s_per_frame"] = a.get("Frame time")
        d["total_elapsed_sec"]          = a.get("Elapsed time")

    if training_path.exists():
        tm = safe_load(training_path)
        if isinstance(tm, list) and tm:
            sizes = [f["Size (MB)"] for f in tm if "Size (MB)" in f]
            if sizes:
                d["total_storage_mb"]          = float(sum(sizes))
                d["avg_storage_per_frame_mb"]  = float(np.mean(sizes[1:])) if len(sizes) > 1 else None

    # FPS is measured at render time by bench_queen.py, not recorded in the
    # prior run.  If a bench_queen.py run exists, prefer its number.
    queen_run = RESULTS / "QUEEN" / "flame_steak_metrics.json"
    if queen_run.exists():
        q = safe_load(queen_run)
        if q.get("render_fps") is not None:
            d["avg_fps"] = q["render_fps"]
    return d


queen = load_queen()


# ── 3DGStream ────────────────────────────────────────────────────────────────
def load_3dgstream():
    d = {}
    metrics_path = RESULTS / "3DGStream/flame_steak/metrics.json"
    if metrics_path.exists():
        d.update(safe_load(metrics_path))
        d["method"] = "3DGStream"
        return d

    # Fallback: parse training log for timing + PSNR
    log_path = RESULTS / "3DGStream/flame_steak/training_log.txt"
    timings, psnrs = [], []
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                m = re.search(r"Frame frame0*(\d+) finished in ([0-9.]+) seconds", line)
                if m:
                    fnum = int(m.group(1))
                    if not (132 <= fnum <= 141):  # exclude contaminated
                        timings.append(float(m.group(2)))
                # PSNR from eval during training
                m2 = re.search(r"PSNR ([0-9.]+)", line)
                if m2:
                    psnrs.append(float(m2.group(1)))

    # Storage: sum all frame model sizes
    storage = []
    full_run = RESULTS / "3DGStream/flame_steak/full_run"
    if full_run.exists():
        for fd in sorted(full_run.iterdir()):
            if fd.name.startswith("frame"):
                ply = fd / "point_cloud/iteration_250/point_cloud.ply"
                ntc = fd / "NTC.pth"
                sz = sum(p.stat().st_size for p in [ply, ntc] if p.exists())
                storage.append(sz / (1024**2))

    return {
        "method": "3DGStream",
        "dataset": "flame_steak",
        "avg_train_time_s_per_frame": float(np.mean(timings)) if timings else None,
        "n_timing_clean_frames": len(timings),
        "avg_psnr":  float(np.mean(psnrs[-300:])) if psnrs else None,
        "avg_ssim":  None,
        "avg_lpips": None,
        "avg_fps":   None,
        "total_storage_mb": float(sum(storage)) if storage else None,
        "avg_storage_per_frame_mb": float(np.mean(storage)) if storage else None,
    }


# ── Swift4D ──────────────────────────────────────────────────────────────────
def load_swift4d():
    # Check per-frame results from Swift4D's metrics.py output
    metrics_path = RESULTS / "Swift4D/flame_steak/metrics.json"
    if metrics_path.exists():
        return safe_load(metrics_path)

    # Parse training log for timing
    log = RESULTS / "Swift4D/flame_steak/training_log.txt"
    d = {"method": "Swift4D", "dataset": "flame_steak"}
    if log.exists():
        with open(log) as f:
            content = f.read()
        # Swift4D trains 13000 iterations total (all frames bundled)
        # Get total training time from log
        times = re.findall(r"Training complete.*?(\d+\.\d+)\s+seconds", content, re.S)
        # Extract PSNR from psnr.txt if exists
    psnr_txt = REPO_ROOT / "baselines" / "Swift4D" / "psnr.txt"
    if psnr_txt.exists():
        with open(psnr_txt) as f:
            lines = f.readlines()
        vals = [float(x.split()[-1]) for x in lines if x.strip() and x.strip().replace('.','').replace('-','').isdigit() == False]
        # Parse psnr values
        psnr_vals = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                try:
                    psnr_vals.append(float(parts[-1]))
                except:
                    pass
        if psnr_vals:
            d["avg_psnr"] = float(np.mean(psnr_vals[-20:]))

    # FPS from render log
    render_log = RESULTS / "Swift4D/flame_steak/render_log.txt"
    if render_log.exists():
        with open(render_log) as f:
            txt = f.read()
        m = re.search(r"FPS:\s*([0-9.]+)", txt)
        if m:
            d["avg_fps"] = float(m.group(1))

    return d


# ── HiCoM ────────────────────────────────────────────────────────────────────
def load_hicom():
    metrics_path = RESULTS / "HiCoM/flame_steak/metrics.json"
    if metrics_path.exists():
        return safe_load(metrics_path)

    d = {"method": "HiCoM", "dataset": "flame_steak"}
    log = RESULTS / "HiCoM/flame_steak/training_log.txt"
    if log.exists():
        with open(log) as f:
            content = f.read()
        # Parse per-frame timing from "Start learning frame N" lines
        frame_times = re.findall(r"Start learning frame (\d+)", content)
        # Parse PSNR from training progress
        psnr_vals = re.findall(r"psnr=([0-9.]+)", content)
        if psnr_vals:
            d["avg_psnr"] = float(np.mean([float(x) for x in psnr_vals[-100:]]))

    return d


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    methods = {
        "QUEEN":     queen,
        "3DGStream": load_3dgstream(),
        "Swift4D":   load_swift4d(),
        "HiCoM":     load_hicom(),
    }

    # GPU string is best-effort; leaves "unknown" if torch is not importable.
    gpu = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass

    table = {
        "dataset": "N3DV flame_steak",
        "gpu": gpu,
        "methods": methods,
    }

    out_path = RESULTS / "metric_table.json"
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)

    # Pretty print
    print("\n" + "="*70)
    print(f"{'Method':<12} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'FPS':>8} {'Train(s/f)':>12} {'Store(MB)':>10}")
    print("-"*70)
    for name, m in methods.items():
        psnr  = f"{m.get('avg_psnr', 0):.2f}"  if m.get('avg_psnr')  else "N/A"
        ssim  = f"{m.get('avg_ssim', 0):.4f}"  if m.get('avg_ssim')  else "N/A"
        lps   = f"{m.get('avg_lpips', 0):.4f}" if m.get('avg_lpips') else "N/A"
        fps   = f"{m.get('avg_fps', 0):.1f}"   if m.get('avg_fps')   else "N/A"
        trn   = f"{m.get('avg_train_time_s_per_frame', 0):.2f}" if m.get('avg_train_time_s_per_frame') else "N/A"
        store = f"{m.get('total_storage_mb', 0):.1f}" if m.get('total_storage_mb') else "N/A"
        print(f"{name:<12} {psnr:>8} {ssim:>8} {lps:>8} {fps:>8} {trn:>12} {store:>10}")
    print("="*70)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
