#!/usr/bin/env python3
"""
Benchmark QUEEN on flame_steak N3DV scene.
Imports prior full run results and measures render FPS from the trained model.

Usage (run inside queen-cu121 conda env):
    python bench_queen.py [--smoke] [--rerun]
"""
import os, sys, time, json, argparse
from pathlib import Path

# Repo-relative paths — `benchmarks/scripts/*.py` → parents[2] = repo root.
REPO_ROOT    = Path(__file__).resolve().parents[2]
REPO         = REPO_ROOT / "baselines" / "QUEEN"
DATA_ROOT    = REPO_ROOT / "data" / "flame_steak" / "flame_steak"
# Minimal prior-run subset; see benchmarks/README.md §"Assets" for layout.
PRIOR_RUN    = REPO_ROOT / "data" / "assets" / "QUEEN" / "flame_steak_prior"
OUTPUT       = REPO_ROOT / "results" / "QUEEN" / "flame_steak"
RESULTS_JSON = REPO_ROOT / "results" / "QUEEN" / "flame_steak_metrics.json"

sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rerun", action="store_true", help="Force a fresh training run instead of importing prior")
    p.add_argument("--fps_only", action="store_true", help="Only measure FPS from existing model")
    return p.parse_args()


def import_prior_quality_metrics():
    """Import quality metrics from the completed prior run."""
    avg_path = PRIOR_RUN / "avg_metrics.json"
    results_path = PRIOR_RUN / "results.json"
    training_path = PRIOR_RUN / "training_metrics.json"

    metrics = {}

    if avg_path.exists():
        with open(avg_path) as f:
            avg = json.load(f)
        metrics["psnr_db_paper"]  = avg.get("PSNR (Val)", avg.get("PSNR (Test)", None))
        metrics["size_mb_avg"]    = avg.get("Size (MB)", None)
        metrics["frame_time_sec_avg"] = avg.get("Frame time", None)
        metrics["total_elapsed_sec"]  = avg.get("Elapsed time", None)

    if results_path.exists():
        with open(results_path) as f:
            res = json.load(f)
        metrics["psnr_db"]  = res.get("PSNR", None)
        metrics["ssim"]     = res.get("SSIM", None)
        metrics["lpips"]    = res.get("LPIPS", None)

    if training_path.exists():
        with open(training_path) as f:
            train = json.load(f)
        # Per-frame times; exclude init frame (index 1, usually 60+s)
        times = [f["Frame time"] for f in train if "Frame time" in f]
        if len(times) > 1:
            incremental_times = times[1:]   # exclude init
            metrics["train_sec_per_frame_mean"]   = sum(incremental_times) / len(incremental_times)
            metrics["train_sec_per_frame_median"] = sorted(incremental_times)[len(incremental_times)//2]
            metrics["train_sec_init"]             = times[0]
            metrics["n_frames_trained"]           = len(times)

        # Per-frame storage
        sizes = [f["Size (MB)"] for f in train if "Size (MB)" in f]
        if sizes:
            metrics["init_model_mb"]           = sizes[0] if sizes else None
            metrics["incremental_mb_per_frame"] = sum(sizes[1:]) / len(sizes[1:]) if len(sizes) > 1 else None

    return metrics


def measure_fps_queen():
    """Measure render FPS using QUEEN's fps.py approach."""
    import torch
    from gaussian_renderer import render, GaussianModel
    from scene import Scene
    from arguments import ModelParams, PipelineParams, get_combined_args
    from argparse import ArgumentParser
    from utils.general_utils import safe_state

    os.chdir(str(REPO))

    # Find a usable trained frame directory to load from
    # Use the prior run's last frame model
    frame_dirs = sorted(PRIOR_RUN.glob("frames/????"))
    if not frame_dirs:
        print("[QUEEN] No frame output dirs found for FPS measurement.")
        return None

    # Use last frame
    last_frame = frame_dirs[-1]
    ply_files  = list(last_frame.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not ply_files:
        print(f"[QUEEN] No point_cloud.ply in {last_frame}")
        return None

    model_path = str(last_frame)
    print(f"[QUEEN] Measuring render FPS from frame {last_frame.name} ...")

    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "--source_path", str(DATA_ROOT),
        "--model_path", model_path,
        "--quiet",
    ])
    args.images = "cam00/images"   # QUEEN expects multi-cam layout

    safe_state(True)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = scene.getTestCameras()
    if not cameras:
        cameras = scene.getTrainCameras()

    print(f"[QUEEN] {len(cameras)} test cameras")

    # Warm-up
    for cam in cameras[:3]:
        with torch.no_grad():
            render(cam, gaussians, pp.extract(args), background)
    torch.cuda.synchronize()

    # Time render of all cameras
    reps, times = 5, []
    for _ in range(reps):
        for cam in cameras:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                render(cam, gaussians, pp.extract(args), background)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    fps = 1.0 / (sum(times) / len(times))
    print(f"[QUEEN] Render FPS: {fps:.1f}")
    return fps


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = {}

    if not args.rerun:
        print("[QUEEN] Importing prior full run metrics ...")
        prior = import_prior_quality_metrics()
        results.update(prior)
        print(f"  PSNR={prior.get('psnr_db', '?'):.2f}, SSIM={prior.get('ssim', '?'):.4f}, LPIPS={prior.get('lpips', '?'):.4f}")
        print(f"  Train avg (incr): {prior.get('train_sec_per_frame_mean', '?'):.2f}s/frame")
        print(f"  Per-frame size (incr): {prior.get('incremental_mb_per_frame', '?'):.2f}MB")

    # FPS measurement requires the trained model
    if not args.smoke:
        try:
            fps = measure_fps_queen()
            if fps:
                results["render_fps"] = fps
        except Exception as e:
            print(f"[QUEEN] FPS measurement failed: {e}")
            # Fallback: QUEEN paper reports ~350 FPS; use per-render timing from training_metrics
            if "frame_time_sec_avg" in results and results["frame_time_sec_avg"]:
                # frame_time includes training; actual render FPS is much higher
                print("[QUEEN] Using paper-reported ~350 FPS as reference")
                results["render_fps_reference"] = 350.0
                results["render_fps_source"] = "paper-reported"

    # Write results
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"QUEEN Benchmark Results (flame_steak)")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<45} {v:.4f}")
        elif isinstance(v, (int, str)):
            print(f"  {k:<45} {v}")
    print(f"\nFull results saved to: {RESULTS_JSON}")

    return results


if __name__ == "__main__":
    main()
