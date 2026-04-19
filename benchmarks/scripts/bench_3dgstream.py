#!/usr/bin/env python3
"""
Benchmark 3DGStream on flame_steak N3DV scene.
Measures: per-frame incremental training time, render FPS, PSNR/SSIM/LPIPS, storage.

Usage (run inside 3dgstream-cu121 conda env):
    python bench_3dgstream.py [--n_frames 300] [--smoke]
"""
import os, sys, time, json, math, argparse, subprocess, shutil
from pathlib import Path

# Repo-relative paths — `benchmarks/scripts/*.py` → parents[2] = repo root.
REPO_ROOT    = Path(__file__).resolve().parents[2]
REPO         = REPO_ROOT / "baselines" / "3DGStream"
INIT_3DGS    = REPO_ROOT / "data" / "assets" / "3DGStream" / "flame_steak_suite" / "flame_steak_init"
VIDEO_PATH   = REPO_ROOT / "data" / "3dgstream_flame_steak_scene"
OUTPUT       = REPO_ROOT / "results" / "3DGStream" / "flame_steak"
RESULTS_JSON = REPO_ROOT / "results" / "3DGStream" / "flame_steak_metrics.json"

sys.path.insert(0, str(REPO))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_frames", type=int, default=300)
    p.add_argument("--smoke", action="store_true", help="Quick 5-frame smoke test")
    p.add_argument("--skip_train", action="store_true", help="Skip training, only eval existing output")
    return p.parse_args()


def run_training(args_ns):
    """Run 3DGStream incremental training with frame-level timing."""
    import re
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.general_utils import safe_state
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from argparse import ArgumentParser, Namespace
    from train_frames import train_one_frame   # per-frame training

    os.chdir(str(REPO))

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=args_ns.n_frames)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--first_load_iteration", type=int, default=15000)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", default=None)
    parser.add_argument("--read_config", action="store_true", default=False)
    parser.add_argument("--config_path", default=None)

    cfg_args = str(OUTPUT / "cfg_args.json")
    train_args = parser.parse_args([
        "--source_path", str(VIDEO_PATH / "frame000000"),
        "--model_path", str(INIT_3DGS),
        "--output_path", str(OUTPUT / "frame000001"),
        "--video_path", str(VIDEO_PATH),
        "--frame_start", "1",
        "--frame_end", str(args_ns.n_frames),
        "--quiet",
    ])

    # Override config paths for our paths
    train_args.source_path = str(VIDEO_PATH / "frame000000")
    train_args.model_path  = str(INIT_3DGS)
    train_args.output_path = str(OUTPUT / "frame000001")
    train_args.video_path  = str(VIDEO_PATH)
    train_args.images      = "images"
    train_args.first_load_iteration = 15000
    train_args.load_iteration = None

    safe_state(train_args.quiet)

    # --- Frame-level training loop with per-frame timing ---
    frame_pattern = re.compile(r"frame(\d+)")
    frames = sorted(
        (f for f in os.listdir(str(VIDEO_PATH)) if frame_pattern.match(f)),
        key=lambda x: int(frame_pattern.match(x).group(1))
    )
    # Skip frame000000 (used for init); start at frame000001
    frames = [f for f in frames if f != "frame000000"]
    frames = frames[:args_ns.n_frames]

    print(f"[3DGStream] Training {len(frames)} incremental frames ...")

    frame_times = []
    model_path  = str(INIT_3DGS)

    for i, frame in enumerate(frames):
        train_args.source_path = str(VIDEO_PATH / frame)
        train_args.output_path = str(OUTPUT / frame)
        train_args.model_path  = model_path

        if i == 0:
            train_args.load_iteration = 15000   # load init model
        else:
            train_args.load_iteration = None    # load from previous frame output

        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / frame).mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        train_one_frame(lp, op, pp, train_args)
        elapsed = time.perf_counter() - t0

        frame_times.append({"frame": frame, "train_sec": elapsed})
        print(f"  [{i+1}/{len(frames)}] {frame}: {elapsed:.2f}s")

        # Next frame reads from this frame's output
        model_path = train_args.output_path

    return frame_times


def measure_fps(frame_times):
    """Measure render FPS from the trained model of the last frame."""
    import torch
    from gaussian_renderer import render, GaussianModel
    from scene import Scene
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser
    from utils.general_utils import safe_state

    os.chdir(str(REPO))

    # Find the last trained frame output directory
    frame_dirs = sorted(OUTPUT.glob("frame*"))
    if not frame_dirs:
        print("[3DGStream] No trained frames found for FPS benchmark.")
        return None

    last_frame_dir = frame_dirs[-1]
    print(f"[3DGStream] Measuring render FPS from {last_frame_dir.name} ...")

    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "--source_path", str(VIDEO_PATH / last_frame_dir.name),
        "--model_path", str(last_frame_dir),
        "--quiet",
    ])
    args.images = "images"

    safe_state(True)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)
    gaussians.eval()

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = scene.getTestCameras()
    if not cameras:
        cameras = scene.getTrainCameras()

    # Warm up
    for cam in cameras[:3]:
        with torch.no_grad():
            render(cam, gaussians, pp.extract(args), background)
    torch.cuda.synchronize()

    # Benchmark
    n_reps = 3
    all_times = []
    for _ in range(n_reps):
        for cam in cameras:
            t0 = time.perf_counter()
            with torch.no_grad():
                render(cam, gaussians, pp.extract(args), background)
            torch.cuda.synchronize()
            all_times.append(time.perf_counter() - t0)

    fps = 1.0 / (sum(all_times) / len(all_times))
    print(f"[3DGStream] Render FPS: {fps:.1f}")
    return fps


def measure_quality(frame_dir: Path):
    """Compute PSNR/SSIM/LPIPS for rendered vs ground-truth images."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    import lpips as lpips_module

    render_dir = frame_dir / "test" / "ours_None" / "renders"
    gt_dir     = frame_dir / "test" / "ours_None" / "gt"

    if not render_dir.exists():
        print(f"[3DGStream] No renders at {render_dir}")
        return None

    loss_fn = lpips_module.LPIPS(net="alex").cuda()

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for fname in sorted(render_dir.glob("*.png")):
        gt_path = gt_dir / fname.name
        if not gt_path.exists():
            continue

        pred = TF.to_tensor(Image.open(fname)).unsqueeze(0).cuda()
        gt   = TF.to_tensor(Image.open(gt_path)).unsqueeze(0).cuda()

        mse = torch.mean((pred - gt) ** 2)
        psnr_vals.append((-10 * torch.log10(mse)).item())

        # SSIM (simple)
        from utils.image_utils import psnr as gs_psnr
        # Reuse LPIPS/SSIM if available
        import pytorch_msssim
        ssim_vals.append(pytorch_msssim.ssim(pred, gt, data_range=1.0).item())

        with torch.no_grad():
            lp = loss_fn(pred * 2 - 1, gt * 2 - 1)
        lpips_vals.append(lp.item())

    if not psnr_vals:
        return None

    return {
        "psnr_db": sum(psnr_vals) / len(psnr_vals),
        "ssim":    sum(ssim_vals) / len(ssim_vals),
        "lpips":   sum(lpips_vals) / len(lpips_vals),
        "n_frames_evaluated": len(psnr_vals),
    }


def measure_storage():
    """Measure storage: init model size + per-frame delta size."""
    init_bytes = sum(f.stat().st_size for f in INIT_3DGS.rglob("*") if f.is_file())

    frame_dirs = sorted(OUTPUT.glob("frame*"))
    if not frame_dirs:
        return {"init_model_mb": init_bytes / 1e6}

    per_frame_bytes = []
    for fd in frame_dirs:
        ply_files = list(fd.glob("**/*.ply"))
        if ply_files:
            # Sum all .ply and param files saved per frame
            total = sum(f.stat().st_size for ply in ply_files for f in [ply])
            per_frame_bytes.append(total)

    avg_delta = sum(per_frame_bytes) / len(per_frame_bytes) if per_frame_bytes else 0
    total_bytes = init_bytes + sum(per_frame_bytes)

    return {
        "init_model_mb":           init_bytes / 1e6,
        "incremental_mb_per_frame": avg_delta / 1e6,
        "total_mb":                total_bytes / 1e6,
    }


def main():
    args = parse_args()
    if args.smoke:
        args.n_frames = 5

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Training ---
    if not args.skip_train:
        frame_times = run_training(args)
        train_times = [ft["train_sec"] for ft in frame_times]
        results["frame_train_times"] = frame_times
        results["train_sec_per_frame_mean"] = sum(train_times) / len(train_times)
        results["train_sec_per_frame_median"] = sorted(train_times)[len(train_times) // 2]
        print(f"[3DGStream] Avg incremental training: {results['train_sec_per_frame_mean']:.2f}s/frame")
    else:
        print("[3DGStream] Skipping training (--skip_train)")

    # --- FPS ---
    fps = measure_fps(results.get("frame_train_times", []))
    if fps:
        results["render_fps"] = fps

    # --- Quality ---
    frame_dirs = sorted(OUTPUT.glob("frame*"))
    if frame_dirs:
        quality = measure_quality(frame_dirs[-1])
        if quality:
            results.update(quality)

    # --- Storage ---
    storage = measure_storage()
    results.update(storage)

    # --- Save results ---
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"3DGStream Benchmark Results (flame_steak)")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:<40} {v}")
    print(f"\nFull results saved to: {RESULTS_JSON}")

    return results


if __name__ == "__main__":
    main()
