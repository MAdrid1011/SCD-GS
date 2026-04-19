"""
Evaluate 3DGStream quality metrics (PSNR/SSIM/LPIPS) and render FPS.
Run AFTER training is complete on all 300 frames.

Usage (from the 3dgstream conda env):
    python benchmarks/scripts/eval_3dgstream.py
"""
import os, sys, time, json, re, math
from pathlib import Path
import numpy as np
import torch

# Repo-relative paths — `benchmarks/scripts/*.py` → parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASEDIR   = str(REPO_ROOT / "baselines" / "3DGStream")
sys.path.insert(0, BASEDIR)

from argparse import Namespace
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from lpipsPyTorch.modules.lpips import LPIPS as LPIPSModel
from utils.image_utils import psnr as compute_psnr
from utils.loss_utils import ssim as compute_ssim

RESULTS_DIR = str(REPO_ROOT / "results" / "3DGStream" / "flame_steak" / "full_run")
DATA_DIR    = str(REPO_ROOT / "data" / "3dgstream_flame_steak_scene")
INIT_MODEL  = str(REPO_ROOT / "data" / "assets" / "3DGStream" / "flame_steak_suite" / "flame_steak_init")
OUTPUT_JSON = str(REPO_ROOT / "results" / "3DGStream" / "flame_steak" / "metrics.json")

WARMUP  = 5
MEASURE = 100


# ── Model loading ─────────────────────────────────────────────────────────────

def load_gaussians(frame_path, sh_degree=1):
    g = GaussianModel(sh_degree)
    ply = os.path.join(frame_path, "point_cloud", "iteration_250", "point_cloud.ply")
    if not os.path.exists(ply):
        ply = os.path.join(frame_path, "point_cloud", "iteration_150", "point_cloud.ply")
    g.load_ply(ply)   # already places tensors on CUDA
    return g


# ── Camera loading via Scene ──────────────────────────────────────────────────

def load_test_cameras(frame_data_dir, sh_degree=1):
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos

    args = Namespace(
        source_path=frame_data_dir,
        model_path=INIT_MODEL,
        images="images",
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=True,
        sh_degree=sh_degree,
    )
    try:
        scene_info = readColmapSceneInfo(frame_data_dir, "images", eval=True)
        return cameraList_from_camInfos(scene_info.test_cameras, 1.0, args)
    except Exception as e:
        return []


# ── Metrics ───────────────────────────────────────────────────────────────────

def frame_metrics(gaussians, cameras, pipe, lpips_fn):
    bg = torch.zeros(3, device="cuda")
    psnrs, ssims, lps = [], [], []
    for cam in cameras:
        with torch.no_grad():
            img = render(cam, gaussians, pipe, bg)["render"].clamp(0, 1)
        gt = cam.original_image.cuda()
        psnrs.append(compute_psnr(img, gt).mean().item())
        ssims.append(compute_ssim(img.unsqueeze(0), gt.unsqueeze(0)).item())
        lps.append(lpips_fn(img.unsqueeze(0)*2-1, gt.unsqueeze(0)*2-1).squeeze().mean().item())
    return np.mean(psnrs), np.mean(ssims), np.mean(lps)


def measure_fps(gaussians, cam, pipe):
    bg = torch.zeros(3, device="cuda")
    for _ in range(WARMUP):
        with torch.no_grad():
            render(cam, gaussians, pipe, bg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        with torch.no_grad():
            render(cam, gaussians, pipe, bg)
    torch.cuda.synchronize()
    return MEASURE / (time.perf_counter() - t0)


def model_size_mb(frame_path):
    sz = 0
    for rel in ["point_cloud/iteration_250/point_cloud.ply",
                "point_cloud/iteration_150/point_cloud.ply",
                "NTC.pth"]:
        p = os.path.join(frame_path, rel)
        if os.path.exists(p):
            sz += os.path.getsize(p)
    return sz / (1024**2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pipe     = Namespace(debug=False, bwd_depth=False,
                         convert_SHs_python=False, compute_cov3D_python=False)
    lpips_fn = LPIPSModel("vgg").cuda()

    frame_dirs = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("frame")])
    print(f"Found {len(frame_dirs)} trained frames")

    # Pre-load test cameras from first frame (same rig throughout)
    print("Loading test cameras from frame000001...")
    ref_cams = load_test_cameras(os.path.join(DATA_DIR, "frame000001"))
    print(f"  {len(ref_cams)} test camera(s) loaded")

    psnr_list, ssim_list, lpips_list, fps_list, storage_list = [], [], [], [], []

    # Sample every 10 frames to keep runtime manageable
    sample = frame_dirs[::10]
    print(f"Evaluating {len(sample)} sampled frames (every 10th)...")

    for fd in sample:
        frame_path = os.path.join(RESULTS_DIR, fd)
        frame_num  = int(re.search(r"\d+", fd).group())
        frame_data = os.path.join(DATA_DIR, fd)

        # Load frame-specific cameras if available, fallback to ref
        cams = load_test_cameras(frame_data)
        if not cams:
            cams = ref_cams
        if not cams:
            print(f"  {fd}: no cameras, skip")
            continue

        try:
            gs = load_gaussians(frame_path)

            # Quality
            p, s, l = frame_metrics(gs, cams, pipe, lpips_fn)
            psnr_list.append(p);  ssim_list.append(s);  lpips_list.append(l)

            # FPS
            fps = measure_fps(gs, cams[0], pipe)
            fps_list.append(fps)

            # Storage
            storage_list.append(model_size_mb(frame_path))

            print(f"  {fd}: PSNR={p:.2f} SSIM={s:.4f} LPIPS={l:.4f} "
                  f"FPS={fps:.1f} store={storage_list[-1]:.1f}MB "
                  f"gs={gs.get_xyz.shape[0]}")
        except Exception as e:
            import traceback
            print(f"  {fd}: ERROR – {e}")
            traceback.print_exc()

    # ── Training timing from log ──────────────────────────────────────────────
    log_path = str(REPO_ROOT / "results" / "3DGStream" / "flame_steak" / "training_log.txt")
    timings = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m = re.search(r"Frame frame0*(\d+) finished in ([0-9.]+) seconds", line)
                if m:
                    fnum = int(m.group(1))
                    if not (132 <= fnum <= 141):   # exclude contaminated slice
                        timings.append(float(m.group(2)))
    print(f"\nTiming: {len(timings)} clean frames, avg={np.mean(timings):.2f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "method": "3DGStream",
        "dataset": "flame_steak",
        "n_frames": len(frame_dirs),
        "avg_psnr":  float(np.mean(psnr_list))  if psnr_list  else None,
        "avg_ssim":  float(np.mean(ssim_list))  if ssim_list  else None,
        "avg_lpips": float(np.mean(lpips_list)) if lpips_list else None,
        "avg_fps":   float(np.mean(fps_list))   if fps_list   else None,
        "avg_train_time_s_per_frame": float(np.mean(timings)) if timings else None,
        "n_timing_clean_frames": len(timings),
        "avg_storage_per_frame_mb": float(np.mean(storage_list)) if storage_list else None,
        "total_storage_mb": float(sum(storage_list)) if storage_list else None,
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== 3DGStream Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSaved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
