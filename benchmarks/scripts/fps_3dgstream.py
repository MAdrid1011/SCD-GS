"""
Measure rendering FPS for 3DGStream on flame_steak.
Run AFTER training is complete. Loads each frame's model and times rendering.
"""
import os
import sys
import time
import json
from pathlib import Path
import torch

# Repo-relative paths — `benchmarks/scripts/*.py` → parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASEDIR   = str(REPO_ROOT / "baselines" / "3DGStream")
sys.path.insert(0, BASEDIR)

from argparse import Namespace
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams

RESULTS_DIR = str(REPO_ROOT / "results" / "3DGStream" / "flame_steak" / "full_run")
DATA_DIR    = str(REPO_ROOT / "data" / "3dgstream_flame_steak_scene")
OUTPUT_JSON = str(REPO_ROOT / "results" / "3DGStream" / "flame_steak" / "fps_results.json")

WARMUP_RENDERS = 5
MEASURE_RENDERS = 50

def load_frame_model(frame_path, sh_degree=1):
    """Load saved 3DGS model for one frame."""
    gaussians = GaussianModel(sh_degree)
    ply_path = os.path.join(frame_path, "point_cloud", "iteration_250", "point_cloud.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(frame_path, "point_cloud", "iteration_150", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def make_fake_camera(width=1352, height=1014, fovx=0.85, fovy=0.64):
    """Create a minimal camera object compatible with 3DGStream renderer."""
    import math
    import numpy as np

    class FakeCamera:
        def __init__(self):
            self.image_height = height
            self.image_width  = width
            self.FoVx = fovx
            self.FoVy = fovy
            # Identity world-view transform
            wvt = torch.eye(4, device="cuda")
            self.world_view_transform = wvt
            self.full_proj_transform  = wvt
            self.camera_center = torch.zeros(3, device="cuda")
    return FakeCamera()


def measure_fps(gaussians, camera, pipe):
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    # Warmup
    for _ in range(WARMUP_RENDERS):
        with torch.no_grad():
            render(camera, gaussians, pipe, background)
    torch.cuda.synchronize()
    # Measure
    t0 = time.perf_counter()
    for _ in range(MEASURE_RENDERS):
        with torch.no_grad():
            render(camera, gaussians, pipe, background)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return MEASURE_RENDERS / elapsed


def main():
    # Build a minimal pipe namespace
    pipe = Namespace(debug=False, bwd_depth=False, convert_SHs_python=False,
                     compute_cov3D_python=False)

    camera = make_fake_camera()

    frame_dirs = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if d.startswith("frame")
    ])
    print(f"Found {len(frame_dirs)} frame dirs")

    fps_list = []
    sample_frames = frame_dirs[::10]  # Sample every 10 frames
    for fd in sample_frames:
        frame_path = os.path.join(RESULTS_DIR, fd)
        try:
            gs = load_frame_model(frame_path)
            gs.cuda()
            fps = measure_fps(gs, camera, pipe)
            fps_list.append(fps)
            print(f"  {fd}: {fps:.1f} FPS  ({gs.get_xyz.shape[0]} Gaussians)")
        except Exception as e:
            print(f"  {fd}: ERROR - {e}")

    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\n==> 3DGStream average FPS: {avg_fps:.2f}")

        result = {
            "method": "3DGStream",
            "dataset": "flame_steak",
            "avg_fps": avg_fps,
            "per_frame_fps": dict(zip(sample_frames, fps_list)),
        }
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
