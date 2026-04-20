"""
Ablation runner for 4DGC — GLASS ASIC Simulator.

4DGC ("Rate-Aware 4D Gaussian Compression for Efficient Streamable FVV",
Hu et al., CVPR 2025) is a streamable descendant of 3DGStream: it replaces
NTC with a multi-resolution Motion Grid + tiny MLP (Motion Estimation Module,
MEM).  The per-frame training loop is structurally identical to 3DGStream:

  • Stage 1: train MEM (mem_optimizer) — scans current frame's views, runs
    backprop on motion grid / MLP / entropy bottleneck weights only.
  • Between stages: ``update_by_mem()`` bulk-writes (Δxyz, Δrot) into _xyz
    / _rotation.  This is the single per-frame bulk delta we care about.
  • Stage 2: train compensated Gaussians (gaussians.optimizer) + periodic
    ``adding_and_prune`` (structural change).

Usage
-----
    python ablation/run_4dgc.py --mode 1 --dataset flame_steak

Output
------
    results/GLASS_ablation/<dataset>/4DGC/mode_<mode>.json

Called by ablation/run_full_sweep.sh in the 4dgc-cu121 conda environment.
Must be run with cwd == baselines/4DGC (the runner switches it internally).

Input data layout (on flame_steak)
----------------------------------
    data/4dgc_flame_steak_scene/colmap_0/     ← frame 0 (keyframe, not used)
                               /colmap_1/     ← first streamed frame
                               /.../images/cam00.png
    data/flame_steak/4dgc_init/               ← output of train_initial.py
        └── cfg_args.json, point_cloud/iteration_<N>/point_cloud.ply, …
    data/flame_steak/4dgc_init/mem.pth        ← Motion_Grid_warmup.py output
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Resolve GLASS root and add required paths ─────────────────────────────────
GLASS_DIR    = Path(__file__).resolve().parents[2]
BASELINE_DIR = GLASS_DIR / "baselines" / "4DGC"

sys.path.insert(0, str(GLASS_DIR))       # import simulator.*
sys.path.insert(0, str(BASELINE_DIR))    # import scene, arguments, mem, …

# ── Simulator imports (must come after sys.path setup) ────────────────────────
from simulator.config import ASICConfig
from simulator.datasets import load_dataset_config
from simulator.metrics.collector import MetricsCollector
from simulator.operators.op_4dgc import Op4DGC
from simulator.pipeline import GLASSPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_frame_dirs(video_path: Path) -> list[str]:
    """Return colmap_<k> sub-directory names in ascending numeric order."""
    pat = re.compile(r"colmap_(\d+)")
    return sorted(
        (f for f in os.listdir(video_path) if pat.match(f)),
        key=lambda x: int(pat.match(x).group(1)),
    )


def _build_args(ds_cfg):
    """Construct an argparse Namespace mirroring 4DGC's train_frames.py.

    Per-dataset overrides from DatasetConfig.extra_args["4dgc"] are applied
    on top of the defaults.
    """
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # Only add args NOT already registered by the three ParamGroup classes.
    parser.add_argument("--frame_start",           type=int, default=1)
    parser.add_argument("--frame_end",             type=int, default=31)
    parser.add_argument("--load_iteration",        type=int, default=None)
    parser.add_argument("--debug_from",            type=int, default=-1)
    parser.add_argument("--test_iterations",  nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations",  nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=[])
    parser.add_argument("--start_checkpoint",      type=str, default=None)
    parser.add_argument("--quiet",            action="store_true")

    video_path = ds_cfg.resolved("source_path_4dgc")
    init_model = ds_cfg.resolved("init_model_4dgc")
    init_mem   = ds_cfg.resolved("init_mem_4dgc")
    base_args = [
        "--source_path", "",
        "--model_path",  str(init_model),
        "--video_path",  str(video_path),
        "--output_path", "",
        "--mem_path",    str(init_mem),
    ]
    args = parser.parse_args(base_args)

    # Apply per-dataset overrides
    extra = ds_cfg.model_args("4dgc")
    for k, v in extra.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Ensure we always save at the end of Stage 1 so the next frame can
    # load from that checkpoint.  train_one_frame() also appends
    # (iterations + iterations_s2) to save_iterations on every call; we
    # snapshot-and-restore save_iterations inside the frame loop to avoid
    # unbounded growth across frames.
    s1_end = args.iterations
    if s1_end not in args.save_iterations:
        args.save_iterations = sorted(set(list(args.save_iterations) + [s1_end]))

    return args, lp, op, pp


# ── Per-mode runner ───────────────────────────────────────────────────────────

def run_one_mode(mode: int, ds_cfg) -> dict:
    """Train ds_cfg.n_frames_ablation frames with ASIC mode ``mode``.

    Uses class-level monkey-patching of ``GaussianModel.training_one_frame_setup``
    / ``training_one_frame_s2_setup`` to intercept the GaussianModel instance
    that 4DGC creates internally per frame — without requiring us to construct
    our own copy.
    """
    print(f"\n{'='*60}")
    print(f"  4DGC  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    # ── Switch cwd so relative paths inside 4DGC resolve correctly ────────────
    os.chdir(BASELINE_DIR)

    # ── 4DGC imports ──────────────────────────────────────────────────────────
    import torch
    from gaussian_renderer import render as gs_render
    from scene import Scene, GaussianModel
    from train_frames import train_one_frame

    args, lp, op, pp = _build_args(ds_cfg)

    config    = ASICConfig(
        mode=mode,
        n_frames_ablation=ds_cfg.n_frames_ablation,
        fps_warmup=ds_cfg.fps_warmup,
        fps_measure=ds_cfg.fps_measure,
    )
    slot      = Op4DGC()
    pipeline  = GLASSPipeline(config, slot)
    collector = MetricsCollector()

    # ── Frame directory list ───────────────────────────────────────────────────
    video_path  = ds_cfg.resolved("source_path_4dgc")
    all_frames  = _sorted_frame_dirs(video_path)
    frame_start = getattr(args, "frame_start", 1)
    frames      = all_frames[frame_start : frame_start + ds_cfg.n_frames_ablation]
    print(f"  Training {len(frames)} frames  (indices {frame_start}…"
          f"{frame_start + len(frames) - 1})")

    # ── Output root ───────────────────────────────────────────────────────────
    out_root = (GLASS_DIR / "results" / "GLASS_ablation"
                / ds_cfg.name / "4DGC" / f"mode_{mode}_run")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Class-level patching to capture GaussianModel + Scene references ──────
    _orig_s1         = GaussianModel.training_one_frame_setup
    _orig_s2         = GaussianModel.training_one_frame_s2_setup
    _orig_scene_init = Scene.__init__

    _ref: dict = {"gaussians": None, "scene": None}

    def _patched_s1(self_g, *a, **kw):
        # Run original setup (creates mem_optimizer + loads warmed-up MEM)
        result = _orig_s1(self_g, *a, **kw)
        _ref["gaussians"] = self_g
        slot.install_hooks(pipeline, gaussians=self_g)
        # Pass args.iterations so the cycle model accounts for MEM training.
        slot.install_mem_delta_hook(self_g, n_mem_iters=args.iterations)
        return result

    def _patched_s2(self_g, *a, **kw):
        # Run original Stage-2 setup (creates the Gaussian optimizer and
        # spawns the initial _added_* Gaussians)
        result = _orig_s2(self_g, *a, **kw)
        slot.install_s2_hooks(self_g)
        return result

    def _patched_scene_init(self_s, *a, **kw):
        _orig_scene_init(self_s, *a, **kw)
        _ref["scene"] = self_s

    GaussianModel.training_one_frame_setup    = _patched_s1
    GaussianModel.training_one_frame_s2_setup = _patched_s2
    Scene.__init__                            = _patched_scene_init

    # ── Frame training loop ───────────────────────────────────────────────────
    model_path          = args.model_path  # init static model for frame 0
    per_frame_delta_bytes: list[int] = []
    per_frame_mem_iters:  list[int]  = []
    last_pp_extract     = None
    last_background     = None

    # Adaptive MEM iteration scheduling (mode >= 1) — same lag-one rule as
    # 3DGStream.  base_iters is the maximum allowed; adaptive_iters is what
    # each frame runs.  prev_save_iter tracks the iteration the PREVIOUS
    # frame saved at so the next frame's load_iteration points to a real
    # checkpoint on disk.
    base_iters     = args.iterations
    prev_save_iter = args.first_load_iteration  # frame 0 loads from init

    try:
        for frame_idx, frame in enumerate(frames):
            frame_src = str(video_path / frame)
            frame_out = str(out_root / frame)
            os.makedirs(frame_out, exist_ok=True)

            # ── Adaptive MEM iteration scheduling (lag-one) ──
            if mode >= 1 and frame_idx > 0:
                prev_ratio     = pipeline._prev_suppression_ratio
                adaptive_iters = max(
                    config.min_ntc_iters_floor,
                    round(base_iters * (1.0 - prev_ratio)),
                )
            else:
                adaptive_iters = base_iters

            args.iterations     = adaptive_iters
            args.load_iteration = (args.first_load_iteration
                                   if frame_idx == 0
                                   else prev_save_iter)
            args.source_path    = frame_src
            args.output_path    = frame_out
            args.model_path     = model_path

            per_frame_mem_iters.append(adaptive_iters)

            # Snapshot-and-restore save_iterations: train_one_frame appends
            # (iterations + iterations_s2) to args.save_iterations on every
            # call, which otherwise would grow unboundedly across frames.
            _save_iters_snapshot = list(args.save_iterations)
            if adaptive_iters not in args.save_iterations:
                args.save_iterations = sorted(
                    set(list(args.save_iterations) + [adaptive_iters])
                )
            with collector.frame_timer():
                train_one_frame(lp, op, pp, args)
            args.save_iterations = _save_iters_snapshot

            prev_save_iter = adaptive_iters

            # ── Post-render bookkeeping ─────────────────────────────────────
            gaussians = _ref["gaussians"]
            scene     = _ref["scene"]
            lp_extract = lp.extract(args)
            pp_extract = pp.extract(args)
            bg_color   = [1, 1, 1] if lp_extract.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            last_pp_extract = pp_extract
            last_background = background

            if gaussians is not None and scene is not None:
                try:
                    cams = scene.getTrainCameras()
                    if cams:
                        with torch.no_grad():
                            rp = gs_render(cams[0], gaussians, pp_extract, background)
                        pipeline.post_render(rp)
                        pipeline.set_render_stats_from_render_pkg(
                            rp, gaussians.active_sh_degree
                        )
                except Exception as exc:
                    print(f"  [frame {frame_idx}] post_render skipped: {exc}")

            per_frame_delta_bytes.append(
                pipeline.stream_stats()["frame_active_bytes"]
            )

            # Uninstall hooks before the next frame creates a new GaussianModel.
            slot.uninstall_hooks()

            # Next frame loads from this frame's S1 checkpoint.
            model_path = frame_out

    finally:
        GaussianModel.training_one_frame_setup    = _orig_s1
        GaussianModel.training_one_frame_s2_setup = _orig_s2
        Scene.__init__                            = _orig_scene_init

    # ── FPS measurement ───────────────────────────────────────────────────────
    print("\n  Measuring FPS …")
    fps = 0.0
    gaussians = _ref["gaussians"]
    scene     = _ref["scene"]
    if gaussians is not None and last_pp_extract is not None:
        try:
            test_cams = (scene.getTestCameras() if scene else []) or []
            if not test_cams and scene:
                test_cams = scene.getTrainCameras()
            if test_cams:
                slot._gaussians = gaussians
                slot._pipeline  = pipeline
                fps = collector.measure_fps(
                    pipeline,
                    lambda cam, g, p, bg: gs_render(cam, g, p, bg),
                    test_cams, last_pp_extract, last_background,
                )
                pipeline.set_render_cycles_from_fps(fps)
        except Exception as exc:
            print(f"  FPS measurement failed: {exc}")

    # ── Quality measurement ───────────────────────────────────────────────────
    print("  Measuring quality …")
    quality = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
    if gaussians is not None and scene is not None:
        try:
            test_cams = scene.getTestCameras()[:20]
            if not test_cams:
                test_cams = scene.getTrainCameras()[:20]
            renders, gts = MetricsCollector.render_test_cameras(
                gaussians, test_cams,
                lambda cam, g, p, bg: gs_render(cam, g, p, bg),
                last_pp_extract, last_background,
            )
            if renders and gts:
                quality = collector.measure_quality(renders, gts)
        except Exception as exc:
            print(f"  Quality measurement failed: {exc}")

    # ── Storage ───────────────────────────────────────────────────────────────
    total_storage_mb = collector.measure_storage(str(out_root))
    delta_storage_mb = collector.measure_delta_storage(per_frame_delta_bytes)

    stats         = pipeline.stream_stats()
    cycle_summary = pipeline.cycle_summary()
    avg_mem_iters = (sum(per_frame_mem_iters) / len(per_frame_mem_iters)
                     if per_frame_mem_iters else base_iters)

    result = {
        "mode":                mode,
        "baseline":            "4DGC",
        "dataset":             ds_cfg.name,
        "n_frames_trained":    len(frames),
        "avg_train_time_s":    collector.avg_frame_time(),
        "all_frame_times_s":   collector.all_frame_times(),
        "render_fps":          fps,
        **quality,
        "total_storage_mb":    total_storage_mb,
        "delta_storage_mb":    delta_storage_mb,
        **stats,
        "incremental":         True,
        "base_ntc_iters":      base_iters,    # MEM iterations (kept under the
        "avg_ntc_iters":       avg_mem_iters, # 3DGStream-compatible key name)
        "per_frame_ntc_iters": per_frame_mem_iters,
        **cycle_summary,
        # Same ASIC train-time formula as 3DGStream: MEM fwd+bwd+Adam over
        # all Stage-1 iters + per-iter render + commit + Stage-2 Adam.
        "asic_train_time_s": (
            cycle_summary.get("asic_ntc_train_render_cycles", 0)
            + cycle_summary.get("asic_ntc_train_cycles", 0)
            + cycle_summary.get("asic_ntc_cycles", 0)
            + cycle_summary.get("asic_gradient_pipeline_cycles", 0)
            + cycle_summary.get("asic_optimizer_cycles", 0)
        ) / config.asic_spec.clock_hz,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_dir  = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name / "4DGC"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_file = json_dir / f"mode_{mode}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {json_file}")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "all_frame_times_s"},
        indent=2,
    ))
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GLASS ASIC ablation runner — 4DGC"
    )
    parser.add_argument("--mode",           type=int, choices=[0, 1, 2, 3],
                        required=True, help="Ablation mode 0-3")
    parser.add_argument("--dataset",        type=str, default="flame_steak",
                        help="Dataset name from DATASET_REGISTRY")
    parser.add_argument("--dataset-config", type=str, default=None,
                        dest="dataset_config",
                        help="Path to YAML with DatasetConfig overrides")
    parser.add_argument("--n-frames", type=int, default=None,
                        dest="n_frames",
                        help="Override n_frames_ablation (default: from dataset config)")
    args = parser.parse_args()

    ds_cfg = load_dataset_config(args.dataset, args.dataset_config)
    if args.n_frames is not None:
        ds_cfg.n_frames_ablation = args.n_frames
    run_one_mode(args.mode, ds_cfg)


if __name__ == "__main__":
    main()
