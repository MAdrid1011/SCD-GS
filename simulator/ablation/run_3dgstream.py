"""
Ablation runner for 3DGStream — GLASS ASIC Simulator.

Executes one ablation mode (0-3) for the 3DGStream baseline on a specified
dataset, measures all required metrics, and writes a JSON result file.

Usage
-----
    python ablation/run_3dgstream.py --mode 1 --dataset flame_steak
    python ablation/run_3dgstream.py --mode 3 --dataset custom \\
        --dataset-config /path/to/custom.yaml

Output
------
    results/GLASS_ablation/<dataset>/3DGStream/mode_<mode>.json

Called by ablation/run_all.sh in the 3dgstream-cu121 conda environment.
Must be executed with the current working directory set to
baselines/3DGStream (or the script sets it itself) so that relative paths
such as ntc_conf_path and ntc_path resolve correctly.
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
BASELINE_DIR = GLASS_DIR / "baselines" / "3DGStream"

sys.path.insert(0, str(GLASS_DIR))       # allows: import simulator.*
sys.path.insert(0, str(BASELINE_DIR))    # allows: import scene, arguments, …

# ── Simulator imports (must come after sys.path setup) ────────────────────────
from simulator.ablation.result_schema import build_result
from simulator.config import ASICConfig
from simulator.datasets import load_dataset_config
from simulator.metrics.collector import MetricsCollector
from simulator.operators.op_3dgstream import Op3DGStream
from simulator.pipeline import GLASSPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sorted_frame_dirs(video_path: Path) -> list[str]:
    """Return frame sub-directory names in ascending numeric order."""
    pat = re.compile(r"frame(\d+)")
    return sorted(
        (f for f in os.listdir(video_path) if pat.match(f)),
        key=lambda x: int(pat.match(x).group(1)),
    )


def _build_args(ds_cfg):
    """Construct an argparse Namespace that mirrors 3DGStream's CLI output.

    Extra per-dataset overrides from DatasetConfig.extra_args["3dgstream"]
    are applied on top of the defaults.
    """
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # Only add args NOT already registered by ModelParams / OptimizationParams.
    # Duplicates (first_load_iteration, batch_size, opt_type) live in those classes.
    parser.add_argument("--frame_start",           type=int, default=1)
    parser.add_argument("--frame_end",             type=int, default=31)
    parser.add_argument("--load_iteration",        type=int, default=None)
    parser.add_argument("--debug_from",            type=int, default=-1)
    parser.add_argument("--test_iterations",  nargs="+", type=int,
                        default=[50, 100, 150])
    parser.add_argument("--save_iterations",  nargs="+", type=int,
                        default=[150])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=[])
    parser.add_argument("--start_checkpoint",      type=str, default=None)
    parser.add_argument("--quiet",            action="store_true")

    # Parse with empty source/model/output; video_path set from DatasetConfig
    video_path = ds_cfg.resolved("source_path_3dgstream")
    base_args = [
        "--source_path", "",
        "--model_path",  str(ds_cfg.resolved("init_model_3dgstream")),
        "--video_path",  str(video_path),
        "--output_path", "",
    ]
    args = parser.parse_args(base_args)

    # Apply dataset-specific overrides from extra_args
    extra = ds_cfg.model_args("3dgstream")
    for k, v in extra.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Ensure we always save at the S1 end so the next frame can load it.
    s1_end = args.iterations
    if s1_end not in args.save_iterations:
        args.save_iterations = sorted(set(list(args.save_iterations) + [s1_end]))

    return args, lp, op, pp


# ── Per-mode runner ───────────────────────────────────────────────────────────

def run_one_mode(mode: int, ds_cfg) -> dict:
    """Train ds_cfg.n_frames_ablation frames with ASIC mode `mode`.

    Uses class-level monkey-patching of GaussianModel.training_one_frame_setup
    and training_one_frame_s2_setup to intercept the GaussianModel instance
    that 3DGStream creates internally — without requiring us to construct our
    own copy.

    The working directory is switched to BASELINE_DIR so that relative paths
    in the baseline (ntc_conf_path, ntc_path, configs/) resolve correctly.
    """
    print(f"\n{'='*60}")
    print(f"  3DGStream  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    # ── Switch cwd so relative NTC paths work ─────────────────────────────────
    os.chdir(BASELINE_DIR)

    # ── 3DGStream imports ─────────────────────────────────────────────────────
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
    slot      = Op3DGStream()
    pipeline  = GLASSPipeline(config, slot)
    collector = MetricsCollector()

    # ── Frame directory list ───────────────────────────────────────────────────
    video_path  = ds_cfg.resolved("source_path_3dgstream")
    all_frames  = _sorted_frame_dirs(video_path)
    frame_start = getattr(args, "frame_start", 1)
    frames      = all_frames[frame_start : frame_start + ds_cfg.n_frames_ablation]
    print(f"  Training {len(frames)} frames  (indices {frame_start}…"
          f"{frame_start + len(frames) - 1})")

    # ── Output root ───────────────────────────────────────────────────────────
    out_root = (GLASS_DIR / "results" / "GLASS_ablation"
                / ds_cfg.name / "3DGStream" / f"mode_{mode}_run")
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Class-level patching — capture GaussianModel + Scene references ────────
    #
    # 3DGStream's training_one_frame() creates its OWN GaussianModel internally.
    # We cannot pass hooks before that creation, so we patch the class methods
    # that run immediately after construction to intercept the live instance.
    #
    _orig_s1        = GaussianModel.training_one_frame_setup
    _orig_s2        = GaussianModel.training_one_frame_s2_setup
    _orig_scene_init = Scene.__init__

    _ref: dict = {"gaussians": None, "scene": None}

    def _patched_s1(self_g, *a, **kw):
        # Run the original setup (creates ntc_optimizer)
        result = _orig_s1(self_g, *a, **kw)
        # Capture reference and install NTC optimizer + delta hooks
        _ref["gaussians"] = self_g
        slot.install_hooks(pipeline, gaussians=self_g)
        # Pass args.iterations so the cycle model can account for NTC training.
        slot.install_ntc_delta_hook(self_g, n_ntc_iters=args.iterations)
        return result

    def _patched_s2(self_g, *a, **kw):
        # Run the original Stage-2 setup (creates optimizer for added Gaussians)
        result = _orig_s2(self_g, *a, **kw)
        # Install GS optimizer + adding_and_prune hooks
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
    per_frame_ntc_iters: list[int]   = []
    last_pp_extract     = None
    last_background     = None

    # For adaptive NTC iteration scheduling (mode >= 1).
    # base_iters is the maximum allowed; adaptive_iters is what each frame runs.
    # prev_save_iter tracks the checkpoint iteration the PREVIOUS frame saved at
    # so the next frame's load_iteration always points to a real checkpoint.
    base_iters     = args.iterations          # e.g. 150 (from DatasetConfig)
    prev_save_iter = args.first_load_iteration  # frame 0 loads from init model

    try:
        for frame_idx, frame in enumerate(frames):
            frame_src = str(video_path / frame)
            frame_out = str(out_root / frame)
            os.makedirs(frame_out, exist_ok=True)

            # ── Adaptive NTC iteration scheduling (lag-one: uses prev frame's ratio) ──
            # Mode 0: always run full base_iters (no UCN).
            # Mode >= 1: scale down by (1 - prev_suppression_ratio), floor at min.
            # Frame 0 always runs base_iters (no suppression history yet).
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

            per_frame_ntc_iters.append(adaptive_iters)

            pipeline.begin_frame()

            # train_one_frame() appends iterations+iterations_s2 to save_iterations
            # on every call; snapshot before to avoid growing list across frames.
            # We also ensure adaptive_iters is in save_iterations so Stage 1 saves
            # at the right checkpoint for the next frame's load_iteration.
            _save_iters_snapshot = list(args.save_iterations)
            if adaptive_iters not in args.save_iterations:
                args.save_iterations = sorted(
                    set(list(args.save_iterations) + [adaptive_iters])
                )
            with collector.frame_timer():
                train_one_frame(lp, op, pp, args)
            args.save_iterations = _save_iters_snapshot

            # Track where this frame saved so the next frame loads from the right place.
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
                        # Capture render stats for next frame's NTC training render model.
                        pipeline.set_render_stats_from_render_pkg(
                            rp, gaussians.active_sh_degree
                        )
                except Exception as exc:
                    print(f"  [frame {frame_idx}] post_render skipped: {exc}")

            # ── Stage 2 render+loss+backward cycles ─────────────────────────
            # 3DGStream trains compensated Gaussians for ``iterations_s2``
            # full render+bwd steps *after* the NTC commit.  ingest_ntc_deltas
            # only accounts for the Stage-1 NTC-driven render iters, so Stage 2
            # would otherwise be missing from asic_ntc_train_render_cycles.
            # We reuse the just-captured render stats (same frame, same scene)
            # to size the cycles, which matches what the GPU actually executed.
            iters_s2 = int(getattr(args, "iterations_s2", 0))
            if iters_s2 > 0 and gaussians is not None:
                pipeline.add_gaussian_training_iters(
                    iters_s2, n_total=int(gaussians.get_xyz.shape[0])
                )

            per_frame_delta_bytes.append(
                pipeline.stream_stats()["frame_active_bytes"]
            )

            # Uninstall hooks from this frame's gaussians before the next
            # frame creates a new GaussianModel.
            slot.uninstall_hooks()

            # Next frame loads from this frame's S1 checkpoint.
            model_path = frame_out

    finally:
        # Always restore class methods, even on error.
        GaussianModel.training_one_frame_setup    = _orig_s1
        GaussianModel.training_one_frame_s2_setup = _orig_s2
        Scene.__init__                            = _orig_scene_init

    # ── FPS measurement ───────────────────────────────────────────────────────
    print("\n  Measuring FPS …")
    fps = 0.0
    fps_diag: dict = {}
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
                fps_diag = pipeline.measure_asic_render_fps(
                    lambda cam, g, p, bg: gs_render(cam, g, p, bg),
                    test_cams, last_pp_extract, last_background,
                    sh_degree=int(gaussians.active_sh_degree),
                )
                fps = fps_diag["asic_fps"]
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

    stats = pipeline.stream_stats()

    cycle_summary = pipeline.cycle_summary()
    avg_ntc_iters = (sum(per_frame_ntc_iters) / len(per_frame_ntc_iters)
                     if per_frame_ntc_iters else base_iters)

    # asic_train_time_s: full ASIC training time per frame.
    #   = (render-train cycles        ← 3DGS render+loss+bwd per NTC iter
    #      + NTC-train cycles         ← NTC fwd+bwd+Adam per NTC iter
    #      + NTC-inference cycles     ← final commit inference pass
    #      + commit pipeline cycles   ← PUOE→UCN→ICE pipeline
    #      + optimizer cycles)        ← Adam on active Gaussians
    #   / clock_hz
    # Mode 0→1 speedup is driven primarily by render-train + NTC-train:
    # both scale with n_iters, which UCN reduces from 150 → ~1-20 in Mode 1.
    asic_train_time_s = (
        cycle_summary.get("asic_ntc_train_render_cycles", 0)
        + cycle_summary.get("asic_ntc_train_cycles", 0)
        + cycle_summary.get("asic_ntc_cycles", 0)
        + cycle_summary.get("asic_gradient_pipeline_cycles", 0)
        + cycle_summary.get("asic_optimizer_cycles", 0)
    ) / config.asic_spec.clock_hz

    result = build_result(
        baseline="3DGStream",
        dataset=ds_cfg.name,
        mode=mode,
        n_frames_trained=len(frames),
        quality=quality,
        render_fps=fps,
        total_storage_mb=total_storage_mb,
        delta_storage_mb=delta_storage_mb,
        asic_train_time_s=asic_train_time_s,
        cycle_summary=cycle_summary,
        stream_stats=stats,
        fps_diagnostic=fps_diag,
        base_train_iters=base_iters,
        avg_train_iters=avg_ntc_iters,
        per_frame_train_iters=per_frame_ntc_iters,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_dir  = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name / "3DGStream"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_file = json_dir / f"mode_{mode}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {json_file}")
    print(json.dumps(result, indent=2))
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GLASS ASIC ablation runner — 3DGStream"
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
