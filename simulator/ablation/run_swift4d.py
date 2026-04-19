"""
Ablation runner for Swift4D — GLASS ASIC Simulator.

Swift4D trains all timestamps jointly (not incrementally per frame) via a
coarse → dynamic → fine stage sequence. Each optimizer iteration processes
all frames simultaneously.

Adaptation notes
----------------
- Each optimizer iteration is treated as one logical "frame" for the CWB.
- Training time is reported as total_time / n_training_iterations (approx.
  per-frame equivalent, flagged as incremental=False in output JSON).
- OpSwift4D wraps gaussians.optimizer.step and hash_optimizer.step.
- FPS is measured on test cameras from the last training scene state.
- Quality is measured the same way as other baselines.

Usage
-----
    python ablation/run_swift4d.py --mode 0 --dataset flame_steak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GLASS_DIR    = Path(__file__).resolve().parents[2]
BASELINE_DIR = GLASS_DIR / "baselines" / "Swift4D"

sys.path.insert(0, str(GLASS_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from simulator.config import ASICConfig
from simulator.datasets import load_dataset_config
from simulator.metrics.collector import MetricsCollector
from simulator.operators.op_swift4d import OpSwift4D
from simulator.pipeline import GLASSPipeline


def _build_swift4d_args(ds_cfg, mode: int):
    """Build an argparse Namespace for Swift4D's training() function."""
    from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

    parser = argparse.ArgumentParser(add_help=False)
    lp  = ModelParams(parser)
    op  = OptimizationParams(parser)
    pp  = PipelineParams(parser)
    hyp = ModelHiddenParams(parser)

    parser.add_argument("--test_iterations",       nargs="+", type=int,
                        default=[1000, 3000, 5000, 7000])
    parser.add_argument("--save_iterations",       nargs="+", type=int,
                        default=[13001])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,  default=None)
    parser.add_argument("--debug_from",            type=int,  default=-1)
    parser.add_argument("--quiet",                 action="store_true")
    parser.add_argument("--expname",               type=str,  default="dynerf/flame_steak")

    src = str(ds_cfg.resolved("source_path_swift4d"))
    out = str(GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name /
              "Swift4D" / f"mode_{mode}_run")
    os.makedirs(out, exist_ok=True)

    base_args = [
        "--source_path", src,
        "--model_path",  out,
        "--expname",     "glass_ablation",
    ]
    args = parser.parse_args(base_args)

    # Apply extra_args
    extra = ds_cfg.model_args("swift4d")
    for k, v in extra.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.model_path = out

    return args, lp, op, pp, hyp


def run_one_mode(mode: int, ds_cfg) -> dict:
    print(f"\n{'='*60}")
    print(f"  Swift4D  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    import torch

    args, lp, op, pp, hyp = _build_swift4d_args(ds_cfg, mode)

    config    = ASICConfig(
        mode=mode,
        n_frames_ablation=ds_cfg.n_frames_ablation,
        fps_warmup=ds_cfg.fps_warmup,
        fps_measure=ds_cfg.fps_measure,
    )
    slot      = OpSwift4D()
    pipeline  = GLASSPipeline(config, slot)
    collector = MetricsCollector()

    per_frame_delta_bytes: list[int] = []
    _state = {"gaussians": None, "scene": None, "pipe_args": None, "background": None}

    # ── Patch scene_reconstruction to intercept gaussians + optimizer ─────────
    import train as swift4d_train
    orig_scene_reconstruction = swift4d_train.scene_reconstruction

    def _patched_scene_reconstruction(
        dataset, opt_, hyper_, pipe_, testing_iterations,
        saving_iterations, checkpoint_iterations, checkpoint,
        debug_from, gaussians, scene, stage, tb_writer, final_iter, timer, args_
    ):
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        _state["gaussians"]  = gaussians
        _state["scene"]      = scene
        _state["pipe_args"]  = pipe_
        _state["background"] = background

        # Install hooks on the Gaussian+hash optimizer
        slot.install_hooks(pipeline,
                           gaussians=gaussians,
                           optimizer=gaussians.optimizer)

        orig_scene_reconstruction(
            dataset, opt_, hyper_, pipe_, testing_iterations,
            saving_iterations, checkpoint_iterations, checkpoint,
            debug_from, gaussians, scene, stage, tb_writer, final_iter, timer, args_
        )
        slot.uninstall_hooks()

    swift4d_train.scene_reconstruction = _patched_scene_reconstruction

    # Also wrap hash_optimizer.step if it exists (install_hooks only wraps
    # gaussians.optimizer; we manually add the hash optimizer here):
    _orig_install = slot.install_hooks

    def _extended_install(pipeline_, gaussians=None, optimizer=None, **kw):
        _orig_install(pipeline_, gaussians=gaussians, optimizer=optimizer, **kw)
        # Wrap hash_optimizer too
        if gaussians is not None and hasattr(gaussians, "hash_optimizer"):
            orig_hash = gaussians.hash_optimizer.step
            def _hash_step(closure=None):
                pipeline_.pre_optimizer_step(gaussians.hash_optimizer)
                return orig_hash(closure)
            gaussians.hash_optimizer.step = _hash_step
            slot._orig_hash_step = orig_hash

    slot.install_hooks = _extended_install

    _orig_uninstall = slot.uninstall_hooks

    def _extended_uninstall():
        _orig_uninstall()
        g = _state["gaussians"]
        if g is not None and hasattr(slot, "_orig_hash_step"):
            g.hash_optimizer.step = slot._orig_hash_step

    slot.uninstall_hooks = _extended_uninstall

    # ── Run Swift4D training ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        from train import training as swift4d_training
        with collector.frame_timer():
            swift4d_training(
                lp.extract(args), hyp.extract(args), op.extract(args), pp.extract(args),
                args.test_iterations, args.save_iterations,
                args.checkpoint_iterations, args.start_checkpoint,
                args.debug_from, args.expname, args,
            )
    except Exception as e:
        print(f"  Swift4D training raised: {e}")
        raise
    finally:
        swift4d_train.scene_reconstruction = orig_scene_reconstruction

    total_time   = time.perf_counter() - t0
    n_steps      = slot.step_count if slot.step_count > 0 else 1
    avg_per_step = total_time / n_steps

    per_frame_delta_bytes.append(pipeline.stream_stats()["frame_active_bytes"])

    # ── FPS measurement ───────────────────────────────────────────────────────
    print("  Measuring FPS …")
    fps = 0.0
    gaussians = _state["gaussians"]
    scene     = _state["scene"]
    pipe_args = _state["pipe_args"]
    background = _state["background"]

    if gaussians is not None and scene is not None and background is not None:
        try:
            from gaussian_renderer import render as gs_render
            test_cams = scene.getTestCameras()
            if not test_cams:
                test_cams = scene.getTrainCameras()

            slot.install_hooks(pipeline, gaussians=gaussians,
                               optimizer=gaussians.optimizer)
            fps = collector.measure_fps(
                pipeline,
                lambda cam, g, p, bg: gs_render(cam, g, p, bg,
                                                 stage="fine", cam_type=None),
                test_cams, pipe_args, background,
            )
            slot.uninstall_hooks()
        except Exception as e:
            print(f"  FPS measurement failed: {e}")

    # ── Quality measurement ───────────────────────────────────────────────────
    quality = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
    if gaussians is not None and scene is not None and background is not None:
        try:
            from gaussian_renderer import render as gs_render
            test_cams = scene.getTestCameras()[:20]
            renders, gts = MetricsCollector.render_test_cameras(
                gaussians, test_cams,
                lambda cam, g, p, bg: gs_render(cam, g, p, bg,
                                                 stage="fine", cam_type=None),
                pipe_args, background,
            )
            if renders and gts:
                quality = collector.measure_quality(renders, gts)
        except Exception as e:
            print(f"  Quality measurement failed: {e}")

    total_storage_mb = collector.measure_storage(args.model_path)
    delta_storage_mb = collector.measure_delta_storage(per_frame_delta_bytes)
    stats = pipeline.stream_stats()

    result = {
        "mode":                  mode,
        "baseline":              "Swift4D",
        "dataset":               ds_cfg.name,
        "n_frames_trained":      ds_cfg.n_frames_ablation,
        "n_optimizer_steps":     slot.step_count,
        "avg_train_time_s":      avg_per_step,
        "total_train_time_s":    total_time,
        "render_fps":            fps,
        **quality,
        "total_storage_mb":      total_storage_mb,
        "delta_storage_mb":      delta_storage_mb,
        **stats,
        "incremental":           False,   # Swift4D is joint training, not per-frame
    }

    out_path = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name / "Swift4D"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"mode_{mode}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_file}")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="GLASS ASIC ablation — Swift4D")
    parser.add_argument("--mode",           type=int, choices=[0,1,2,3], required=True)
    parser.add_argument("--dataset",        type=str, default="flame_steak")
    parser.add_argument("--dataset-config", type=str, default=None, dest="dataset_config")
    args = parser.parse_args()
    ds_cfg = load_dataset_config(args.dataset, args.dataset_config)
    run_one_mode(args.mode, ds_cfg)


if __name__ == "__main__":
    main()
