"""
Ablation runner for QUEEN — GLASS ASIC Simulator.

QUEEN trains all frames jointly using compressed Gaussian splatting with
latent decoders and gate modules. The training loop iterates through
frame_idx in [start_idx, max_frames] with per-frame inner iteration loops.

Integration strategy
---------------------
We run QUEEN's full training() function with a monkey-patched optimizer so
that pipeline.pre_optimizer_step() fires on each optimizer.step() call.
Pipeline.begin_frame() is called by an outer wrapper around the frame-level
part of the training loop. Since we cannot easily split QUEEN's monolithic
training() function, we:
  1. Patch gaussians.optimizer.step before calling training()
  2. Track frame boundaries via a frame_hook installed into the render_mask
     wrapper (QUEEN's per-frame rendering trigger)
  3. After training() returns, measure FPS and quality on test cameras

Usage
-----
    python ablation/run_queen.py --mode 0 --dataset flame_steak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GLASS_DIR    = Path(__file__).resolve().parents[2]
BASELINE_DIR = GLASS_DIR / "baselines" / "QUEEN"

sys.path.insert(0, str(GLASS_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from simulator.config import ASICConfig
from simulator.datasets import load_dataset_config
from simulator.metrics.collector import MetricsCollector
from simulator.operators.op_queen import OpQUEEN
from simulator.pipeline import GLASSPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_queen_config() -> dict:
    """Load QUEEN's dynerf.yaml as the default config dict."""
    import yaml
    from collections import defaultdict
    cfg_path = BASELINE_DIR / "configs" / "dynerf.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    return defaultdict(lambda: {}, raw)


def _build_queen_args(ds_cfg, mode: int):
    """Build an argparse Namespace matching QUEEN's CLI."""
    from arguments import (ModelParams, OptimizationParams, PipelineParams,
                            QuantizeParams, OptimizationParamsInitial,
                            OptimizationParamsRest)

    config = _load_queen_config()

    parser = argparse.ArgumentParser(add_help=False)
    lp  = ModelParams(parser,              config["model_params"])
    opi = OptimizationParamsInitial(parser, config["opt_params_initial"])
    opr = OptimizationParamsRest(parser,   config["opt_params_rest"])
    pp  = PipelineParams(parser,           config["pipe_params"])
    qp  = QuantizeParams(parser,           config["quantize_params"])
    # OptimizationParams is NOT a ParamGroup — it takes extracted opi/opr objects.
    # Build it after parse_args so we have the final extracted values.

    parser.add_argument("--test_iterations",         nargs="+", type=int,
                        default=[1000, 2000, 3000])
    parser.add_argument("--save_iterations",         nargs="+", type=int,
                        default=[3000])
    parser.add_argument("--checkpoint_iterations",   nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",        type=str,  default=None)
    parser.add_argument("--start_frame_idx",         type=int,  default=1)
    parser.add_argument("--debug_from",              type=int,  default=-1)
    parser.add_argument("--quiet",                   action="store_true")
    parser.add_argument("--use_xyz_legacy",          action="store_true")
    # Note: --max_frames and --start_idx are registered by ModelParams above.

    src = str(ds_cfg.resolved("source_path_queen"))
    out = str(GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name /
              "QUEEN" / f"mode_{mode}_run")
    os.makedirs(out, exist_ok=True)

    base_args = ["--source_path", src, "--model_path", out]
    args = parser.parse_args(base_args)

    # Apply extra_args
    extra = ds_cfg.model_args("queen")
    for k, v in extra.items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.max_frames = ds_cfg.n_frames_ablation
    args.model_path = out

    # When a pre-trained init model is available, load it and start the
    # training loop at frame 2, skipping the expensive frame-1 initialisation.
    # This mirrors 3DGStream's offline init-model workflow.
    init_model_path = ds_cfg.resolved("init_model_queen")
    if init_model_path and init_model_path.exists():
        # Resolve the latest saved iteration inside the init model directory.
        from utils.system_utils import searchForMaxIteration
        loaded_iter = searchForMaxIteration(str(init_model_path / "point_cloud"))
        # Symlink the init PLY into the output dir so Scene.load_ply() can
        # find it at model_path/point_cloud/iteration_{loaded_iter}/point_cloud.ply
        # while QUEEN still writes tensorboard / checkpoints to model_path (out).
        import shutil
        pc_link = Path(out) / "point_cloud" / f"iteration_{loaded_iter}"
        pc_link.mkdir(parents=True, exist_ok=True)
        ply_dst = pc_link / "point_cloud.ply"
        ply_src = init_model_path / "point_cloud" / f"iteration_{loaded_iter}" / "point_cloud.ply"
        if not ply_dst.exists():
            shutil.copy2(str(ply_src), str(ply_dst))
        args.load_iteration = loaded_iter
        args.start_frame_idx = 2
        # +1 so the dataset loader covers frame-1 (init) + n incremental frames.
        args.max_frames = ds_cfg.n_frames_ablation + 1
        # Test/save iterations must fall within the incremental frame budget
        # (epochs_rest * n_cams = 200 iters/frame).
        args.test_iterations = [100, 200]
        args.save_iterations = []
        print(f"  [init model] {init_model_path} (iter {loaded_iter}), "
              f"start_frame_idx=2, max_frames={args.max_frames}")

    # Build OptimizationParams from the extracted init/rest objects
    op = OptimizationParams(opi.extract(args), opr.extract(args))

    return args, lp, op, pp, qp, opi, opr


def run_one_mode(mode: int, ds_cfg, n_frames: int | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  QUEEN  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    from arguments import (ModelParams, OptimizationParams, PipelineParams,
                            QuantizeParams, OptimizationParamsInitial,
                            OptimizationParamsRest)
    from gaussian_renderer import render_mask
    from scene import Scene, GaussianModel
    import torch

    args, lp, op, pp, qp, opi, opr = _build_queen_args(ds_cfg, mode)

    # Whether Frame 1 is loaded from a pre-trained init model (start_frame_idx=2).
    using_init_model = (args.start_frame_idx > 1)

    # Override n_frames when --n-frames is given (quick diagnostic runs).
    # When using an init model, add +1 so the dataset covers the skipped frame.
    effective_n_frames = ds_cfg.n_frames_ablation
    if n_frames is not None:
        effective_n_frames = n_frames
        args.max_frames = n_frames + (1 if using_init_model else 0)
        print(f"  [n_frames override] effective={n_frames}, max_frames={args.max_frames}")

    config    = ASICConfig(
        mode=mode,
        n_frames_ablation=effective_n_frames,
        fps_warmup=ds_cfg.fps_warmup,
        fps_measure=ds_cfg.fps_measure,
        # UCN configuration for QUEEN:
        # - ``frame_level_ucn=True`` is pinned explicitly (not left to the
        #   auto-structural detector) because auto-detection only fires at
        #   the start of frame 2, and frame 1 with 188 K Gaussians × 6 attrs
        #   × 150 steps on the dict (slow) path saturates a CPU core for
        #   tens of seconds while the GPU sits idle.  The fast path requires
        #   ``frame_level_ucn=True`` from step 0, so we must set it up front.
        #   The auto-detector respects this user override and will not flip
        #   it back (see ``_user_frame_level_ucn`` in GLASSPipeline).
        # - ``merge_cos_threshold`` stays at 0.90 (the default); the actual
        #   per-frame threshold is then moved by the feedback controller
        #   (``cfg.adaptive_ucn_threshold``, enabled by default) toward
        #   the target suppression ratio.
        # - ``min_delta_norm=1e-6`` is QUEEN-specific: the latent-decoder
        #   (``ldec_*``) path has extremely small per-parameter gradients
        #   that the default 1e-5 floor would zero out entirely and kill
        #   the decoder's adaptation.  Not an UCN aggressiveness knob.
        frame_level_ucn=True,
        min_delta_norm=1e-6,
    )
    slot      = OpQUEEN()
    pipeline  = GLASSPipeline(config, slot)
    collector = MetricsCollector()

    # ── Instrument QUEEN's training function with timing & hook installs ──────
    # We wrap training() to intercept its gaussians.optimizer.step calls.
    # Since training() creates gaussians internally, we patch render_mask to
    # get a reference to the gaussians object and install hooks lazily.

    frame_times: list[float] = []
    per_frame_delta_bytes: list[int] = []
    _state = {"gaussians": None, "optimizer": None, "installed": False,
               "frame_t0": None, "frame_idx": 0,
               "cur_frame_idx": None, "n_gaussians": 0,
               "in_test_report": False,
               # Per-frame training iteration counter: incremented on every
               # training render (grad_enabled) so we know how many iterations
               # ran in each frame.  Used by add_gaussian_training_iters() to
               # model the 3DGS render+loss+bwd cycles for asic_train_time_s.
               "train_iters_this_frame": 0,
               # Total trainable parameters in QUEEN's network param groups
               # (decoder + gate).  Measured once at first hook install and
               # used by add_network_training() to model decoder/gate backprop.
               "n_network_params": 0}

    # ── Quality measurement: wrap training_report ─────────────────────────────
    # QUEEN calls training_report(is_test=True) at the LAST iteration of every
    # frame.  Inside training_report, it renders scene.getTestCameras() with
    # renderFunc and computes PSNR.  We wrap training_report so that we can
    # inject a capturing renderFunc and also compute SSIM/LPIPS on the same
    # (render, gt) pairs — giving metrics identical to QUEEN's internal values.
    _per_frame_pairs: list[list] = []   # list of [(render_cpu, gt_cpu), ...] per frame

    # Rolling window of recent cameras for FPS measurement
    _recent_cams: list = []
    _MAX_RECENT_CAMS = 40

    orig_render_mask = None

    def _patched_render_mask(cam, gaussians, pipe_, bg, **kwargs):
        # ── Hook (re-)installation ────────────────────────────────────────────
        # QUEEN calls training_setup() at the start of EVERY frame, which
        # replaces gaussians.optimizer with a new object.  We must reinstall
        # the hook each time the optimizer identity changes.
        if gaussians is not None and not _state.get("in_test_report"):
            current_opt = getattr(gaussians, "optimizer", None)
            if current_opt is not None and id(current_opt) != _state.get("_opt_id"):
                # Uninstall from the old optimizer (if any)
                if _state["installed"]:
                    slot.uninstall_hooks()
                    _state["installed"] = False
                # Install on the new optimizer
                _state["gaussians"] = gaussians
                _state["optimizer"] = current_opt
                _state["_opt_id"]   = id(current_opt)
                slot.install_hooks(pipeline, gaussians=gaussians, optimizer=current_opt)
                _state["installed"] = True
                # Measure network (decoder+gate) parameter count from the
                # optimizer's param groups.  Done once (first install); QUEEN
                # re-creates the optimizer each frame but keeps the same
                # network architecture, so param count is stable.
                if _state["n_network_params"] == 0:
                    # Identify network parameter groups as those whose name
                    # starts with "ldec_" or is in network_attr_groups.
                    # QUEEN uses abbreviated names for Gaussian attributes
                    # (e.g. "sc" for scaling, "rot" for rotation) and "ldec_<attr>"
                    # for latent decoders.  Gaussian groups can be identified
                    # as groups that do NOT start with "ldec_" and are not
                    # explicitly listed in network_attr_groups.
                    net_params = 0
                    for pg in current_opt.param_groups:
                        pg_name = pg.get("name", "")
                        pg_count = sum(p.numel() for p in pg["params"])
                        is_network = (
                            pg_name.startswith("ldec_")
                            or pg_name in slot.network_attr_groups
                        )
                        if is_network:
                            net_params += pg_count
                    _state["n_network_params"] = net_params
                    print(f"  [network params] ldec+gate total: {net_params:,}")
                # Begin first frame only on very first install
                if _state["cur_frame_idx"] is None:
                    pipeline.begin_frame()
                    _state["cur_frame_idx"] = getattr(cam, "frame_idx", 0)

        # Track latest known Gaussian count so we can pass it to stream_stats()
        # after slot.uninstall_hooks() clears slot._gaussians.
        if gaussians is not None:
            try:
                _state["n_gaussians"] = int(gaussians.get_xyz.shape[0])
            except Exception:
                pass

        # ── Frame-boundary detection → begin_frame() ─────────────────────────
        # QUEEN's training is monolithic so we must infer frame boundaries from
        # cam.frame_idx transitions.  At each transition we advance the pipeline
        # frame counter so stream / cycle / fabric state resets correctly.
        # Skip this during training_report(is_test=True): test cameras have
        # a different frame_idx (they reference future / held-out frames) and
        # would trigger a spurious begin_frame() that clears cycle accumulators
        # filled by the current training frame.
        cam_frame = getattr(cam, "frame_idx", None)
        if (not _state.get("in_test_report")
                and cam_frame is not None
                and _state.get("cur_frame_idx") is not None
                and cam_frame != _state["cur_frame_idx"]):
            # Flush training cycles for the frame that just completed
            # before resetting the pipeline state for the new frame.
            n_iters = _state["train_iters_this_frame"]
            pipeline.add_gaussian_training_iters(
                n_iters,
                n_total=_state.get("n_gaussians", 0),
            )
            pipeline.add_network_training(
                _state.get("n_network_params", 0),
                n_iters,
            )
            _state["train_iters_this_frame"] = 0
            pipeline.begin_frame()
            _state["cur_frame_idx"] = cam_frame

        # ── post_render: feed visibility mask into ILSF / Salience ───────────
        # We call the real render first, then pass the result to post_render so
        # the LiveStateFabric and SalienceFirstIssue see per-frame visibility.
        result = orig_render_mask(cam, gaussians, pipe_, bg, **kwargs)
        if torch.is_grad_enabled():
            # Only during training renders (not eval / FPS warmup)
            pipeline.post_render(result)
            if not _state.get("in_test_report"):
                # Capture scene stats for add_gaussian_training_iters() on the
                # NEXT frame boundary.  Lag-one design: frame N's render stats
                # are used to model frame N's training cycles at begin of frame N+1.
                sh_deg = 0
                if gaussians is not None:
                    try:
                        sh_deg = int(getattr(gaussians, "active_sh_degree", 0))
                    except Exception:
                        pass
                pipeline.set_render_stats_from_render_pkg(result, sh_deg)
                _state["train_iters_this_frame"] += 1

        # Rolling window for FPS measurement
        _recent_cams.append(cam)
        if len(_recent_cams) > _MAX_RECENT_CAMS:
            _recent_cams.pop(0)

        return result

    # ── Patch train.training_report to capture test-camera (render, gt) pairs ─
    # training() and training_report() are both defined in train.py (same module).
    # Python resolves the call at runtime via the module's global namespace, so
    # patching train.training_report intercepts every call from training().
    import sys as _sys
    # Ensure train is imported first so we can patch its namespace
    from train import training as queen_training, training_report as _orig_tr
    import train as _train_mod
    _orig_training_report = _orig_tr   # save original

    def _capturing_training_report(
            tb_writer, wandb_enabled, model_args, frame_idx, iteration,
            Ll1, loss, l1_loss, size, elapsed, is_test, scene,
            renderFunc, renderArgs, **kwargs):
        """Wrap training_report to capture test-camera renders for PSNR/SSIM."""
        if not is_test:
            return _orig_training_report(
                tb_writer, wandb_enabled, model_args, frame_idx, iteration,
                Ll1, loss, l1_loss, size, elapsed, is_test, scene,
                renderFunc, renderArgs, **kwargs)

        # Build a capturing renderFunc that records (render, gt) for test cameras
        frame_pairs: list[tuple] = []
        n_test_cams = len(scene.getTestCameras())

        def _capturing_renderFunc(viewpoint, gaussians_, *ra, **kw):
            pkg = renderFunc(viewpoint, gaussians_, *ra, **kw)
            # Only capture the first n_test_cams calls (test cameras);
            # the remaining calls are val (training) cameras.
            if len(frame_pairs) < n_test_cams:
                render = pkg["render"].detach().clamp(0, 1).cpu()
                gt     = viewpoint.original_image.float().clamp(0, 1).cpu()
                frame_pairs.append((render, gt))
            return pkg

        # Guard against test-camera frame_idx triggering begin_frame() and
        # resetting the cycle accumulators that training just filled.
        _state["in_test_report"] = True
        try:
            result = _orig_training_report(
                tb_writer, wandb_enabled, model_args, frame_idx, iteration,
                Ll1, loss, l1_loss, size, elapsed, is_test, scene,
                _capturing_renderFunc, renderArgs, **kwargs)
        finally:
            _state["in_test_report"] = False

        if frame_pairs:
            _per_frame_pairs.append(frame_pairs)

        return result

    _train_mod.training_report = _capturing_training_report

    import gaussian_renderer as _gr
    orig_render_mask = _gr.render_mask
    _gr.render_mask = _patched_render_mask
    # Also patch the name in train.py's namespace so the main loop uses our hook
    _train_mod.render_mask = _patched_render_mask

    # ── Run QUEEN training ─────────────────────────────────────────────────────
    t0_total = time.perf_counter()
    try:
        with collector.frame_timer():
            queen_training(
                lp.extract(args), op, pp.extract(args),
                qp.extract(args),
                args.test_iterations, args.save_iterations,
                args.checkpoint_iterations, args.start_checkpoint,
                args.debug_from, args,
            )
    except Exception as e:
        print(f"  QUEEN training raised: {e}")
        raise
    finally:
        _gr.render_mask = orig_render_mask
        _train_mod.render_mask = orig_render_mask
        _train_mod.training_report = _orig_training_report
        if _state["installed"]:
            slot.uninstall_hooks()
        # Flush remaining training iters for the last frame (no subsequent
        # begin_frame() is called after the final training frame).
        n_iters_last = _state["train_iters_this_frame"]
        pipeline.add_gaussian_training_iters(
            n_iters_last,
            n_total=_state.get("n_gaussians", 0),
        )
        pipeline.add_network_training(
            _state.get("n_network_params", 0),
            n_iters_last,
        )
        _state["train_iters_this_frame"] = 0

    total_time = time.perf_counter() - t0_total
    avg_frame_time = total_time / max(effective_n_frames, 1)

    _n_cached = _state.get("n_gaussians", 0)
    per_frame_delta_bytes.append(
        pipeline.stream_stats(n_gaussians=_n_cached)["frame_active_bytes"]
    )

    gaussians = _state["gaussians"]

    n_frames_with_quality = len(_per_frame_pairs)
    print(f"  Intercepted training_report test renders: {n_frames_with_quality} frames, "
          f"{sum(len(p) for p in _per_frame_pairs)} test-camera pairs total")

    # ── Aggregate PSNR / SSIM / LPIPS from intercepted test-camera renders ────
    quality = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
    fps = 0.0
    _n_gaussians_for_stats: int = 0
    lpips_val = float("nan")

    if _per_frame_pairs:
        try:
            from torchmetrics.image import (
                PeakSignalNoiseRatio as PSNR,
                StructuralSimilarityIndexMeasure as SSIM,
                LearnedPerceptualImagePatchSimilarity as LPIPS,
            )
            psnr_fn  = PSNR(data_range=1.0)
            ssim_fn  = SSIM(data_range=1.0)
            lpips_fn = LPIPS(net_type="alex")

            frame_psnrs, frame_ssims, frame_lpips = [], [], []
            for pairs in _per_frame_pairs:
                ps, ss, ls = [], [], []
                for render, gt in pairs:
                    r_ = render.unsqueeze(0)
                    g_ = gt.unsqueeze(0)
                    ps.append(float(psnr_fn(r_, g_).item()))
                    ss.append(float(ssim_fn(r_, g_).item()))
                    ls.append(float(lpips_fn(r_, g_).item()))
                frame_psnrs.append(sum(ps) / len(ps))
                frame_ssims.append(sum(ss) / len(ss))
                frame_lpips.append(sum(ls) / len(ls))

            avg_psnr  = sum(frame_psnrs) / len(frame_psnrs)
            avg_ssim  = sum(frame_ssims) / len(frame_ssims)
            avg_lpips = sum(frame_lpips) / len(frame_lpips)
            lpips_val = avg_lpips
            quality = {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips}
            print(f"  Avg quality over {len(_per_frame_pairs)} frames "
                  f"(intercepted test renders): "
                  f"PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")
        except Exception as e:
            import traceback
            print(f"  Quality aggregation failed: {e}")
            traceback.print_exc()

    # ── FPS measurement ────────────────────────────────────────────────────────
    print(f"  Measuring FPS … ({len(_recent_cams)} recent cams for timing)")
    if gaussians is not None and _recent_cams:
        try:
            bg_color   = [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            pp_extract = pp.extract(args)
            opt_for_fps = (getattr(gaussians, "optimizer", None)
                           or _state["optimizer"])
            if opt_for_fps is not None:
                slot.install_hooks(pipeline, gaussians=gaussians,
                                   optimizer=opt_for_fps)
            else:
                slot._gaussians = gaussians

            fps = collector.measure_fps(
                pipeline,
                lambda cam, g, p, bg_: orig_render_mask(cam, g, p, bg_),
                _recent_cams, pp_extract, background,
            )
            print(f"  render_fps = {fps:.1f}")
            pipeline.set_render_cycles_from_fps(fps)

            # Save n_gaussians before uninstall clears slot._gaussians so that
            # stream_stats() can compute accurate fabric tier_counts later.
            _n_gaussians_for_stats = slot.n_gaussians() or _state.get("n_gaussians", 0)
            slot.uninstall_hooks()
        except Exception as e:
            import traceback
            print(f"  FPS measurement failed: {e}")
            traceback.print_exc()

    # Save model checkpoint so measure_storage finds a .ply file
    if gaussians is not None:
        try:
            import os as _os
            ply_path = _os.path.join(args.model_path, "point_cloud", "final",
                                     "point_cloud.ply")
            gaussians.save_ply(ply_path, mask=None)
            print(f"  Saved PLY → {ply_path}")
        except Exception as e:
            print(f"  PLY save failed: {e}")

    out_dir = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name / "QUEEN"
    total_storage_mb = collector.measure_storage(args.model_path)
    delta_storage_mb = collector.measure_delta_storage(per_frame_delta_bytes)
    stats = pipeline.stream_stats(n_gaussians=_n_gaussians_for_stats)
    cycle_summary = pipeline.cycle_summary()

    result = {
        "mode":             mode,
        "baseline":         "QUEEN",
        "dataset":          ds_cfg.name,
        "n_frames_trained": effective_n_frames,
        "avg_train_time_s": avg_frame_time,
        "render_fps":       fps,
        **quality,
        "total_storage_mb": total_storage_mb,
        "delta_storage_mb": delta_storage_mb,
        **stats,
        "incremental":      True,
        **cycle_summary,
        # asic_commit_time_s: ASIC commit-path only (gradient capture + Adam).
        # Minimal view: excludes render cost, useful for comparing mode 0 → 1 → 2 → 3.
        "asic_commit_time_s": (
            cycle_summary.get("asic_gradient_pipeline_cycles", 0)
            + cycle_summary.get("asic_optimizer_cycles", 0)
        ) / config.asic_spec.clock_hz,
        # asic_train_time_s: FULL ASIC incremental training time per frame.
        #   = render_train cycles  (n_iters × 3DGS render+loss+bwd, modelled via
        #                           add_gaussian_training_iters at frame boundary)
        #   + gradient pipeline    (PUOE gradient capture)
        #   + optimizer            (Adam on active Gaussians)
        #   / clock_hz
        # Directly comparable to avg_train_time_s (GPU wall-clock per frame):
        #   avg_train_time_s     → GPU incremental training time
        #   asic_train_time_s    → ASIC incremental training time at clock_hz
        # Speedup = avg_train_time_s / asic_train_time_s
        "asic_train_time_s": (
            cycle_summary.get("asic_ntc_train_render_cycles", 0)
            + cycle_summary.get("asic_gradient_pipeline_cycles", 0)
            + cycle_summary.get("asic_optimizer_cycles", 0)
        ) / config.asic_spec.clock_hz,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mode_{mode}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_file}")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="GLASS ASIC ablation — QUEEN")
    parser.add_argument("--mode",           type=int, choices=[0,1,2,3], required=True)
    parser.add_argument("--dataset",        type=str, default="flame_steak")
    parser.add_argument("--dataset-config", type=str, default=None, dest="dataset_config")
    parser.add_argument("--n-frames",       type=int, default=None,
                        help="Override number of frames to train (default: dataset config)")
    args = parser.parse_args()
    ds_cfg = load_dataset_config(args.dataset, args.dataset_config)
    run_one_mode(args.mode, ds_cfg, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
