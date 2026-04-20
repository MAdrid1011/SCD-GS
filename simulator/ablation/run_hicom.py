"""
Ablation runner for HiCoM — GLASS ASIC Simulator.

HiCoM uses a libgs-based pipeline (Trainer + Module). Training is driven by:
  Trainer.training_loop()
    → Trainer.frame_training_loop()       ← frame boundary
      → Module.training_step()            ← forward + backward
      → Module.post_training_step()       ← optimizer.step() calls

Integration strategy
---------------------
1. Construct HiCoM's pipeline from a config file via load_pipeline("hicom").
2. After the pipeline is initialised (trainer and module are accessible),
   call slot.install_hooks(pipeline, trainer=trainer, module=module).
3. Wrap trainer.training_loop() with a frame-level timer so we get
   per-frame training times.
4. Call trainer.training_loop() (which internally calls frame_training_loop()
   for each frame); hooks fire inside.
5. After training, measure FPS and quality using the module's validation logic.

Usage
-----
    python ablation/run_hicom.py --mode 0 --dataset flame_steak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GLASS_DIR    = Path(__file__).resolve().parents[2]
BASELINE_DIR = GLASS_DIR / "baselines" / "HiCoM"

sys.path.insert(0, str(GLASS_DIR))
sys.path.insert(0, str(BASELINE_DIR))

from simulator.config import ASICConfig
from simulator.datasets import load_dataset_config
from simulator.metrics.collector import MetricsCollector
from simulator.operators.op_hicom import OpHiCoM
from simulator.pipeline import GLASSPipeline


def _build_hicom_config(ds_cfg, mode: int):
    """Return a HiCoM OmegaConf override dict + output dir for the dataset.

    Notes
    -----
    - ``datamodule`` is not a top-level key in HiCoM's Config — the data layer
      is rooted at ``data.*``.
    - ``trainer.num_frames`` does not exist; HiCoM reads the frame count from
      ``datamodule.scene.train_dataset.num_frames``.  Our ablation caps the
      number of incremental frames directly inside ``_timed_training_loop``.
    - When a frame-0 checkpoint is available (``init_model_hicom``), we set
      the top-level ``ckpt_path`` so ``trainer.setup`` restores gaussians and
      deformation state before we enter the incremental loop.
    """
    out_dir = str(GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name /
                  "HiCoM" / f"mode_{mode}_run")
    os.makedirs(out_dir, exist_ok=True)

    extra = ds_cfg.model_args("hicom")
    num_incr_steps = int(extra.get("trainer.num_incr_steps", 200))
    num_init_steps = int(extra.get("trainer.num_init_steps", 15000))

    overrides: dict = {
        "data.root":                 str(ds_cfg.resolved("source_path_hicom")),
        "output_dir":                out_dir,
        "experiment_name":           f"mode_{mode}",
        "trainer.num_incr_steps":    num_incr_steps,
        "trainer.num_init_steps":    num_init_steps,
        "trainer.saving_ckpt_steps": [],
        "trainer.testing_steps":     [],
    }

    init_ckpt = ds_cfg.resolved("init_model_hicom")
    if init_ckpt.exists() and init_ckpt.is_file():
        overrides["ckpt_path"] = str(init_ckpt)

    return overrides, out_dir


def run_one_mode(mode: int, ds_cfg, n_frames: int | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  HiCoM  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    import torch

    if n_frames is not None:
        ds_cfg.n_frames_ablation = int(n_frames)

    overrides, out_dir = _build_hicom_config(ds_cfg, mode)

    config   = ASICConfig(
        mode=mode,
        n_frames_ablation=ds_cfg.n_frames_ablation,
        fps_warmup=ds_cfg.fps_warmup,
        fps_measure=ds_cfg.fps_measure,
    )
    slot      = OpHiCoM()
    pipeline  = GLASSPipeline(config, slot)
    collector = MetricsCollector()

    per_frame_delta_bytes: list[int] = []
    _state = {"trainer": None, "module": None}

    try:
        from libgs.pipeline.loader import load_pipeline
        from omegaconf import OmegaConf as oc

        Config, PipelineCls = load_pipeline("hicom")

        # Preserve list-valued overrides (e.g. saving_ckpt_steps: []) that
        # OmegaConf.from_dotlist would otherwise stringify.
        dotlist_overrides = {k: v for k, v in overrides.items()
                             if not isinstance(v, list)}
        struct_overrides  = {k: v for k, v in overrides.items()
                             if isinstance(v, list)}

        default_cfg_path = BASELINE_DIR / "config" / "dynerf.yaml"
        base_cfg = (oc.load(str(default_cfg_path))
                    if default_cfg_path.exists() else oc.create({}))
        cli_cfg  = oc.from_dotlist(
            [f"{k}={v}" for k, v in dotlist_overrides.items()]
        )
        merged   = oc.merge(base_cfg, cli_cfg)
        for dotted, val in struct_overrides.items():
            oc.update(merged, dotted, val, merge=False)

        from libgs.utils.config import to_structured
        cfg, _ = to_structured(merged, Config)

        hicom_pipeline = PipelineCls(cfg)
        trainer = hicom_pipeline.trainer
        module  = hicom_pipeline.module
        _state["trainer"] = trainer
        _state["module"]  = module

        # ── Manually drive trainer.setup (normally invoked by pipeline.run())
        # so we can (a) load the frame-0 ckpt, (b) reset global_step so the
        # incremental loop starts at step 1, and (c) install hooks *after*
        # the Gaussian tensors have been restored.
        ckpt_path = cfg.ckpt_path if cfg.ckpt_path else None
        trainer.setup(module, ckpt_path=ckpt_path)
        if ckpt_path is not None:
            # After on_load_checkpoint: current_frame=0, global_step=num_init_steps.
            # Clear global_step so frame_training_loop enters its inner
            # `range(self.global_step+1, num_steps+1)` starting at 1.
            trainer.global_step = 0
            print(f"  Loaded frame-0 ckpt → {ckpt_path}")
            print(f"    current_frame = {trainer.datamodule.current_frame}"
                  f"   n_gaussians = {module.gaussians.get_xyz.shape[0]}")

        slot.install_hooks(pipeline, trainer=trainer, module=module)

        n_total_frames = trainer.datamodule.num_frames
        # ``current_frame`` points at the last trained frame (0 after ckpt load
        # or -1 from scratch); incremental frames are [current_frame+1, …].
        first_incr    = trainer.datamodule.current_frame + 1
        last_incr_exc = min(n_total_frames,
                            first_incr + ds_cfg.n_frames_ablation)

        # ``slot.install_hooks`` already wrapped ``trainer.frame_training_loop``
        # so that ``pipeline.begin_frame()`` fires at the start of every frame.
        # We must NOT call it a second time here — doing so would reset the
        # per-frame byte / unique-target caches after the hook already
        # populated them, and silently zero out the stream stats.
        _n_gaussians_for_stats = 0
        # HiCoM trains every incremental frame for ``num_incr_steps`` render +
        # loss + backward passes (default 200: 100 deform-stage1 + 100 GS-
        # stage2).  We account for these as 3DGS-style render_training iters
        # on the ASIC cycle model; this is the only remaining contributor to
        # asic_train_time_s that the ASIC hooks don't already capture.
        n_iters_per_frame = int(overrides.get("trainer.num_incr_steps", 200))
        per_frame_iter_counts: list[int] = []

        def _timed_training_loop():
            nonlocal _n_gaussians_for_stats
            for current_frame in range(first_incr, last_incr_exc):
                trainer.datamodule.current_frame = current_frame
                trainer.global_step = 0

                # ── Path 2 — UCN→iter scheduler (mode >= 1, frame > first) ──
                # Mirrors run_3dgstream's adaptive schedule.  The
                # ``previous frame's`` per-byte UCN suppression ratio is
                # sourced from the pipeline's feedback accumulators, which
                # are populated by both pre_optimizer_step (HiCoM's dense
                # per-iter Adam path) and ingest_ntc_deltas.  Setting
                # ``trainer.config.num_incr_steps`` here feeds both the GPU
                # loop (``Trainer.num_steps`` reads it dynamically) and the
                # ASIC cycle model (``add_gaussian_training_iters`` below),
                # so wall-clock quality and simulator cycles agree on the
                # reduced iteration count.  Frame 0 of the ablation window
                # (``current_frame == first_incr``) always runs the full
                # base iteration count — no UCN history yet.
                is_first_ablation_frame = (current_frame == first_incr)
                if mode >= 1 and not is_first_ablation_frame:
                    prev_ratio = pipeline.current_frame_suppression_ratio()
                    adaptive_steps = max(
                        int(config.min_ntc_iters_floor),
                        round(n_iters_per_frame * (1.0 - prev_ratio)),
                    )
                else:
                    adaptive_steps = n_iters_per_frame
                trainer.config.num_incr_steps = adaptive_steps
                per_frame_iter_counts.append(adaptive_steps)

                with collector.frame_timer():
                    trainer.frame_training_loop()
                per_frame_delta_bytes.append(
                    pipeline.stream_stats()["frame_active_bytes"]
                )
                _n_gaussians_for_stats = slot.n_gaussians()
                # Book the ASIC render + loss + backward cycles for the
                # frame we just finished, using the ACTUAL number of steps
                # the Trainer executed (adaptive_steps) so that the cycle
                # model and wall-clock measurement agree.  The lag-one
                # render stats fed by op_hicom._post_step() already
                # describe a representative viewpoint from the same frame,
                # so cycle sizing is accurate without any magic-number
                # scaling.
                pipeline.add_gaussian_training_iters(
                    adaptive_steps, n_total=_n_gaussians_for_stats,
                )

        trainer.training_loop = _timed_training_loop
        trainer.training_loop()  # we already ran trainer.setup ourselves
        slot.uninstall_hooks()

    except Exception as e:
        print(f"  HiCoM training raised: {e}")
        raise

    # ── FPS measurement ───────────────────────────────────────────────────────
    print("  Measuring FPS …")
    fps = 0.0
    gaussians = None
    try:
        module = _state["module"]
        # Re-install hooks *before* get_gaussians(), since uninstall_hooks()
        # above cleared slot._module and _MergedGaussianAccessor needs it.
        if module is not None:
            slot.install_hooks(pipeline, trainer=_state["trainer"], module=module)
            gaussians = slot.get_gaussians()
        if gaussians is not None:

            # Use validation dataloader cameras
            val_loaders = module.datamodule.val_dataloader()
            if not isinstance(val_loaders, (list, tuple)):
                val_loaders = [val_loaders]
            val_cams = list(val_loaders[0])[:ds_cfg.fps_measure + ds_cfg.fps_warmup]

            if val_cams:
                def _render_fn(cam, g, pipe_, bg):
                    # Temporarily point the renderer at the MaskedGaussianProxy
                    # (``g``) built by ``pipeline.make_render_proxy()`` so the
                    # fabric render mask (mode >= 2) and salience keep-mask
                    # (mode >= 3) actually shape the rasterised workload.
                    # ``module.forward`` itself never rebinds
                    # ``self.renderer.gaussians`` (only ``pre_training_step``
                    # does, and that is not called during inference), so a
                    # simple try/finally swap is race-free.  Mode 0 / 1 pass
                    # the unmasked proxy, so their render cost is unchanged.
                    renderer = module.renderer
                    saved    = renderer.gaussians
                    renderer.gaussians = g
                    try:
                        return module.forward(cam.to(module.device))
                    finally:
                        renderer.gaussians = saved

                fps = collector.measure_fps(
                    pipeline,
                    _render_fn,
                    val_cams, None, None,
                )
                # Back-calc render cycles from measured FPS so that
                # cycle_summary() reports ``asic_render_cycles`` / the
                # ASIC equivalents — no magic numbers, sourced from the
                # spec clock_hz.
                if fps > 0:
                    pipeline.set_render_cycles_from_fps(fps)
            slot.uninstall_hooks()
    except Exception as e:
        print(f"  FPS measurement failed: {e}")

    # ── Quality measurement ───────────────────────────────────────────────────
    quality = {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
    try:
        module = _state["module"]
        if module is not None:
            val_results = []
            val_loaders = module.datamodule.val_dataloader()
            if not isinstance(val_loaders, (list, tuple)):
                val_loaders = [val_loaders]
            renders, gts = [], []
            for viewpoint in list(val_loaders[0])[:20]:
                vp = viewpoint.to(module.device)
                with torch.no_grad():
                    rp = module.forward(vp)
                renders.append(rp["render"].detach().cpu())
                gts.append(vp.image.cpu())
            if renders and gts:
                quality = collector.measure_quality(renders, gts)
    except Exception as e:
        print(f"  Quality measurement failed: {e}")

    total_storage_mb = collector.measure_storage(out_dir)
    delta_storage_mb = collector.measure_delta_storage(per_frame_delta_bytes)
    # Pass the cached Gaussian count so tier_counts stay accurate even though
    # slot.uninstall_hooks() has already cleared slot._module.
    stats         = pipeline.stream_stats(n_gaussians=_n_gaussians_for_stats)
    cycle_summary = pipeline.cycle_summary()

    result = {
        "mode":             mode,
        "baseline":         "HiCoM",
        "dataset":          ds_cfg.name,
        "n_frames_trained": len(per_frame_delta_bytes),
        "avg_train_time_s": collector.avg_frame_time(),
        "render_fps":       fps,
        **quality,
        "total_storage_mb": total_storage_mb,
        "delta_storage_mb": delta_storage_mb,
        **stats,
        "incremental":      True,
        **cycle_summary,
        # asic_commit_time_s: gradient pipeline + optimizer only (mirrors
        # run_queen / run_3dgstream convention).  Minimal view for
        # mode 0 → 1 → 2 → 3 comparison of the commit path alone.
        "asic_commit_time_s": (
            cycle_summary.get("asic_gradient_pipeline_cycles", 0)
            + cycle_summary.get("asic_optimizer_cycles", 0)
        ) / config.asic_spec.clock_hz,
        # asic_train_time_s: full ASIC incremental training time per frame.
        #   = n_iters × (render + loss + backward) cycles (from
        #                 add_gaussian_training_iters)
        #   + gradient pipeline  (PUOE / ICE etc.)
        #   + optimizer          (Adam on active Gaussians)
        # Directly comparable to ``avg_train_time_s`` (GPU wall-clock).
        "asic_train_time_s": (
            cycle_summary.get("asic_ntc_train_render_cycles", 0)
            + cycle_summary.get("asic_gradient_pipeline_cycles", 0)
            + cycle_summary.get("asic_optimizer_cycles", 0)
        ) / config.asic_spec.clock_hz,
    }

    out_path = GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name / "HiCoM"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"mode_{mode}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved → {out_file}")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="GLASS ASIC ablation — HiCoM")
    parser.add_argument("--mode",           type=int, choices=[0,1,2,3], required=True)
    parser.add_argument("--dataset",        type=str, default="flame_steak")
    parser.add_argument("--dataset-config", type=str, default=None, dest="dataset_config")
    parser.add_argument("--n-frames",       type=int, default=None,
                        help="Override number of incremental frames to train"
                             " (default: dataset config)")
    args = parser.parse_args()
    ds_cfg = load_dataset_config(args.dataset, args.dataset_config)
    run_one_mode(args.mode, ds_cfg, n_frames=args.n_frames)


if __name__ == "__main__":
    main()
