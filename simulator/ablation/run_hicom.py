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
    """Return a HiCoM OmegaConf config dict for the specified dataset."""
    # HiCoM is configured via yaml config files; we construct an override dict
    # and merge it over the default hicom config.
    out_dir = str(GLASS_DIR / "results" / "GLASS_ablation" / ds_cfg.name /
                  "HiCoM" / f"mode_{mode}_run")
    os.makedirs(out_dir, exist_ok=True)

    extra = ds_cfg.model_args("hicom")
    num_incr_steps = extra.get("trainer.num_incr_steps", 200)
    num_init_steps = extra.get("trainer.num_init_steps", 15000)

    overrides = {
        "datamodule.root":            str(ds_cfg.resolved("source_path_hicom")),
        "output_dir":                  out_dir,
        "trainer.num_steps":           num_incr_steps,
        "trainer.num_init_steps":      num_init_steps,
        "trainer.num_frames":          ds_cfg.n_frames_ablation + 1,
        "trainer.saving_ckpt_steps":   [],
        "trainer.testing_steps":       [num_incr_steps],
    }
    if ds_cfg.resolved("init_model_hicom").exists():
        overrides["trainer.init_checkpoint"] = str(
            ds_cfg.resolved("init_model_hicom"))

    return overrides, out_dir


def run_one_mode(mode: int, ds_cfg) -> dict:
    print(f"\n{'='*60}")
    print(f"  HiCoM  |  ASIC mode {mode}  |  dataset: {ds_cfg.name}")
    print(f"{'='*60}")

    import torch

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

        # Build config with overrides
        cli_overrides = [f"{k}={v}" for k, v in overrides.items()]
        # If a default config file exists, use it
        default_cfg_path = BASELINE_DIR / "config" / "hicom.yaml"
        if default_cfg_path.exists():
            base_cfg = oc.load(str(default_cfg_path))
        else:
            base_cfg = oc.create({})

        cli_cfg = oc.from_dotlist(cli_overrides)
        merged  = oc.merge(base_cfg, cli_cfg)

        from libgs.utils.config import to_structured
        cfg, args = to_structured(merged, Config)

        hicom_pipeline = PipelineCls(cfg)
        trainer = hicom_pipeline.trainer
        module  = hicom_pipeline.module
        _state["trainer"] = trainer
        _state["module"]  = module

        # Install hooks
        slot.install_hooks(pipeline, trainer=trainer, module=module)

        # Wrap training_loop to measure per-frame time
        orig_loop = trainer.training_loop

        def _timed_training_loop():
            for current_frame in range(
                trainer.datamodule.current_frame + 1,
                trainer.datamodule.num_frames,
            ):
                trainer.datamodule.current_frame = current_frame
                pipeline.begin_frame()
                with collector.frame_timer():
                    trainer.frame_training_loop()
                per_frame_delta_bytes.append(
                    pipeline.stream_stats()["frame_active_bytes"]
                )

        trainer.training_loop = _timed_training_loop

        # Run training
        hicom_pipeline.run()
        slot.uninstall_hooks()

    except Exception as e:
        print(f"  HiCoM training raised: {e}")
        raise

    # ── FPS measurement ───────────────────────────────────────────────────────
    print("  Measuring FPS …")
    fps = 0.0
    gaussians = None
    try:
        module  = _state["module"]
        gaussians = slot.get_gaussians() if module else None
        if gaussians is not None:
            slot.install_hooks(pipeline, trainer=_state["trainer"], module=module)

            # Use validation dataloader cameras
            val_loaders = module.datamodule.val_dataloader()
            if not isinstance(val_loaders, (list, tuple)):
                val_loaders = [val_loaders]
            val_cams = list(val_loaders[0])[:ds_cfg.fps_measure + ds_cfg.fps_warmup]

            if val_cams:
                def _render_fn(cam, g, pipe_, bg):
                    return module.forward(cam.to(module.device))

                fps = collector.measure_fps(
                    pipeline,
                    _render_fn,
                    val_cams, None, None,
                )
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
    stats = pipeline.stream_stats()

    result = {
        "mode":             mode,
        "baseline":         "HiCoM",
        "dataset":          ds_cfg.name,
        "n_frames_trained": ds_cfg.n_frames_ablation,
        "avg_train_time_s": collector.avg_frame_time(),
        "render_fps":       fps,
        **quality,
        "total_storage_mb": total_storage_mb,
        "delta_storage_mb": delta_storage_mb,
        **stats,
        "incremental":      True,
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
    args = parser.parse_args()
    ds_cfg = load_dataset_config(args.dataset, args.dataset_config)
    run_one_mode(args.mode, ds_cfg)


if __name__ == "__main__":
    main()
