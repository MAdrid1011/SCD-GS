#!/usr/bin/env python3
"""GPU-side motivation trace for HiCoM.

Runs inside ``hicom-cu121`` conda env.  Mirrors the same config-override and
checkpoint-load pattern used by ``simulator/ablation/run_hicom.py`` so paths,
frame boundaries, and checkpoint loading all match.

Frame cap: HiCoM's config has no ``num_incr_frames`` key — the total is set
by the dataset.  We cap it by monkey-patching ``trainer.training_loop`` the
same way ``scripts/train_hicom_init.py`` does (stop after N incremental
frames).

Usage::

    conda run -n hicom-cu121 python scripts/trace_hicom.py \\
        --dataset flame_steak --n_frames 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baselines" / "HiCoM"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

from scripts.trace_for_motivation import MotivationTracer, patch_hicom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="flame_steak")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stable-threshold", type=float, default=1e-4)
    p.add_argument("--stable-window", type=int, default=5)
    p.add_argument("--emit-stability-cdf", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from simulator.datasets import load_dataset_config

    ds_cfg = load_dataset_config(args.dataset)
    out_path = args.out or (
        REPO_ROOT
        / "results"
        / "motivation_trace"
        / "HiCoM"
        / f"{args.dataset}.json"
    )

    from libgs.pipeline.loader import load_pipeline  # type: ignore
    from libgs.utils.config import to_structured  # type: ignore
    from omegaconf import OmegaConf as oc  # type: ignore

    extra = ds_cfg.model_args("hicom")
    num_init_steps = int(extra.get("trainer.num_init_steps", 15000))
    num_incr_steps = int(extra.get("trainer.num_incr_steps", 200))

    out_dir = REPO_ROOT / "results" / "motivation_trace" / "HiCoM" / "work"
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "data.root": str(ds_cfg.resolved("source_path_hicom")),
        "output_dir": str(out_dir),
        "experiment_name": "motivation_trace",
        "trainer.num_incr_steps": num_incr_steps,
        "trainer.num_init_steps": num_init_steps,
        "trainer.saving_ckpt_steps": [],
        "trainer.testing_steps": [],
    }
    init_ckpt = ds_cfg.resolved("init_model_hicom")
    if init_ckpt.exists() and init_ckpt.is_file():
        overrides["ckpt_path"] = str(init_ckpt)

    Config, PipelineCls = load_pipeline("hicom")

    dotlist = {k: v for k, v in overrides.items() if not isinstance(v, list)}
    struct = {k: v for k, v in overrides.items() if isinstance(v, list)}
    default_cfg = BASELINE_DIR / "config" / "dynerf.yaml"
    base_cfg = oc.load(str(default_cfg)) if default_cfg.exists() else oc.create({})
    merged = oc.merge(base_cfg, oc.from_dotlist([f"{k}={v}" for k, v in dotlist.items()]))
    for dotted, val in struct.items():
        oc.update(merged, dotted, val, merge=False)
    cfg, _ = to_structured(merged, Config)

    hicom_pipeline = PipelineCls(cfg)
    trainer = hicom_pipeline.trainer

    n_frames_to_run = int(args.n_frames)

    def _capped_training_loop():
        """Run at most ``n_frames_to_run`` incremental frames, then stop.

        NOTE: ``trainer.fit()`` calls ``trainer.setup()`` which reloads the
        checkpoint and sets ``global_step`` to the checkpoint step (e.g.
        15000).  Without resetting it here, the first
        ``frame_training_loop`` would see ``range(15001, 201)`` and skip all
        200 iters.
        """
        trainer.global_step = 0
        dm = trainer.datamodule
        start = dm.current_frame + 1
        end = min(dm.num_frames, start + n_frames_to_run)
        for f in range(start, end):
            dm.current_frame = f
            trainer.frame_training_loop()

    trainer.training_loop = _capped_training_loop  # type: ignore[assignment]

    tracer = MotivationTracer(
        baseline="HiCoM",
        scene=args.dataset,
        out_path=out_path,
        stable_threshold=args.stable_threshold,
        stable_window=args.stable_window,
        emit_stability_cdf=args.emit_stability_cdf,
    )
    with patch_hicom(tracer):
        hicom_pipeline.run()

    tracer.close()
    print(f"[trace_hicom] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
