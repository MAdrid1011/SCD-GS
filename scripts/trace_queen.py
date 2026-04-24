#!/usr/bin/env python3
"""GPU-side motivation trace for QUEEN.

Runs inside ``queen-cu121`` conda env.  Closely mirrors
``simulator/ablation/run_queen.py::_build_queen_args`` — QUEEN has no
per-scene yaml in ``baselines/QUEEN/configs/`` for flame_steak; instead we
use ``dynerf.yaml`` as the ParamGroup base and inject source/model paths via
``ds_cfg.resolved(...)``.  Init-model PLY symlinking follows the same
``searchForMaxIteration`` logic as the ablation runner.

Frame cap: QUEEN's training loop runs for ``max_frames`` frames.  We set
``max_frames = n_frames`` (+1 when an init model is present, which reserves
frame 1 as the loaded initial model).

Usage::

    conda run -n queen-cu121 python scripts/trace_queen.py \\
        --dataset flame_steak --n_frames 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baselines" / "QUEEN"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

from scripts.trace_for_motivation import MotivationTracer, patch_queen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="flame_steak")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stable-threshold", type=float, default=1e-4)
    p.add_argument("--stable-window", type=int, default=5)
    p.add_argument("--emit-stability-cdf", action="store_true")
    return p.parse_args()


def _load_queen_config() -> dict:
    import yaml
    from collections import defaultdict

    cfg_path = BASELINE_DIR / "configs" / "dynerf.yaml"
    raw = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    return defaultdict(lambda: {}, raw)


def _build_queen_args(ds_cfg, n_frames: int, out_root: Path):
    from arguments import (  # type: ignore
        ModelParams,
        OptimizationParams,
        OptimizationParamsInitial,
        OptimizationParamsRest,
        PipelineParams,
        QuantizeParams,
    )

    cfg = _load_queen_config()
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser, cfg["model_params"])
    opi = OptimizationParamsInitial(parser, cfg["opt_params_initial"])
    opr = OptimizationParamsRest(parser, cfg["opt_params_rest"])
    pp = PipelineParams(parser, cfg["pipe_params"])
    qp = QuantizeParams(parser, cfg["quantize_params"])

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_frame_idx", type=int, default=1)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_xyz_legacy", action="store_true")

    src = str(ds_cfg.resolved("source_path_queen"))
    out = str(out_root)
    os.makedirs(out, exist_ok=True)

    base_args = ["--source_path", src, "--model_path", out, "--quiet"]
    args = parser.parse_args(base_args)

    for k, v in ds_cfg.model_args("queen").items():
        if hasattr(args, k):
            setattr(args, k, v)
    args.max_frames = n_frames
    args.model_path = out

    init_model_path = ds_cfg.resolved("init_model_queen")
    if init_model_path.exists():
        from utils.system_utils import searchForMaxIteration  # type: ignore

        loaded_iter = searchForMaxIteration(str(init_model_path / "point_cloud"))
        pc_link = Path(out) / "point_cloud" / f"iteration_{loaded_iter}"
        pc_link.mkdir(parents=True, exist_ok=True)
        ply_dst = pc_link / "point_cloud.ply"
        ply_src = (
            init_model_path / "point_cloud" / f"iteration_{loaded_iter}" / "point_cloud.ply"
        )
        if not ply_dst.exists() and ply_src.exists():
            shutil.copy2(str(ply_src), str(ply_dst))
        args.load_iteration = loaded_iter
        args.start_frame_idx = 2
        args.max_frames = n_frames + 1
        args.test_iterations = []
        args.save_iterations = []
        print(
            f"[trace_queen] init model {init_model_path} (iter {loaded_iter}), "
            f"start_frame_idx=2, max_frames={args.max_frames}"
        )

    op = OptimizationParams(opi.extract(args), opr.extract(args))
    return args, lp, op, pp, qp


def main() -> int:
    args = parse_args()

    from simulator.datasets import load_dataset_config

    ds_cfg = load_dataset_config(args.dataset)
    out_path = args.out or (
        REPO_ROOT / "results" / "motivation_trace" / "QUEEN" / f"{args.dataset}.json"
    )

    out_root = REPO_ROOT / "results" / "motivation_trace" / "QUEEN" / "work"
    out_root.mkdir(parents=True, exist_ok=True)

    qargs, lp, op, pp, qp = _build_queen_args(ds_cfg, args.n_frames, out_root)

    from train import training as queen_training  # type: ignore

    tracer = MotivationTracer(
        baseline="QUEEN",
        scene=args.dataset,
        out_path=out_path,
        stable_threshold=args.stable_threshold,
        stable_window=args.stable_window,
        emit_stability_cdf=args.emit_stability_cdf,
    )
    with patch_queen(tracer):
        queen_training(
            lp.extract(qargs),
            op,
            pp.extract(qargs),
            qp.extract(qargs),
            qargs.test_iterations,
            qargs.save_iterations,
            qargs.checkpoint_iterations,
            qargs.start_checkpoint,
            qargs.debug_from,
            qargs,
        )

    tracer.close()
    print(f"[trace_queen] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
