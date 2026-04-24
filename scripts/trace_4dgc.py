#!/usr/bin/env python3
"""GPU-side motivation trace for 4DGC.

Runs inside ``4dgc-cu121`` conda env.  Mirrors ``simulator/ablation/run_4dgc.py``
for path resolution and frame-directory layout (``colmap_<k>``), plus the
Motion-Grid warmup weight at ``init_mem_4dgc``.

Usage::

    conda run -n 4dgc-cu121 python scripts/trace_4dgc.py \\
        --dataset flame_steak --n_frames 5
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baselines" / "4DGC"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

from scripts.trace_for_motivation import MotivationTracer, patch_4dgc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="flame_steak")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stable-threshold", type=float, default=1e-4)
    p.add_argument("--stable-window", type=int, default=5)
    p.add_argument("--emit-stability-cdf", action="store_true")
    return p.parse_args()


def _sorted_frame_dirs(video_path: Path) -> list[str]:
    pat = re.compile(r"colmap_(\d+)")
    return sorted(
        (f for f in os.listdir(video_path) if pat.match(f)),
        key=lambda x: int(pat.match(x).group(1)),
    )


def main() -> int:
    args = parse_args()

    from simulator.datasets import load_dataset_config

    ds_cfg = load_dataset_config(args.dataset)
    out_path = args.out or (
        REPO_ROOT
        / "results"
        / "motivation_trace"
        / "4DGC"
        / f"{args.dataset}.json"
    )

    os.chdir(BASELINE_DIR)

    from argparse import ArgumentParser as BaseArgParser

    from arguments import ModelParams, OptimizationParams, PipelineParams  # type: ignore
    from train_frames import train_one_frame  # type: ignore

    parser = BaseArgParser(add_help=False)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=args.n_frames)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    video_path = ds_cfg.resolved("source_path_4dgc")
    init_model = ds_cfg.resolved("init_model_4dgc")
    init_mem = ds_cfg.resolved("init_mem_4dgc")
    base_args = [
        "--source_path", "",
        "--model_path", str(init_model),
        "--video_path", str(video_path),
        "--output_path", "",
        "--mem_path", str(init_mem),
        "--quiet",
    ]
    train_args = parser.parse_args(base_args)
    for k, v in ds_cfg.model_args("4dgc").items():
        if hasattr(train_args, k):
            setattr(train_args, k, v)
    s1_end = train_args.iterations
    if s1_end not in train_args.save_iterations:
        train_args.save_iterations = sorted(
            set(list(train_args.save_iterations) + [s1_end])
        )

    all_frames = _sorted_frame_dirs(video_path)
    frame_start = getattr(train_args, "frame_start", 1)
    frames = all_frames[frame_start : frame_start + args.n_frames]

    out_root = REPO_ROOT / "results" / "motivation_trace" / "4DGC" / "work"
    out_root.mkdir(parents=True, exist_ok=True)

    tracer = MotivationTracer(
        baseline="4DGC",
        scene=args.dataset,
        out_path=out_path,
        stable_threshold=args.stable_threshold,
        stable_window=args.stable_window,
        emit_stability_cdf=args.emit_stability_cdf,
    )

    model_path = str(init_model)
    prev_save_iter = train_args.first_load_iteration

    with patch_4dgc(tracer):
        for i, frame in enumerate(frames):
            train_args.source_path = str(video_path / frame)
            train_args.output_path = str(out_root / frame)
            train_args.model_path = model_path
            train_args.load_iteration = (
                train_args.first_load_iteration if i == 0 else prev_save_iter
            )
            Path(train_args.output_path).mkdir(parents=True, exist_ok=True)

            _save_iters_snapshot = list(train_args.save_iterations)
            if train_args.iterations not in train_args.save_iterations:
                train_args.save_iterations = sorted(
                    set(list(train_args.save_iterations) + [train_args.iterations])
                )
            train_one_frame(lp, op, pp, train_args)
            train_args.save_iterations = _save_iters_snapshot

            prev_save_iter = train_args.iterations
            model_path = train_args.output_path

    tracer.close()
    print(f"[trace_4dgc] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
