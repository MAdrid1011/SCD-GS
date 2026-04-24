#!/usr/bin/env python3
"""GPU-side motivation trace for 3DGStream.

Runs inside ``3dgstream-cu121`` conda env.  Path resolution follows the same
``simulator.datasets.load_dataset_config`` pattern as
``simulator/ablation/run_3dgstream.py`` so the trace and the ASIC ablation
always read from the same on-disk layout.

Usage::

    conda run -n 3dgstream-cu121 python scripts/trace_3dgstream.py \\
        --dataset flame_steak --n_frames 5

Output::

    results/motivation_trace/3DGStream/<dataset>.json
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baselines" / "3DGStream"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

from scripts.trace_for_motivation import MotivationTracer, patch_3dgstream


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="flame_steak")
    p.add_argument("--n_frames", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--stable-threshold",
        type=float,
        default=1e-4,
        help="Absolute τ_stable for the C3 five-frame stability window; "
        "Gaussians with mean ||Δθ||² ≤ τ over the window are counted as stable.",
    )
    p.add_argument(
        "--stable-window",
        type=int,
        default=5,
        help="Window size W (in frames) for the stability snapshot deque.",
    )
    p.add_argument(
        "--emit-stability-cdf",
        action="store_true",
        help="Dump a 50-point S^(W) CDF per frame (used by the Step-6 pilot).",
    )
    return p.parse_args()


def _sorted_frame_dirs(video_path: Path) -> list[str]:
    pat = re.compile(r"frame(\d+)")
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
        / "3DGStream"
        / f"{args.dataset}.json"
    )

    os.chdir(BASELINE_DIR)

    from argparse import ArgumentParser as BaseArgParser

    from arguments import ModelParams, OptimizationParams, PipelineParams  # type: ignore
    from utils.general_utils import safe_state  # type: ignore
    from train_frames import train_one_frame  # type: ignore
    import torch._dynamo  # type: ignore
    # safe_state() swaps sys.stdout for a wrapper lacking .isatty(), which
    # breaks torch._dynamo's debug-logging.  Fall back to eager execution of
    # any @torch.compile'd function (e.g. 3DGStream's rotation_compose) so
    # dynamo does not get invoked at all.  Numerically identical to the
    # compiled path, just a bit slower — fine for motivation trace.
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

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

    video_path = ds_cfg.resolved("source_path_3dgstream")
    init_model = ds_cfg.resolved("init_model_3dgstream")
    base_args = [
        "--source_path", "",
        "--model_path", str(init_model),
        "--video_path", str(video_path),
        "--output_path", "",
        "--quiet",
    ]
    train_args = parser.parse_args(base_args)
    # Apply per-dataset overrides (e.g. iterations)
    for k, v in ds_cfg.model_args("3dgstream").items():
        if hasattr(train_args, k):
            setattr(train_args, k, v)
    # Ensure we save at S1 end so next frame can load
    s1_end = train_args.iterations
    if s1_end not in train_args.save_iterations:
        train_args.save_iterations = sorted(
            set(list(train_args.save_iterations) + [s1_end])
        )

    safe_state(True)

    all_frames = _sorted_frame_dirs(video_path)
    frames = all_frames[1 : 1 + args.n_frames]  # skip colmap_0 / frame000000

    out_root = REPO_ROOT / "results" / "motivation_trace" / "3DGStream" / "work"
    out_root.mkdir(parents=True, exist_ok=True)

    tracer = MotivationTracer(
        baseline="3DGStream",
        scene=args.dataset,
        out_path=out_path,
        stable_threshold=args.stable_threshold,
        stable_window=args.stable_window,
        emit_stability_cdf=args.emit_stability_cdf,
    )

    model_path = str(init_model)
    # ``load_iteration`` must be the iter the *previous* frame's save_iterations
    # wrote to (frame 0 loads from init's first_load_iteration, frames N>0 load
    # from frame N-1's S1 save at ``train_args.iterations``).  Without this
    # chain, ``Scene.__init__`` falls into ``create_from_pcd`` and rebuilds
    # the model from the source COLMAP point cloud (~few thousand points)
    # instead of carrying forward the trained ~280K Gaussians.  See
    # ``simulator/ablation/run_3dgstream.py`` for the same pattern.
    prev_save_iter = train_args.first_load_iteration
    with patch_3dgstream(tracer):
        for i, frame in enumerate(frames):
            train_args.source_path = str(video_path / frame)
            train_args.output_path = str(out_root / frame)
            train_args.model_path = model_path
            train_args.load_iteration = (
                train_args.first_load_iteration if i == 0 else prev_save_iter
            )
            Path(train_args.output_path).mkdir(parents=True, exist_ok=True)
            train_one_frame(lp, op, pp, train_args)
            prev_save_iter = train_args.iterations
            model_path = train_args.output_path

    tracer.close()
    print(f"[trace_3dgstream] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
