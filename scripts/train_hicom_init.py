"""One-shot trainer: produce HiCoM's frame-0 checkpoint for ablation.

HiCoM officially trains frame 0 from scratch (15 000 iters) every run.  For
GLASS-ASIC ablation we want mode 0/1/2/3 to differ **only** in the incremental
frames, so we cache the frame-0 result once and reload it as the ablation
starting point.

This script:
  1.  Constructs HiCoM's Pipeline with saving_ckpt_steps=[num_init_steps]
      so the frame-0 ckpt is written the moment the 15 000-th iteration ends.
  2.  Monkey-patches ``trainer.training_loop`` to break out after frame 0
      (we don't want to spend time on incremental frames here).
  3.  Runs exactly one training pass, then copies the resulting
      ``ckpt-0-<N>.pth`` to ``data/<dataset>/hicom_init/ckpt-0-<N>.pth``.

Usage
-----
Run inside the ``hicom-cu121`` conda env::

    python scripts/train_hicom_init.py --dataset flame_steak

Re-runs are idempotent: if the destination ckpt already exists, the script
exits immediately (pass ``--force`` to overwrite).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baselines" / "HiCoM"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

from simulator.datasets import load_dataset_config  # noqa: E402


def train_frame0(dataset_name: str, force: bool) -> Path:
    ds_cfg = load_dataset_config(dataset_name)

    dest_dir = REPO_ROOT / "data" / dataset_name / "hicom_init"
    dest_dir.mkdir(parents=True, exist_ok=True)

    extra = ds_cfg.model_args("hicom")
    num_init_steps = int(extra.get("trainer.num_init_steps", 15000))
    dest_ckpt = dest_dir / f"ckpt-0-{num_init_steps}.pth"

    if dest_ckpt.exists() and not force:
        print(f"[skip] frame-0 ckpt already exists → {dest_ckpt}")
        print("       pass --force to retrain.")
        return dest_ckpt

    tmp_out_dir = REPO_ROOT / "results" / "hicom_init_tmp" / dataset_name
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured
    from omegaconf import OmegaConf as oc

    Config, PipelineCls = load_pipeline("hicom")

    default_cfg = BASELINE_DIR / "config" / "dynerf.yaml"
    overrides = {
        "data.root":                   str(ds_cfg.resolved("source_path_hicom")),
        "output_dir":                  str(tmp_out_dir),
        "experiment_name":             "frame0_init",
        "trainer.num_init_steps":      num_init_steps,
        "trainer.num_incr_steps":      1,
        "trainer.saving_ckpt_steps":   [num_init_steps],
        "trainer.saving_ckpt_every_n_frames": 1,
        "trainer.testing_steps":       [],
    }
    cli_cfg = oc.from_dotlist([f"{k}={v}" for k, v in overrides.items()])
    base_cfg = oc.load(str(default_cfg))
    merged = oc.merge(base_cfg, cli_cfg)
    cfg, _ = to_structured(merged, Config)

    print(f"[train_hicom_init] dataset = {dataset_name}")
    print(f"[train_hicom_init] data.root = {cfg.data.root}")
    print(f"[train_hicom_init] num_init_steps = {num_init_steps}")
    print(f"[train_hicom_init] tmp output_dir = {tmp_out_dir}")

    pipeline = PipelineCls(cfg)

    # Cap training at the end of frame 0 — the checkpoint at step=num_init_steps
    # is written inside frame_training_loop() before we break here.
    orig_training_loop = pipeline.trainer.training_loop

    def _one_frame_only():
        dm = pipeline.trainer.datamodule
        dm.current_frame = 0
        pipeline.trainer.frame_training_loop()
        print("[train_hicom_init] frame 0 finished; stopping.")

    pipeline.trainer.training_loop = _one_frame_only

    t0 = time.time()
    pipeline.run()
    dt = time.time() - t0
    print(f"[train_hicom_init] frame-0 training took {dt:.1f} s")

    produced = pipeline.trainer.output_dir / f"ckpt-0-{num_init_steps}.pth"
    if not produced.exists():
        raise RuntimeError(
            f"expected checkpoint not found: {produced}. Check "
            f"trainer.saving_ckpt_steps wiring."
        )

    shutil.copy2(produced, dest_ckpt)
    size_mb = dest_ckpt.stat().st_size / (1024 * 1024)
    print(f"[train_hicom_init] saved → {dest_ckpt}  ({size_mb:.1f} MB)")

    return dest_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Train HiCoM frame 0 once and cache the checkpoint."
    )
    parser.add_argument("--dataset", type=str, default="flame_steak")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite existing cached ckpt.")
    args = parser.parse_args()

    ckpt = train_frame0(args.dataset, args.force)
    print(f"\nDone. Use it with:\n"
          f"  ds_cfg.init_model_hicom = \"{ckpt.relative_to(REPO_ROOT)}\"")


if __name__ == "__main__":
    main()
