"""
DatasetConfig — per-dataset paths and training parameters.
DATASET_REGISTRY — built-in registry of known datasets.
load_dataset_config() — load by name or YAML override.

Adding a new dataset
--------------------
Option A (built-in): add an entry to DATASET_REGISTRY below.
Option B (YAML file): pass --dataset-config /path/to/my.yaml to any runner.
  The YAML may override any DatasetConfig field, including extra_args.

YAML example (meet_room.yaml)
------------------------------
name: meet_room
source_path_3dgstream: data/meet_room/meet_room
source_path_queen:      data/meet_room/meet_room
source_path_hicom:      data/meet_room/hicom_meet_room_scene
source_path_swift4d:    data/meet_room/meet_room
init_model_3dgstream:   data/meet_room/3dgstream_meet_room_scene
n_frames_ablation: 30
extra_args:
  3dgstream:
    iterations: 200
    iterations_s2: 50
  queen:
    epochs: 5
    max_frames: 30
  hicom:
    trainer.num_incr_steps: 200
  swift4d:
    configs: arguments/dynerf/meet_room.py
    test_iterations: "1000"
    save_iterations: "1000"
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Repo root — `simulator/datasets.py` → parent = simulator/, then parent = repo root.
# Derived at import time so the module is portable across checkouts.
# Override at runtime with: `os.environ["GLASS_ROOT"] = "/path"` before import,
# or by assigning to `simulator.datasets.GLASS_DIR` after import.
import os
GLASS_DIR = Path(os.environ.get("GLASS_ROOT", Path(__file__).resolve().parents[1]))


@dataclass
class DatasetConfig:
    """Holds all dataset-specific paths and per-model training overrides.

    Per-model source paths may be absolute paths or paths relative to
    GLASS_DIR. Use .resolved(field_name) to obtain the absolute Path.
    """

    name: str = "unnamed"

    # ── Data source paths ────────────────────────────────────────────────────
    # Each model expects a slightly different directory layout; separate fields
    # allow each model to point to its own correctly-formatted copy.
    source_path_3dgstream: str = ""   # COLMAP frame* sub-directory layout
    source_path_queen:     str = ""   # cam*/ multi-camera layout
    source_path_hicom:     str = ""   # libgs-compatible layout
    source_path_swift4d:   str = ""   # DyNeRF / COLMAP layout

    # ── Pre-trained frame-0 model roots ──────────────────────────────────────
    # 3DGStream: the output directory written by a prior train.py run; the
    #   runner will resolve the latest point_cloud inside it.
    # HiCoM: libgs checkpoint directory; empty string → train from scratch.
    # QUEEN: directory containing point_cloud/iteration_<N>/point_cloud.ply
    #   written by a prior QUEEN train.py run for frame 1 only (full epochs).
    #   When set, run_queen.py loads this PLY and starts incremental training
    #   from frame 2, skipping the expensive frame-1 initialisation.
    init_model_3dgstream: str = ""
    init_model_hicom:     str = ""
    init_model_queen:     str = ""

    # ── Ablation training scope ───────────────────────────────────────────────
    n_frames_ablation: int = 30
    """Number of incremental frames to train per ablation run.
    Frame 0 (initial reconstruction) is excluded from per-frame timing."""

    # ── FPS measurement ───────────────────────────────────────────────────────
    fps_warmup:  int = 5
    fps_measure: int = 50

    # ── Per-model extra arguments ─────────────────────────────────────────────
    # Keys: "3dgstream", "queen", "hicom", "swift4d"
    # Values: dict of argparse-style kwarg overrides forwarded to each baseline.
    # Boolean flags are expressed as True/False (converted to store_true args).
    extra_args: dict[str, dict[str, Any]] = field(default_factory=dict)

    def resolved(self, key: str) -> Path:
        """Return the absolute Path for any string path field."""
        val = getattr(self, key, "")
        if not val:
            return Path("")
        p = Path(val)
        return p if p.is_absolute() else GLASS_DIR / p

    def model_args(self, model: str) -> dict[str, Any]:
        """Return per-model extra args dict (safe default = empty dict)."""
        return self.extra_args.get(model, {})


# ── Built-in registry ─────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "flame_steak": DatasetConfig(
        name="flame_steak",
        # 3DGStream uses a per-frame directory layout (frame000001/, frame000002/, …)
        # with COLMAP calibration in the top-level distorted/ folder.
        source_path_3dgstream="data/3dgstream_flame_steak_scene",
        source_path_queen    ="data/flame_steak/flame_steak",
        source_path_hicom    ="data/hicom_flame_steak_scene",
        source_path_swift4d  ="data/flame_steak/flame_steak",
        # Pre-trained init models — all paths relative to GLASS_DIR.
        init_model_3dgstream ="data/flame_steak/3dgstream_init",
        init_model_hicom     ="",   # HiCoM trains frame-0 from scratch
        # QUEEN Frame-1 PLY (188 K Gaussians, 9 995-iter checkpoint).
        # run_queen.py loads this and starts incremental training from frame 2.
        init_model_queen     ="data/flame_steak/queen_init",
        n_frames_ablation    =30,
        fps_warmup  =5,
        fps_measure =50,
        extra_args={
            "3dgstream": {
                # Match the settings used in the benchmark full_run.
                "sh_degree":            1,    # init model was trained with SH deg 1
                "iterations":           150,  # NTC (Stage-1) iterations per frame
                "iterations_s2":        100,  # Stage-2 Gaussian refinement iters
                "first_load_iteration": 15000,  # load iter from init model on frame 0
                "frame_start":          1,    # first incremental frame index
                "batch_size":           1,
                # Relative paths resolve from baselines/3DGStream (cwd at run time).
                "ntc_conf_path":   "configs/cache/cache_F_4.json",
                "ntc_path":        "ntc/flame_steak_ntc_params_F_4.pth",
                # Disable in-training eval to avoid 3DGStream's training_report
                # bug (last_test_psnr stays float when test cameras are absent).
                # Quality is measured separately by MetricsCollector.
                "test_iterations": [],
            },
            "queen": {
                # Frame 1 is loaded from init_model_queen (188 K Gaussian PLY).
                # run_queen.py sets start_frame_idx=2 and max_frames=n+1 so the
                # training loop skips frame 1 and only runs incremental frames.
                # epochs_rest=10 (from dynerf.yaml) gives 10*20=200 iters/frame.
                "max_frames": 30,
                "start_idx":  0,
                # White background matches N3DV default
                "white_background": False,
                # Disable depth initialisation: MiDaS weights not present and
                # lambda_depth=0.0 in dynerf.yaml so depth supervision is off anyway.
                "depth_init": False,
            },
            "hicom": {
                "trainer.num_incr_steps": 200,
                "trainer.num_init_steps": 15000,
            },
            "swift4d": {
                "configs":         "arguments/dynerf/flame_steak.py",
                "test_iterations": "1000 3000 5000 7000 9000 11000 13001",
                "save_iterations": "13001",
                "expname":         "dynerf/flame_steak",
            },
        },
    ),

    # ── Placeholder entries — fill in paths when data is available ────────────
    # "meet_room": DatasetConfig(
    #     name="meet_room",
    #     source_path_3dgstream="data/meet_room/meet_room",
    #     source_path_queen    ="data/meet_room/meet_room",
    #     source_path_hicom    ="data/hicom_meet_room_scene",
    #     source_path_swift4d  ="data/meet_room/meet_room",
    #     init_model_3dgstream ="data/meet_room/3dgstream_scene",
    #     n_frames_ablation    =30,
    #     extra_args={ ... },
    # ),
    # "coffee_martini": DatasetConfig( ... ),
}


def load_dataset_config(
    name: str,
    override_yaml: str | None = None,
) -> DatasetConfig:
    """Load a DatasetConfig by registry name, then apply optional YAML overrides.

    Parameters
    ----------
    name:
        Key in DATASET_REGISTRY, e.g. "flame_steak".
        If not found, a blank DatasetConfig(name=name) is returned and the
        caller must supply all required paths via override_yaml.
    override_yaml:
        Path to a YAML file whose top-level keys match DatasetConfig fields.
        Missing keys are left at their current values.
        The 'extra_args' key is *deep-merged* (not replaced wholesale).

    Returns
    -------
    DatasetConfig
        A fully populated (or partially filled) configuration object.
    """
    if name in DATASET_REGISTRY:
        cfg = copy.deepcopy(DATASET_REGISTRY[name])
    else:
        cfg = DatasetConfig(name=name)

    if override_yaml:
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for --dataset-config support. "
                "Install it with: pip install pyyaml"
            )
        with open(override_yaml) as fh:
            overrides: dict = yaml.safe_load(fh) or {}

        for key, val in overrides.items():
            if key == "extra_args":
                # Deep-merge: only update the sub-keys that are present
                for model_key, model_val in val.items():
                    if model_key in cfg.extra_args:
                        cfg.extra_args[model_key].update(model_val)
                    else:
                        cfg.extra_args[model_key] = model_val
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
            else:
                import warnings
                warnings.warn(
                    f"YAML key '{key}' is not a DatasetConfig field; ignoring.",
                    stacklevel=2,
                )

    return cfg
