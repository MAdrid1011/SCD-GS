# GPU Baseline Benchmarks

This directory houses the **GPU-side** head-to-head harness for the four shipped baselines (QUEEN, 3DGStream, HiCoM, Swift4D). It sits alongside [`simulator/ablation/`](../simulator/ablation/) — the two together produce the two columns of numbers compared in the paper:

- `benchmarks/` → GPU wall-clock + quality + FPS + storage (what the baseline actually does)
- `simulator/ablation/` → ASIC-projected cycle counts + the same quality/FPS/storage passed through the GLASS pipeline

---

## Layout

```text
benchmarks/
├── run_all_serial.sh            master runner — 7 steps, one baseline at a time
├── configs/
│   ├── bench_config.yaml        single source of truth for per-baseline paths
│   ├── 3dgstream_flame_steak.json       smoke (5-frame) training config
│   └── 3dgstream_flame_steak_full.json  full (300-frame) training config
└── scripts/
    ├── bench_3dgstream.sh       thin wrapper around 3DGStream/train_frames.py
    ├── bench_3dgstream.py       Python driver with per-frame timing + FPS + quality
    ├── eval_3dgstream.py        post-training PSNR / SSIM / LPIPS + FPS evaluator
    ├── fps_3dgstream.py         standalone FPS measurement (minimal)
    ├── bench_queen.py           QUEEN FPS + quality from prior-run subset
    └── collect_metrics.py       aggregate all 4 baselines into one table
```

---

## Path convention

**Every path you see in this tree is repo-relative.** At runtime:

- Python scripts resolve `REPO_ROOT = Path(__file__).resolve().parents[2]` once at the top and express every constant as `REPO_ROOT / "data" / …`.
- Shell scripts derive `$GLASS` the same way (via `BASH_SOURCE`) and pass it into embedded Python via env vars.
- JSON configs contain repo-relative paths; the caller (`bench_3dgstream.{py,sh}`) expands them against the resolved repo root before handing them to the baseline's own CLI.

If you move the repo, **nothing needs to change**.

Override the detected repo root by exporting `GLASS=/your/path` before running any script.

---

## Assets (required; not in git)

`.gitignore` excludes `data/`, so the **assets themselves live on your disk, not in the repo**. The scripts expect the following layout:

```text
data/
├── flame_steak/flame_steak/                        raw N3DV frames (you provide)
├── 3dgstream_flame_steak_scene/                    per-frame COLMAP output, 300 frames
├── hicom_flame_steak_scene/                        HiCoM-preprocessed variant
└── assets/
    ├── 3DGStream/flame_steak_suite/
    │   ├── flame_steak_init/                       static 3DGS init model (24 MB)
    │   │   └── point_cloud/iteration_15000/
    │   │       └── point_cloud.ply
    │   └── frame000000/                            COLMAP output for frame 0 (79 MB)
    │       ├── distorted/
    │       ├── images/
    │       └── sparse/
    └── QUEEN/flame_steak_prior/                    minimal prior-run (~33 MB) — see below
        ├── avg_metrics.json
        ├── cameras.json
        ├── cfg_args
        ├── per_frame.json
        ├── results.json
        ├── training_metrics.json
        └── frames/<LAST>/point_cloud/iteration_<K>/point_cloud.ply
```

### How to populate each asset

| Asset | Origin | How to (re)build |
|-------|--------|------------------|
| `data/flame_steak/flame_steak/` | N3DV dataset | Download from [Neural 3D Video Synthesis](https://github.com/facebookresearch/Neural_3D_Video) and unpack per-cam videos. |
| `data/3dgstream_flame_steak_scene/` | per-frame COLMAP | Run 3DGStream's own preprocessing on the raw N3DV frames (see [`baselines/3DGStream/README.md`](../baselines/3DGStream/README.md)). Output layout: `frame000000/`, `frame000001/`, … each with `images/` + `sparse/` + `distorted/`. |
| `data/hicom_flame_steak_scene/` | HiCoM-preprocessed variant | Run HiCoM's preprocessing (see [`baselines/HiCoM/README.md`](../baselines/HiCoM/README.md)). |
| `data/assets/3DGStream/flame_steak_suite/flame_steak_init/` | static 3DGS init | Train vanilla 3DGS on `frame000000` for 15 000 iterations. Reference command: `python baselines/3DGStream/train.py -s <frame000000> --iterations 15000 --model_path data/assets/3DGStream/flame_steak_suite/flame_steak_init`. |
| `data/assets/3DGStream/flame_steak_suite/frame000000/` | COLMAP frame 0 | Same as `data/3dgstream_flame_steak_scene/frame000000/` — either a copy or a symlink. |
| `data/assets/QUEEN/flame_steak_prior/` | completed QUEEN run, minimal subset | Run `simulator/ablation/run_queen.py --n-frames 300 --mode 0` once; then copy the 4 JSON files + `cfg_args` + **only the last frame's** `point_cloud.ply` into this layout. `bench_queen.py` will read the JSONs (for quality/timing) and the PLY (for FPS). |

Total disk budget for the assets: **~110 MB**, plus the preprocessed scene directories (~8 GB for 3DGStream's 300-frame COLMAP output, ~55 MB for HiCoM).

### Why a "prior-run subset" for QUEEN?

QUEEN is a bundled-training baseline — its training doesn't expose a per-frame optimizer.step() that we can hook, so `bench_queen.py` doesn't re-train but instead sources quality / timing numbers from a completed run's JSONs and measures FPS from the last frame's saved `point_cloud.ply`. Keeping only the essential ~33 MB (rather than the full 7.5 GB output tree) keeps the reference portable without sacrificing any number that `collect_metrics.py` needs.

---

## Running

```bash
# Full run (≈ hours, needs all four conda envs installed):
bash benchmarks/run_all_serial.sh

# Smoke (3DGStream only, 10 frames):
bash benchmarks/scripts/bench_3dgstream.sh 10

# QUEEN — no retraining, just reads the prior-run subset + measures FPS:
conda run -n queen-cu121 python benchmarks/scripts/bench_queen.py

# Aggregate an existing result set into results/metric_table.json:
conda run -n 3dgstream-cu121 python benchmarks/scripts/collect_metrics.py
```

Override the environment names used by the master runner:

```bash
QUEEN_ENV=queen-cu121 HICOM_ENV=hicom-cu121 DGS_ENV=3dgstream-cu121 \
    bash benchmarks/run_all_serial.sh
```

Override the repo root (for out-of-tree checkouts, CI, etc.):

```bash
GLASS=/path/to/GLASS bash benchmarks/run_all_serial.sh
```

---

## Output schema

All scripts write their raw result JSONs under `results/<BASELINE>/…`; `collect_metrics.py` reads those and emits the single aggregated file:

```text
results/
├── 3DGStream/flame_steak/
│   ├── training_log.txt
│   ├── eval_log.txt
│   ├── metrics.json           ← read by collect_metrics.py (3DGStream row)
│   ├── fps_results.json
│   └── full_run/              ← per-frame trained models
├── QUEEN/
│   ├── bench_log.txt
│   └── flame_steak_metrics.json ← read by collect_metrics.py (QUEEN row)
├── Swift4D/flame_steak/{training,render,metrics}_log.txt
├── HiCoM/flame_steak/{training_log.txt, output/}
└── metric_table.json          ← unified PSNR / SSIM / LPIPS / FPS / storage table
```

`collect_metrics.py` prints the table to stdout as well:

```
Method       PSNR    SSIM   LPIPS      FPS  Train(s/f)  Store(MB)
QUEEN       31.42  0.9435  0.1470   240.3        4.11      141.0
3DGStream   30.14  0.9128  0.1902    54.7        8.23      ...
Swift4D      ...     ...     ...     ...          ...       ...
HiCoM        ...     ...     ...     ...          ...       ...
```

Every number in the QUEEN row is read from `data/assets/QUEEN/flame_steak_prior/results.json`, `avg_metrics.json`, and `training_metrics.json` — **no constants are hard-coded** in `collect_metrics.py`. Substitute your own prior-run subset to bench a different scene.

---

## See also

- [`docs/ablation_modes.md`](../docs/ablation_modes.md) — the ASIC-side companion (Mode 0–3 JSONs).
- [`docs/adapting_new_baseline.md`](../docs/adapting_new_baseline.md) — how to wire a new baseline into both `simulator/` *and* this `benchmarks/` harness.
- [`simulator/ablation/run_all.sh`](../simulator/ablation/run_all.sh) — ASIC ablation master runner (analogous to `benchmarks/run_all_serial.sh`).
