# GLASS — Live-State Dataflow Architecture for Streamable Gaussian Systems

A Python ASIC simulator for an edge accelerator that trains streamable Gaussian Splatting models frame-by-frame. The simulator measures cycle counts, memory traffic, and quality of four progressive ablation configurations (Mode 0 – Mode 3) across four published baselines: **QUEEN**, **3DGStream**, **HiCoM**, and **4DGC**.

---

## Why this repo

Edge deployment of per-frame Gaussian Splatting updates is **memory-bandwidth-bound**, not compute-bound. GLASS proposes three hardware units that target that bottleneck:

| Unit | Stage | Purpose |
|------|-------|---------|
| **UCN** — Update-Stream Consolidation  | 1 | Fold temporally correlated per-Gaussian gradients; suppress noisy ones |
| **ILSF** — Incremental Live-State Fabric | 2 | Classify Gaussians into ASC / RSB / CSS tiers; zero grads for cold tier |
| **Salience-First Issue**                 | 3 | PROMOTE / FULL / SKIP filter on the render queue |

This simulator quantifies the speed-up, bandwidth savings, and quality impact of each unit on real baselines so the silicon team can prioritise correctly.

---

## Repo layout

```text
GLASS/
├── simulator/              # ← this repo's main product
│   ├── config.py            ablation modes + all tunable knobs
│   ├── pipeline.py          GLASSPipeline — stage hub, fast paths for modes 0-3
│   ├── datasets.py          per-scene configuration registry + YAML loader
│   ├── core/
│   │   ├── cycle_model.py       ASIC spec + per-stage cycle accountant
│   │   ├── consolidation.py     UCN (CWB + scale math)
│   │   ├── live_state_fabric.py hot/warm/cold tier classifier (ILSF)
│   │   ├── salience_issue.py    Stage-4 PROMOTE/FULL/SKIP classifier
│   │   ├── update_descriptor.py UpdateDescriptor / AttrClass taxonomy
│   │   └── proxy.py             MaskedGaussianProxy for tier-masked renders
│   ├── operators/
│   │   ├── base_operator.py     BaseOperatorSlot — 3 hook contract
│   │   ├── op_queen.py          QUEEN — latent-decoder + gate
│   │   ├── op_3dgstream.py      3DGStream — NTC delta-norm ingestion
│   │   ├── op_hicom.py          HiCoM — hierarchical commit
│   │   └── op_4dgc.py           4DGC — Motion_Grid MEM + entropy bottleneck
│   ├── metrics/
│   │   └── collector.py         PSNR / SSIM / LPIPS / FPS / storage
│   └── ablation/                per-baseline runner scripts (in PR #4)
├── benchmarks/                  benchmarking harness (in PR #5)
├── baselines/                   git submodules — pristine upstream
│   ├── QUEEN                    @ NVlabs/queen
│   ├── 3DGStream                @ SJoJoK/3DGStream
│   ├── HiCoM                    @ gqk/HiCoM
│   └── 4DGC                     @ qianghu-huber/4DGC
├── data/                        datasets + pre-trained init weights  (gitignored)
└── results/                     ablation outputs  (gitignored)
```

---

## Quick start

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/MAdrid1011/GLASS.git
cd GLASS
# or, post-clone:
git submodule update --init --recursive
```

### 2. Environment

The simulator has no unusual dependencies beyond what the baselines require. The QUEEN environment (CUDA 12.1 / PyTorch 2.x) is the most capable shared environment:

```bash
conda env create -f baselines/QUEEN/environment.yml
conda activate queen-cu121
pip install torchmetrics pyyaml      # add-on for metrics + dataset YAMLs
```

### 3. Run one ablation mode (after PR #4 lands)

```bash
# QUEEN, mode 1, 3 frames — smoke test
python simulator/ablation/run_queen.py --mode 1 --n-frames 3

# 3DGStream, mode 2, 10 frames
python simulator/ablation/run_3dgstream.py --mode 2 --n-frames 10
```

Results land in `results/GLASS_ablation/<scene>/<BASELINE>/mode_<N>.json`.

### 4. Inspect a result

```bash
cat results/GLASS_ablation/flame_steak/QUEEN/mode_1.json
```

Key fields (canonical key order in `simulator/ablation/result_schema.py`):
- `asic_train_time_s` — headline: ASIC incremental-training time per frame at `ASICSpec.clock_hz` (default 1 GHz)
- `asic_commit_time_s` — commit-path only (`gradient_pipeline + optimizer`); isolates Mode 0→1→2 movement
- `psnr` / `ssim` / `lpips` — averaged over all test cameras on all trained frames
- `render_fps` — ASIC-modelled FPS via `add_render_inference` on the mode-aware masked proxy
- `render_fps_gpu_measured` — GPU wall-clock FPS on the same proxy (diagnostic)
- `total_storage_mb` / `delta_storage_mb` — final checkpoint + run-level incremental stream
- `frame_*` / `run_*` bytes and `*_suppression_ratio` — UCN evidence (Challenge 1)
- `n_asc_rsb` / `n_css` / `n_known` / `n_total` — LiveStateFabric residency (Challenge 2)
- `base_train_iters` / `avg_train_iters` / `per_frame_train_iters` — per-frame training-iter schedule (null for QUEEN's monolithic `opt.iterations`)
- `asic_<unit>_cycles` — per-unit totals; see [`docs/cycle_accounting.md`](docs/cycle_accounting.md)

GPU wall-clock timings are deliberately not written — they are simulator-host artefacts, not ASIC properties. Older result JSONs can be migrated in place with `python -m simulator.ablation.migrate_results`.

---

## The four ablation modes

| Mode | Stages enabled | What it measures |
|------|----------------|------------------|
| 0    | Capture only (PUOE) | Base ASIC — descriptor bookkeeping overhead |
| 1    | + UCN               | Consolidation-only: bandwidth and training-cycle savings |
| 2    | + ILSF              | Add hot/cold tier render mask + cold-grad zero |
| 3    | + Salience          | Add PROMOTE / FULL / SKIP classifier on render queue |

See [`docs/ablation_modes.md`](docs/ablation_modes.md) for a field-by-field walkthrough of what each mode enables.

---

## Documentation map

| Doc | Audience | Purpose |
|-----|----------|---------|
| [`docs/architecture.md`](docs/architecture.md) | Everyone | Stage 0→4 data flow + every component's role |
| [`docs/ablation_modes.md`](docs/ablation_modes.md) | Researchers | What each mode turns on; how to read `mode_*.json` |
| [`docs/adapting_new_baseline.md`](docs/adapting_new_baseline.md) | Contributors | **Step-by-step guide to plugging in a new Gaussian-splatting baseline** |
| [`docs/cycle_accounting.md`](docs/cycle_accounting.md) | Hardware eng. | ASIC spec + every cycle formula with justification |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Contributors | Branch / commit / PR conventions + parity-test rules |

---

## Adding a new streamable Gaussian baseline

The simulator was designed around the lesson learnt from QUEEN and 3DGStream integration: **everything model-specific belongs in a 100-200-line operator slot; the pipeline never imports the baseline**. Follow [`docs/adapting_new_baseline.md`](docs/adapting_new_baseline.md) for the exact checklist.

---

## Design principles

- **No magic numbers.** Every threshold, capacity, or cycle coefficient is either a dataclass field in [`simulator/config.py`](simulator/config.py) / [`simulator/core/cycle_model.py`](simulator/core/cycle_model.py) or a numerical-stability constant (documented at the call site).
- **Bit-exact fast paths.** Every performance-oriented rewrite comes with a parity argument against the dict-based reference path — see the docstrings in `pipeline.py`. Changing a fast path without matching the reference is rejected at review.
- **Model-agnostic core.** `simulator/core/` imports nothing from `baselines/`; all wiring is via `operators/op_<name>.py`.

---

## License

TBD (private repo during development). Upstream baselines retain their own licenses — see each entry under `baselines/`.
