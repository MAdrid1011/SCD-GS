# Ablation Modes

Four progressive modes are defined in `simulator/config.py::AblationMode`:

```python
class AblationMode(IntEnum):
    BASE_ASIC     = 0    # Capture only (PUOE)
    CONSOLIDATION = 1    # + UCN
    LIVE_FABRIC   = 2    # + ILSF hot/cold tier mask
    SALIENCE      = 3    # + Salience-First Issue
```

Every subsequent mode strictly adds optimisation on top of the previous one. This is the property we exploit when reading the ablation: a field that changes between Mode N-1 and Mode N is attributable to the unit added at Mode N.

---

## Mode 0 — Base ASIC (capture only)

**What runs:** Stage 0 (PUOE) — per-Gaussian norm computation and descriptor capture. No gradient modification; every `param.grad` is untouched. Render mask = identity.

**Why it exists:** establishes the bandwidth and cycle cost of simply *observing* every per-Gaussian gradient. Everything Mode 1+ saves is relative to Mode 0.

**Must-be-non-zero:** `asic_puoe_cycles`, `asic_optimizer_cycles`, `frame_total_bytes`.
**Must-be-zero:** `asic_ucn_cycles`, `asic_ice_cycles`, `asic_salience_cycles`, `frame_suppressed_bytes`.

---

## Mode 1 — + UCN (Consolidation)

**What runs:** Stage 1. For every (gid, attr) the ConsolidationUnit computes a per-row `grad_scale ∈ {0, 1}` (0 = suppressed, 1 = keep). Main + fallback scales are unified into a per-gid minimum and multiplied into `param.grad`. ICE cycles accrue via `add_ice_scale`.

**What the numbers will say:** compared to Mode 0 you should see:
- `frame_suppressed_bytes > 0` and `frame_suppression_ratio > 0`;
- `asic_ice_cycles > 0`, `asic_ucn_cycles > 0`;
- `asic_optimizer_cycles` decreases (Adam only runs on non-suppressed rows);
- `asic_train_time_s` drops significantly vs Mode 0 on baselines with high cross-frame stability (QUEEN: ~3×; 3DGStream: ~1.3×);
- Quality (PSNR / SSIM / LPIPS) should change by < 0.3 dB vs Mode 0 on well-tuned thresholds.

**Config knobs to tune per baseline:** `merge_cos_threshold`, `min_delta_norm`, `frame_level_ucn`, `reset_cwb_each_frame`. See `docs/adapting_new_baseline.md` §"Common pitfalls" §3 and §4.

**v2 additions (enabled by default, no per-baseline tuning required):**
- `adaptive_ucn_threshold` — per-frame PI controller that moves `merge_cos_threshold` toward `adaptive_ucn_target_ratio` (0.70). Stops `merge_cos_threshold=0.90` from over-suppressing on HiCoM-style baselines (PSNR collapse observed at > 99% suppression).
- `ucn_drive_fabric_demotion` + `ucn_demotion_streak` — UCN's "suppressed for N consecutive events" signal demotes stable Gaussians to CSS so Mode ≥ 2 can skip them in render+backward too (Path 1).
- `run_*.py` adaptive iteration scheduler — reads `pipeline.current_frame_suppression_ratio()` to early-exit the frame's training loop (Path 2). Currently wired for 3DGStream (NTC) and HiCoM; skipped for QUEEN (monolithic iteration schedule).

Full mechanism write-up: `GLASS.md` Appendix A.

---

## Mode 2 — + ILSF (Live-State Fabric)

**What runs:** Stage 2 — the fabric classifies Gaussians into ASC (hot) / RSB (warm) / CSS (cold) tiers. Cold-tier gradients are zeroed before `optimizer.step()`; render call uses `MaskedGaussianProxy` to only paint ASC ∪ (RSB ∩ recent_delta > threshold). Mode 1's `add_ice_scale` is replaced by `add_ice_merged(n_hot, n_cold, d)` — a single fused scale + cold-zero pass that costs ~33% less memory bandwidth.

**What the numbers will say:**
- `asic_ice_cycles` **decreases** vs Mode 1 (merged pass is cheaper).
- `asic_optimizer_cycles` drops sharply — Adam only runs on hot Gaussians (ASC + qualifying RSB), typically ~3k out of ~200k for QUEEN.
- `render_fps` jumps roughly proportional to `(asc+rsb)/n_total` — for QUEEN on flame_steak we see ~3× (140 → 430 FPS).
- Training wall-clock may be slightly **higher** than Mode 1 (the fabric's Python sort is not free); the ASIC estimate still improves because cycles reflect the silicon, not the simulator host.
- Quality is typically within 0.1-0.3 dB of Mode 1.

**Config knobs:** `asb_capacity` (≈ 3000), `rsb_capacity` (≈ 10000), `rsb_delta_thresh`, `hotness_alpha`, `hotness_update_reward`, `hotness_render_reward`. Values ship tuned for the flame_steak baseline set; if your scene has markedly different N or dynamism, start by scaling `asb_capacity + rsb_capacity` so the sum is ~6-8% of N.

---

## Mode 3 — + Salience

**What runs:** Stage 4 — on top of the tier mask, each Gaussian is further classified as `PROMOTE` / `FULL` / `SKIP` based on an EMA salience score with a stability gate. `pipeline.post_render()` pushes visibility events into the classifier; `pipeline.begin_frame()` + structural-change events reset dirty entries.

**What the numbers will say:**
- `asic_salience_cycles > 0` and `asic_post_render_cycles` roughly equals `asic_salience_cycles` (they overlap on critical path).
- Small additional savings vs Mode 2 — typically the delta is within noise for short runs, meaningful on long-running streams with highly stable regions.
- Quality should match Mode 2 (within 0.05 dB) as long as `full_path_fraction ≥ 0.10` (a safeguard; see `config.py`).

**Config knobs:** `salience_ema_alpha`, `promote_thresh`, `skip_thresh`, `stability_window`, `stability_var_max`, `full_path_fraction`.

---

## Reading `mode_*.json`

Every mode produces a JSON with the same schema. The headline fields:

| Field | Unit | Meaning |
|-------|------|---------|
| `avg_train_time_s` | s/frame | GPU wall-clock per incremental frame (drop-first-frame + top-10% trimmed mean) |
| `asic_train_time_s` | s | ASIC-projected incremental training time at `ASICSpec.clock_hz` |
| `asic_commit_time_s` | s | Subset of ASIC time that is UCN + ICE + Optimizer (the streamed commit phase) |
| `psnr` / `ssim` / `lpips` | — | Averaged over all test cameras on all trained frames |
| `render_fps` | fps | `pipeline.measure_fps()` — tier mask active in Mode ≥ 2 |
| `total_storage_mb` | MB | Final checkpoint footprint on disk |
| `delta_storage_mb` | MB | Sum of per-frame update bytes across the run |
| `frame_total_bytes` | B | Bytes emitted by the last frame's update stream |
| `frame_suppressed_bytes` | B | Subset of the above suppressed by UCN |
| `frame_suppression_ratio` | — | `suppressed / total` |
| `frame_unique_targets` | — | Number of distinct Gaussians touched in the last frame |
| `run_total_bytes` / `run_suppressed_bytes` | B | Run-level accumulators |
| `asic_<unit>_cycles` | cycles | Per-unit totals; see `docs/cycle_accounting.md` |

### Cross-mode expectations — fast sanity check

For a well-tuned run on the same scene + baseline:

```
psnr(0)        ≈ psnr(1)   ≈ psnr(2)   ≈ psnr(3)          (within ±0.3 dB)
frame_suppr(0) = 0
frame_suppr(1) > 0
frame_suppr(2) ≥ frame_suppr(1)
frame_suppr(3) ≥ frame_suppr(2)

asic_train_time(0) > asic_train_time(1) > asic_train_time(2) ≳ asic_train_time(3)
```

The last line is the "progressive improvement" property. If Mode 2 is slower than Mode 1 in ASIC time, you almost certainly have `add_ice_merged` double-counting (regression check: Mode 1's `add_ice_scale` must *not* fire in Mode ≥ 2 — see `pipeline.py` comment around `_apply_consolidation`).

---

## Worked example — three baselines on flame_steak (10 frames)

Numbers below come from the v2 UCN run (Path 1 + Path 2 + adaptive threshold enabled). The corresponding design notes live in `GLASS.md` Appendix A.

**QUEEN** (188 K Gaussians, dense per-frame Adam — optimizer-bound):

| Mode | `avg_train_time_s` | `asic_train_time_s` | PSNR | SSIM | Render FPS |
|------|-------------------:|--------------------:|-----:|-----:|-----------:|
| 0    | 17.6               | **1.957**           | 29.85 | 0.931 | 152 |
| 1    | 15.8               | **0.565** (3.5× ↓)  | 29.20 | 0.924 | 152 |
| 2    | 20.5               | **0.568**           | 29.18 | 0.923 | 437 |
| 3    | 26.9               | **0.572**           | 29.15 | 0.923 | 449 |

**3DGStream** (NTC-bound — the bottleneck is network training, not Adam):

| Mode | `asic_train_time_s` | `avg_ntc_iters` | PSNR | Render FPS |
|------|--------------------:|----------------:|-----:|-----------:|
| 0    | **9.46**            | 150             | 17.14 | 143 |
| 1    | **0.65** (14.5× ↓)  | 63.7            | 17.05 | 143 |
| 2    | **0.65**            | 63.7            | 17.02 | 944 |
| 3    | **0.65**            | 63.7            | 17.00 | 983 |

**HiCoM** (render+backward-bound — base Gaussians hard-frozen):

| Mode | `asic_train_time_s` | `num_incr_steps` (avg) | PSNR | Render FPS |
|------|--------------------:|-----------------------:|-----:|-----------:|
| 0    | **0.960**           | 200                    | 25.75 | 76  |
| 1    | **0.051** (18.8× ↓) | ≈37                    | 24.80 | 76  |
| 2    | **0.051**           | ≈37                    | 24.72 | 287 |
| 3    | **0.051**           | ≈37                    | 24.66 | 303 |

Cross-baseline takeaways:

- **Mode 1 speed-up source shifts per baseline**: QUEEN gets it from Path 1 (`grad_scale → 0` skips Adam); 3DGStream gets it from Path 2 (adaptive threshold converges over ~3 frames, then the iter scheduler halves NTC training); HiCoM gets it from both Path 1 (fabric demotion cuts render cost) *and* Path 2 (early exit), compounded by fewer densification steps keeping `n_visible` small — see `GLASS.md §A.3` "cascade effect".
- **Render-FPS jump in Mode 2** is present on all three baselines now (3DGStream 6.6×, QUEEN 2.9×, HiCoM 3.8×). Pre-v2 fix, HiCoM showed 0× — the proxy wasn't reaching its renderer. See `docs/adapting_new_baseline.md` pitfall #7(b).
- **Quality delta Mode 0 → Mode 3** is ≤ 1.1 dB PSNR for all three baselines, and PSNR is monotonically non-increasing along 0→3. Adaptive threshold keeps the knee-point reliable without per-baseline tuning.
- **3DGStream's 3-frame smoke vs 10-frame ablation discrepancy**: at 3 frames, `asic_train_time(Mode 1) ≈ 0.95 × Mode 0` — the adaptive threshold hasn't converged to steady-state yet. The 14.5× number only shows up once the feedback controller has had enough frames to settle (≈ 4–5 frames on this scene). If you are benchmarking a new baseline, run at least 10 frames before drawing conclusions about Mode 1's asymptotic speed-up.

`avg_train_time_s` (GPU wall-clock) **rising** in Modes 2/3 for QUEEN is a *simulator-host* artefact (unoptimised Python in `LiveStateFabric.get_render_mask`), not an ASIC property. The hardware estimate (`asic_train_time_s`) is what the paper should report.

---

## What Mode to use when

| Goal | Mode |
|------|------|
| Measure capture-only overhead / descriptor traffic | 0 |
| Isolate the effect of consolidation | 1 vs 0 |
| Measure hot/cold tier + render-FPS impact | 2 vs 1 |
| Measure salience's marginal effect | 3 vs 2 |
| End-to-end GLASS vs baseline comparison | 0 vs 3 |
| Quality regression test on a new baseline | All four (should each be within 0.3 dB of the GPU run) |
