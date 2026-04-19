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

## Worked example — QUEEN on flame_steak (3 frames)

| Mode | `avg_train_time_s` | `asic_train_time_s` | PSNR | SSIM | Render FPS |
|------|-------------------:|--------------------:|-----:|-----:|-----------:|
| 0    | 17.58              | **1.97**            | 29.82 | 0.930 | 140.6 |
| 1    | 16.00              | **0.59** (3.3× ↓)   | 29.58 | 0.925 | 140.6 |
| 2    | 20.40              | **0.59**            | 29.58 | 0.925 | 430.2 |
| 3    | 26.90              | **0.59**            | 29.58 | 0.925 | 447.7 |

Takeaways:
- The ASIC training-time drop is carried almost entirely by UCN (Mode 1).
- The render-FPS jump comes from the tier mask (Mode 2).
- Quality loss from the full stack is 0.24 dB PSNR — well inside acceptable range.
- `avg_train_time_s` rising in Modes 2/3 is a *simulator-host* artefact (unoptimised Python in `LiveStateFabric`), not an ASIC property. The hardware estimate (`asic_train_time_s`) is what matters.

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
