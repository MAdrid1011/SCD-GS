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
- `asic_ntc_train_render_cycles` drops proportionally too: `pipeline._train_render_active_counts()` scales `n_visible` / `tile_touches` by the hot-tier ratio so `add_render_training_iter` sees the same masked population that inference does. This is what GLASS §6 calls "Dual-Consumer Access" — update and render consumers share on-fabric residency, so ILSF's savings apply to both training render+backward cycles and inference render cycles, not just inference.
- `render_fps` jumps roughly by `1 / (N_active_M0 / N_active_Mi)` — see the three-column contribution table below for exact numbers.
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

The full canonical key order — identical for every baseline — lives in
`simulator/ablation/result_schema.py` (`SCHEMA_KEYS`). Highlights:

| Field | Unit | Meaning |
|-------|------|---------|
| `asic_train_time_s` | s | Headline — ASIC per-frame incremental training time at `ASICSpec.clock_hz`. Sum of the relevant `asic_*_cycles` buckets divided by `clock_hz` (each runner documents which buckets apply). |
| `asic_commit_time_s` | s | Commit path only: `(gradient_pipeline + optimizer) / clock_hz`. Isolates Mode 0→1→2 movement from NTC-training / render cycles. |
| `psnr` / `ssim` / `lpips` | — | Averaged over all test cameras on all trained frames |
| `render_fps` | fps | **Headline.** ASIC modelled FPS = `clock_hz / add_render_inference(n_active, n_visible, tile_touches, H, W, sh_degree)`; tier mask is baked into `n_active` / `n_visible` / `tile_touches` via `make_render_proxy()` for Mode ≥ 2. |
| `render_fps_gpu_measured` | fps | Diagnostic. GPU wall-clock FPS on the same `MaskedGaussianProxy`; never used for cycle accounting. |
| `render_n_active` / `render_n_visible` / `render_tile_touches` / `render_H` / `render_W` | — | Inputs that fed `add_render_inference`. Makes `render_fps` derivable by hand from any `mode_*.json`. |
| `total_storage_mb` | MB | Final checkpoint footprint on disk |
| `delta_storage_mb` | MB | Sum of per-frame update bytes across the run |
| `frame_total_bytes` | B | Bytes emitted by the last frame's update stream |
| `frame_suppressed_bytes` | B | Subset of the above suppressed by UCN |
| `frame_suppression_ratio` | — | `suppressed / total` |
| `frame_unique_targets` | — | Number of distinct Gaussians touched in the last frame |
| `run_total_bytes` / `run_suppressed_bytes` / `run_suppression_ratio` | B / B / ratio | Run-level accumulators |
| `base_train_iters` / `avg_train_iters` / `per_frame_train_iters` | — | Per-frame training-iteration schedule. Populated by 3DGStream / 4DGC / HiCoM (UCN feedback controls iter count in Mode ≥ 1); `null` for QUEEN whose `opt.iterations` is monolithic. |
| `asic_<unit>_cycles` | cycles | Per-unit totals; see `docs/cycle_accounting.md` |

GPU wall-clock timings (formerly `avg_train_time_s` / `all_frame_times_s`) are *not* written — they are simulator-host artefacts, not ASIC properties — and the redundant `asic_time_s_1ghz = asic_total_cycles / clock_hz` is also omitted since it silently excluded NTC-training cycles and was easily mistaken for `asic_train_time_s`. To migrate any older result files in place, run `python -m simulator.ablation.migrate_results`.

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

## Worked example — four baselines on flame_steak (10 frames)

Numbers from the 2026-04 ablation sweep (`results/GLASS_ablation/flame_steak/summary.csv`). All four baselines run through the same `result_schema.build_result` canonical JSON.

**3DGStream** (NTC-bound — the bottleneck is network training, not Adam):

| Mode | `asic_train_time_s` | `avg_train_iters` | PSNR | Render FPS |
|------|--------------------:|------------------:|-----:|-----------:|
| 0    | **9.885**           | 150               | 17.14 | 797.5  |
| 1    | **0.957** (10.3× ↓) | ≈63               | 16.97 | 796.1  |
| 2    | **0.763**           | ≈63               | 16.95 | 2811.7 |
| 3    | **0.758**           | ≈63               | 16.92 | 2913.7 |

**QUEEN** (188 K Gaussians, dense per-frame Adam — optimizer-bound):

| Mode | `asic_train_time_s` | PSNR | SSIM | Render FPS |
|------|--------------------:|-----:|-----:|-----------:|
| 0    | **1.958**           | 29.84 | 0.929 | 918.8  |
| 1    | **0.565** (3.5× ↓)  | 29.15 | 0.919 | 955.1  |
| 2    | **0.307**           | 29.15 | 0.919 | 2956.6 |
| 3    | **0.305**           | 29.15 | 0.919 | 2962.4 |

**HiCoM** (render+backward-bound — base Gaussians hard-frozen):

| Mode | `asic_train_time_s` | `avg_train_iters` (≡ `num_incr_steps`) | PSNR | Render FPS |
|------|--------------------:|---------------------------------------:|-----:|-----------:|
| 0    | **0.964**           | 200                                    | 26.00 | 269.2  |
| 1    | **0.053** (18.2× ↓) | ≈37                                    | 24.65 | 257.1  |
| 2    | **0.012**           | ≈37                                    | 24.31 | 2622.3 |
| 3    | **0.012**           | ≈37                                    | 24.61 | 3014.7 |

**4DGC** (EntropyBottleneck + MEM training + per-frame grid + render+backward — all paths active):

| Mode | `asic_train_time_s` | `avg_train_iters` (MEM) | PSNR | Render FPS |
|------|--------------------:|------------------------:|-----:|-----------:|
| 0    | **3.682**           | 100                     | 32.95 | 434.5  |
| 1    | **1.247** (3.0× ↓)  | ≈73                     | 32.19 | 434.1  |
| 2    | **0.820**           | ≈73                     | 32.29 | 3644.8 |
| 3    | **0.604**           | ≈73                     | 32.51 | 3890.3 |

### Technique → cycle bucket → quantified contribution

Each of UCN / ILSF / Salience primarily moves one or two cycle buckets. The table below pins the mechanism, the bucket it reduces, and the observed absolute gain from the flame_steak sweep above:

| Technique | Primary cycle bucket it shrinks | Mechanism | Observed gain on flame_steak (10 f) |
|---|---|---|---|
| **UCN** (M0 → M1) | `asic_ntc_train_cycles` + `asic_ntc_train_render_cycles` + `asic_optimizer_cycles` | Adaptive iteration scheduler cuts `n_iters` (`base → ~37-73`); `grad_scale=0` skips Adam on suppressed rows | 3DGS 10.3×, QUEEN 3.5×, HiCoM 18.2×, 4DGC 3.0× on `asic_train_time_s` |
| **ILSF** (M1 → M2) | `asic_ntc_train_render_cycles` + `asic_optimizer_cycles` + `asic_render_cycles` | Hot-tier mask (ASC ∪ hot-RSB) scales `n_active` / `n_visible` / `tile_touches` in both `add_render_training_iter` (GLASS §6 Dual-Consumer Access) and `add_render_inference` | ASIC train: 3DGS 1.25×, QUEEN 1.84×, HiCoM 4.4×, 4DGC 1.52×. Inference FPS: 3DGS 3.5×, QUEEN 3.1×, HiCoM 10.2×, 4DGC 8.4× |
| **Salience** (M2 → M3) | `asic_ntc_train_render_cycles` (low-value tail) + tile work in `asic_render_cycles` | PROMOTE/FULL classifier drops stable low-contribution Gaussians from tile expansion before rasterize | ASIC train: 4DGC 1.36× (largest); others already bottom out at compute floor (0.012 s / 0.30 s). Inference FPS: +3-15% across all baselines |

(Absolute cycle counts per bucket per mode are in the `mode_*.json` files; this is a pointer, not a replacement.)

### Cross-baseline takeaways

- **Mode 1 speed-up source shifts per baseline**: QUEEN gets it from Path 1 (`grad_scale → 0` skips Adam); 3DGStream gets it from Path 2 (adaptive threshold converges over ~3 frames, then the iter scheduler halves NTC training); HiCoM gets it from both Path 1 (fabric demotion cuts render cost) *and* Path 2 (early exit), compounded by fewer densification steps keeping `n_visible` small — see `GLASS.md §A.3` "cascade effect". 4DGC gets it from Path 2 throttling MEM training iterations while UCN-suppressed bytes never hit the NTC path.
- **Render-FPS jump in Mode 2** is present on all four baselines now and the cross-baseline spread is physically motivated: `(M2 FPS) / (M0 FPS) ≈ (N_active,M0 · avg_tile_touches_M0) / (N_active,M2 · avg_tile_touches_M2)`. HiCoM and 4DGC see the biggest jumps because they have the lowest hot-tier ratios (1.4% and 4.5% respectively on flame_steak). Pre-2026-04 fix, `render_fps` was back-calculated from GPU wall-clock on the masked proxy, which folded simulator-host kernel overhead into the ASIC number and produced misleading spreads (e.g. 4DGC M2/M0 = 27× from artificially low GPU M0). The current `add_render_inference` formula models ASIC render cycles directly from EWA-splatting phases 1-3 so the ratios are bandwidth- and throughput-consistent.
- **Quality delta Mode 0 → Mode 3** is ≤ 1.7 dB PSNR for all four baselines, and PSNR is monotonically non-increasing only up to noise on short 10-frame runs. Adaptive threshold keeps the knee-point reliable without per-baseline tuning.
- **3DGStream's 3-frame smoke vs 10-frame ablation discrepancy**: at 3 frames, `asic_train_time(Mode 1) ≈ 0.95 × Mode 0` — the adaptive threshold hasn't converged to steady-state yet. The 10× number only shows up once the feedback controller has had enough frames to settle (≈ 4–5 frames on this scene). If you are benchmarking a new baseline, run at least 10 frames before drawing conclusions about Mode 1's asymptotic speed-up.

A note on GPU wall-clock: the four runners still measure per-frame wall-clock via `MetricsCollector.frame_timer()` for internal profiling, but this number is deliberately **not** emitted to `mode_*.json`. It is a simulator-host artefact (unoptimised Python in `LiveStateFabric.get_render_mask` can make QUEEN Modes 2/3 look *slower* on the host, which is the opposite of the ASIC behaviour). The headline the paper reports is `asic_train_time_s` + `render_fps`. `render_fps_gpu_measured` is retained per-mode as a diagnostic so the ASIC-vs-host gap is auditable.

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
