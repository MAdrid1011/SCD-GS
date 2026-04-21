# Cycle Accounting — `CycleAccumulator` Reference

Every timing claim in a `mode_*.json` JSON is derived from a sum of `CycleAccumulator.add_*()` calls. This document catalogs each method, the hardware it models, and the formula it uses so you can verify any number yourself.

All methods live in `simulator/core/cycle_model.py`. Every coefficient is either a field on `ASICSpec` or a byte-accurate constant (4 B per float32, 2 B per float16, 8 B = radii+opacity pair, 16 B per descriptor). There are no magic numbers.

---

## ASIC hardware specification (`ASICSpec`)

| Field | Default | Justification |
|-------|---------|---------------|
| `clock_hz`          | `1e9`   | 1 GHz target — matches the edge-ASIC budget used in the paper |
| `mac_units`         | `128`   | 128 × 32-bit FMA = 512-bit datapath; fits in <2 mm² at 28 nm @ 1 GHz (cf. Eyeriss v2, TPU micro-core sizing) |
| `sram_bw_bytes`     | `256`   | 4 banks × 64 B/cycle; 64 B = 1 cache line (ARM / TSMC); 4 banks allow simultaneous read + write |
| `sqrt_latency_cyc`  | `15`    | Industry 12–18 cy for fp32 sqrt (ARM Cortex-A55 FPU = 14 cy, TSMC 28 nm ASIC typical = 15 cy) |
| `cmp_units`         | `256`   | 2 × `mac_units`; comparator cells are ≈⅛ the area of a FMA cell |
| `dram_bw_bytes`     | `64`    | LPDDR5X-8000 @ 64-bit: 8000 Mbps × 64/8 = 64 GB/s → 64 B/cycle @ 1 GHz |
| `adam_ops_per_float`| `10`    | Exact Adam step op count (`m←β₁m+…`, `v←β₂v+…`, `m̂`, `v̂`, `√`, `+ε`, `div`, `sub`) |

All values are `ASICSpec` dataclass fields — override one in `ASICConfig(asic_spec=ASICSpec(mac_units=256, ...))` to model a different ASIC without touching any simulation code.

---

## Per-unit cycle methods

Each heading links the ASIC unit name to its simulator method. All formulas below use:
- `N` = Gaussian count for the call
- `D` = attribute dimension per Gaussian (per param group)
- `bw_sram` = `spec.sram_bw_bytes`
- `bw_dram` = `spec.dram_bw_bytes`
- `mac` = `spec.mac_units`, `cmp` = `spec.cmp_units`
- `⌈·/·⌉` = integer ceiling division (`_ceil`)

### `add_puoe(n, d)` — per-Gaussian norm capture

Reads `N×D` grad floats, computes one L2 norm per Gaussian (`D` MACs + 1 sqrt), writes `N×16 B` descriptors.

```
compute  = ⌈N / mac⌉ × max(D, sqrt_latency_cyc)      # per-Gaussian time pipelined
mem_read = ⌈N × D × 4 / bw_sram⌉
write    = ⌈N × 16    / bw_sram⌉
cycles   = max(compute, mem_read) + write
```

Memory-bound for `D ≥ 15`; sqrt-bound for `D < 15`.

### `add_puoe_network(n_params)` — aggregate reduction

One scalar norm over every network parameter (NTC weights, latent decoder, gate, hash grid). The single sqrt is trivial against the `N` MACs.

```
compute  = ⌈n_params / mac⌉
mem_read = ⌈n_params × 4 / bw_sram⌉
write    = ⌈16 / bw_sram⌉                            # 1 aggregate descriptor
cycles   = max(compute, mem_read) + write
```

Always memory-bound at the default spec (`bw_sram=256 < 4·mac=512`).

### `add_ucn(n)` — Consolidation EMA + scale emit

Per-Gaussian: 1 FMA (EMA update), 2 CMP, 1 DIV, 1 SUB, 1 CMP → **7 ops** on the comparator fabric. Memory: 1 float read (prev mean), 2 floats written (new mean + grad_scale).

```
compute = ⌈N × 7 / cmp⌉
mem_rd  = ⌈N × 4 / bw_sram⌉
mem_wr  = 2 × ⌈N × 4 / bw_sram⌉
cycles  = max(compute, mem_rd) + mem_wr
```

Called **once per optimizer step** with the combined main + fallback count. Calling it more than once would change the total because of `⌈·⌉` — this is a parity invariant.

### `add_ice_scale(n, d)` — ICE scale-only (Mode 1)

Reads `N×(D+1)×4 B` (grad + scale factor), computes `N×D` mults, writes `N×D×4 B`.

```
read    = ⌈N × (D+1) × 4 / bw_sram⌉
compute = ⌈N × D / mac⌉
write   = ⌈N × D × 4 / bw_sram⌉
cycles  = max(read, compute) + write
```

### `add_ice_merged(n_hot, n_cold, d)` — ICE fused scale + cold-zero (Mode ≥ 2)

Single read-modify-write pass. Hot rows get multiplied by scale; cold rows get zeroed (no MAC). Saves ~33% memory BW vs two separate passes because the cold-mask-only pass is eliminated.

```
N       = n_hot + n_cold
read    = ⌈N × D × 4 / bw_sram⌉
compute = ⌈n_hot × D / mac⌉                          # only hot rows do MACs
write   = ⌈N × D × 4 / bw_sram⌉
cycles  = max(read, compute) + write
```

### `add_ilsf_update(n_updated)` — hotness EMA from gradient events

```
cycles = ⌈n_updated / mac⌉ + ⌈n_updated × 4 / bw_sram⌉
```

Runs on a disjoint memory bus (hotness scores vs grad_scale); absorbed into the gradient pipeline via `max()` in `summary()`.

### `add_ilsf_render(n_visible)` — hotness EMA from render-visibility

Same shape as `add_ilsf_update`, driven by the visibility filter post-render. Parallel with `add_salience` (disjoint inputs/outputs) — critical path is `max(ilsf_render, salience)`.

### `add_ilsf_sort(n)` — once per frame in `begin_frame`

Sorts `N` hotness scores to derive tier membership.

```
cycles = ⌈N / cmp⌉ + ⌈N × 4 / bw_sram⌉
```

Fires concurrently with the first optimizer step's gradient pipeline (ICE ≈ 200 K cycles/step); sort is ≈ 5.5 K cycles → **completely hidden**, so `summary()` records it but does **not** add it to the critical path.

### `add_salience(n)` — Mode ≥ 3 post-render score update

Per Gaussian: 1 MUL (opacity × radii), 1 DIV (normalise), 1 FMA (EMA). Memory: 8 B read (radii + opacity), 4 B write.

```
compute = ⌈N × 3 / mac⌉
mem_rd  = ⌈N × 8 / bw_sram⌉
write   = ⌈N × 4 / bw_sram⌉
cycles  = max(mem_rd, compute) + write
```

### `add_optimizer_update(n_active, total_d)` — Adam step

`adam_ops_per_float = 10` exact ops per scalar θ. Memory: read (grad + m + v + θ) = 16 B, write (m + v + θ) = 12 B.

```
compute = ⌈n_active × total_d × 10 / mac⌉
read    = ⌈n_active × total_d × 16 / bw_sram⌉
write   = ⌈n_active × total_d × 12 / bw_sram⌉
cycles  = max(compute, read) + write
```

`n_active` depends on mode: Mode 0 = all touched, Mode 1 = UCN-active, Mode ≥ 2 = fabric render-mask.

### `add_ntc_forward(n, mlp_macs, hash_params, enc_dim, out_dim)` — 3DGStream NTC inference

Two-phase: (1) HashGrid fp16 load from DRAM + trilinear interpolation, (2) FullyFusedMLP with outputs written back to DRAM. All five arguments are **measured from the live tcnn model** at hook install time (see `Op3DGStream.install_ntc_delta_hook`).

```
# Phase 1 (HashGrid)
hash_load    = ⌈n_hash_params × 2 / bw_dram⌉         # fp16
hash_compute = ⌈n × enc_dim × 2 / mac⌉                # 2 MACs per tri-lerp axis pair
phase1       = max(hash_load, hash_compute)
             + ⌈n × enc_dim × 4 / bw_sram⌉

# Phase 2 (FullyFusedMLP)
mlp_compute  = ⌈n × mlp_macs / mac⌉
enc_read     = ⌈n × enc_dim × 4 / bw_sram⌉
phase2       = max(mlp_compute, enc_read)
             + ⌈n × out_dim × 4 / bw_dram⌉           # delta attrs → DRAM

cycles       = phase1 + phase2
```

### `add_ntc_training(n, mlp_macs, hash_params, enc_dim, out_dim, n_iters)` — 3DGStream NTC training loop

One iteration = forward + backward + Adam over `(n_hash_params + n_mlp_macs_per_g)` total NTC params. Total is that one-iteration sum multiplied by `n_iters`. See the method docstring for the full per-phase breakdown.

### `add_render_training_iter(n, n_visible, tile_touches, H, W, sh_degree, n_iters)` — forward render + photometric loss backward

Charged per NTC / MEM / gaussian training iteration. Six sequential EWA-splatting phases — (1) preprocessing forward, (2) 32-bit radix sort over `tile_touches`, (3) alpha compositing forward, (4) photometric L1+SSIM loss, (5) alpha compositing backward, (6) preprocessing backward. Reads scene stats captured by `pipeline.post_render()`; the lag-one usage (previous frame's stats feed the current frame's schedule) mirrors `_prev_suppression_ratio`.

Under Mode ≥ 2 the `n / n_visible / tile_touches` arguments are **active-ratio-scaled** by `pipeline._train_render_active_counts()` (GLASS §6 "Dual-Consumer Access" — update- and render-side consumers share the same on-fabric residency). This makes `asic_ntc_train_render_cycles` respond monotonically to ILSF (Mode 2) and Salience (Mode 3), not just inference FPS.

### `add_render_inference(n_active, n_visible, tile_touches, H, W, sh_degree)` — per-frame inference render (headline FPS)

Mirrors phases 1-3 of `add_render_training_iter` (preprocess + sort + alpha-compositing forward) and explicitly excludes phases 4-6 (loss, compositing backward, preprocessing backward — training-only). Per-frame cost; overwrites `_cyc["render"]` like the legacy `set_render_cycles_from_fps`.

```
sh_params = 3 × (sh_degree+1)²               # exact SH coefficient count
per_g_macs = sh_params + 15 + 4              # SH dot + EWA cov (RSS^TR^T + 2D proj) + 3D→2D proj

# Phase 1 — preprocessing (over N_active, hit-visible only writes)
p1_compute = ⌈N_active × per_g_macs              / mac⌉
p1_read    = ⌈N_active × (11 + sh_params) × 4    / bw_dram⌉
p1_write   = ⌈N_visible × 10 × 4                 / bw_dram⌉
phase1     = max(p1_compute, p1_read) + p1_write

# Phase 2 — 32-bit radix sort (tile||depth KV, 32 passes, read+write amortised)
p2_compute = ⌈tile_touches × 32 / cmp⌉
p2_bw      = ⌈tile_touches × 8 × 2 / bw_dram⌉
phase2     = max(p2_compute, p2_bw)

# Phase 3 — alpha compositing forward (19 MACs per Gaussian-tile intersection)
p3_compute = ⌈tile_touches × 19 / mac⌉
p3_read    = ⌈tile_touches × 9 × 4 / bw_dram⌉     # cov2D+mean2D+color+opacity
p3_write   = ⌈H × W × 3 × 4 / bw_dram⌉            # rendered image
phase3     = max(p3_compute, p3_read) + p3_write

cycles     = phase1 + phase2 + phase3
asic_fps   = clock_hz / cycles
```

All five arguments are measured from a single forward pass through the `MaskedGaussianProxy` that `make_render_proxy()` builds for the current ablation mode:

- `N_active` = `proxy.get_xyz.shape[0]` (hot ratio × N under Mode ≥ 2)
- `N_visible` = `(render_pkg["radii"] > 0).sum()`
- `tile_touches` = `N_visible × π × (avg_r/16)²`  (tile_size=16 from `diff_gaussian_rasterization/config.h`)
- `H`, `W` from `render_pkg["render"].shape[-2:]`
- `sh_degree` from `gaussians.active_sh_degree`

**Why this replaces `set_render_cycles_from_fps` as the FPS headline.** The four baselines' GPU-side `render_fn` differ in SH degree, Gaussian scale distribution, kernel-launch overhead and AMP / TF32 flags. Back-calculating `render_cycles = clock_hz / gpu_fps` folded all those simulator-host artefacts into the ASIC number, making Mode-0 FPS heterogeneous across baselines (e.g. 4DGC 42 FPS vs 3DGStream 143 FPS on the same 280 k-Gaussian scene) for reasons unrelated to the GLASS architecture. Modelling from EWA-splatting first principles yields ASIC FPS that depends only on the masked-proxy geometry the ASIC would see, and is directly comparable across baselines.

### `set_render_cycles_from_fps(fps)` — legacy GPU FPS fallback

Back-calculates `render_cycles = clock_hz / fps` from a measured GPU wall-clock FPS. Retained for backwards compatibility with ablation runs that predate `add_render_inference`, and for any path without render-stat access. Not used on the main FPS measurement path any more — every runner now calls `pipeline.measure_asic_render_fps(...)`, which captures stats and invokes `add_render_inference` internally, co-measuring `render_fps_gpu_measured` as a diagnostic.

### `add_mlp_training(n_params, n_iters)` — generic backward + Adam MLP loop

Used for baselines (QUEEN, HiCoM) whose network params don't need the specialised NTC split. `n_params` × `adam_ops_per_float` × `n_iters`.

---

## From cycles to wall-clock — `summary()`

```python
gradient_pipeline = max(puoe, ucn, ice, ilsf_update)      # streaming 3-stage pipeline
post_render        = max(ilsf_render, salience)            # parallel units
asic_overhead      = gradient_pipeline + post_render
total              = asic_overhead + optimizer + render
commit_cycles      = gradient_pipeline + optimizer

asic_commit_time_s = commit_cycles / clock_hz              # commit-path only
# asic_train_time_s is assembled per-baseline by result_schema.build_result
# (see below); the old asic_time_s_1ghz = total/clock_hz field was dropped
# because it silently excluded ntc_train cycles and was easy to misread.
```

**Why `max()` rather than `sum()` for the pipeline:** PUOE, UCN, and ICE form a 3-stage streaming pipeline reading the same descriptor stream on different banks. `ilsf_update` reads the same stream on yet another bus — also absorbed.

**`ilsf_sort` missing:** hidden by the concurrent ICE pass. The cycles are recorded but never included in a sum.

**`render` meaning:** set once per frame via `add_render_inference(...)` from first-principles EWA-splatting phases 1-3, using the masked proxy's geometry under the current ablation mode. The headline `render_fps = clock_hz / render_cycles` is therefore a pure ASIC model number. `render_fps_gpu_measured` is also stored (wall-clock on the same proxy) for auditing; it's never used for cycle accounting.

**`asic_commit_time_s` vs `asic_train_time_s`:** use `commit` for progressive Mode 0 → 1 → 2 comparisons because NTC cycles are identical across modes and would mask differences. Use `train` as the headline absolute ASIC training time for cross-baseline comparisons.

---

## Emitted JSON keys from `summary()`

```
asic_ntc_train_render_cycles     # per-iter forward-render + photometric loss × n_iters (3DGStream / 4DGC / HiCoM / QUEEN)
asic_ntc_train_cycles            # NTC / MEM / MLP training loop (fwd+bwd+Adam) × n_iters
asic_ntc_cycles                  # NTC forward inference (per frame)
asic_puoe_cycles
asic_ucn_cycles
asic_ice_cycles
asic_ilsf_cycles                 # ilsf_update + ilsf_render (combined)
asic_salience_cycles
asic_optimizer_cycles
asic_render_cycles               # add_render_inference (phases 1-3, modelled)
asic_gradient_pipeline_cycles    # max(puoe, ucn, ice, ilsf_update)
asic_post_render_cycles          # max(ilsf_render, salience)
asic_total_cycles                # overhead + optimizer + render
asic_commit_time_s               # (pipeline + optimizer) / clock_hz
```

Each ablation runner assembles `asic_train_time_s` from a
baseline-specific subset of these buckets and passes the result through
`simulator.ablation.result_schema.build_result`, which writes the
canonical JSON schema.  Current per-baseline formulas:

| Baseline   | `asic_train_time_s` × `clock_hz` |
|------------|----------------------------------|
| 3DGStream  | `ntc_train_render + ntc_train + ntc + gradient_pipeline + optimizer` |
| 4DGC       | `ntc_train_render + ntc_train + ntc + gradient_pipeline + optimizer` |
| QUEEN      | `ntc_train_render + gradient_pipeline + optimizer` (no NTC inference path) |
| HiCoM      | `ntc_train_render + gradient_pipeline + optimizer` |

---

## Verifying a number

1. From a `mode_*.json`, pick any `asic_<unit>_cycles` value.
2. Look up the corresponding `add_<unit>` method above.
3. From the `mode_*_run/` output directory or your own trace, collect the `(n, d, ...)` arguments the pipeline would have passed per-frame per-step.
4. Sum the formula by hand (or in a notebook).
5. It should match to the single cycle.

If it doesn't, something is miscounted — see [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) §"Parity-test rules" for what to do next.
