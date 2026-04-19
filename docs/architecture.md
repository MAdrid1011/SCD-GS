# GLASS Architecture — Stage 0 → 4 Data Flow

This doc explains what happens in the ASIC for **one optimizer step** of incremental training, and how the simulator mirrors it. Read this before [`adapting_new_baseline.md`](adapting_new_baseline.md).

---

## Big picture

```
                 ┌──────────────────────────────────────────────────────┐
    .backward()  │                                                      │
    ────────────▶│ Stage 0: PUOE — per-Gaussian ‖∇‖ capture            │
                 │        (simulator/pipeline.py: _capture_stream,      │
                 │         fast paths: _pre_optimizer_step_mode*_fast)  │
                 └──────────────────────────────┬───────────────────────┘
                                                │ UpdateStream (descriptors)
                                                ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ Stage 1: UCN — Update-Stream Consolidation           │
                 │        (simulator/core/consolidation.py)             │
                 │  • Correlation Window Buffer (CWB) per (gid, attr)   │
                 │  • cos-similarity proxy → grad_scale ∈ {0, 1}        │
                 │  • mode ≥ 1                                          │
                 └──────────────────────────────┬───────────────────────┘
                                                │ per-gid grad_scale
                                                ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ Stage 2: ICE + ILSF                                  │
                 │        (pipeline.py: _apply_consolidation,           │
                 │         _apply_cold_mask)                            │
                 │  • Apply scale × grad; in mode ≥ 2 merged with       │
                 │    cold-mask zeroing (single RMW pass)               │
                 │  • Live-State Fabric decides which Gaussians are     │
                 │    ASC (hot) / RSB (warm) / CSS (cold)               │
                 └──────────────────────────────┬───────────────────────┘
                                                ▼
                 ┌──────────────────────────────────────────────────────┐
                 │          optimizer.step()   ← standard Adam          │
                 └──────────────────────────────┬───────────────────────┘
                                                │ render call
                                                ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ Stage 3: Render via MaskedGaussianProxy              │
                 │        (simulator/core/proxy.py)                     │
                 │  • Render mask = ASC ∪ (RSB ∩ recent_delta > thr)    │
                 │  • mode ≥ 2                                          │
                 └──────────────────────────────┬───────────────────────┘
                                                ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ Stage 4: Salience — PROMOTE / FULL / SKIP            │
                 │        (simulator/core/salience_issue.py)            │
                 │  • mode ≥ 3; tracks variance for stability           │
                 │    gate; overlays on tier mask                       │
                 └──────────────────────────────────────────────────────┘
```

All cycle accounting flows through one `CycleAccumulator` (`simulator/core/cycle_model.py`) that the pipeline calls as it executes.

---

## The common currency: `UpdateDescriptor`

Every per-Gaussian update that the ASIC cares about is one of these, produced in Stage 0:

```python
@dataclass
class UpdateDescriptor:
    target_id:      int          # Gaussian index ∈ [0, N);  -1 for network params
    attr_class:     AttrClass    # XYZ | ROTATION | SCALING | OPACITY | FEATURES | NETWORK
    delta_norm:     float        # L2 norm of the gradient for this (gid, attr)
    version:        int          # monotonic frame counter
    grad_scale:     float        # filled in by UCN (Stage 1); default 1.0
    struct_changed: bool         # set by the operator on spawn/split/prune
```

Stage 1 writes `grad_scale`; Stage 2 multiplies it into `param.grad`; Stages 3/4 read `target_id` + `delta_norm`; the metrics layer reads `attr_class.delta_bytes()` for traffic totals.

The taxonomy is decided by `AttrClass.from_group_name()`:

```python
# simulator/core/update_descriptor.py
if   "xyz"  in name.lower(): return AttrClass.XYZ
elif "rot"  in name.lower(): return AttrClass.ROTATION
elif "scal" in name.lower(): return AttrClass.SCALING
elif "opac" in name.lower(): return AttrClass.OPACITY
elif any(t in name.lower() for t in ("f_dc","f_rest","feat","sh_")):
                             return AttrClass.FEATURES
else:                        return AttrClass.NETWORK   # decoder, gate, ntc, hash, deform, …
```

This is the **only** place where baseline-specific naming leaks into the core. If your baseline uses a new name (e.g. `"sh_offsets"`), either pick a tag that contains one of the existing substrings, or extend this function.

---

## Stage 0 — PUOE (Per-Unit Output Enqueue)

**File:** `simulator/pipeline.py` → `_capture_stream()` (reference) and `_pre_optimizer_step_mode*_fast()` (fast paths).

**What the hardware does:** for every per-Gaussian `param.grad` row, compute the L2 norm (one MAC+SQRT lane per row).

**What the simulator does:** walk `optimizer.param_groups`, separate Gaussian groups from network groups via the operator slot's `gaussian_attr_groups` / `network_attr_groups` lists, and either:
- push one `UpdateDescriptor` per (gid, attr) in the reference path, or
- compute per-row norms as a single `.norm(dim=1)` call in the fast paths, keeping the tensor on-device.

**Cycle entries:** `add_puoe(n, d)` for Gaussian rows; `add_puoe_network(n_params)` for the aggregate reduction of a network tensor.

---

## Stage 1 — UCN (Update-Stream Consolidation)

**File:** `simulator/core/consolidation.py`.

**Idea:** if a Gaussian's current-step norm looks just like its own recent history (cos-similarity proxy `min / max > merge_cos_threshold`), it is already converged → fold the update (scale 0). If it is below `min_delta_norm` it is pure noise → also zero. Otherwise keep at scale 1.

**State:** per-(gid, attr) **Correlation Window Buffer** (CWB) of the last `cwb_window` frames or iterations. Two equivalent implementations live side-by-side:

| | dict-based (reference) | tensor-based (fast) |
|-|-|-|
| API | `compute_scales(stream)` | `compute_scales_batched({attr: norms})` + `compute_scales_fallback_batched(attr, norms)` |
| Used by | all modes' slow path | Mode 1/2/3 fast paths in `pipeline.py` |
| Parity | — | Pushes through the SAME cwb_batched; per-row head pointers + float64 accumulators keep means bit-exact |

**Config knobs:** `cwb_window`, `merge_cos_threshold`, `min_delta_norm`, `adaptive_ucn_threshold`, `frame_level_ucn`, `reset_cwb_each_frame`. Every one is a dataclass field in `ASICConfig`.

**Cycle entry:** `add_ucn(n)` — one combined call per optimizer step over main + fallback descriptors. Calling `add_ucn` more than once per step would change the total because of the `ceil()` inside the formula (this is the critical parity invariant).

---

## Stage 2 — ICE (Incremental Commit Engine) + ILSF gating

**File:** `simulator/pipeline.py` → `_apply_consolidation()` and `_apply_cold_mask()`.

**Mode 1** applies UCN's per-gid-min scale tensor into `param.grad` and logs `add_ice_scale(n, d)` per main param group.

**Mode ≥ 2** additionally reads the Live-State Fabric render mask (`fabric.get_cold_mask()`), zeros grads for CSS (cold) Gaussians in the same read-modify-write pass, and logs a single `add_ice_merged(n_hot, n_cold, d)` that accounts for the fused scale + zero operation (saves ~33% memory BW vs two separate passes). The scalar `add_ice_scale` is **not** called in mode ≥ 2 — the parity invariant enforced by the fast path.

Fallback (`ldec_*` on QUEEN) tensors follow the same scale path but are **not** cold-masked (they are not per-Gaussian).

---

## Stage 3 — ILSF (Incremental Live-State Fabric)

**File:** `simulator/core/live_state_fabric.py`.

Per Gaussian state:
- `hotness[gid]`    — EMA of gradient-update events + render-visible events. Config: `hotness_alpha`, `hotness_update_reward`, `hotness_render_reward`.
- `recent_delta[gid]` — per-frame `max(delta_norm)` seen this frame. Reset at `begin_frame()`.

**Tiering:** `get_render_mask(N)` sorts by hotness once, marks the top `asb_capacity` as ASC (always rendered, always updated), the next `rsb_capacity` as RSB (rendered only if `recent_delta > rsb_delta_thresh`), and everything else CSS (cold).

**Cycle entries:** `add_ilsf_update(n_updated)` each step (per-event bookkeeping), `add_ilsf_sort(n)` once per frame, `add_ilsf_render(n_visible)` each render.

**Parallelism note** (`add_ilsf_sort` lives on a disjoint memory region): absorbed by the gradient pipeline's critical path — see `cycle_model.py::summary()`.

---

## Stage 4 — Salience-First Issue

**File:** `simulator/core/salience_issue.py`.

Over-arching goal: on top of the fabric's tier mask, further bucket each Gaussian into `PROMOTE` / `FULL` / `SKIP` based on an EMA salience score (`salience_ema_alpha`) with a stability gate (`stability_window`, `stability_var_max`). A minimum `full_path_fraction` of Gaussians is always `FULL` as a quality safeguard regardless of score.

Only `mode >= 3` consults the salience classifier when building the render mask.

**Cycle entry:** `add_salience(n)` per frame.

---

## Rendering with tier masks — `MaskedGaussianProxy`

**File:** `simulator/core/proxy.py`.

Rather than rewrite each baseline's `render()`, the pipeline wraps the Gaussian model with a proxy that exposes the standard `get_xyz / get_opacity / …` API but filters rows by the live render mask. All four baselines' render backends then "just work" on the proxy.

---

## Cumulative run state

The simulator keeps run-level counters in the `GLASSPipeline` instance:

| Field | Meaning |
|-------|---------|
| `_stream` | Reference-path descriptor stream for the current frame (reset in `begin_frame`) |
| `_cycles` | `CycleAccumulator` — cycle totals across the entire run |
| `_total_bytes_all` / `_suppressed_bytes_all` | run-level traffic accumulators |
| `_frame_total_bytes` / `_frame_suppressed_bytes` / `_frame_unique_targets` | per-frame (used by fast paths + `stream_stats()`) |
| `_prev_suppression_ratio` | feeds the adaptive NTC-iteration scheduler (mode ≥ 1) |
| `_prev_render_stats` | render package from last `post_render` — parameterises `add_render_training_iter` |

A run concludes with `_cycles.summary()` returning the critical-path breakdown — this is what the metrics collector emits into `mode_*.json`.
