# Adapting a New Streamable Gaussian Baseline

This is the hands-on checklist distilled from integrating **QUEEN**, **3DGStream**, and **HiCoM**. Work through the steps in order; at the end of each step you should be able to run a targeted sanity check before moving on.

---

## Prerequisite mental model

Before writing any code, internalise three invariants the simulator relies on:

1. **The pipeline never imports your baseline.** Everything baseline-specific lives in *exactly one file*: `simulator/operators/op_<name>.py` (≈ 100–350 lines, see existing ops for reference).
2. **Per-optimizer-step hook is the main entry.** `pipeline.pre_optimizer_step(optimizer, struct_changed=…)` must be called *before* the real `optimizer.step()`. Everything else (traffic, cycle accounting, cold-mask, salience) is triggered from there.
3. **`optimizer.param_groups` drives descriptor capture.** Make sure each param group has a stable `name` string that `AttrClass.from_group_name()` recognises (`xyz`, `rot…`, `scal…`, `opac…`, `f_dc`/`f_rest`/`feat…`/`sh_…` → Gaussian; everything else → NETWORK).

---

## Step 1 — Inventory the baseline

Before writing code, answer these on paper:

1. **Where is the per-frame training loop?** Pinpoint the function that iterates frames and the function that runs one optimizer step. For the existing baselines:
   - QUEEN: `baselines/QUEEN/train.py::training()` outer frame loop; `gaussians.optimizer.step()` inner.
   - 3DGStream: `baselines/3DGStream/train.py::training_one_frame()`; **two** optimizers (`ntc_optimizer` stage-1, `gaussians.optimizer` stage-2) + `update_by_ntc()` between them.
   - HiCoM: libgs `Trainer.frame_training_loop()` + `Module.post_training_step()`.
2. **List every param-group name** appearing in `optimizer.param_groups`. Separate them into:
   - Per-Gaussian (shape `[N, …]` where N = number of Gaussians) → `gaussian_attr_groups`.
   - Network-level (MLP / latent decoder / hash-grid / gate / NTC) → `network_attr_groups`.
3. **Where do structural changes happen?** Densification / pruning / spawning. You will need the list of newly-added Gaussian IDs so the simulator's salience unit can mark them dirty (Mode 3). Look for signatures like `densify_and_prune`, `adding_and_prune`, `clone_and_split`.
4. **Is there a non-Adam update path?** 3DGStream has NTC: before Stage 2 a neural network writes fresh attribute values in bulk, bypassing `optimizer.step()`. Those updates must flow through `pipeline.ingest_ntc_deltas()` (see Step 4). HiCoM's deformation MLP does its updates through a normal optimizer — no special path needed.

**Sanity check:** you should be able to draw a flow chart that labels, for each code line in the frame loop, which Stage (0 / 1 / 2 / 3 / 4) of the ASIC it corresponds to.

---

## Step 2 — Write the `Op<Name>` operator slot

Start from the simplest live example, `simulator/operators/op_queen.py` (≈ 115 LOC). Copy it, rename the class, and fill in the five required pieces.

### 2.1 Class-level group declarations

```python
class OpMyBaseline:
    gaussian_attr_groups: list[str] = [
        "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
        # add baseline-specific Gaussian groups here, e.g. "deform_xyz"
    ]
    network_attr_groups: list[str] = [
        # every non-per-Gaussian name that appears in optimizer.param_groups
        "hash", "deform", …
    ]
```

If your baseline has added-Gaussian groups with a prefix (as 3DGStream does — `added_xyz`, `added_f_rest`, …), list all of them. `from_group_name()` resolves the `AttrClass` via substring match, so `added_xyz` → `AttrClass.XYZ` and everything still works.

### 2.2 Three required methods

```python
def install_hooks(self, pipeline, *, gaussians, optimizer, **ctx):
    """Patch the training loop so pipeline hooks fire at the right times."""

def uninstall_hooks(self) -> None:
    """Restore every callable you patched."""

def n_gaussians(self) -> int: ...
def get_gaussians(self): ...   # returns an object with get_xyz / get_opacity / …
```

### 2.3 Minimum hook: `optimizer.step` wrap

```python
orig_step = optimizer.step
def _step(closure=None):
    pipeline.pre_optimizer_step(optimizer)
    return orig_step(closure)
optimizer.step   = _step
self._orig_step  = orig_step
```

The same pattern scales to multiple optimizers (HiCoM wraps 2-3; 3DGStream wraps 2) — just keep one `self._orig_<name>` per patched attribute so `uninstall_hooks` can restore them.

### 2.4 Structural-change capture

If densification happens, wrap the function that creates new Gaussians and collect the new IDs into `self._pending_struct_ids`. Pass them on the next optimizer hook:

```python
def _step(closure=None):
    pipeline.pre_optimizer_step(
        optimizer, struct_changed=self._pending_struct_ids
    )
    self._pending_struct_ids = []
    return orig_step(closure)
```

See `Op3DGStream._adding_and_prune` for the reference implementation.

### 2.5 Test before moving on

```python
from simulator.config    import ASICConfig
from simulator.pipeline  import GLASSPipeline
from simulator.operators.op_mybaseline import OpMyBaseline

cfg  = ASICConfig(mode=0)         # Mode 0 = capture only, safest first run
slot = OpMyBaseline()
pipe = GLASSPipeline(cfg, slot)
slot.install_hooks(pipe, gaussians=<your_g>, optimizer=<your_opt>)
# ... run one frame of your baseline's training ...
print(pipe._cycles.summary())
slot.uninstall_hooks()
```

If `asic_puoe_cycles > 0` and no exceptions were raised, the descriptor path works.

---

## Step 3 — Get render masking working (Mode ≥ 2)

For Mode 2+ the ASIC only renders the hot tier. The simulator achieves this by wrapping the Gaussian model in a `MaskedGaussianProxy` (`simulator/core/proxy.py`) that shadows `get_xyz / get_opacity / …` with filtered tensors.

Two things you may need to do:

1. **Ensure `get_gaussians()` returns something proxy-compatible** — standard 3DGS-style `GaussianModel` already works. For hybrid models that have separate base + incremental Gaussian sets (HiCoM), wrap them in a tiny accessor that concatenates along dim 0 (see `_MergedGaussianAccessor` in `op_hicom.py`).
2. **If your baseline uses a custom `render()` signature** (e.g. QUEEN's `render_mask(gaussians, cam, pipe, bg, mask=None)`), the ablation runner calls it with the proxy as `gaussians` and `mask=None`. The proxy already represents the filtered set.

---

## Step 4 — Non-Adam bulk update path (optional — only if your baseline has one)

Use this step if attribute updates ever bypass `optimizer.step()`. 3DGStream's NTC is the canonical example.

Wrap the function that writes new attribute values in bulk. Snapshot the tensors before, compute per-Gaussian deltas after, and feed them to `pipeline.ingest_ntc_deltas()`:

```python
def install_bulk_hook(self, gaussians, *, n_iters_max: int):
    pipeline    = self._pipeline
    orig_update = gaussians.apply_my_network   # <- the bulk writer

    # Measure architecture parameters (MLP MACs, hash params, encoding dim, …)
    # from the live model ONCE — no magic numbers.
    # See Op3DGStream.install_ntc_delta_hook for the exact tcnn introspection.

    def _hooked_update():
        old = {k: getattr(gaussians, k).clone()
               for k in ("_xyz", "_rotation", "_features_dc", "_features_rest")}
        orig_update()
        parts = []
        for k in old:
            nv, ov = getattr(gaussians, k), old[k]
            if nv.shape == ov.shape and nv.shape[0] > 0:
                parts.append((nv - ov).reshape(nv.shape[0], -1))
        if parts:
            with torch.no_grad():
                delta   = torch.cat(parts, dim=1)
                norms   = delta.norm(dim=1)
                total_d = delta.shape[1]
            pipeline.ingest_ntc_deltas(
                norms, total_d,
                ntc_macs_per_g=<measured>,
                n_hash_params =<measured>,
                enc_dim       =<measured>,
                ntc_output_dim=<measured>,
                n_ntc_iters_max=n_iters_max,
            )

    gaussians.apply_my_network = _hooked_update
    self._orig_update_bulk     = orig_update
```

Every argument to `ingest_ntc_deltas()` comes from **measurement** of your live model, never a constant. See `Op3DGStream.install_ntc_delta_hook` for the reference.

---

## Step 5 — Init-model mechanism

Streamable baselines start from a **pre-trained Frame-1 (or Frame-0) model** and only do incremental training from Frame-2 onward. That initial checkpoint is where the interesting "live-state" history begins.

Pattern (QUEEN reference):

1. **Store the checkpoint path** on disk at a predictable location (we use `data/<scene>/<baseline>_init/`; see `.gitignore` — the weights themselves are never committed).
2. **Add a field to `DatasetConfig`** (`simulator/datasets.py`):
   ```python
   init_model_mybaseline: str = ""    # path to Frame-1 output dir or PLY
   ```
3. **Load the checkpoint before incremental training starts.** Typically the baseline's own train.py has a `load_iteration=<N>` flag; if not, you'll need to patch a `load_state_dict` call.
4. **Skip Frame-1 training.** Add a `--start_frame_idx` CLI arg that (a) advances the data loader by `start_frame_idx - 1` frames and (b) bypasses the full-epochs frame-1 training block.
5. **Populate the history buffers the incremental step assumes exist.** QUEEN keeps per-Gaussian `prev_atts` / `prev_latents` to compute residuals; if the init path skips those, the very first incremental frame will have an uninitialised buffer.  See `GaussianModel.init_prev_atts_from_current()` for the pattern.

3DGStream's equivalent is simpler: its init_model is a directory written by a prior train.py run; `run_3dgstream.py` just points its `--start_checkpoint` at that dir. No code patch required.

### Sanity check

Run your baseline for Mode 0 with N=3 frames using the init model. You should see:
- Frame 1: skipped / instantaneous (checkpoint loaded)
- Frame 2-3: full incremental training (seconds, not minutes)
- `n_frames_trained = 3` in the emitted JSON

If Frame 1 still takes a long time, the init-model load is not short-circuiting.

---

## Step 6 — Ablation runner

Copy `simulator/ablation/run_queen.py` (shipping in PR #4) as a template and adapt. The runner's job is:

1. Parse args (`--mode`, `--n-frames`, `--dataset-config`, `--scene`).
2. Load `DatasetConfig` + build `ASICConfig(mode=args.mode)`.
3. Instantiate `slot = OpMyBaseline()`, `pipeline = GLASSPipeline(cfg, slot)`.
4. Call baseline's `training(...)` function. Your slot's `install_hooks` fires; the pipeline now runs.
5. After training:
   - Measure FPS: `pipeline.measure_fps(render_fn, cameras, pipe, bg)`.
   - Measure quality: `MetricsCollector.measure_quality(renders, gts)`.
   - Measure storage: `.measure_storage(output_path)`.
   - Write `results/GLASS_ablation/<scene>/<BASELINE>/mode_<N>.json`.

Keep the JSON schema identical to other baselines (run `jq keys results/.../QUEEN/mode_0.json` to see it); the cross-baseline comparison scripts rely on it.

---

## Step 7 — Sanity checklist before opening a PR

After your first 3-frame run completes in Mode 0, open the resulting JSON and check:

| Field | Mode 0 | Mode 1 | Mode ≥ 2 | Notes |
|-------|--------|--------|----------|-------|
| `asic_puoe_cycles`       | > 0 | > 0 | > 0  | always non-zero if gradients flow |
| `asic_ucn_cycles`        | = 0 | > 0 | > 0  | UCN only runs mode ≥ 1 |
| `asic_ice_cycles`        | = 0 | > 0 | > 0  | ICE only runs mode ≥ 1 |
| `asic_optimizer_cycles`  | > 0 | > 0 | > 0  | — |
| `asic_salience_cycles`   | = 0 | = 0 | = 0/>0 | non-zero only mode ≥ 3 + structural changes |
| `frame_suppressed_bytes` | = 0 | > 0 | > 0  | UCN's bytes saved |
| `frame_unique_targets`   | ≈ N | ≈ N | ≈ N  | total Gaussians touched this frame |
| `psnr` / `ssim` / `lpips`| all populated | all populated | all populated | |

If any of the "> 0" entries are zero, revisit the corresponding stage.

---

## Common pitfalls (from the QUEEN / 3DGStream integration)

### 1. Densification changes N mid-frame
Multiple optimizer steps can happen while `N` varies. The pipeline tolerates this (scale tensors are per-step and broadcast-safe), but your `n_gaussians()` implementation must always return the **current** count, and structural IDs must be appended to `_pending_struct_ids` only when Gaussians are *added*, not pruned.

### 2. `struct_changed is None` vs `[]`
The fast paths in `pipeline.py` require `struct_changed is None` as a gating condition. Pass `[]` (empty list) on steps where nothing changed — do **not** pass `None` except on steps where you truly don't know. Passing `None` forces the slow path.

### 3. Frame-level vs iteration-level UCN
`cfg.frame_level_ucn=True` accumulates per-iteration norms into a single CWB entry per frame. This is what you want for baselines whose optimizer converges within a frame (QUEEN). Leave it `False` for baselines where cross-frame stability is the suppression signal (3DGStream/NTC). Setting it wrong manifests as either zero UCN savings or catastrophic quality loss.

### 4. `reset_cwb_each_frame`
Set this to `True` for baselines that re-initialise their optimizer state every frame (QUEEN), otherwise UCN sees stale history. Leave `False` for baselines where the optimizer carries state (3DGStream Stage 2 Adam).

### 5. `adaptive_ucn_threshold` + fast path (updated in v2 — adaptive now works)
Since the v2 UCN generalisation the vectorised fast path **does support** `adaptive_ucn_threshold`. The gate is now `cfg.frame_level_ucn == True` **and** `struct_changed is None`; the adaptive threshold is read per-frame through `get_effective_threshold()` and broadcast into the comparator. If you are porting code written against the old gate, drop any `adaptive_ucn_threshold=False` override you put in just to enable the fast path.

### 6. `optimizer.param_groups[*]["name"]` missing
Some baselines build the optimizer with no `name` in each group dict. The pipeline treats unnamed groups as network-level (safest default) but you lose per-attribute taxonomy. Patch your baseline to set `name` explicitly — this is always a one-line fix in `gaussian_model.py`.

### 7. FPS measurement crashes with tier mask — or, worse, *silently* ignores it
Two failure modes:
- **(a)** `render()` reads `gaussians._xyz` (raw backing tensor) instead of `gaussians.get_xyz` — `MaskedGaussianProxy` doesn't intercept and the mask is silently bypassed.
- **(b)** Your ablation runner's `_render_fn` accepts a `g` parameter (the proxy) but then calls `module.forward(cam)` internally — the module holds its own reference to the unmasked `self.renderer.gaussians` and the `g` argument never actually reaches the renderer. This was the HiCoM Mode 2/3 bug that made render-FPS look unchanged even though the fabric mask was active.

The canonical fix for (b) is a try/finally swap inside `_render_fn`:

```python
def _render_fn(cam, g, pipe_, bg):
    renderer = module.renderer
    saved    = renderer.gaussians
    renderer.gaussians = g            # route the proxy through the real render path
    try:
        return module.forward(cam.to(module.device))
    finally:
        renderer.gaussians = saved
```

Verify fix: Mode 2 render-FPS must jump by roughly `n_total / (asc + rsb)` relative to Mode 0/1. If it is flat, one of (a)/(b) is still wired wrong.

### 8. Dense-iter baseline + auto-structural UCN detector — explicit `frame_level_ucn=True` is mandatory
`cfg.auto_structural_ucn` (default on) *decides* at frame-2 whether to run frame-level UCN. **Frame 1 always runs with whatever `cfg.frame_level_ucn` was at init time.** For dense-iter baselines (QUEEN: 150 Adam steps × 188 K Gaussians × 6 attrs ≈ 170 M Python ops/frame) that means frame 1 falls through to the dict (slow) path and saturates a single CPU core for tens of seconds while the GPU sits idle. Symptom: `nvidia-smi` shows 0% GPU, `top` shows one core at 100%, wall time per frame ≫ expected.

**Fix**: in your `run_<baseline>.py`, pass `frame_level_ucn=True` explicitly to `pipeline.init(...)`. The auto-detector respects user overrides and will not flip it back (see `_user_frame_level_ucn` in `GLASSPipeline`).

Rule of thumb: if your baseline runs > 20 optimizer steps per frame on > 100 K Gaussians, set `frame_level_ucn=True` at construction time. Treat the auto-detector as a safety net for light baselines, not a primary API.

### 9. Path 2 adaptive iteration scheduler — read from `current_frame_suppression_ratio()`, not `_prev_suppression_ratio`
v2 introduced a unified per-frame suppression-ratio signal, `pipeline.current_frame_suppression_ratio()`, that is populated by **all three** UCN paths (NTC bulk-delta, dict slow path, and vectorised fast path). The legacy `_prev_suppression_ratio` attribute is only updated by `begin_frame()` and by NTC ingest; if you scheduled iterations off it directly you'd miss QUEEN/HiCoM's signals entirely.

If you want Path 2–style early-exit iteration scheduling in your runner:

```python
base_iters = trainer.config.num_incr_steps          # or equivalent
prev_ratio = pipeline.current_frame_suppression_ratio()
adaptive_steps = max(
    cfg.min_ntc_iters_floor,                        # hard floor from ASICConfig
    int(round(base_iters * (1.0 - prev_ratio))),
)
trainer.config.num_incr_steps = adaptive_steps
```

See `simulator/ablation/run_hicom.py::_timed_training_loop` for the reference. Skip this step if your baseline's iteration budget is monolithically coupled to other schedules (SH-degree ramp, densification cadence, LR decay) — QUEEN is such a case.

### 10. UCN → Fabric demotion — `record_ucn_suppression` must be wired on every path
When Path 1 (`cfg.ucn_drive_fabric_demotion=True`, default) is enabled, the pipeline routes `grad_scale == 0` descriptors to `fabric.record_ucn_suppression(gid)` instead of `fabric.record_update(gid, norm)`. All three descriptor paths (NTC, dict, fast) in `pipeline.py` already do this. If you add a fourth path (uncommon), mirror the same `if d.grad_scale <= 0.0: fabric.record_ucn_suppression(...) else: fabric.record_update(...)` branch; otherwise your baseline will never demote stable Gaussians to CSS and Path 1's render-cycle savings evaporate.

---

## Where to get help

- Read the docstring at the top of `simulator/pipeline.py` — it lists every hook and when it fires.
- `docs/architecture.md` explains what the simulator expects at each stage.
- `docs/cycle_accounting.md` documents every `add_*()` on `CycleAccumulator` so you can check your own numbers.
- Open an issue with your `Op<Name>.install_hooks` diff and a 3-frame JSON; that is usually enough to reproduce integration bugs.
