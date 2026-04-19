# Contributing

Thanks for your interest. This simulator is research infrastructure, not a product — the standards below exist to keep the scientific conclusions reproducible and the code honest.

---

## Ground rules

### 1. No magic numbers

Every hardware constant lives in `simulator/config.py::ASICSpec` with a citation for its default. Every baseline-architecture constant (NTC hash params, MLP MACs, encoding dim, …) is **measured from the live model** at hook install time, never hard-coded.

If you find yourself wanting to write `0.7` or `5.5e3` or `int(1.3 * n)` anywhere in `simulator/core/` or `simulator/pipeline.py`, stop:

- If it's a hardware quantity → add a field to `ASICSpec` with a citation.
- If it's a scheduling/threshold knob → add a field to `ASICConfig` (or a more specific config dataclass) with a default set from measurement or from an ablation sweep whose results you commit to a comment.
- If it's a baseline-specific measurement → read it from the model/optimizer at setup time.

Reviewers will reject a PR that introduces a literal without one of the three justifications above.

### 2. Bit-exact fast paths

`simulator/pipeline.py` contains a "slow path" (per-descriptor Python loop) and a "fast path" (tensorised). For any given `(cfg, stream)` the **two must produce identical statistics** — cycle counts, byte counts, per-param gradient scales. Floating-point accumulators on the fast path use `torch.float64` where the slow path used NumPy defaults (see `BatchedCorrelationWindowBuffer._batched_frame_sum`).

Before opening a PR that touches either path:

1. Temporarily force the slow path by tightening the dispatcher guard (`self.cfg.mode == 1 → False`).
2. Run `python simulator/ablation/run_queen.py --mode 1 --n-frames 3`.
3. Save the resulting JSON aside.
4. Restore the dispatcher, rerun.
5. `diff` the two JSONs. The only fields allowed to differ are wall-clock timings (`avg_train_time_s`, `asic_train_time_s` is allowed to be *faster* but its cycle components must match).

If any cycle count or byte count differs by one, a regression has been introduced — fix it before asking for review.

### 3. Model-agnostic core

Code under `simulator/core/` and `simulator/pipeline.py` must not `import` anything from `baselines/` or from a specific baseline's operator file. Everything baseline-specific goes in `simulator/operators/op_<name>.py`. If your change requires touching the core, stop and describe the need in an issue first — we've successfully integrated 4 baselines with this invariant held.

### 4. Every `add_*` on `CycleAccumulator` has a docstring formula

Before merging a new `add_<unit>()` method, verify:

- The docstring includes the exact formula it implements, with every term parenthesised and labelled.
- Every coefficient in that formula appears either in `ASICSpec` or as a byte-accurate constant (4 / 8 / 16 B, etc.).
- The formula is dimensionally consistent (cycles, not bytes or ops).
- The method returns the cycles it added, so callers that need the value can use it.

See `docs/cycle_accounting.md` for the reference style.

---

## Development workflow

### Environments

Each baseline has its own conda env — they have conflicting dependency pins (different PyTorch, different diff-gaussian-rasterization forks, tcnn vs no-tcnn, …). Typical names on a dev box:

- `queen-cu121`
- `3dgstream`
- `hicom`
- `swift4d`

The simulator itself runs in any of them because it only depends on `torch` and standard-library modules.

### Reproducing the ablation headline numbers

```bash
# from repo root, inside the matching conda env:
python simulator/ablation/run_queen.py --mode 0 --n-frames 3
python simulator/ablation/run_queen.py --mode 1 --n-frames 3
python simulator/ablation/run_queen.py --mode 2 --n-frames 3
python simulator/ablation/run_queen.py --mode 3 --n-frames 3
```

Results land in `results/GLASS_ablation/<scene>/QUEEN/mode_{0,1,2,3}.json`. Compare against the table in `docs/ablation_modes.md` — a fresh run on the same machine should reproduce it within ±2% wall-clock, ±0.1 dB PSNR, and cycle counts bit-exactly.

### Running the parity test (if you bring it back)

`simulator/tests/test_mode1_fastpath_parity.py` was removed before the initial public commit to keep the repo tidy. If you're touching the fast path, resurrect it locally (the template is in the git history) or write a new one specific to your change; the key comparisons it must make are listed in §"Bit-exact fast paths" above.

---

## Pull-request etiquette

### One concern per PR

Mirror the PR breakdown used to seed this repo:

1. Baseline submodules only.
2. Simulator core only.
3. Documentation only.
4. Ablation runners only.
5. Benchmark harness only.

If your change spans categories, split it.

### Commit message

Describe **the why**, not the what. A good message for a pipeline change:

> Mode-2 ICE path: fuse scale + cold-zero into a single RMW.
>
> The slow path counted `add_ice_scale(n, d) + add_ice_cold_mask(n_cold, d)` —
> two passes with two reads. ASIC HW does this in one RMW. Replacing both
> with `add_ice_merged(n_hot, n_cold, d)` drops `asic_ice_cycles` ~33% on
> the flame_steak QUEEN run without changing any byte totals.

The header line fits in < 72 chars. The body reads like a design note, not a changelog.

### What the CI (= you, manually, until we add it) must check

- `python -m simulator.ablation.run_queen --mode 0 --n-frames 2` exits 0.
- `python -m simulator.ablation.run_queen --mode 1 --n-frames 2` exits 0.
- Cycle-count bit-parity for any `(cfg, stream)` you touched.
- No new literal numbers under `simulator/core/` or `simulator/pipeline.py`.
- Linter is happy (`ruff check simulator/`).
- Any public API you changed has an updated docstring.

---

## Code style

- `from __future__ import annotations` at the top of every new `.py` under `simulator/`.
- Type-annotate every public function signature (`from typing import …` sparingly — prefer built-in `list[…] | None` syntax).
- Prefer pure functions; if you need state, use a dataclass.
- No inline comments that restate the code. Comments explain **intent** and **invariants** — especially invariants the fast path depends on.
- Keep `simulator/pipeline.py` readable despite its length: every new helper goes in `simulator/core/`, never inlined.

---

## Reporting an issue

Please include:

1. The exact `git log -1 --oneline` of the repo you ran from.
2. The conda env name and `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
3. The full command line.
4. The first 100 lines and last 200 lines of stdout/stderr.
5. The generated `results/GLASS_ablation/…/mode_*.json` if the run finished.
6. What you expected and what you observed.

Bugs with no reproduction are difficult to chase — one carefully-prepared issue is worth ten hurried ones.

---

## Adding a new baseline

See [`docs/adapting_new_baseline.md`](adapting_new_baseline.md) for the seven-step checklist. Open a draft PR as soon as Step 5 ("sanity check") passes; we'll help review the remaining steps.

---

## License

By contributing you agree that your code is released under the same license as the rest of the repository (see `LICENSE` at the repo root).
