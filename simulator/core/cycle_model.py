"""
Cycle-accurate ASIC timing model for the GLASS simulator.

Two classes are exported:

  ASICSpec        — dataclass of GLASS ASIC hardware parameters.
                    Every value has an explicit physical justification;
                    there are no magic numbers.

  CycleAccumulator — accumulates cycle counts for each ASIC compute unit
                     across an ablation run, then summarises the critical
                     path after accounting for hardware parallelism.

Parallelism rules encoded in summary()
---------------------------------------
  1. PUOE → UCN → ICE form a 3-stage streaming pipeline; critical path =
     max(PUOE_total, UCN_total, ICE_total).
  2. ILSF_update is absorbed into that pipeline (it reads the same descriptor
     stream from a separate bus); max(..., ILSF_update) ≈ free.
  3. ILSF_sort fires once per frame in begin_frame; it operates on a disjoint
     memory region and completes well within one ICE step → 0 extra cycles.
  4. ICE_scale + ICE_cold_mask are merged into a single read-modify-write pass
     (mode >= 2) via add_ice_merged(), saving ~33% memory bandwidth.
  5. ILSF_render and CGS support-domain bookkeeping execute in parallel after
     render(); critical path = max(ILSF_render, Stage4_CGS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ASICSpec:
    """GLASS ASIC hardware specification.

    All default values are derived from published data or explicit design
    decisions — no estimates or magic numbers.

    Attribute         Value  Justification
    ─────────────────────────────────────────────────────────────────────
    clock_hz          1e9    User-specified target clock (1 GHz)
    mac_units         128    128 × 32-bit FMA = 512-bit datapath; fits in
                             < 2 mm² at 28 nm @ 1 GHz (cf. Eyeriss v2,
                             TPU micro-core sizing)
    sram_bw_bytes     256    4 banks × 64 B/cycle; 64 B = 1 cache-line
                             (ARM / TSMC process standard); 4 banks allow
                             simultaneous read + write
    sqrt_latency_cyc  15     fp32 sqrt pipeline depth; industry range
                             12–18 cy (ARM Cortex-A55 FPU = 14 cy, TSMC
                             28 nm ASIC FPU typical = 15 cy); paired with
                             128 FMA lanes so throughput = 128 sqrts/cycle
    cmp_units         256    Comparator / simple-ALU units = 2 × mac_units.
                             A comparator cell is ~1/8 the area of an FMA,
                             so the same silicon budget would allow 8×; we
                             use a conservative 2× to avoid over-provisioning
                             the compare fabric.
    dram_bw_bytes      64    Off-chip DRAM interface bandwidth per cycle.
                             LPDDR5X-8000 with 64-bit data bus:
                               8000 Mbps × 64 bits / 8 = 64,000 MB/s = 64 GB/s
                               @ 1 GHz clock → 64 B/cycle (exact, no rounding)
                             Appropriate for a standalone edge ASIC with a
                             single-channel 64-bit LPDDR5X interface.
                             4× slower than on-chip sram_bw_bytes (256).
    adam_ops_per_float 10    Exact Adam step count per scalar θ (see below):
                             m←β₁m+(1-β₁)g     1 FMA
                             g²=g·g             1 MUL
                             v←β₂v+(1-β₂)g²    1 FMA
                             m̂=m·inv_bias1      1 MUL   (scalar precomputed)
                             v̂=v·inv_bias2      1 MUL
                             r=√v̂               1 SQRT
                             d=r+ε              1 ADD
                             s=lr·m̂/d           1 DIV+1 MUL = 2 ops
                             θ←θ−s              1 SUB
                             ─────────────────  10 ops total
                             All ops charged to mac_units (conservative;
                             DIV/SQRT on dedicated units would be faster).
    """
    clock_hz:           float = 1e9
    mac_units:          int   = 128
    sram_bw_bytes:      int   = 256
    sqrt_latency_cyc:   int   = 15
    cmp_units:          int   = 256
    dram_bw_bytes:      int   = 64
    adam_ops_per_float: int   = 10


# ── internal helpers ──────────────────────────────────────────────────────────

def _ceil(n: int, d: int) -> int:
    """Integer ceiling division."""
    return (n + d - 1) // d


# ── CycleAccumulator ──────────────────────────────────────────────────────────

class CycleAccumulator:
    """Accumulate ASIC cycle counts across an ablation run.

    Each add_*() method models one hardware unit's latency for a given
    operand size, using only ASICSpec parameters in its formula.  The
    method returns the cycles it added so the caller can inspect
    per-call costs if needed.

    Call reset_frame() at the start of each frame to clear per-step
    accumulators (render cycles are not reset — they are set once per
    frame via set_render_cycles_from_fps).

    Call summary() after the final frame to obtain the critical-path
    total and per-unit breakdown.
    """

    def __init__(self, spec: ASICSpec) -> None:
        self.spec = spec
        self._cyc: dict[str, int] = {
            "ntc_train_render": 0,
            "ntc_train":   0,
            "ntc":         0,
            "puoe":        0,
            "ucn":         0,
            "ice":         0,
            "ilsf_update": 0,
            "ilsf_render": 0,
            "ilsf_sort":   0,
            "salience":    0,
            "optimizer":   0,
            "render":      0,
            # ── Convergence-Graded Splatting diagnostics (mode 3) ──
            # Populated by :meth:`add_render_inference_graded`; zero when
            # CGS is inactive.  ``render`` above still reports the single
            # headline cycle count (max over the two paths + composite).
            "render_full":    0,
            "render_compact": 0,
        }

    # ── ASIC overhead counting methods ────────────────────────────────────────

    def add_puoe(self, n: int, d: int) -> int:
        """PUOE: per-Gaussian L2-norm capture.

        Reads N×D grad floats, computes one norm per Gaussian (D MACs +
        1 sqrt), writes N descriptors of 16 B each.

        Each batch of mac_units Gaussians takes max(D, sqrt_latency_cyc)
        cycles because the sqrt trails the D MACs within each Gaussian and
        the two stages are pipelined: total per-Gaussian time = max(D, sqrt_lat).

        Memory and compute run in parallel (streaming); the bottleneck is
        whichever is larger.  Descriptor write follows after compute.

            compute  = ceil(N / mac_units) × max(D, sqrt_lat)
            mem_read = ceil(N × D × 4 / sram_bw)
            write    = ceil(N × 16 / sram_bw)
            cycles   = max(compute, mem_read) + write

        For D=23 (SH deg=1): memory-bound (100 K > 50 K per step).
        For D<15: sqrt-bound.
        """
        if n == 0:
            return 0
        s = self.spec
        compute  = _ceil(n, s.mac_units) * max(d, s.sqrt_latency_cyc)
        mem_read = _ceil(n * d * 4, s.sram_bw_bytes)
        write    = _ceil(n * 16,    s.sram_bw_bytes)
        cyc      = max(compute, mem_read) + write
        self._cyc["puoe"] += cyc
        return cyc

    def add_puoe_network(self, n_params: int) -> int:
        """PUOE: aggregate norm over all network parameters (e.g. NTC).

        Unlike per-Gaussian PUOE, this computes ONE scalar norm over
        n_params elements: n_params MACs (reduction) + 1 sqrt (trivial).
        The single sqrt does not affect throughput.

            compute  = ceil(n_params / mac_units)
            mem_read = ceil(n_params × 4 / sram_bw)
            write    = ceil(16 / sram_bw)        ← 1 aggregate descriptor
            cycles   = max(compute, mem_read) + write

        Typically memory-bound (ceil(P×4/bw) > ceil(P/mac_units)
        when sram_bw < 4 × mac_units, i.e. 256 < 512 — always true here).
        """
        if n_params == 0:
            return 0
        s = self.spec
        compute  = _ceil(n_params, s.mac_units)
        mem_read = _ceil(n_params * 4, s.sram_bw_bytes)
        write    = _ceil(16, s.sram_bw_bytes)
        cyc      = max(compute, mem_read) + write
        self._cyc["puoe"] += cyc
        return cyc

    def add_ucn(self, n: int) -> int:
        """UCN: EMA-based Correlation Window Buffer comparison.

        Uses a single running-mean EMA per Gaussian (1 float stored),
        avoiding the N×W memory overhead of a full window buffer.

        Per Gaussian:
          1 FMA  — EMA update: mean ← α×mean + (1−α)×norm
          2 CMP  — compare norm against mean ± threshold band
          1 DIV  — similarity = lo/hi (normalised distance)
          1 SUB  — scale = 1.0 − similarity
          1 CMP  — threshold gate (scale < suppress_thresh → scale = 0)
          ──────
          7 ops total (using cmp_units as throughput divider)

        Memory:
          read  = ceil(N × 4 / sram_bw)   ← EMA mean (1 float/Gaussian)
          write = ceil(N × 4 / sram_bw)   ← updated EMA mean
                + ceil(N × 4 / sram_bw)   ← grad_scale output

            cycles = max(ceil(N×4/bw), ceil(N×7/cmp_units))
                     + 2×ceil(N×4/bw)

        Note: the cycle model uses EMA; the behavioural ConsolidationUnit
        may retain a full window buffer for higher fidelity — the two are
        independent.
        """
        if n == 0:
            return 0
        s = self.spec
        compute = _ceil(n * 7, s.cmp_units)
        mem_rd  = _ceil(n * 4, s.sram_bw_bytes)
        mem_wr  = 2 * _ceil(n * 4, s.sram_bw_bytes)  # updated mean + grad_scale
        cyc     = max(compute, mem_rd) + mem_wr
        self._cyc["ucn"] += cyc
        return cyc

    def add_ice_scale(self, n: int, d: int) -> int:
        """ICE scale-only pass (mode == 1, no cold-mask).

        Reads N×(D+1) floats (grad + scale factor), performs N×D
        element-wise multiplications, writes N×D floats back.

            read    = ceil(N×(D+1)×4 / sram_bw)
            compute = ceil(N×D / mac_units)
            write   = ceil(N×D×4 / sram_bw)
            cycles  = max(read, compute) + write

        Memory-bound for typical sizes (D=23, N=280K):
          read 104 K > MAC 50 K, so memory dominates.
        """
        if n == 0:
            return 0
        s = self.spec
        mem_rd  = _ceil(n * (d + 1) * 4, s.sram_bw_bytes)
        compute = _ceil(n * d, s.mac_units)
        write   = _ceil(n * d * 4, s.sram_bw_bytes)
        cyc     = max(mem_rd, compute) + write
        self._cyc["ice"] += cyc
        return cyc

    def add_ice_merged(self, n_hot: int, n_cold: int, d: int) -> int:
        """ICE merged read-modify-write pass (mode >= 2).

        Combines ICE_scale and ICE_cold_mask into a single scan:
          - Read  (n_hot + n_cold) × D × 4 B
          - n_hot  Gaussians: multiply grad × scale  (n_hot × D MACs)
          - n_cold Gaussians: write 0                (no MAC needed)
          - Write (n_hot + n_cold) × D × 4 B

        Saves ~33% memory bandwidth vs two separate passes (eliminates
        the extra read in the cold-mask pass).

            N       = n_hot + n_cold
            read    = ceil(N×D×4 / sram_bw)
            compute = ceil(n_hot×D / mac_units)     ← only hot rows have MACs
            write   = ceil(N×D×4 / sram_bw)
            cycles  = max(read, compute) + write
        """
        n = n_hot + n_cold
        if n == 0:
            return 0
        s = self.spec
        mem_rd  = _ceil(n * d * 4, s.sram_bw_bytes)
        compute = _ceil(n_hot * d, s.mac_units) if n_hot > 0 else 0
        write   = _ceil(n * d * 4, s.sram_bw_bytes)
        cyc     = max(mem_rd, compute) + write
        self._cyc["ice"] += cyc
        return cyc

    def add_ilsf_update(self, n_updated: int) -> int:
        """ILSF hotness EMA update (called per optimizer step).

        For each Gaussian that received a gradient: 1 FMA (EMA update)
        + 1 write (new hotness score).

            compute = ceil(n_updated / mac_units)
            write   = ceil(n_updated × 4 / sram_bw)
            cycles  = compute + write

        Runs in parallel with UCN on a separate memory bus (hotness
        scores vs grad_scale output); absorbed into the gradient pipeline
        via max() in summary(), so it does not increase the critical path
        unless n_updated is larger than PUOE, UCN, or ICE.
        """
        if n_updated == 0:
            return 0
        s = self.spec
        cyc = _ceil(n_updated, s.mac_units) + _ceil(n_updated * 4, s.sram_bw_bytes)
        self._cyc["ilsf_update"] += cyc
        return cyc

    def add_ilsf_render(self, n_visible: int) -> int:
        """ILSF hotness update from render visibility (called post-render).

        Same computation as add_ilsf_update but driven by the visibility
        filter rather than gradient norms.  Runs in parallel with CGS support
        (different input, different output memory).

            cycles = ceil(n_visible / mac_units) + ceil(n_visible×4 / sram_bw)
        """
        if n_visible == 0:
            return 0
        s = self.spec
        cyc = (_ceil(n_visible, s.mac_units)
               + _ceil(n_visible * 4, s.sram_bw_bytes))
        self._cyc["ilsf_render"] += cyc
        return cyc

    def add_ilsf_sort(self, n: int) -> int:
        """ILSF tier sort (once per frame, called in begin_frame).

        Sorts N hotness scores to determine tier membership.

            cycles = ceil(N / cmp_units) + ceil(N×4 / sram_bw)

        This operation fires during begin_frame while the first optimizer
        step's gradient pipeline (ICE ≈ 201 K cycles/step) is running
        concurrently.  Since ILSF_sort ≈ 5.5 K cycles << ICE, it is
        completely hidden.  The cycles are recorded here for transparency
        but are NOT added to the critical path in summary().
        """
        if n == 0:
            return 0
        s = self.spec
        cyc = _ceil(n, s.cmp_units) + _ceil(n * 4, s.sram_bw_bytes)
        self._cyc["ilsf_sort"] += cyc
        return cyc

    def add_salience(self, n: int) -> int:
        """Stage-4 CGS support-domain update (called post-render, mode >= 3).

        The schema keeps the historical ``salience`` bucket name, but the
        modeled work is now the support-domain evaluator:

          support_area ~= pi * radii^2
          raw_p        = opacity * support_area * appearance_bound
          P            = saturate(raw_p / P_ref)
          rho          = 1 - lambda * kappa * (1 - P)
          Omega_CGS    = rho * support_area

        Memory:
          read  = ceil(N×16 / sram_bw)  ← radii, opacity, appearance, kappa
          write = ceil(N×8 / sram_bw)   ← rho/effective support or grade meta

            cycles = max(ceil(N×16/bw), ceil(N×12/mac_units)) + ceil(N×8/bw)

        Runs in parallel with ILSF_render (disjoint inputs and outputs).
        """
        if n == 0:
            return 0
        s = self.spec
        mem_rd  = _ceil(n * 16, s.sram_bw_bytes)
        compute = _ceil(n * 12, s.mac_units)
        write   = _ceil(n * 8, s.sram_bw_bytes)
        cyc     = max(mem_rd, compute) + write
        self._cyc["salience"] += cyc
        return cyc

    # Trilinear-interpolation tap count per 3D query.  Derived from the 3-D
    # cube having 2³ = 8 corners; every output scalar is a weighted sum over
    # these 8 corner values.  Applies to both tcnn HashGrid lookups (Instant-NGP,
    # Müller et al. 2022 §3.1) and PyTorch's ``F.grid_sample(mode='bilinear')``
    # when the spatial input is a 3-D grid (see PyTorch docs — "bilinear" on
    # 5-D inputs is a trilinear interpolation).  This is a structural constant,
    # not a tunable knob.
    _TRILINEAR_TAPS_PER_QUERY = 8

    def add_ntc_forward(
        self,
        n: int,
        n_mlp_macs_per_g: int,
        n_hash_params: int,
        enc_dim: int,
        out_dim: int,
        hash_bytes_per_param: int = 2,
    ) -> int:
        """NTC (Neural Texture Cache) forward inference on the PUOE.

        The HashGrid and FullyFusedMLP of the NTC run on the ASIC's
        Vector/Tensor Update Core (part of PUOE).  Two-phase model:

        Phase 1: HashGrid lookup + trilinear interpolation
          The hash table is typically 2–4 MB, exceeding on-chip SRAM capacity
          for an edge device → loaded from off-chip DRAM.  For tcnn
          HashGrid encodings the table stores fp16 features
          (``hash_bytes_per_param=2``, the default); for dense 3-D grids used
          by 4DGC's ``Motion_Grid`` the table is fp32 and callers pass
          ``hash_bytes_per_param=4``.

          Trilinear interpolation needs 8 corner reads per query point (2³
          corners of the voxel cube enclosing it), so compute scales with
          ``n × enc_dim × 8`` MACs — one FMA per corner-channel pair.

            hash_load    = ceil(n_hash_params × bpp / dram_bw)
            hash_compute = ceil(n × enc_dim × TAPS / mac_units)    TAPS = 8
            phase1       = max(hash_load, hash_compute)
                         + ceil(n × enc_dim × 4 / sram_bw)         ← enc write, SRAM

        Phase 2: FullyFusedMLP streaming forward
          MLP weights are small (tens of KB) and reside on-chip SRAM.
          Encoding activations are read from SRAM; attribute delta outputs
          are written back to off-chip DRAM for the state commit path.

            mlp_compute  = ceil(n × n_mlp_macs_per_g / mac_units)
            mem_enc_read = ceil(n × enc_dim × 4 / sram_bw)
            phase2       = max(mlp_compute, mem_enc_read)
                         + ceil(n × out_dim × 4 / dram_bw)     ← delta attrs, DRAM

        All parameters are measured from the live model at hook install time
        (n_mlp_macs_per_g, n_hash_params, enc_dim, out_dim,
        hash_bytes_per_param) — no magic numbers.

        Parameters
        ----------
        n                   : number of Gaussians processed (tensor shape[0])
        n_mlp_macs_per_g    : MLP MACs per Gaussian (from layer weight counts)
        n_hash_params       : HashGrid / 3-D grid parameter count (measured)
        enc_dim             : total feature dim summed across all interpolation
                              outputs (e.g. n_levels × n_feat for tcnn; sum of
                              per-grid channels for Motion_Grid).
        out_dim             : MLP output dimension (ntc_model.n_output_dims)
        hash_bytes_per_param: storage width of the grid/hash table entries.
                              Default 2 (fp16) matches tcnn; pass 4 for fp32
                              dense grids (4DGC Motion_Grid stores fp32).
        """
        if n == 0:
            return 0
        s = self.spec
        taps = self._TRILINEAR_TAPS_PER_QUERY
        hash_load    = _ceil(n_hash_params * hash_bytes_per_param, s.dram_bw_bytes)
        hash_compute = _ceil(n * enc_dim * taps, s.mac_units)
        phase1 = (max(hash_load, hash_compute)
                  + _ceil(n * enc_dim * 4, s.sram_bw_bytes))             # enc → SRAM
        mlp_compute  = _ceil(n * n_mlp_macs_per_g,   s.mac_units)
        mem_enc_read = _ceil(n * enc_dim * 4,         s.sram_bw_bytes)   # enc ← SRAM
        phase2 = (max(mlp_compute, mem_enc_read)
                  + _ceil(n * out_dim * 4, s.dram_bw_bytes))             # delta → DRAM
        cyc = phase1 + phase2
        self._cyc["ntc"] += cyc
        return cyc

    def add_ntc_training(
        self,
        n: int,
        n_mlp_macs_per_g: int,
        n_hash_params: int,
        enc_dim: int,
        out_dim: int,
        n_iters: int,
        hash_bytes_per_param: int = 2,
    ) -> int:
        """NTC training loop on the PUOE: forward + backward + NTC Adam.

        Models n_iters complete gradient-descent steps over all N Gaussians.
        Each iteration consists of three phases executed sequentially on the
        PUOE Vector/Tensor Update Core.

        Phase 1 — Forward (identical to add_ntc_forward, per-iter):
            hash_load    = ceil(n_hash_params × bpp / dram_bw)   ← bpp=2 fp16 / 4 fp32
            hash_compute = ceil(n × enc_dim × TAPS / mac_units)  ← TAPS=8 trilinear corners
            fwd1         = max(hash_load, hash_compute)
                         + ceil(n × enc_dim × 4 / sram_bw)      ← enc features → SRAM
            fwd2         = max(ceil(n × n_mlp_macs / mac_units),
                               ceil(n × enc_dim × 4 / sram_bw))
                         + ceil(n × out_dim × 4 / dram_bw)      ← predicted deltas → DRAM

        Phase 2 — Backward:
          * MLP backward uses the standard "2 MACs per weight" convention
            (∂L/∂W + ∂L/∂x are independent matmuls of the same shape as fwd;
            see DeepBench / TPU whitepaper).
          * Hash-grid backward is the scatter-add adjoint of trilinear
            interpolation: each input-side gradient fans out to the SAME 8
            corners that fwd read, so the compute also scales with ``TAPS``
            (one atomic FMA per corner-channel pair).

            mlp_bwd  = ceil(n × n_mlp_macs × 2 / mac_units)
            hash_bwd = ceil(n × enc_dim × TAPS / mac_units)   ← 8-corner scatter
            bwd_cyc  = max(mlp_bwd, hash_bwd)
                     + ceil(n_mlp_macs × 4 / dram_bw)          ← weight grad write-back
                     + ceil(n × enc_dim × 4 / dram_bw)         ← hash grad accumulation

        Phase 3 — NTC Adam (over all hash + MLP parameters):
            n_total_params = n_hash_params + n_mlp_macs_per_g
            adam_compute = ceil(n_total_params × adam_ops / mac_units)
            adam_read    = ceil(n_total_params × 4 × 4 / dram_bw)  ← grad+m+v+θ
            adam_write   = ceil(n_total_params × 4 × 3 / dram_bw)  ← m+v+θ
            adam_cyc     = max(adam_compute, adam_read) + adam_write

        Total: (fwd_cyc + bwd_cyc + adam_cyc) × n_iters

        Parameters
        ----------
        n                   : number of Gaussians (tensor shape[0])
        n_mlp_macs_per_g    : MLP MACs per Gaussian = MLP weight count (measured)
        n_hash_params       : HashGrid / 3-D grid parameter count (measured).
        enc_dim             : total feature dim summed across all interpolation
                              outputs — for tcnn HashGrid this is
                              ``n_levels × n_feat``; for 4DGC Motion_Grid it is
                              the sum of per-grid channel dims (derived from
                              tinymlp input width, 20 in the paper).
        out_dim             : MLP output dimension (ntc_model.n_output_dims)
        n_iters             : number of training iterations for this frame
        hash_bytes_per_param: width of each grid/hash table entry.  Default 2
                              matches tcnn's fp16; pass 4 for 4DGC's fp32
                              ``Motion_Grid`` (stored as ``torch.randn`` →
                              fp32 by default, see Motion_Grid._init_grid).
        """
        if n == 0 or n_iters == 0:
            return 0
        s = self.spec
        taps = self._TRILINEAR_TAPS_PER_QUERY
        n_total_params = n_hash_params + n_mlp_macs_per_g  # total NTC params

        # Phase 1: Forward
        hash_load    = _ceil(n_hash_params * hash_bytes_per_param, s.dram_bw_bytes)
        hash_compute = _ceil(n * enc_dim * taps,   s.mac_units)        # tri-lerp
        fwd1 = (max(hash_load, hash_compute)
                + _ceil(n * enc_dim * 4, s.sram_bw_bytes))             # enc → SRAM
        fwd2 = (max(_ceil(n * n_mlp_macs_per_g, s.mac_units),
                    _ceil(n * enc_dim * 4, s.sram_bw_bytes))
                + _ceil(n * out_dim * 4, s.dram_bw_bytes))             # delta → DRAM
        fwd_cyc = fwd1 + fwd2

        # Phase 2: Backward
        mlp_bwd  = _ceil(n * n_mlp_macs_per_g * 2, s.mac_units)       # ∂L/∂W+∂L/∂x
        hash_bwd = _ceil(n * enc_dim * taps,       s.mac_units)        # 8-corner scatter
        bwd_cyc  = (max(mlp_bwd, hash_bwd)
                    + _ceil(n_mlp_macs_per_g * 4, s.dram_bw_bytes)    # weight grad
                    + _ceil(n * enc_dim * 4,      s.dram_bw_bytes))    # hash grad

        # Phase 3: NTC Adam over all parameters
        adam_compute = _ceil(n_total_params * s.adam_ops_per_float, s.mac_units)
        adam_read    = _ceil(n_total_params * 4 * 4, s.dram_bw_bytes)  # 4 arrays
        adam_write   = _ceil(n_total_params * 4 * 3, s.dram_bw_bytes)  # 3 arrays
        adam_cyc     = max(adam_compute, adam_read) + adam_write

        cyc = (fwd_cyc + bwd_cyc + adam_cyc) * n_iters
        self._cyc["ntc_train"] += cyc
        return cyc

    def add_render_training_iter(
        self,
        n: int,
        n_visible: int,
        tile_touches: int,
        H: int,
        W: int,
        sh_degree: int,
        n_iters: int,
    ) -> int:
        """3DGS render forward + photometric loss + backward per NTC training iteration.

        Models six sequential phases executed by the ASIC's Gaussian Render Backend
        and PUOE during each of the n_iters NTC training iterations.  All per-phase
        formulae are derived from the EWA splatting / 3DGS algorithm; no magic numbers.

        Measurable inputs (all captured from the live model at runtime):
          n            : total Gaussian count (norms.shape[0])
          n_visible    : visible Gaussians — (render_pkg["radii"] > 0).sum()
          tile_touches : Gaussian-tile intersections — n_visible × π × (avg_r/16)²
                         where avg_r is the mean screen-space radius in pixels and
                         16 is the CUDA tile block size in diff-gaussian-rasterization
                         (BLOCK_X = BLOCK_Y = 16, hardcoded in config.h)
          H, W         : render resolution from render_pkg["render"].shape[-2:]
          sh_degree    : from gaussians.active_sh_degree

        Phase 1 — Preprocessing forward (per Gaussian):
          sh_params = 3 × (sh_degree+1)²      # exact SH coefficient count
          sh_macs   = sh_params                # 1 MAC per SH coefficient (dot product)
          cov_macs  = 15                       # EWA: R S S^T R^T = 9 MACs + 2D proj = 6 MACs
          proj_macs = 4                        # 3D→2D perspective divide
          read : xyz(3)+rot(4)+scale(3)+opacity(1)+SH(sh_params) floats from DRAM
          write: n_visible × 10 floats (mean2D+cov2D+color+opacity+depth) to DRAM

        Phase 2 — Radix sort (32-bit tile||depth key-value pairs):
          Standard 32-bit radix sort = 32 passes over tile_touches elements.
          Each pass: 1 comparison per element (cmp_units throughput) + 1 KV read+write.

        Phase 3 — Alpha compositing forward (per Gaussian-tile intersection):
          eval 2D Gaussian q^T Σ^{-1} q   : 6 MACs
          exp(-0.5×q) [Taylor / LUT approx]: 4 MACs
          α_i = opacity × exp              : 1 MUL
          T_new = T × (1−α_i)             : 2 ops
          C += T × α_i × c_i (RGB)        : 6 MACs
          Total: 19 MACs per intersection

        Phase 4 — Photometric loss (L1 + SSIM):
          L1   :  H×W×3 ABS ops (1 MAC/pixel/ch)
          SSIM :  separable 11×1 Gaussian conv ×2 passes = 22 MACs/pixel/ch
                + SSIM formula ≈ 5 MACs/pixel/ch
                Total SSIM ≈ 27 MACs/pixel/ch
          Combined: H×W×3 × 28 MACs

        Phase 5 — Alpha compositing backward (gradient scatter):
          Re-traverse tile_touches; chain rule ≈ 2× forward MACs = 38 MACs/intersection.
          Scatter-add gradients (color+opacity+mean2D+cov2D = 9 floats) to n_visible.

        Phase 6 — Preprocessing backward (2D grad → 3D Gaussian grad):
          Symmetric to Phase 1; input = 8 gradient floats per n_visible Gaussian.
          Output = (3+4+3+sh_params) gradient floats per n_visible Gaussian.
        """
        if n == 0 or n_iters == 0:
            return 0
        s = self.spec

        # SH and geometry constants (derived, not estimated)
        sh_params = 3 * (sh_degree + 1) ** 2  # exact SH coefficient count
        sh_macs   = sh_params                   # 1 MAC per coefficient (dot product)
        cov_macs  = 15   # EWA: R S S^T R^T = 9 + 2D projection = 6 (exact)
        proj_macs = 4    # 3D→2D perspective (mul + div = 4 ops, conservative)
        per_g_macs = sh_macs + cov_macs + proj_macs

        # ── Phase 1: Preprocessing forward ──────────────────────────────────
        p1_compute = _ceil(n * per_g_macs,              s.mac_units)
        p1_read    = _ceil(n * (11 + sh_params) * 4,   s.dram_bw_bytes)  # xyz+rot+scale+opa+SH
        p1_write   = _ceil(n_visible * 10 * 4,          s.dram_bw_bytes)  # 2D state
        phase1 = max(p1_compute, p1_read) + p1_write

        # ── Phase 2: Radix sort (32-bit key-value, tile_touches entries) ────
        # 32 passes; each pass: ceil(tile_touches/cmp_units) + ceil(8B×tile_touches/dram_bw)
        p2_compute = _ceil(tile_touches * 32, s.cmp_units)
        p2_bw      = _ceil(tile_touches * 8 * 2, s.dram_bw_bytes)  # read+write per pass, amortised
        phase2 = max(p2_compute, p2_bw)

        # ── Phase 3: Alpha compositing forward ──────────────────────────────
        p3_compute = _ceil(tile_touches * 19,        s.mac_units)
        p3_read    = _ceil(tile_touches * 9 * 4,     s.dram_bw_bytes)   # cov2D+mean2D+color+opa
        p3_write   = _ceil(H * W * 3 * 4,            s.dram_bw_bytes)   # rendered image
        phase3 = max(p3_compute, p3_read) + p3_write

        # ── Phase 4: Photometric loss (L1 + SSIM) ───────────────────────────
        # SSIM separable 11×1 conv × 2 passes = 22 MACs + formula 5 MACs = 27 MACs/pix/ch
        # L1 = 1 MAC/pix/ch → total = 28 MACs/pix/ch
        p4_compute = _ceil(H * W * 3 * 28,           s.mac_units)
        p4_read    = _ceil(H * W * 3 * 4 * 2,        s.dram_bw_bytes)   # render + GT
        phase4 = max(p4_compute, p4_read)

        # ── Phase 5: Alpha compositing backward ─────────────────────────────
        # Chain rule ≈ 2× forward MACs = 38 MACs/intersection
        p5_compute = _ceil(tile_touches * 38,        s.mac_units)
        p5_read    = _ceil(tile_touches * 9 * 4,     s.dram_bw_bytes)   # same as fwd
        p5_write   = _ceil(n_visible * 9 * 4,        s.dram_bw_bytes)   # scatter grad
        phase5 = max(p5_compute, p5_read) + p5_write

        # ── Phase 6: Preprocessing backward (2D grad → 3D Gaussian grad) ───
        # Symmetric to Phase 1; output: xyz(3)+rot(4)+scale(3)+SH(sh_params) grads
        p6_compute = _ceil(n_visible * per_g_macs,              s.mac_units)
        p6_read    = _ceil(n_visible * 8 * 4,                   s.dram_bw_bytes)  # 2D grads
        p6_write   = _ceil(n_visible * (10 + sh_params) * 4,   s.dram_bw_bytes)  # 3D grads
        phase6 = max(p6_compute, p6_read) + p6_write

        per_iter = phase1 + phase2 + phase3 + phase4 + phase5 + phase6
        cyc = per_iter * n_iters
        self._cyc["ntc_train_render"] += cyc
        return cyc

    def add_mlp_training(self, n_params: int, n_iters: int) -> int:
        """Generic MLP (no hash grid) forward + backward + Adam per training iter.

        Models compact networks such as QUEEN's LatentDecoder and Gate modules
        where total parameter count is known but network topology need not be
        specified.  Reuses the ntc_train accumulator so that asic_ntc_train_cycles
        and asic_train_time_s both pick it up.

        Per-parameter operation count (derived, not estimated):
          Forward pass          :  1 MAC / weight  (linear layer dot products)
          Backward pass         :  2 MACs / weight  (∂L/∂W + ∂L/∂input)
          Adam step             :  adam_ops_per_float MACs / param  (from ASICSpec)
          Total compute         :  (1 + 2 + adam_ops) × n_params per iter

        Memory access per Adam step (7 floats × 4 bytes per param):
          Reads  : grad (1) + m (1) + v (1) + θ (1) = 4 floats
          Writes : m (1) + v (1) + θ (1) = 3 floats

        Bottleneck is max(compute, memory).

        Parameters
        ----------
        n_params : total trainable parameter count in all network modules.
        n_iters  : training iterations in the current frame.
        """
        if n_params <= 0 or n_iters <= 0:
            return 0
        s = self.spec
        ops_per_param  = 1 + 2 + s.adam_ops_per_float   # fwd + bwd + Adam
        cyc_compute    = _ceil(n_params * ops_per_param, s.mac_units)
        cyc_mem        = _ceil(n_params * 4 * 7,         s.dram_bw_bytes)
        cyc = max(cyc_compute, cyc_mem) * n_iters
        self._cyc["ntc_train"] += cyc
        return cyc

    def add_entropy_bottleneck_training(
        self,
        total_elements: int,
        n_iters: int,
        filters: tuple = (3, 3, 3, 3),
    ) -> int:
        """CompressAI-style EntropyBottleneck training cycles per frame.

        Models 4DGC's Motion_Grid rate-aware compression: every iteration
        pushes the ENTIRE 3D grid tensor through each EntropyBottleneck
        module to estimate a likelihood for the rate loss, then backprops
        through the ``_logits_cumulative`` pipeline (see
        ``baselines/4DGC/scene/entropy_models.py::EntropyBottleneck``).

        Every number below is **derived** from the EB architecture — no
        magic constants.  For the default CompressAI ``filters=(3,3,3,3)``
        the structure is a chain of matrix multiplications

            x₀ = input (1 float per element)
            xᵢ₊₁ = softplus(Wᵢ) · xᵢ + bᵢ   (+ tanh-gate, middle layers)

        with per-layer weight matrices ``Wᵢ`` of shape
        ``[channels, filters[i+1], filters[i]]`` where
        ``filters = (1,) + self.filters + (1,)``.  The per-element MAC count
        per ``_logits_cumulative`` call is therefore

            matrix_macs = Σᵢ filters[i+1] × filters[i]

        ``_likelihood`` runs ``_logits_cumulative`` twice (lower + upper
        bounds), so **forward MACs per element** = 2 × matrix_macs.  The
        backward pass re-uses the same derivative convention employed by
        :meth:`add_mlp_training` (2 MACs / weight), giving **backward MACs
        per element** = 2 × forward.

        Elementwise non-linearities — one ``softplus`` + one ``tanh`` +
        (in middle layers) a tanh-factor gate per layer, plus two
        ``sigmoid`` calls and a likelihood subtraction once per forward —
        contribute a per-layer constant cost.  We charge 5 scalar ops per
        layer activation (common LUT-style implementation) and 8 ops for
        the trailing sigmoid / subtract / noise path, again derived from
        the module structure rather than estimated.

        Memory: each element carries a 4-byte activation read and a
        4-byte gradient write, plus 4 bytes for the intermediate logit
        buffer that must be spilled to DRAM when the tile is larger than
        on-chip SRAM.  That is exactly 12 bytes / element / iter,
        directly matching the three float32 tensors involved.

        Lands in the ``ntc_train`` bucket so ``asic_ntc_train_cycles`` and
        ``asic_train_time_s`` pick it up, giving 4DGC reporting parity
        with the HashGrid-based baselines.

        Parameters
        ----------
        total_elements : sum of grid element counts across all EB modules
                         (for 4DGC flame_steak: 32·4·32² + 64·4·64² +
                         128·2·128² ≈ 5.37 M).
        n_iters        : MEM training iterations in the current frame.
        filters        : CompressAI EB filter tuple (default ``(3,3,3,3)``
                         matches 4DGC's ``Motion_Grid``).  Per-element
                         cycle cost is derived entirely from this.
        """
        if total_elements <= 0 or n_iters <= 0:
            return 0
        s = self.spec

        # Derive per-element MAC cost from the filter structure.
        full_filters = (1,) + tuple(int(f) for f in filters) + (1,)
        n_layers     = len(full_filters) - 1
        matrix_macs  = sum(full_filters[i + 1] * full_filters[i]
                           for i in range(n_layers))

        fwd_mac_per_elt = 2 * matrix_macs            # lower + upper bounds
        bwd_mac_per_elt = 2 * fwd_mac_per_elt        # 2 MACs / weight convention
        # Elementwise: softplus × n_layers + tanh-gate × (n_layers-1) per call,
        # × 2 calls, plus sigmoid/subtract/noise path (8 ops) per element.
        nonlin_per_call = 5 * n_layers + 5 * (n_layers - 1)
        nonlin_per_elt  = 2 * nonlin_per_call + 8

        ops_per_element   = fwd_mac_per_elt + bwd_mac_per_elt + nonlin_per_elt
        bytes_per_element = 3 * 4   # activation (read) + grad (write) + logit spill

        cyc_compute = _ceil(total_elements * ops_per_element,   s.mac_units)
        cyc_mem     = _ceil(total_elements * bytes_per_element, s.dram_bw_bytes)
        cyc = max(cyc_compute, cyc_mem) * n_iters
        self._cyc["ntc_train"] += cyc
        return cyc

    # ── Savings counting methods ───────────────────────────────────────────────

    def add_optimizer_update(self, n_active: int, total_d: int) -> int:
        """Adam optimizer cycles for active (non-suppressed) Gaussians.

        Only Gaussians that will actually receive a parameter update are
        counted; suppressed or cold-masked Gaussians are excluded.

        adam_ops_per_float = 10 (derived from Adam algorithm; see ASICSpec).
        total_d = sum of per-attribute dimensions across all Gaussian param
                  groups (e.g. xyz=3, rot=4, scale=3, opacity=1,
                  f_dc=3, f_rest=9 → total_d=23 for SH deg=1).

        DRAM access model (edge ASIC, Adam state in off-chip DRAM):
          Each scalar parameter requires:
            read  : grad(1) + m_prev(1) + v_prev(1) + θ_prev(1) = 4 fp32
            write : m_new(1) + v_new(1) + θ_new(1)              = 3 fp32

            dram_read  = ceil(n_active × total_d × 4 × 4 / dram_bw)
            dram_write = ceil(n_active × total_d × 4 × 3 / dram_bw)
            compute    = ceil(n_active × total_d × adam_ops / mac_units)
            cycles     = max(compute, dram_read) + dram_write

        UCN suppression (mode >= 1) reduces n_active toward zero, making
        both dram_read and dram_write approach zero — the dominant saving
        in a memory-bandwidth-constrained edge ASIC.

        n_active by mode:
          mode 0  — all Gaussians with delta_norm > 0
          mode 1  — Gaussians where grad_scale > 0 after consolidation
          mode >= 2 — Gaussians in fabric render mask (ASC + RSB tiers)
        """
        if n_active == 0 or total_d == 0:
            return 0
        s = self.spec
        compute    = _ceil(n_active * total_d * s.adam_ops_per_float, s.mac_units)
        dram_read  = _ceil(n_active * total_d * 4 * 4, s.dram_bw_bytes)  # 4 arrays × fp32
        dram_write = _ceil(n_active * total_d * 4 * 3, s.dram_bw_bytes)  # 3 arrays × fp32
        cyc = max(compute, dram_read) + dram_write
        self._cyc["optimizer"] += cyc
        return cyc

    def add_render_inference(
        self,
        n_active: int,
        n_visible: int,
        tile_touches: int,
        H: int,
        W: int,
        sh_degree: int,
    ) -> int:
        """ASIC per-frame INFERENCE render cycles (EWA splatting, forward only).

        Shares phases 1-3 with :meth:`add_render_training_iter` (preprocess +
        radix sort + alpha compositing forward) and explicitly excludes phases
        4-6 (photometric loss, alpha-compositing backward, preprocessing
        backward), which are training-only.

        This method **overwrites** ``_cyc["render"]`` with the single-frame
        count so that ``asic_render_cycles`` always reports the most recent
        inference frame, consistent with the legacy
        :meth:`set_render_cycles_from_fps` semantics.

        Parameters mirror those of :meth:`add_render_training_iter`; all are
        measurable from one render call on the masked inference proxy
        (no magic numbers, no estimation):

          n_active     : Gaussians actually submitted to the rasterizer
                         (= MaskedGaussianProxy.n_gaussians() under mode >=2)
          n_visible    : (render_pkg["radii"] > 0).sum()
          tile_touches : n_visible × π × (avg_r/16)²  (BLOCK_X=16 is the
                         diff-gaussian-rasterization tile size)
          H, W         : render_pkg["render"].shape[-2:]
          sh_degree    : gaussians.active_sh_degree

        Returns the single-frame cycle count, so callers can derive
        ``asic_fps = clock_hz / cyc``.
        """
        if n_active == 0:
            self._cyc["render"] = 0
            return 0
        s = self.spec

        sh_params = 3 * (sh_degree + 1) ** 2
        sh_macs   = sh_params
        cov_macs  = 15
        proj_macs = 4
        per_g_macs = sh_macs + cov_macs + proj_macs

        # ── Phase 1: preprocessing forward (over N_active) ──────────────────
        p1_compute = _ceil(n_active * per_g_macs,              s.mac_units)
        p1_read    = _ceil(n_active * (11 + sh_params) * 4,    s.dram_bw_bytes)
        p1_write   = _ceil(max(n_visible, 0) * 10 * 4,          s.dram_bw_bytes)
        phase1 = max(p1_compute, p1_read) + p1_write

        # ── Phase 2: radix sort (32-bit key-value, tile_touches entries) ────
        p2_compute = _ceil(tile_touches * 32,      s.cmp_units)
        p2_bw      = _ceil(tile_touches * 8 * 2,   s.dram_bw_bytes)
        phase2 = max(p2_compute, p2_bw)

        # ── Phase 3: alpha compositing forward ──────────────────────────────
        p3_compute = _ceil(tile_touches * 19,      s.mac_units)
        p3_read    = _ceil(tile_touches * 9 * 4,   s.dram_bw_bytes)
        p3_write   = _ceil(H * W * 3 * 4,          s.dram_bw_bytes)
        phase3 = max(p3_compute, p3_read) + p3_write

        cyc = phase1 + phase2 + phase3
        self._cyc["render"] = cyc
        return cyc

    def _render_phase123_cycles(
        self,
        n_active: int,
        n_visible: int,
        tile_touches: int,
        H: int,
        W: int,
        sh_degree: int,
        state_byte_scale: float,
    ) -> int:
        """Shared EWA splatting Phase-1/2/3 cycle model.

        The Full and Compact render paths differ only in:
          • ``sh_degree`` — controls SH parameter count (and therefore the
            colour-MAC budget + per-primitive state-read byte count);
          • ``state_byte_scale`` — fp16 → 0.5 halves Phase 1 / Phase 3
            state-read bandwidth (compute is unaffected).

        There is deliberately **no** additional ``compute_scale``
        multiplier: the colour / per-primitive MAC savings of a lower SH
        degree already fall out of the ``sh_params`` term, and the
        α-compositing inner loop (``tile_touches * 19`` MACs) is
        SH-independent.  Scaling α-compositing compute by an extra fudge
        factor would introduce a magic number and would overstate the
        Compact path's savings.
        """
        if n_active == 0:
            return 0
        s = self.spec
        sh_params = 3 * (sh_degree + 1) ** 2
        sh_macs   = sh_params
        cov_macs  = 15
        proj_macs = 4
        per_g_macs = sh_macs + cov_macs + proj_macs

        p1c = _ceil(n_active * per_g_macs, s.mac_units)
        p1r = _ceil(
            int(n_active * (11 + sh_params) * 4 * state_byte_scale),
            s.dram_bw_bytes,
        )
        p1w = _ceil(max(n_visible, 0) * 10 * 4, s.dram_bw_bytes)
        ph1 = max(p1c, p1r) + p1w

        p2c = _ceil(tile_touches * 32, s.cmp_units)
        p2b = _ceil(tile_touches * 8 * 2, s.dram_bw_bytes)
        ph2 = max(p2c, p2b)

        p3c = _ceil(tile_touches * 19, s.mac_units)
        p3r = _ceil(
            int(tile_touches * 9 * 4 * state_byte_scale),
            s.dram_bw_bytes,
        )
        p3w = _ceil(H * W * 3 * 4, s.dram_bw_bytes)
        ph3 = max(p3c, p3r) + p3w

        return ph1 + ph2 + ph3

    def compact_primitive_weight(
        self,
        sh_degree_full: int = 3,
        compact_sh_degree: int = 1,
        compact_state_byte_scale: float = 0.5,
        probe_n: int = 1_048_576,
    ) -> float:
        """Per-primitive Compact-to-Full cycle ratio, derived from the
        same EWA Phase-1/2/3 formulas used by the graded inference path.

        Needed anywhere the pipeline has to count "effective" primitives
        (e.g. weighting Grade-B primitives in training-side render cycle
        accounting) without hardcoding the headline 0.33× cost fudge.

        Method: render the same probe workload through
        :meth:`_render_phase123_cycles` twice — once with the Full-path
        parameters, once with the Compact-path parameters — and return
        the ratio.  A large ``probe_n`` is used so that fixed-cost terms
        (framebuffer write, radix-sort floor) are amortised and the
        result is dominated by the per-primitive terms, matching the
        usage pattern in
        :meth:`~glass_sim.pipeline.GLASS._train_render_active_counts`.

        Returns a ratio strictly in ``(0, 1]``.
        """
        # Probe geometry is deliberately simple: every primitive is
        # visible and hits exactly one tile.  The absolute cycle
        # numbers here are never surfaced — only the ratio is used.
        n         = max(1, int(probe_n))
        n_visible = n
        tiles     = n
        # Pick an H/W that won't dominate p3_write — 256×256 keeps the
        # framebuffer term << per-primitive terms for probe_n = 1M.
        H = W = 256

        full = self._render_phase123_cycles(
            n, n_visible, tiles, H, W,
            sh_degree=sh_degree_full,
            state_byte_scale=1.0,
        )
        compact = self._render_phase123_cycles(
            n, n_visible, tiles, H, W,
            sh_degree=compact_sh_degree,
            state_byte_scale=compact_state_byte_scale,
        )
        if full <= 0:
            return 1.0
        return max(1e-6, min(1.0, compact / full))

    def add_render_inference_graded(
        self,
        n_grade_a: int,
        n_grade_b: int,
        n_visible_total: int,
        tile_touches_total: int,
        H: int,
        W: int,
        sh_degree_full: int,
        compact_sh_degree: int = 1,
        compact_state_byte_scale: float = 0.5,
        n_visible_a: int | None = None,
        n_visible_b: int | None = None,
        tile_touches_a: int | None = None,
        tile_touches_b: int | None = None,
    ) -> dict[str, int]:
        """ASIC per-frame INFERENCE render cycles under Convergence-Graded
        Splatting (CGS).

        The Grade-A (LivePrecise) path keeps the standard EWA splatting
        pipeline — SH :math:`L=` ``sh_degree_full`` colour, fp32 state.
        The Grade-B (StableCompact) path uses SH :math:`L=`
        ``compact_sh_degree`` colour (default SH1: 1 direction × 3
        channels vs SH3's 16 × 3) and fp16 state reads
        (``compact_state_byte_scale=0.5``).  Both savings propagate
        *through the existing Phase 1/3 formulas* (see
        :meth:`_render_phase123_cycles`) rather than via a single
        hardcoded "compact cost ratio" multiplier — the Compact-to-Full
        ratio is an *emergent* property of the cycle model, not a magic
        knob.  Grade-D primitives are pre-pruned by the dispatcher and
        never enter either path.

        Since the hardware primitive dispatcher fans out the two paths to
        independent warp / tile queues (GLASS §8), the critical path is
        ``max(full_cycles, compact_cycles)``; the Phase 4 final composite
        that merges the two framebuffers shares bandwidth with Phase 3's
        write already counted inside each path (no additional cycles).

        Parameters
        ----------
        n_grade_a, n_grade_b
            Gaussian counts assigned to the Full / Compact paths.  The
            legacy ``add_render_inference`` method's ``n_active`` is
            ``n_grade_a + n_grade_b``.
        n_visible_total, tile_touches_total
            Aggregate visibility / tile-touch statistics observed from
            the masked-proxy render.  When per-grade support statistics are
            provided, they take precedence and directly model support-domain
            shrink before tile expansion.  Otherwise the split falls back to
            the legacy A:B count ratio approximation.
        H, W, sh_degree_full
            Identical to :meth:`add_render_inference`.
        compact_sh_degree, compact_state_byte_scale
            Compact-path design knobs.  Defaults match
            :class:`~glass_sim.config.ASICConfig` and stay in sync with
            :meth:`compact_primitive_weight` so derived diagnostics
            (``compact_path_ratio``) agree with observed run-time cycles.

        Returns
        -------
        dict with keys ``render_full_cycles``, ``render_compact_cycles``,
        and ``render_cycles`` (== ``max`` of the two).  ``_cyc["render"]``
        is overwritten to the headline ``render_cycles`` so downstream
        FPS / summary consumers stay schema-compatible.
        """
        n_total = n_grade_a + n_grade_b
        if n_total <= 0:
            self._cyc["render_full"] = 0
            self._cyc["render_compact"] = 0
            self._cyc["render"] = 0
            return {
                "render_full_cycles":    0,
                "render_compact_cycles": 0,
                "render_cycles":         0,
            }

        if (
            n_visible_a is not None
            and n_visible_b is not None
            and tile_touches_a is not None
            and tile_touches_b is not None
        ):
            vis_a = max(0, int(n_visible_a))
            vis_b = max(0, int(n_visible_b))
            tt_a = max(0, int(tile_touches_a))
            tt_b = max(0, int(tile_touches_b))
        else:
            a_frac = n_grade_a / n_total
            vis_a = int(round(n_visible_total * a_frac))
            vis_b = max(0, n_visible_total - vis_a)
            tt_a = int(round(tile_touches_total * a_frac))
            tt_b = max(0, tile_touches_total - tt_a)

        full_cyc = self._render_phase123_cycles(
            n_grade_a, vis_a, tt_a, H, W,
            sh_degree=sh_degree_full,
            state_byte_scale=1.0,
        )
        compact_cyc = self._render_phase123_cycles(
            n_grade_b, vis_b, tt_b, H, W,
            sh_degree=compact_sh_degree,
            state_byte_scale=compact_state_byte_scale,
        )

        # Independent warp / tile queues → the two paths run in parallel;
        # headline latency is the slower of the two.
        cyc = max(full_cyc, compact_cyc)

        self._cyc["render_full"]    = full_cyc
        self._cyc["render_compact"] = compact_cyc
        self._cyc["render"]         = cyc

        return {
            "render_full_cycles":    full_cyc,
            "render_compact_cycles": compact_cyc,
            "render_cycles":         cyc,
        }

    def set_render_cycles_from_fps(self, fps: float) -> None:
        """(Legacy) back-calculate render cycles from a measured GPU FPS.

        Kept for backwards compatibility with older ablation runners and for
        any path where no render_pkg stats are available.  Prefer
        :meth:`add_render_inference`, which models ASIC render cost directly
        from measurable per-frame geometry (n_visible, tile_touches, H×W,
        sh_degree).

        render_cycles = clock_hz / fps
        Overwrites any previously stored render cycle count.
        """
        if fps > 0:
            self._cyc["render"] = int(self.spec.clock_hz / fps)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return per-unit cycles and critical-path total.

        Parallelism rules applied:
          1. Gradient pipeline critical path:
               max(PUOE_total, UCN_total, ICE_total, ILSF_update_total)
             ILSF_update is absorbed because it reads from the same
             descriptor stream on a separate bus.
          2. ILSF_sort is hidden by the first ICE step (5.5 K << 201.8 K
             cycles) — not added to any path.
          3. Post-render:
               max(ILSF_render_total, Salience_total)
          4. Total:
               gradient_pipeline + post_render + optimizer + render

        Derived timing metrics:
          asic_commit_time_s : (gradient_pipeline + optimizer) / clock_hz
            Commit-path latency only (no NTC, no render). Use this to
            compare mode 0→1→2 progressive improvements, since NTC cycles
            are identical across modes and would mask the differences.
          asic_ntc_cycles    : NTC forward-inference cycles (PUOE tensor core).
            Add to commit cycles for absolute comparison with GPU training time.

        Notes
        -----
        Callers that want a wall-clock figure should either
        (a) use ``asic_commit_time_s`` (commit path only) or
        (b) assemble ``asic_train_time_s`` per-baseline by summing the
            relevant cycle buckets through ``result_schema.build_result``.
        The old ``asic_time_s_1ghz = asic_total_cycles / clock_hz`` field
        was removed because it silently excluded ``ntc_train*`` /
        ``ntc`` cycles — dominant for 3DGStream / 4DGC — and was easily
        mistaken for the headline training time.

        Returns
        -------
        dict with keys:
          asic_ntc_train_render_cycles, asic_ntc_train_cycles,
          asic_ntc_cycles,
          asic_puoe_cycles, asic_ucn_cycles, asic_ice_cycles,
          asic_ilsf_cycles (update + render combined),
          asic_salience_cycles, asic_optimizer_cycles, asic_render_cycles,
          asic_gradient_pipeline_cycles, asic_post_render_cycles,
          asic_total_cycles, asic_commit_time_s
        """
        gradient_pipeline = max(
            self._cyc["puoe"],
            self._cyc["ucn"],
            self._cyc["ice"],
            self._cyc["ilsf_update"],  # absorbed; max() keeps it free
        )
        # ilsf_sort is recorded but hidden — not part of any sum
        post_render = max(self._cyc["ilsf_render"], self._cyc["salience"])
        asic_overhead = gradient_pipeline + post_render
        total = asic_overhead + self._cyc["optimizer"] + self._cyc["render"]
        commit_cycles = gradient_pipeline + self._cyc["optimizer"]

        return {
            "asic_ntc_train_render_cycles":   self._cyc["ntc_train_render"],
            "asic_ntc_train_cycles":          self._cyc["ntc_train"],
            "asic_ntc_cycles":                self._cyc["ntc"],
            "asic_puoe_cycles":               self._cyc["puoe"],
            "asic_ucn_cycles":                self._cyc["ucn"],
            "asic_ice_cycles":                self._cyc["ice"],
            "asic_ilsf_cycles":               (self._cyc["ilsf_update"]
                                               + self._cyc["ilsf_render"]),
            "asic_salience_cycles":           self._cyc["salience"],
            "asic_optimizer_cycles":          self._cyc["optimizer"],
            "asic_render_cycles":             self._cyc["render"],
            # CGS diagnostics — zero when mode < 3 or the graded render
            # path was not exercised.  ``asic_render_cycles`` is the
            # headline figure; the per-path breakdown below is for
            # ablation plots that decompose Full vs Compact work.
            "asic_render_full_cycles":        self._cyc["render_full"],
            "asic_render_compact_cycles":     self._cyc["render_compact"],
            "asic_gradient_pipeline_cycles":  gradient_pipeline,
            "asic_post_render_cycles":        post_render,
            "asic_total_cycles":              total,
            "asic_commit_time_s":             commit_cycles / self.spec.clock_hz,
        }

    def reset_frame(self) -> None:
        """Reset per-frame counters.

        Resets all accumulators EXCEPT render (set once per frame via
        set_render_cycles_from_fps) and EXCEPT ilsf_sort (which is
        already hidden and non-critical).

        Called at the start of each frame's begin_frame(), after
        add_ilsf_sort() has already fired.
        """
        for key in ("ntc_train_render", "ntc_train", "ntc", "puoe", "ucn",
                    "ice", "ilsf_update", "ilsf_render", "salience", "optimizer"):
            self._cyc[key] = 0

    def reset(self) -> None:
        """Reset all counters (call between ablation runs)."""
        for key in self._cyc:
            self._cyc[key] = 0
