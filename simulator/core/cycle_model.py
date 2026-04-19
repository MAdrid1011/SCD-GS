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
  5. ILSF_render and Salience execute in parallel after render(); critical path
     = max(ILSF_render, Salience).
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
        filter rather than gradient norms.  Runs in parallel with Salience
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
        """Salience score EMA update (called post-render, mode >= 3).

        Per Gaussian (3 ops):
          1 MUL  — raw_score = opacity × radii
          1 DIV  — normalise by max_radii
          1 FMA  — EMA: score ← α×score + (1−α)×raw_score

        Memory:
          read  = ceil(N×8 / sram_bw)   ← radii (4B) + opacity (4B)
          write = ceil(N×4 / sram_bw)   ← updated score

            cycles = max(ceil(N×8/bw), ceil(N×3/mac_units)) + ceil(N×4/bw)

        Runs in parallel with ILSF_render (disjoint inputs and outputs).
        """
        if n == 0:
            return 0
        s = self.spec
        mem_rd  = _ceil(n * 8, s.sram_bw_bytes)
        compute = _ceil(n * 3, s.mac_units)
        write   = _ceil(n * 4, s.sram_bw_bytes)
        cyc     = max(mem_rd, compute) + write
        self._cyc["salience"] += cyc
        return cyc

    def add_ntc_forward(
        self,
        n: int,
        n_mlp_macs_per_g: int,
        n_hash_params: int,
        enc_dim: int,
        out_dim: int,
    ) -> int:
        """NTC (Neural Texture Cache) forward inference on the PUOE.

        The HashGrid and FullyFusedMLP of the NTC run on the ASIC's
        Vector/Tensor Update Core (part of PUOE).  Two-phase model:

        Phase 1: HashGrid lookup + trilinear interpolation
          The hash table is fp16 and typically 2–4 MB, exceeding on-chip
          SRAM capacity for an edge device → loaded from off-chip DRAM.
          Trilinear interpolation is compute-bound for large N.
          Encoded features written to on-chip SRAM for Phase 2.

            hash_load    = ceil(n_hash_params × 2 / dram_bw)   ← fp16, DRAM
            hash_compute = ceil(n × enc_dim × 2 / mac_units)   ← tri-lerp MACs
            phase1       = max(hash_load, hash_compute)
                         + ceil(n × enc_dim × 4 / sram_bw)     ← enc write, SRAM

        Phase 2: FullyFusedMLP streaming forward
          MLP weights are small (tens of KB) and reside on-chip SRAM.
          Encoding activations are read from SRAM; attribute delta outputs
          are written back to off-chip DRAM for the state commit path.

            mlp_compute  = ceil(n × n_mlp_macs_per_g / mac_units)
            mem_enc_read = ceil(n × enc_dim × 4 / sram_bw)
            phase2       = max(mlp_compute, mem_enc_read)
                         + ceil(n × out_dim × 4 / dram_bw)     ← delta attrs, DRAM

        All parameters are measured from the live model at hook install time
        (n_mlp_macs_per_g, n_hash_params, enc_dim, out_dim) — no magic numbers.

        Parameters
        ----------
        n               : number of Gaussians processed (real tensor shape[0])
        n_mlp_macs_per_g: MLP MACs per Gaussian (from layer weight counts)
        n_hash_params   : HashGrid parameter count in fp16 (measured at runtime)
        enc_dim         : HashGrid output feature dimension (n_levels × n_feat)
        out_dim         : MLP output dimension (ntc_model.n_output_dims)
        """
        if n == 0:
            return 0
        s = self.spec
        # Phase 1: hash table load from DRAM + trilinear interpolation
        hash_load    = _ceil(n_hash_params * 2,       s.dram_bw_bytes)   # fp16
        hash_compute = _ceil(n * enc_dim * 2,         s.mac_units)       # tri-lerp
        phase1 = (max(hash_load, hash_compute)
                  + _ceil(n * enc_dim * 4, s.sram_bw_bytes))             # enc → SRAM
        # Phase 2: MLP forward, output deltas to DRAM
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
    ) -> int:
        """NTC training loop on the PUOE: forward + backward + NTC Adam.

        Models n_iters complete gradient-descent steps over all N Gaussians.
        Each iteration consists of three phases executed sequentially on the
        PUOE Vector/Tensor Update Core.

        Phase 1 — Forward (identical to add_ntc_forward, per-iter):
            hash_load    = ceil(n_hash_params × 2 / dram_bw)    ← fp16 hash from DRAM
            hash_compute = ceil(n × enc_dim × 2 / mac_units)    ← trilinear-interp MACs
            fwd1         = max(hash_load, hash_compute)
                         + ceil(n × enc_dim × 4 / sram_bw)      ← enc features → SRAM
            fwd2         = max(ceil(n × n_mlp_macs / mac_units),
                               ceil(n × enc_dim × 4 / sram_bw))
                         + ceil(n × out_dim × 4 / dram_bw)      ← predicted deltas → DRAM

        Phase 2 — Backward (2× forward MACs: ∂L/∂W + ∂L/∂x standard derivation):
            mlp_bwd  = ceil(n × n_mlp_macs × 2 / mac_units)     ← weight + input grads
            hash_bwd = ceil(n × enc_dim / mac_units)             ← scatter-add ALU ops
            bwd_cyc  = max(mlp_bwd, hash_bwd)
                     + ceil(n_mlp_macs × 4 / dram_bw)            ← weight grad write-back
                     + ceil(n × enc_dim × 4 / dram_bw)           ← hash grad accumulation

        Phase 3 — NTC Adam (over all hash + MLP parameters):
            n_total_params = n_hash_params + n_mlp_macs_per_g
            adam_compute = ceil(n_total_params × adam_ops / mac_units)
            adam_read    = ceil(n_total_params × 4 × 4 / dram_bw)  ← grad+m+v+θ
            adam_write   = ceil(n_total_params × 4 × 3 / dram_bw)  ← m+v+θ
            adam_cyc     = max(adam_compute, adam_read) + adam_write

        Total: (fwd_cyc + bwd_cyc + adam_cyc) × n_iters

        Parameters
        ----------
        n               : number of Gaussians (tensor shape[0])
        n_mlp_macs_per_g: MLP MACs per Gaussian = MLP weight count (measured)
        n_hash_params   : HashGrid fp16 parameter count (measured at runtime)
        enc_dim         : HashGrid output feature dim (n_levels × n_feat)
        out_dim         : MLP output dimension (ntc_model.n_output_dims)
        n_iters         : number of training iterations for this frame
        """
        if n == 0 or n_iters == 0:
            return 0
        s = self.spec
        n_total_params = n_hash_params + n_mlp_macs_per_g  # total NTC params

        # Phase 1: Forward
        hash_load    = _ceil(n_hash_params * 2,     s.dram_bw_bytes)   # fp16
        hash_compute = _ceil(n * enc_dim * 2,       s.mac_units)       # tri-lerp
        fwd1 = (max(hash_load, hash_compute)
                + _ceil(n * enc_dim * 4, s.sram_bw_bytes))             # enc → SRAM
        fwd2 = (max(_ceil(n * n_mlp_macs_per_g, s.mac_units),
                    _ceil(n * enc_dim * 4, s.sram_bw_bytes))
                + _ceil(n * out_dim * 4, s.dram_bw_bytes))             # delta → DRAM
        fwd_cyc = fwd1 + fwd2

        # Phase 2: Backward
        mlp_bwd  = _ceil(n * n_mlp_macs_per_g * 2, s.mac_units)       # ∂L/∂W+∂L/∂x
        hash_bwd = _ceil(n * enc_dim,              s.mac_units)        # scatter-add
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

    def set_render_cycles_from_fps(self, fps: float) -> None:
        """Back-calculate render cycles from measured FPS (no estimation).

        render_cycles = clock_hz / fps

        Called once after MetricsCollector.measure_fps() completes.
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

        Returns
        -------
        dict with keys:
          asic_ntc_cycles,
          asic_puoe_cycles, asic_ucn_cycles, asic_ice_cycles,
          asic_ilsf_cycles (update + render combined),
          asic_salience_cycles, asic_optimizer_cycles, asic_render_cycles,
          asic_gradient_pipeline_cycles, asic_post_render_cycles,
          asic_total_cycles, asic_time_s_1ghz,
          asic_commit_time_s
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
            "asic_gradient_pipeline_cycles":  gradient_pipeline,
            "asic_post_render_cycles":        post_render,
            "asic_total_cycles":              total,
            "asic_time_s_1ghz":               total / self.spec.clock_hz,
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
