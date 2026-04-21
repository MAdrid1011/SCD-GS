#!/usr/bin/env bash
# =============================================================================
# Serial GPU-baseline benchmark runner: 3DGStream → HiCoM → QUEEN → aggregate.
# (4DGC is benchmarked via simulator/ablation/run_4dgc.py, which doubles as its
# head-to-head harness; see that runner for per-frame stage-1/stage-2 timing.)
# Each task runs exclusively on the GPU; no concurrent execution.
#
# This is the *GPU* head-to-head harness that sits alongside the ASIC-cycle
# ablation (simulator/ablation/run_all.sh).  Together they produce the two
# columns of numbers compared in the paper.
#
# Usage
# -----
#   bash benchmarks/run_all_serial.sh              # full 300-frame run
#   bash benchmarks/run_all_serial.sh 10           # smoke (10 frames, 3DGStream only)
#
# Override repo root or conda envs via env vars if needed:
#   GLASS=/path bash benchmarks/run_all_serial.sh
#   QUEEN_ENV=queen-cu121 HICOM_ENV=hicom-cu121 bash benchmarks/run_all_serial.sh
# =============================================================================
set -euo pipefail

# Derive repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLASS_DIR="${GLASS:-$(cd "$SCRIPT_DIR/.." && pwd)}"

RESULTS="$GLASS_DIR/results"
SCRIPTS="$GLASS_DIR/benchmarks/scripts"

N_FRAMES="${1:-300}"

# Conda envs per baseline (override via env var to use a different name).
DGS_ENV="${DGS_ENV:-3dgstream-cu121}"
QUEEN_ENV="${QUEEN_ENV:-queen-cu121}"
HICOM_ENV="${HICOM_ENV:-hicom-cu121}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── STEP 1: 3DGStream incremental training ────────────────────────────────
log "Step 1 — 3DGStream incremental training (N_FRAMES=$N_FRAMES) ..."
mkdir -p "$RESULTS/3DGStream/flame_steak"
GLASS="$GLASS_DIR" CONDA_ENV="$DGS_ENV" \
    bash "$SCRIPTS/bench_3dgstream.sh" "$N_FRAMES" \
    2>&1 | tee "$RESULTS/3DGStream/flame_steak/training_log.txt"
log "Step 1 — 3DGStream training DONE."

# ─── STEP 2: 3DGStream quality + FPS evaluation ────────────────────────────
log "Step 2 — 3DGStream eval (PSNR/SSIM/LPIPS + FPS) ..."
(cd "$GLASS_DIR/baselines/3DGStream" &&
 conda run --no-capture-output -n "$DGS_ENV" \
    env TORCHDYNAMO_DISABLE=1 \
    python "$SCRIPTS/eval_3dgstream.py" \
    2>&1 | tee "$RESULTS/3DGStream/flame_steak/eval_log.txt")
log "Step 2 — 3DGStream eval DONE."

# ─── STEP 3: HiCoM training ───────────────────────────────────────────────
log "Step 3 — HiCoM training ..."
mkdir -p "$RESULTS/HiCoM/flame_steak"
(cd "$GLASS_DIR/baselines/HiCoM" &&
 conda run --no-capture-output -n "$HICOM_ENV" \
    python main.py \
    --config=config/dynerf.yaml \
    "data.root=$GLASS_DIR/data/hicom_flame_steak_scene" \
    "output_dir=$RESULTS/HiCoM/flame_steak/output" \
    "experiment_name=flame_steak" \
    "trainer.num_incr_steps=200" \
    2>&1 | tee "$RESULTS/HiCoM/flame_steak/training_log.txt")
log "Step 3 — HiCoM training DONE."

# ─── STEP 4: QUEEN (reuses prior-run subset, no retraining) ───────────────
log "Step 4 — QUEEN FPS measurement (quality from prior-run subset) ..."
mkdir -p "$RESULTS/QUEEN"
conda run --no-capture-output -n "$QUEEN_ENV" \
    python "$SCRIPTS/bench_queen.py" \
    2>&1 | tee "$RESULTS/QUEEN/bench_log.txt"
log "Step 4 — QUEEN DONE."

# ─── STEP 5: Aggregate ────────────────────────────────────────────────────
log "Step 5 — aggregating all results ..."
conda run --no-capture-output -n "$DGS_ENV" \
    python "$SCRIPTS/collect_metrics.py" \
    2>&1 | tee "$RESULTS/metric_table.log"

log "ALL DONE. Results in $RESULTS/"
