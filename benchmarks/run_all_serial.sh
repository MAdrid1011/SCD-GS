#!/usr/bin/env bash
# =============================================================================
# Serial GPU-baseline benchmark runner: 3DGStream → Swift4D → HiCoM → aggregate.
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

# ─── STEP 3: Swift4D training (bundled, 13000 iterations) ──────────────────
log "Step 3 — Swift4D training (all frames bundled) ..."
mkdir -p "$RESULTS/Swift4D/flame_steak"
(cd "$GLASS_DIR/baselines/Swift4D" &&
 conda run --no-capture-output -n "$QUEEN_ENV" \
    python train.py \
    -s "$GLASS_DIR/data/flame_steak/flame_steak" \
    --port 6017 \
    --expname "dynerf/flame_steak" \
    --configs arguments/dynerf/flame_steak.py \
    --test_iterations 13001 \
    --save_iterations 13001 \
    2>&1 | tee "$RESULTS/Swift4D/flame_steak/training_log.txt")
log "Step 3 — Swift4D training DONE."

# ─── STEP 4: Swift4D render + quality ─────────────────────────────────────
log "Step 4 — Swift4D render + metrics ..."
(cd "$GLASS_DIR/baselines/Swift4D" &&
 SWIFT4D_MODEL="$(ls -dt output/dynerf/flame_steak 2>/dev/null | head -1)"
 if [ -n "$SWIFT4D_MODEL" ]; then
     conda run --no-capture-output -n "$QUEEN_ENV" \
         python render.py \
         --model_path "$SWIFT4D_MODEL" \
         --skip_train --skip_video \
         --iteration 13000 \
         --configs arguments/dynerf/flame_steak.py \
         2>&1 | tee "$RESULTS/Swift4D/flame_steak/render_log.txt"

     conda run --no-capture-output -n "$QUEEN_ENV" \
         python metrics.py \
         --model_path "$SWIFT4D_MODEL" \
         2>&1 | tee "$RESULTS/Swift4D/flame_steak/metrics_log.txt"
 else
     log "  Swift4D model directory not found; skipping render + metrics."
 fi)
log "Step 4 — Swift4D eval DONE."

# ─── STEP 5: HiCoM training ───────────────────────────────────────────────
log "Step 5 — HiCoM training ..."
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
log "Step 5 — HiCoM training DONE."

# ─── STEP 6: QUEEN (reuses prior-run subset, no retraining) ───────────────
log "Step 6 — QUEEN FPS measurement (quality from prior-run subset) ..."
mkdir -p "$RESULTS/QUEEN"
conda run --no-capture-output -n "$QUEEN_ENV" \
    python "$SCRIPTS/bench_queen.py" \
    2>&1 | tee "$RESULTS/QUEEN/bench_log.txt"
log "Step 6 — QUEEN DONE."

# ─── STEP 7: Aggregate ────────────────────────────────────────────────────
log "Step 7 — aggregating all results ..."
conda run --no-capture-output -n "$DGS_ENV" \
    python "$SCRIPTS/collect_metrics.py" \
    2>&1 | tee "$RESULTS/metric_table.log"

log "ALL DONE. Results in $RESULTS/"
