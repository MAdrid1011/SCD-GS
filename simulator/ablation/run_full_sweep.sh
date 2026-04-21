#!/usr/bin/env bash
# Drive a full 10-frame × 4-mode ablation sweep for all three baselines.
# Runs strictly serially (one GPU, one environment at a time) so logs stay
# deterministic and the GPU never gets contended.
set -uo pipefail

GLASS="${GLASS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="$GLASS/results/GLASS_ablation/sweep_logs"
mkdir -p "$LOG_DIR"

# --- conda bootstrap (non-interactive shell) ---------------------------------
# shellcheck disable=SC1091
source /home/madrid/anaconda3/etc/profile.d/conda.sh

run_one () {
    local env_name="$1" baseline="$2" mode="$3"
    local log="$LOG_DIR/${baseline}_mode${mode}.log"
    echo "[$(date '+%H:%M:%S')] ===== $baseline mode=$mode (env=$env_name) ====="
    conda activate "$env_name"
    (cd "$GLASS" && \
        python -u "simulator/ablation/run_${baseline,,}.py" \
               --mode "$mode" --n-frames 10) \
        >"$log" 2>&1
    local rc=$?
    conda deactivate
    echo "[$(date '+%H:%M:%S')]   exit=$rc  log=$log"
    return $rc
}

# HiCoM — 4 modes × 10 frames (primary deliverable of this PR).
for m in 0 1 2 3; do
    run_one hicom-cu121    HiCoM     "$m" || echo "  HiCoM m=$m FAILED"
done

# QUEEN — 4 modes × 10 frames (extend existing 3-frame runs).
for m in 0 1 2 3; do
    run_one queen-cu121    QUEEN     "$m" || echo "  QUEEN m=$m FAILED"
done

# 3DGStream — only mode 0 needs the 10-frame extension (1-3 already done).
run_one 3dgstream-cu121  3DGStream 0 || echo "  3DGStream m=0 FAILED"

# 4DGC — 4 modes × 10 frames (requires prior train_initial.py + Motion_Grid
# warmup on data/<scene>/4dgc_init/; see simulator/datasets.py for paths).
# Assumes a conda env named ``4dgc-cu121`` with 3DGStream's rasterizer/simple-knn
# plus compressai + range-coder installed on top.
for m in 0 1 2 3; do
    run_one 4dgc-cu121       4DGC      "$m" || echo "  4DGC m=$m FAILED"
done

echo "[$(date '+%H:%M:%S')] ====== FULL SWEEP DONE ======"
