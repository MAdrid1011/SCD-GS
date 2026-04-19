#!/usr/bin/env bash
# =============================================================================
# run_all.sh — GLASS ASIC ablation master runner
#
# Executes all 4 baselines × 4 modes (16 runs total) sequentially, each in
# its correct conda environment, then calls collect_ablation.py to aggregate
# results into a CSV + ASCII table.
#
# Usage
# -----
#   bash run_all.sh                                 # default dataset: flame_steak
#   bash run_all.sh meet_room                       # future dataset
#   bash run_all.sh custom /path/to/my.yaml         # fully custom via YAML
#
# Arguments
# ---------
#   $1  Dataset name (default: flame_steak)
#   $2  Optional: path to YAML file with DatasetConfig overrides
#
# Output
# ------
#   results/GLASS_ablation/<dataset>/<baseline>/mode_<mode>.json
#   results/GLASS_ablation/<dataset>/<baseline>/mode_<mode>.log  ← combined stdout+stderr
#   results/GLASS_ablation/<dataset>/summary.csv
#   results/GLASS_ablation/<dataset>/summary.txt
# =============================================================================

set -euo pipefail

# Derive repo root from this script's location so the runner is portable
# across checkouts.  Override with `GLASS=/path/to/repo bash run_all.sh`.
SIM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLASS="${GLASS:-$(cd "$SIM/../.." && pwd)}"

DATASET="${1:-flame_steak}"
DATASET_YAML="${2:-}"

extra_args=""
if [[ -n "$DATASET_YAML" ]]; then
    extra_args="--dataset-config $DATASET_YAML"
fi

# Conda environments per baseline
declare -A ENVS
ENVS["3dgstream"]="3dgstream-cu121"
ENVS["queen"]="queen-cu121"
ENVS["hicom"]="hicom-cu121"
ENVS["swift4d"]="queen-cu121"   # Swift4D shares the queen env (same CUDA/PyTorch)

BASELINES=("3dgstream" "queen" "hicom" "swift4d")

total_runs=$(( ${#BASELINES[@]} * 4 ))
run_idx=0

echo "============================================================"
echo " GLASS ASIC Ablation Run"
echo " Dataset  : $DATASET"
echo " YAML cfg : ${DATASET_YAML:-<none>}"
echo " Total runs: $total_runs  (${#BASELINES[@]} baselines × 4 modes)"
echo "============================================================"

for BASELINE in "${BASELINES[@]}"; do
    ENV="${ENVS[$BASELINE]}"
    SCRIPT="$SIM/run_${BASELINE}.py"

    for MODE in 0 1 2 3; do
        run_idx=$(( run_idx + 1 ))
        LOG_DIR="$GLASS/results/GLASS_ablation/$DATASET/$BASELINE"
        LOG_FILE="$LOG_DIR/mode_${MODE}.log"

        mkdir -p "$LOG_DIR"

        echo ""
        echo "------------------------------------------------------------"
        echo " [$run_idx/$total_runs] $BASELINE | mode=$MODE | env=$ENV"
        printf " Started: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"

        set +e   # allow individual runs to fail without aborting all
        conda run --no-capture-output -n "$ENV" \
            python "$SCRIPT" \
            --mode "$MODE" \
            --dataset "$DATASET" \
            $extra_args \
            2>&1 | tee "$LOG_FILE"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [[ $exit_code -ne 0 ]]; then
            echo "⚠  WARNING: $BASELINE mode=$MODE exited with code $exit_code"
            echo "   Log saved at: $LOG_FILE"
        else
            echo "✓  $BASELINE mode=$MODE completed successfully."
        fi

        printf " Finished: %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo ""
echo "============================================================"
echo " All training runs complete. Aggregating results …"
echo "============================================================"

COLLECT_LOG="$GLASS/results/GLASS_ablation/$DATASET/collect.log"
conda run --no-capture-output -n "${ENVS[3dgstream]}" \
    python "$SIM/collect_ablation.py" \
    --dataset "$DATASET" \
    $extra_args \
    2>&1 | tee "$COLLECT_LOG"

echo ""
echo "============================================================"
echo " ALL DONE"
echo " Results directory: $GLASS/results/GLASS_ablation/$DATASET/"
echo " Summary CSV:  $GLASS/results/GLASS_ablation/$DATASET/summary.csv"
echo " Summary table: $GLASS/results/GLASS_ablation/$DATASET/summary.txt"
echo "============================================================"
