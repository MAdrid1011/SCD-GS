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
#   bash run_all.sh                                       # dataset=flame_steak, n_frames from DatasetConfig
#   bash run_all.sh flame_steak "" 10                     # override to 10 frames
#   bash run_all.sh custom /path/to/my.yaml               # fully custom via YAML
#   bash run_all.sh custom /path/to/my.yaml 10            # YAML + frame override
#
# Arguments
# ---------
#   $1  Dataset name                     (default: flame_steak)
#   $2  Optional: path to YAML overrides (default: empty)
#   $3  Optional: n_frames override      (default: empty ⇒ use DatasetConfig.n_frames_ablation)
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
N_FRAMES="${3:-}"

extra_args=""
if [[ -n "$DATASET_YAML" ]]; then
    extra_args="--dataset-config $DATASET_YAML"
fi
if [[ -n "$N_FRAMES" ]]; then
    extra_args="$extra_args --n-frames $N_FRAMES"
fi

# Conda environments per baseline
declare -A ENVS
ENVS["3dgstream"]="3dgstream-cu121"
ENVS["queen"]="queen-cu121"
ENVS["hicom"]="hicom-cu121"
ENVS["4dgc"]="4dgc-cu121"

BASELINES=("3dgstream" "queen" "hicom" "4dgc")

total_runs=$(( ${#BASELINES[@]} * 4 ))
run_idx=0

echo "============================================================"
echo " GLASS ASIC Ablation Run"
echo " Dataset  : $DATASET"
echo " YAML cfg : ${DATASET_YAML:-<none>}"
echo " n_frames : ${N_FRAMES:-<DatasetConfig default>}"
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
collect_args="--dataset $DATASET"
if [[ -n "$DATASET_YAML" ]]; then
    collect_args="$collect_args --dataset-config $DATASET_YAML"
fi
# Intentionally omit --n-frames here: the aggregator only reads existing
# mode_*.json files and has no concept of training scope.
conda run --no-capture-output -n "${ENVS[3dgstream]}" \
    python "$SIM/collect_ablation.py" $collect_args \
    2>&1 | tee "$COLLECT_LOG"

echo ""
echo "============================================================"
echo " ALL DONE"
echo " Results directory: $GLASS/results/GLASS_ablation/$DATASET/"
echo " Summary CSV:  $GLASS/results/GLASS_ablation/$DATASET/summary.csv"
echo " Summary table: $GLASS/results/GLASS_ablation/$DATASET/summary.txt"
echo "============================================================"
