#!/usr/bin/env bash
# =============================================================================
# bench_3dgstream.sh — thin wrapper around baselines/3DGStream/train_frames.py
#
# Resolves repo-root and assets via BASH_SOURCE so the script is portable
# across checkouts (override GLASS=/path to pin a different root).
#
# Usage
# -----
#   bash benchmarks/scripts/bench_3dgstream.sh             # full 300-frame run
#   bash benchmarks/scripts/bench_3dgstream.sh 10          # smoke (10 frames)
# =============================================================================
set -euo pipefail

# Derive repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLASS="${GLASS:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

REPO="$GLASS/baselines/3DGStream"
INIT_3DGS="$GLASS/data/assets/3DGStream/flame_steak_suite/flame_steak_init"
FRAME0="$GLASS/data/assets/3DGStream/flame_steak_suite/frame000000"
DATA_ROOT="$GLASS/data/flame_steak/flame_steak"
VIDEO_PATH="$GLASS/data/3dgstream_flame_steak_scene"
OUTPUT="$GLASS/results/3DGStream/flame_steak"
ENV="${CONDA_ENV:-3dgstream-cu121}"
N_FRAMES="${1:-300}"

mkdir -p "$OUTPUT"

echo "[3DGStream] Starting benchmark: $N_FRAMES frames"
echo "[3DGStream] Repo root : $GLASS"
echo "[3DGStream] Init 3DGS : $INIT_3DGS"
echo "[3DGStream] Video path: $VIDEO_PATH"
echo "[3DGStream] Output    : $OUTPUT"

# Forward config to the embedded Python via env vars so the heredoc is
# decoupled from the shell's quoting rules.
export REPO INIT_3DGS FRAME0 DATA_ROOT VIDEO_PATH OUTPUT N_FRAMES

conda run --no-capture-output -n "$ENV" python - <<'PYEOF'
import os, sys, shlex, subprocess

REPO       = os.environ["REPO"]
INIT_3DGS  = os.environ["INIT_3DGS"]
FRAME0     = os.environ["FRAME0"]
DATA_ROOT  = os.environ["DATA_ROOT"]
VIDEO_PATH = os.environ["VIDEO_PATH"]
OUTPUT     = os.environ["OUTPUT"]
N_FRAMES   = int(os.environ["N_FRAMES"])

sys.path.insert(0, REPO)
os.makedirs(OUTPUT, exist_ok=True)

cmd = [
    sys.executable, os.path.join(REPO, "train_frames.py"),
    "--init_path",   INIT_3DGS,
    "--source_path", FRAME0,
    "--model_path",  os.path.join(OUTPUT, "gaussians"),
    "--data_root",   DATA_ROOT,
    "--video_path",  VIDEO_PATH,
    "--n_frames",    str(N_FRAMES),
    "--sh_degree",   "1",
    "--eval",
]
print("[3DGStream bench] Running:", " ".join(shlex.quote(c) for c in cmd))
rc = subprocess.call(cmd)
sys.exit(rc)
PYEOF

echo "[3DGStream] Training finished. Results under: $OUTPUT"
