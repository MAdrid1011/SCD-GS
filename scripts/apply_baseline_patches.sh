#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/apply_baseline_patches.sh [check|apply|reverse|status]

Commands:
  check    Verify that all baseline patches apply cleanly.
  apply    Apply all baseline patches. This is the default command.
  reverse  Reverse all baseline patches.
  status   Show submodule working-tree status.
EOF
}

command="${1:-apply}"

PATCHES=(
  "baselines/3DGStream:patches/baselines/3DGStream.patch"
  "baselines/4DGC:patches/baselines/4DGC.patch"
  "baselines/QUEEN:patches/baselines/QUEEN.patch"
)

run_for_each_patch() {
  local git_apply_args=("$@")
  local item submodule patch

  for item in "${PATCHES[@]}"; do
    submodule="${item%%:*}"
    patch="${item#*:}"

    if [[ ! -d "${ROOT_DIR}/${submodule}/.git" && ! -f "${ROOT_DIR}/${submodule}/.git" ]]; then
      echo "missing submodule: ${submodule}" >&2
      exit 1
    fi
    if [[ ! -f "${ROOT_DIR}/${patch}" ]]; then
      echo "missing patch: ${patch}" >&2
      exit 1
    fi

    echo "==> ${submodule}: git apply ${git_apply_args[*]} ${patch}"
    git -C "${ROOT_DIR}/${submodule}" apply \
      --whitespace=nowarn \
      "${git_apply_args[@]}" \
      "${ROOT_DIR}/${patch}"
  done
}

show_status() {
  local item submodule
  for item in "${PATCHES[@]}"; do
    submodule="${item%%:*}"
    echo "==> ${submodule}"
    git -C "${ROOT_DIR}/${submodule}" status -sb
  done
}

case "${command}" in
  check)
    run_for_each_patch --check
    ;;
  apply)
    run_for_each_patch --check
    run_for_each_patch
    ;;
  reverse)
    run_for_each_patch --reverse --check
    run_for_each_patch --reverse
    ;;
  status)
    show_status
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
