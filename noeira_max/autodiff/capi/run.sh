#!/usr/bin/env bash
# M3: train with the MAX train step from Mojo, through the C API, with no
# Python in the process. Exports the step (Python, once), builds the Mojo
# driver, runs it, and compares its losses with the same compiled step
# driven from Python.
#
#   noeira_max/autodiff/capi/run.sh [--tiny] [STEPS]     # default 200 steps
#   OUT=/some/dir noeira_max/autodiff/capi/run.sh         # keep the artifacts
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
OUT="${OUT:-${TMPDIR:-/tmp}/noeira_capi_step}"
EXTRA=""
if [[ "${1:-}" == "--tiny" ]]; then
    EXTRA="--tiny"
    shift
fi
STEPS="${1:-200}"
cd "$ROOT"

noeira_max/autodiff/run.sh noeira_max/autodiff/capi/export_step.py "$OUT" --steps "$STEPS" $EXTRA
# Build and run inside the activated env: it sets MODULAR_HOME, which the C
# API needs to find its runtime libraries. `mojo run` cannot resolve -lmax.
pixi run --manifest-path "$MAIN/pixi.toml" -e "${AUTODIFF_ENV:-default}" bash -c '
    mojo build noeira_max/autodiff/capi/train_step.mojo -o "$0/train_step" \
        -Xlinker -L"$CONDA_PREFIX/lib" -Xlinker -lmax &&
    "$0/train_step" "$0" "$1" > "$0/mojo.txt"' "$OUT" "$STEPS"
noeira_max/autodiff/run.sh noeira_max/autodiff/capi/compare.py "$OUT" "$STEPS"
