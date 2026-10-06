#!/usr/bin/env bash
# One MLP, eager on noeira's kernels or staged as a MAX graph built in Mojo,
# on the same weight memory: builds and runs the CPU crossover
# sweep, then draws it (crossover.svg next to the log).
#
#   noeira_max/staged_vs_eager/run.sh          # CPU, host memory
#   noeira_max/staged_vs_eager/run.sh --gpu    # CUDA, noeira device buffers lent to MAX
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${MAXRT_ENV:-default}"
OUT="${OUT:-${TMPDIR:-/tmp}/staged_vs_eager}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
PIXI=(pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV")
cd "$ROOT"
mkdir -p "$OUT"

"${PIXI[@]}" mojo build -I . -I noeira_max/graph_mojo -I noeira_max/capi_mojo \
    noeira_max/staged_vs_eager/staged_vs_eager.mojo -o "$OUT/staged_vs_eager" \
    -Xlinker -L"$LIB" -Xlinker -lmax
"${PIXI[@]}" env -u LD_PRELOAD PYTHONPATH="$ROOT" "$OUT/staged_vs_eager" "$OUT" "$@" | tee "$OUT/crossover.log"
"${PIXI[@]}" python noeira_max/staged_vs_eager/plot.py "$OUT/crossover.log" "$OUT/crossover.svg"
