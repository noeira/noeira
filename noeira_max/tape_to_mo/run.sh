#!/usr/bin/env bash
# Builds and runs the tape -> MO -> MAX gate (noeira_max/tape_to_mo/).
#   noeira_max/tape_to_mo/run.sh                 # CPU (env apple, host device)
#   TAPE_MO_ENV=default noeira_max/tape_to_mo/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${TAPE_MO_ENV:-apple}"
OUT="${OUT:-${TMPDIR:-/tmp}/tape_to_mo_$ENV}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
mkdir -p "$OUT"
cd "$ROOT"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
    mojo build -I . -I noeira_max -I noeira_max/capi_mojo \
    noeira_max/tape_to_mo/test_tape_to_mo.mojo -o "$OUT/test_tape_to_mo" \
    -Xlinker -L"$LIB" -Xlinker -lmax -Xlinker -rpath -Xlinker "$LIB"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" env -u LD_PRELOAD "$OUT/test_tape_to_mo" "$OUT" ${TAPE_MO_ARGS:-}
