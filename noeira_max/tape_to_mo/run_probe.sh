#!/usr/bin/env bash
# Builds and runs the kernel probe (probe_kernels.mojo) on the accelerator.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${TAPE_MO_ENV:-apple}"
OUT="${OUT:-${TMPDIR:-/tmp}/tape_to_mo_probe_$ENV}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
mkdir -p "$OUT"
cd "$ROOT"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
    mojo build -I . -I noeira_max -I noeira_max/capi_mojo \
    noeira_max/tape_to_mo/probe_kernels.mojo -o "$OUT/probe_kernels" \
    -Xlinker -L"$LIB" -Xlinker -lmax -Xlinker -rpath -Xlinker "$LIB"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" env -u LD_PRELOAD "$OUT/probe_kernels" "$OUT" ${PROBE_ARGS:-}
