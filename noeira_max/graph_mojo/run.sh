#!/usr/bin/env bash
# Regenerates the Mojo graph builder from the installed MAX's op stubs,
# builds and runs the parity tests (Mojo-built graphs against max.graph's,
# bit for bit), then the example that builds, compiles and runs a graph from
# one Mojo program.
#
#   noeira_max/graph_mojo/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${MAXRT_ENV:-default}"
OUT="${OUT:-${TMPDIR:-/tmp}/graph_mojo}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
PIXI=(pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV")
INCLUDES=(-I noeira_max/graph_mojo -I noeira_max/capi_mojo)
cd "$ROOT"
mkdir -p "$OUT"

"${PIXI[@]}" python noeira_max/graph_mojo/gen/extract_schema.py
"${PIXI[@]}" python noeira_max/graph_mojo/gen/gen_mojo_builder.py
for program in tests/test_parity examples/mlp_from_mojo; do
    "${PIXI[@]}" mojo build "${INCLUDES[@]}" "noeira_max/graph_mojo/$program.mojo" \
        -o "$OUT/$(basename "$program")" -Xlinker -L"$LIB" -Xlinker -lmax
done
# The programs import max.graph through Python: run them inside the env,
# with this checkout on PYTHONPATH and without noeira's CUDA interposer.
echo "== parity: Mojo-built graphs against max.graph"
"${PIXI[@]}" env -u LD_PRELOAD PYTHONPATH="$ROOT" "$OUT/test_parity" "$OUT"
echo "== build, compile and run from one Mojo program"
"${PIXI[@]}" env -u LD_PRELOAD PYTHONPATH="$ROOT" "$OUT/mlp_from_mojo" "$OUT"
