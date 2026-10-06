#!/usr/bin/env bash
# The MAX-from-Mojo MLP table on one GPU (noeira_max/README.md), every column from
# this box:
#   (a), (b)  benchmark_interop.mojo: MAX through Python, from Mojo
#   (c)-(e)   bench_capi.mojo: MAX through its C API (maxrt), no Python
#   (f)       benchmark_nn_baseline.mojo: noeira's own nn
# MAX runs never see noeira's CUDA interposer, which `pixi run` preloads on
# Linux; the nn baseline runs as noeira always does, with it.
#
#   noeira_max/capi_mojo/bench/run.sh            # (c)-(e) only
#   noeira_max/capi_mojo/bench/run.sh --all      # every column
#   noeira_max/capi_mojo/bench/run.sh --stream   # M1.4: Mojo kernels on MAX's own stream
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${MAXRT_ENV:-default}"
OUT="${OUT:-${TMPDIR:-/tmp}/maxrt_bench}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
PIXI=(pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV")
cd "$ROOT"
mkdir -p "$OUT"

"${PIXI[@]}" env -u LD_PRELOAD PYTHONPATH="$ROOT" \
    python noeira_max/capi_mojo/bench/make_mlp_mefs.py "$OUT"
if [[ "${1:-}" == "--stream" ]]; then
    # The one MAX run WITH noeira's interposer preloaded: it is how the
    # benchmark finds MAX's stream, which the C API does not expose.
    "${PIXI[@]}" mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/bench/bench_shared_stream.mojo \
        -o "$OUT/bench_shared_stream" -Xlinker -L"$LIB" -Xlinker -lmax
    echo "== M1.4: one stream for Mojo and MAX"
    "${PIXI[@]}" "$OUT/bench_shared_stream" "$OUT"
    exit 0
fi
"${PIXI[@]}" mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/bench/bench_capi.mojo \
    -o "$OUT/bench_capi" -Xlinker -L"$LIB" -Xlinker -lmax
echo "== (c)-(e): MAX C API from Mojo"
"${PIXI[@]}" env -u LD_PRELOAD "$OUT/bench_capi" "$OUT"

if [[ "${1:-}" == "--all" ]]; then
    "${PIXI[@]}" mojo build -I . noeira_max/benchmark_interop.mojo -o "$OUT/bench_interop"
    echo "== (a), (b): MAX through Python, from Mojo"
    "${PIXI[@]}" env -u LD_PRELOAD "$OUT/bench_interop"
    echo "== (f): noeira nn"
    "${PIXI[@]}" mojo run -I . noeira_max/benchmark_nn_baseline.mojo
fi
