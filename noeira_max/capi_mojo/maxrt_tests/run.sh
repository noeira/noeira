#!/usr/bin/env bash
# Exports the test MEFs (Python, once), builds the maxrt test binary (Mojo),
# and runs it twice: inside the pixi env, then outside it with MODULAR_HOME
# unset, which exercises maxrt's own runtime configuration (config.mojo). On a
# machine with an NVIDIA GPU, it then checks the Mojo <-> MAX device round trip
# (probe_cuda_context.mojo).
#
#   noeira_max/capi_mojo/maxrt_tests/run.sh
#   MAXRT_ENV=apple noeira_max/capi_mojo/maxrt_tests/run.sh    # + Metal
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${MAXRT_ENV:-default}"
OUT="${OUT:-${TMPDIR:-/tmp}/maxrt_tests_$ENV}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
cd "$ROOT"

pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
    python noeira_max/capi_mojo/maxrt_tests/make_mefs.py "$OUT"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
    mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/maxrt_tests/test_maxrt.mojo \
    -o "$OUT/test_maxrt" -Xlinker -L"$LIB" -Xlinker -lmax

echo "== inside pixi ($ENV): MODULAR_HOME from activation"
pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" env -u LD_PRELOAD "$OUT/test_maxrt" "$OUT"
echo "== outside pixi: MODULAR_HOME unset, maxrt writes its own modular.cfg"
env -u MODULAR_HOME -u CONDA_PREFIX -u LD_PRELOAD "$OUT/test_maxrt" "$OUT"

if command -v nvidia-smi > /dev/null; then
    echo "== CUDA: a Mojo DeviceContext and MAX share device memory"
    pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
        mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/maxrt_tests/probe_cuda_context.mojo \
        -o "$OUT/probe_cuda_context" -Xlinker -L"$LIB" -Xlinker -lmax -Xlinker -lcuda
    pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV" \
        env -u LD_PRELOAD "$OUT/probe_cuda_context" "$OUT"
fi
