#!/usr/bin/env bash
# Train steps defined in Mojo, trained on MAX from Mojo.
#
#   noeira_max/train_from_mojo/run.sh [--gpu] [--shape small|ppo|sac] [--steps N]
#       the gate: an MLP step built in Mojo against the same step built by the
#       Python prototype; a mutated step, which must fail it; a step with
#       noeira's LayerNorm kernel pair (custom ops); then the 2-layer GPT with
#       both of noeira's kernel pairs (LayerNorm and attention); each against
#       the same step built in Python
#   noeira_max/train_from_mojo/run.sh --nn [--gpu] [--steps N] [--no-bump]
#       the hybrid: the step trains noeira nn layers' own weights in place
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
ENV="${MAXRT_ENV:-default}"
OUT="${OUT:-${TMPDIR:-/tmp}/train_from_mojo}"
LIB="$MAIN/.pixi/envs/$ENV/lib"
PIXI=(pixi run --manifest-path "$MAIN/pixi.toml" -e "$ENV")
INCLUDES=(-I noeira_max/train_from_mojo -I noeira_max/graph_mojo -I noeira_max/capi_mojo)
cd "$ROOT"
mkdir -p "$OUT"
# The programs build graphs and differentiate through Python: run them inside
# the env, with this checkout on PYTHONPATH and without noeira's CUDA
# interposer.
RUN=("${PIXI[@]}" env -u LD_PRELOAD PYTHONPATH="$ROOT")

if [[ "${1:-}" == "--nn" ]]; then
    shift
    "${PIXI[@]}" mojo build -I . "${INCLUDES[@]}" noeira_max/train_from_mojo/train_nn_mlp.mojo \
        -o "$OUT/train_nn_mlp" -Xlinker -L"$LIB" -Xlinker -lmax
    "${RUN[@]}" "$OUT/train_nn_mlp" "$OUT" "$@"
    exit
fi

for program in train_mlp train_ln_mlp train_gpt; do
    "${PIXI[@]}" mojo build "${INCLUDES[@]}" "noeira_max/train_from_mojo/$program.mojo" \
        -o "$OUT/$program" -Xlinker -L"$LIB" -Xlinker -lmax
done
echo "== the gate"
"${RUN[@]}" "$OUT/train_mlp" "$OUT" "$@"
echo "== a mutated step (twice the learning rate) must fail it"
"${RUN[@]}" "$OUT/train_mlp" "$OUT" "$@" --mutate
echo "== a step with a Mojo kernel pair (LayerNorm custom ops)"
GPU=(); [[ " $* " == *" --gpu "* ]] && GPU=(--gpu)
"${RUN[@]}" "$OUT/train_ln_mlp" "$OUT" ${GPU[@]+"${GPU[@]}"}  # bash 3.2: an empty array under set -u
echo "== the GPT step, with the LayerNorm and attention kernel pairs"
"${RUN[@]}" "$OUT/train_gpt" "$OUT" ${GPU[@]+"${GPU[@]}"}
