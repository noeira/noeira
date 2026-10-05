#!/usr/bin/env bash
# The M2 benchmark session (plan §4, M2): every column of the table, one
# after the other, on one GPU. Each command prints a `RESULT {json}` line;
# all of them are collected in $OUT. Compile minutes are rented minutes:
# run `--smoke` first to check the box end to end at a tiny size.
#
#   noeira_max/autodiff/bench/run_5090.sh            # the full table
#   noeira_max/autodiff/bench/run_5090.sh --smoke    # 1 layer, 5 steps
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
RUN="$ROOT/noeira_max/autodiff/run.sh"
TORCH="env -u LD_PRELOAD $MAIN/.pixi/envs/act-ref/bin/python"
OUT="${OUT:-$ROOT/bench_5090_$(date +%Y%m%d_%H%M%S).log}"
export AUTODIFF_ENV="${AUTODIFF_ENV:-nvidia}"

STEPS=200
SIZE=()
if [[ "${1:-}" == "--smoke" ]]; then
    STEPS=5
    SIZE=(--layers 1)
fi
cd "$ROOT"

run() {  # run <label> <command...>
    echo "=== $1" | tee -a "$OUT"
    shift
    "$@" 2>&1 | tee -a "$OUT" | grep -E "RESULT|steady|compiled|Error|error" || true
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$OUT"

# torch, the twin's own recipe (flash attention), three modes.
for mode in eager compile cudagraph; do
    run "torch $mode" $TORCH tools/nn/torch_nn_reference.py gpt --mode "$mode" --bench-steps "$STEPS"
done
# torch with the same attention math as the MAX step (scores materialised).
for mode in eager compile; do
    run "torch $mode, MATH attention" $TORCH noeira_max/autodiff/bench/torch_twin_math.py gpt \
        --mode "$mode" --bench-steps "$STEPS"
done

# MAX: the one-graph train step, executed, then captured and replayed.
for mode in execute capture; do
    run "max $mode" "$RUN" noeira_max/autodiff/bench/bench_gpt_max.py --mode "$mode" \
        --bench-steps "$STEPS" "${SIZE[@]}"
done

# Compile time against depth (cold: a unique graph name per run).
if [[ "${1:-}" != "--smoke" ]]; then
    run "max compile scaling" "$RUN" noeira_max/autodiff/bench/compile_scaling.py
fi

if [[ "${1:-}" == "--smoke" ]]; then
    echo "results in $OUT"
    exit 0
fi

# noeira's own GPT, the third column, has no step-time flag: it is a whole
# fit (5000 iterations + eval), compared with the twin's whole fit.
run "torch fit, compile" $TORCH tools/nn/torch_nn_reference.py gpt --mode compile
for example in gpt_tinyshakespeare_training_gpu gpt_tinyshakespeare_training_bf16_gpu; do
    run "noeira $example" pixi run --manifest-path "$MAIN/pixi.toml" -e nvidia \
        mojo run -I "$MAIN" "$MAIN/examples/nn/transformer/$example.mojo"
done

echo "results in $OUT"
