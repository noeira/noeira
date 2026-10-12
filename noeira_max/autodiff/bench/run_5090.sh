#!/usr/bin/env bash
# The GPT train-step benchmarks on one GPU, one after the other: torch, the
# MAX step, compile time against depth, whole fits, and the kernel-backed
# rules. Each command prints a `RESULT {json}` line;
# all of them are collected in $OUT. Compile minutes are rented minutes:
# run `--smoke` first to check the box end to end at a tiny size.
#
#   noeira_max/autodiff/bench/run_5090.sh                # every section
#   noeira_max/autodiff/bench/run_5090.sh --smoke        # torch + max: 1 layer, 5 steps
#   noeira_max/autodiff/bench/run_5090.sh torch max      # some of: torch max scaling fit kernels mlp
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
RUN="$ROOT/noeira_max/autodiff/run.sh"
# torch.compile (triton) needs two things the act-ref env does not give a bare
# python: the CUDA driver header, to build its launcher with gcc (borrowed
# from the default env: CUDA 12.9, the version torch is built against), and
# CONDA_PREFIX, by which conda-forge's triton finds the env's ptxas for sm_120.
ACT="$MAIN/.pixi/envs/act-ref"
CUDA_INCLUDE="$MAIN/.pixi/envs/default/targets/x86_64-linux/include"
TORCH="env -u LD_PRELOAD CONDA_PREFIX=$ACT CPATH=$CUDA_INCLUDE $ACT/bin/python"
OUT="${OUT:-$ROOT/bench_5090_$(date +%Y%m%d_%H%M%S).log}"
export AUTODIFF_ENV="${AUTODIFF_ENV:-default}"  # `default` is the nvidia feature

STEPS=200
SIZE=()
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=1
    STEPS=5
    SIZE=(--layers 1)
    shift
fi
SECTIONS=("$@")
if [[ ${#SECTIONS[@]} -eq 0 ]]; then
    SECTIONS=(torch max)
    (( SMOKE )) || SECTIONS+=(scaling fit)
fi
want() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }
cd "$ROOT"

run() {  # run <label> <command...>
    echo "=== $1" | tee -a "$OUT"
    shift
    "$@" 2>&1 | tee -a "$OUT" | grep -E "RESULT|steady|compiled|Error|error" || true
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$OUT"

if want torch; then
    # torch, the twin's own recipe (flash attention), three modes.
    for mode in eager compile cudagraph; do
        run "torch $mode" $TORCH tools/nn/torch_nn_reference.py gpt --mode "$mode" \
            --bench-steps "$STEPS"
    done
    # torch with the same attention math as the MAX step (scores materialised).
    for mode in eager compile; do
        run "torch $mode, MATH attention" $TORCH noeira_max/autodiff/bench/torch_twin_math.py \
            gpt --mode "$mode" --bench-steps "$STEPS"
    done
fi

if want max; then
    # MAX: the one-graph train step, executed, then captured and replayed.
    for mode in execute capture; do
        run "max $mode" "$RUN" noeira_max/autodiff/bench/bench_gpt_max.py --mode "$mode" \
            --bench-steps "$STEPS" "${SIZE[@]}"
    done
fi

if want kernels; then
    # Kernel-backed rules (Mojo custom ops, forward + backward +
    # residuals): LayerNorm, and noeira's fused attention. Each alone, then
    # in the 6-layer step, against the composite rules.
    run "kernels: layer norm, alone" "$RUN" noeira_max/autodiff/bench/layer_norm_kernel.py --device gpu
    run "kernels: attention, alone" "$RUN" noeira_max/autodiff/bench/attention_kernel.py --device gpu
    for kinds in "composite composite" "kernel composite" "composite kernel" "kernel kernel"; do
        set -- $kinds
        run "kernels: max execute, layer norm $1, attention $2" "$RUN" \
            noeira_max/autodiff/bench/bench_gpt_max.py --mode execute --layer-norm "$1" \
            --attention "$2" --bench-steps "$STEPS" "${SIZE[@]}"
    done
fi

if want mlp; then
    # Small MLP train steps at the shapes of noeira's RL networks (mlp_step.py):
    # a SAC critic and a PPO network, each executed with a cold compile, then
    # captured, then profiled (captured) and summarised into kernels per step,
    # GEMM paths and memsets (nsys_summary.py). No dataset needed.
    PROF="${PROF:-${OUT%.log}_mlp_nsys}"
    mkdir -p "$PROF"
    for shape in sac ppo; do
        run "mlp $shape, execute, cold compile" "$RUN" noeira_max/autodiff/bench/mlp_step.py \
            --shape "$shape" --mode execute --cold --bench-steps "$STEPS"
        run "mlp $shape, capture" "$RUN" noeira_max/autodiff/bench/mlp_step.py \
            --shape "$shape" --mode capture --bench-steps "$STEPS"
        run "mlp $shape, nsys" nsys profile -o "$PROF/$shape" --force-overwrite=true \
            --cuda-graph-trace=node "$RUN" noeira_max/autodiff/bench/mlp_step.py \
            --shape "$shape" --mode capture --bench-steps 100 --json "$PROF/$shape.json"
        nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
            --format csv --force-export=true --force-overwrite=true \
            -o "$PROF/$shape" "$PROF/$shape.nsys-rep" > "$PROF/$shape.stats.log" 2>&1
        run "mlp $shape, kernels per step" "$RUN" noeira_max/autodiff/bench/nsys_summary.py "$PROF/$shape"
    done
    echo "mlp profiles in $PROF"
fi

if want scaling; then
    # Compile time against depth (each graph is new: see the script).
    run "max compile scaling" "$RUN" noeira_max/autodiff/bench/compile_scaling.py
fi

if want fit; then
    # noeira's own GPT has no step-time flag: it is a whole fit (5000
    # iterations + eval), compared with the twin's whole fit.
    run "torch fit, compile" $TORCH tools/nn/torch_nn_reference.py gpt --mode compile
    for example in gpt_tinyshakespeare_training_gpu gpt_tinyshakespeare_training_bf16_gpu; do
        run "noeira $example" pixi run --manifest-path "$MAIN/pixi.toml" -e "$AUTODIFF_ENV" \
            mojo run -I "$MAIN" "$MAIN/examples/nn/transformer/$example.mojo"
    done
fi

echo "results in $OUT"
