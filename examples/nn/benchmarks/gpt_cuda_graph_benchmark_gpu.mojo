"""GPT training throughput — batch build (host / device) x step (eager / graph).

Times steady-state next-token training throughput of the `AutoregressiveTrainer`
on four columns:

  - batch build: HOST (`make_batch` + one-hot on the CPU + a resident upload
    that waits for the previous copy) or DEVICE (`DEVICE_BATCH`: the corpus
    resident as int32, windows drawn and one-hot written by a kernel);
  - step: EAGER (one launch per primitive, ~hundreds per step, plus a loss
    read-back each step) or GRAPH (`USE_TRAIN_CUDA_GRAPH`: the device compute
    captured once and replayed — with DEVICE, the batch build is inside it).

host/eager is the trainer before the device batch; device/graph is the step
with no host work at all but the cosine-LR push.

Each mode: build the GPT, run BENCH_ITERS as warmup (the capture mode captures
the graph on the first step), synchronize, then time a second BENCH_ITERS of
pure training (no eval). Steps/s + speedup are printed. The models are built
SEQUENTIALLY (each dies before the next) so peak memory == one model.

⚠️ Real capture is NVIDIA-only; on non-NVIDIA `maybe_capture_replay` runs
eagerly, so each graph column repeats its eager one. On NVIDIA, whether capture wins depends on how
launch-bound the config is (smaller BATCH/seq = more launch-bound = bigger win).

Run on NVIDIA:
    pixi run -e nvidia mojo run -I . \
        examples/nn/benchmarks/gpt_cuda_graph_benchmark_gpu.mojo

Switches (`mojo build -D`, a `mojo run -D` can reuse a stale compile):
  - `GPT_BENCH_DTYPE=bf16`: the bf16-flow GPT (bf16 activations; fp32 master
    weights, loss, softmax / LayerNorm stats and AdamW), as
    `transformer/gpt_tinyshakespeare_training_bf16_gpu.mojo` — the twin of
    PyTorch's `autocast(bfloat16)` (`tools/nn/torch_nn_reference.py --amp bf16`);
  - `GPT_BENCH_COLS=devgraph`: only the device-batch / graph column (the one
    compared with PyTorch's `reduce-overhead`), one model to compile not four.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.sys.defines import get_defined_string
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.constants import DT
from noeira.nn.models.gpt import GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie
from noeira.nn.optimizer.adam import AdamW
from noeira.nn.training.autoregressive_trainer import AutoregressiveTrainer
from noeira.nn.core.initializer import Normal


# Same nanoGPT-class config as the training example.
comptime VOCAB = 65
comptime SEQ = 256
comptime EMBED = 384
comptime HEADS = 6
comptime LAYERS = 6
comptime FF_MULT = 4
comptime BATCH = 64

comptime BASE_LR: Scalar[DT] = 1e-3
comptime BETA2: Scalar[DT] = 0.99
comptime WD: Scalar[DT] = 0.1
comptime MIN_LR_SCALE: Float64 = 0.1
comptime WARMUP_ITERS = 100
comptime USE_MAX_ATTN = True
comptime DROPOUT_P: Float64 = 0.2
comptime GRAD_CLIP: Scalar[DT] = 1.0

comptime DTYPE_NAME = get_defined_string["GPT_BENCH_DTYPE", "fp32"]()
comptime ADT = DType.bfloat16 if DTYPE_NAME == "bf16" else DT
comptime ALL_COLS = get_defined_string["GPT_BENCH_COLS", "all"]() == "all"

# Per-phase iteration count (warmup phase + timed phase each run this many).
comptime BENCH_ITERS = 300

comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
    UInt64(0xC0FFEE), USE_MAX_ATTN, ADT,
]


def bench[
    CAP: Bool, DEV: Bool
](ctx: DeviceContext, ref text: String) raises -> Float64:
    """Build the GPT once, warm up BENCH_ITERS (captures the graph when CAP),
    then time a second BENCH_ITERS of pure training. Returns steps/s. The
    trainer is destroyed at return so the next mode peaks at one model."""
    comptime AR = AutoregressiveTrainer[
        GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu",
        USE_TRAIN_CUDA_GRAPH=CAP, DEVICE_BATCH=DEV,
    ]
    var tok = CharTokenizer(text)
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)

    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = AdamW(lr=BASE_LR, beta2=BETA2, wd=WD)
    var artr = AR.make_from(
        net^, optim^, tok^, split^, ctx,
        BASE_LR, WARMUP_ITERS, BENCH_ITERS, MIN_LR_SCALE, GRAD_CLIP,
    )
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN, ADT,
    ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN, ADT,
    ](artr.net)
    ctx.synchronize()

    # Warmup phase: allocates module buffers + (CAP) captures the graph.
    _ = artr.fit(eval_every=0, n_val_windows=0, print_progress=False)
    ctx.synchronize()

    # Timed phase: pure training, no eval.
    var t0 = perf_counter_ns()
    _ = artr.fit(eval_every=0, n_val_windows=0, print_progress=False)
    ctx.synchronize()
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    return Float64(BENCH_ITERS) / elapsed


def main() raises:
    seed(42)
    print("=" * 70)
    print("GPT training throughput — batch build x step")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS) + " batch=" + String(BATCH)
        + " | bench_iters=" + String(BENCH_ITERS) + " (×2 phases/mode)"
        + " | activations " + String(ADT)
    )

    print("\n[data] loading TinyShakespeare...")
    var text = load_text()

    # ONE DeviceContext for both modes: the CUDA-graph interceptor records the
    # Mojo stream from the first kernel launch globally, so capture must run on
    # the SAME context/stream (two contexts → cuStreamBeginCapture fails 400).
    # The eager trainer is freed before the capture trainer is built (each in
    # its own `bench` scope), so peak memory is still one model.
    var ctx = DeviceContext()

    comptime if ALL_COLS:
        var he = bench[False, False](ctx, text)
        print("  host batch,   eager: " + _row(he, he))
        var hg = bench[True, False](ctx, text)
        print("  host batch,   graph: " + _row(hg, he))
        var de = bench[False, True](ctx, text)
        print("  device batch, eager: " + _row(de, he))
        var dg = bench[True, True](ctx, text)
        print("  device batch, graph: " + _row(dg, he))
    else:
        var dg = bench[True, True](ctx, text)
        print("  device batch, graph: " + _row(dg, dg))
    print("=" * 70)


def _row(sps: Float64, base: Float64) -> String:
    return (
        String(sps) + " steps/s | " + String(1000.0 / sps) + " ms/step | "
        + String(sps / base) + "x host/eager"
    )
