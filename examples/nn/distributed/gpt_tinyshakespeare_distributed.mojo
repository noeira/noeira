"""TinyShakespeare char-GPT on N ranks — DDP or ZeRO-1/2/3, via
`DistributedAutoregressiveTrainer`.

The single-GPU example (`examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo`)
on N ranks. Two pieces change:

  - **The strategy** builds the model, one replica per rank, and owns the
    optimizer state and the collectives. It is a type, picked here by `-D ZERO`:
        ZERO=0  DataParallel[NET, N]     DDP: gradients all-reduced
        ZERO=1  Zero1[NET, N]            Adam state sharded over the ranks
        ZERO=2  ZeroSharded[NET, N, 2]   gradients sharded too
        ZERO=3  ZeroSharded[NET, N, 3]   weights sharded too
  - **`DistributedAutoregressiveTrainer`** drives any strategy through one
    interface (`ShardingStrategy`). Each rank builds its rows of the global
    batch on its own device; `fit`, `val_loss` and `generate` are the
    single-GPU trainer's.

`train` below is the same code for the four strategies; only the type it is
called with differs. The model is `GPTMarked`, the GPT with every block a
sharding unit (ZeRO-2/3 reduce and gather block by block), which trains bit
for bit as `GPTDropTied`.

The ranks run on N GPUs when the process sees N NVIDIA GPUs (MAX `comm`),
and otherwise all on one device: the simulator, with the same data flow and
no speedup, so the example also runs on a Mac. On NVIDIA it trains nanoGPT's
char-Shakespeare config with the single-GPU example's batch of 64, split over
the ranks; elsewhere a small config.

Defines (`mojo build -D ...`, then run the binary):
  NGPUS=N     ranks (default 2)
  ZERO=s      0 = DDP (default), 1, 2 or 3
  NN_ITERS=k  training steps

NVIDIA, 4 GPUs, ZeRO-3:
    pixi run -e nvidia mojo build -I . -D NGPUS=4 -D ZERO=3 \
        examples/nn/distributed/gpt_tinyshakespeare_distributed.mojo -o /tmp/gpt_dist && /tmp/gpt_dist
Mac (simulated ranks). `-Xlinker -ld_classic`: the nested strategy, trainer
and model types mangle to a symbol longer than the new macOS linker accepts.
    pixi run -e apple mojo build -I . -Xlinker -ld_classic -D ZERO=3 \
        examples/nn/distributed/gpt_tinyshakespeare_distributed.mojo -o /tmp/gpt_dist && /tmp/gpt_dist
"""

from std.random import seed
from std.math import log
from std.time import perf_counter_ns
from std.sys import get_defined_int, default_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Normal
from noeira.nn.distributed.process_group import (
    ProcessGroup,
    BACKEND_SHARED,
    backend_name,
)
from noeira.nn.distributed.strategy import ShardingStrategy
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero import Zero1
from noeira.nn.distributed.zero_sharded import ZeroSharded
from noeira.nn.distributed.autoregressive_trainer import (
    DistributedAutoregressiveTrainer,
)
from noeira.nn.distributed.gpt_marked import (
    GPTMarked,
    gpt_marked_scale_residual_proj,
    gpt_marked_wire_tie,
)
from noeira.core.fmt import fit


comptime NGPUS = get_defined_int["NGPUS", 2]()
comptime ZERO = get_defined_int["ZERO", 0]()

# ── nanoGPT char-Shakespeare on NVIDIA; a small config elsewhere ──
comptime FULL = default_accelerator().is_nvidia_gpu()
comptime VOCAB = 65
comptime SEQ = 256 if FULL else 64
comptime EMBED = 384 if FULL else 64
comptime HEADS = 6 if FULL else 4
comptime LAYERS = 6 if FULL else 2
comptime FF_MULT = 4
comptime BATCH = 64 if FULL else 16
"""The global batch, split evenly over the ranks."""
comptime B_RANK = BATCH // NGPUS

comptime BASE_LR: Scalar[DT] = 1e-3
comptime BETA2: Scalar[DT] = 0.99
comptime WD: Scalar[DT] = 0.1
comptime TOTAL_ITERS = get_defined_int["NN_ITERS", 5000 if FULL else 400]()
comptime WARMUP_ITERS = 100 if FULL else 20
comptime EVAL_INTERVAL = 250 if FULL else 100
comptime MIN_LR_SCALE: Float64 = 0.1
comptime N_VAL_WINDOWS = 256 if FULL else 64
comptime DROPOUT_P: Float64 = 0.2
comptime GRAD_CLIP: Scalar[DT] = 1.0
comptime SEED_BASE = UInt64(0xC0FFEE)
comptime USE_MAX_ATTN = True
comptime UNITS = ZERO >= 2
"""Every block a sharding unit: what ZeRO-2/3 reduce and gather by."""
comptime MAX_COLLECTIVE = 16 << 20
"""The largest allreduce on a GPU box, in fp32 elements (DDP's gradient
arena: 10.8M for the full GPT); MAX sizes its Signal buffers by it."""

comptime NET = GPTMarked[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P, SEED_BASE,
    USE_MAX_ATTN, UNITS,
]


def _nanogpt_init[M: Module](mut net: M, ctx: DeviceContext) raises:
    """nanoGPT's 1/sqrt(2L) residual-projection init and the weight-tied LM
    head, on one rank's replica (M is NET, the strategy's model type)."""
    ref gpt = rebind[NET](net)
    gpt_marked_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED_BASE, USE_MAX_ATTN, UNITS,
    ](gpt, Optional(ctx))
    gpt_marked_wire_tie[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED_BASE, USE_MAX_ATTN, UNITS,
    ](gpt)


def train[S: ShardingStrategy]() raises:
    seed(42)
    print("=" * 70)
    print(
        "TinyShakespeare GPT training — " + S.name() + " on "
        + String(NGPUS) + " ranks (DistributedAutoregressiveTrainer)"
    )
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS) + " batch=" + String(BATCH)
        + " (" + String(B_RANK) + " per rank)"
    )
    print(
        "  base_lr=" + String(BASE_LR) + " wd=" + String(WD)
        + " dropout_p=" + String(DROPOUT_P)
        + " grad_clip=" + String(GRAD_CLIP)
    )
    print(
        "  total_iters=" + String(TOTAL_ITERS)
        + " warmup=" + String(WARMUP_ITERS)
        + " eval_interval=" + String(EVAL_INTERVAL)
        + " n_val_windows=" + String(N_VAL_WINDOWS)
    )

    print("\n[data] loading TinyShakespeare...")
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error(
            "vocab mismatch: " + String(tok.vocab_size) + " vs " + String(VOCAB)
        )
    var split = train_val_split(tok.encode(text), 0.1)

    # ── The strategy: one replica per rank, the optimizer, the collectives ──
    var ctx = DeviceContext()
    var pg = ProcessGroup[S.RANKS].devices_or_simulator(ctx, MAX_COLLECTIVE)
    if pg.backend == BACKEND_SHARED:
        print("[init] " + String(NGPUS) + " ranks simulated on one device")
    else:
        print(
            "[init] " + String(NGPUS) + " ranks on " + String(NGPUS)
            + " GPUs, MAX comm " + backend_name(pg.backend)
        )
    var dist = S.make[Normal[0.0, 0.02]](pg^, lr=BASE_LR, beta2=BETA2, wd=WD)
    for r in range(S.RANKS):
        var c = dist.ctx(r)
        _nanogpt_init(dist.net(r), c)

    # ── The trainer: the same code for DDP and every ZeRO stage ──
    var tr = DistributedAutoregressiveTrainer[S, VOCAB, SEQ, B_RANK].make_from(
        dist^, tok^, split^,
        BASE_LR, WARMUP_ITERS, TOTAL_ITERS, MIN_LR_SCALE, GRAD_CLIP,
    )
    print(
        "  weights, gradients and Adam state per rank: "
        + fit(String(Float64(tr.dist.state_bytes_per_rank(0)) / 1e6), 6)
        + " MB"
    )

    var val_init = tr.val_loss(N_VAL_WINDOWS)
    print(
        "\n[iter 0] val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var val_final = tr.fit(
        eval_every=EVAL_INTERVAL, n_val_windows=N_VAL_WINDOWS
    )
    var t1 = perf_counter_ns()
    print("\n  training time: " + fit(String(Float64(t1 - t0) / 1e9), 6) + " s")
    print(
        "\n[final] val_loss=" + String(val_final)
        + " (start " + String(val_init) + ")"
    )

    var prompt = String("ROMEO:")
    print("\n[sample] greedy (T=0.0):")
    print(prompt + tr.generate(prompt, 200, temperature=0.0))
    print("\n[sample] temperature (T=0.8):")
    print(prompt + tr.generate(prompt, 200, temperature=0.8))
    print("\n" + "=" * 70)


def main() raises:
    comptime assert ZERO >= 0 and ZERO <= 3, "ZERO is 0 (DDP), 1, 2 or 3"
    comptime assert (
        BATCH % NGPUS == 0
    ), "the global batch must split evenly over the ranks"
    comptime if ZERO == 0:
        train[DataParallel[NET, NGPUS]]()
    elif ZERO == 1:
        train[Zero1[NET, NGPUS]]()
    elif ZERO == 2:
        train[ZeroSharded[NET, NGPUS, 2]]()
    else:
        train[ZeroSharded[NET, NGPUS, 3]]()
