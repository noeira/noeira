"""TinyShakespeare char-GPT training — nn GPU, via `AutoregressiveTrainer`.

The full nanoGPT recipe expressed entirely through the framework:

  - **Model** (`GPTDropTied`): a dropout GPT whose LM head is a bias-less
    `TiedLinear` sharing the token-embedding table (nanoGPT's
    `lm_head.weight = wte.weight`). Tying is structural — no per-step code.
  - **`gpt_scale_residual_proj`** : 1/√(2L) c_proj scaled init (once).
  - **`gpt_wire_tie`** : point the head at the embedding buffers (once).
  - **`optim.max_grad_norm`** : nanoGPT's grad-norm clip (1.0), native to AdamW.
  - **`AutoregressiveTrainer`** : the task-specialized driver. Its `fit()`
    packages the whole run — streaming random-window sampling, cosine LR
    warmup/decay, the packaged `train_step`, and periodic per-token val CE.
    `val_loss` / `val_top1` / `generate` cover eval + sampling.

So training + eval + generation are a handful of calls; there is no
hand-written loop, no manual forward/vjp/step, and no tying plumbing. See
`noeira/nn/storage/training/autoregressive_trainer.mojo`,
`noeira/nn/storage/primitives/tied_linear.mojo`, and
`noeira/nn/storage/models/gpt.mojo`.

Default config is sized for NVIDIA. On Apple it OOMs — shrink SEQ→64,
EMBED→64, LAYERS→2, BATCH→16, TOTAL_ITERS→400 for a dev run.

Run on NVIDIA:
    pixi run -e nvidia mojo run -I . examples/nn/transformer/gpt_tinyshakespeare_training_gpu.mojo
"""

from std.random import seed
from std.math import log, exp
from std.time import perf_counter_ns
from std.sys import get_defined_int, is_defined
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.constants import DT
from noeira.nn.models.gpt import GPTDropTied
from noeira.nn.models.gpt import gpt_scale_residual_proj, gpt_wire_tie
from noeira.nn.optimizer.adam import AdamW
from noeira.nn.training.autoregressive_trainer import (
    AutoregressiveTrainer,
)
from noeira.nn.core.initializer import Normal
from noeira.core.fmt import fit


# ── Full nanoGPT-class config (NVIDIA) ──
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

comptime TOTAL_ITERS = get_defined_int["NN_ITERS", 5000]()
comptime WARMUP_ITERS = 100
comptime EVAL_INTERVAL = 250
comptime MIN_LR_SCALE: Float64 = 0.1
comptime N_VAL_WINDOWS = 256

# Attention kernel path: True → batched-GEMM (fast on NVIDIA); False →
# portable serial per-(b,h) kernels. Bit-identical on Metal.
comptime USE_MAX_ATTN = True
# Dropout (nanoGPT char-Shakespeare uses 0.2) — without it the small corpus
# is memorized and greedy generation degenerates. Off during eval/gen.
comptime DROPOUT_P: Float64 = 0.2
# 1/√(2L) c_proj scaled init + nanoGPT's grad-norm clip.
comptime SCALED_INIT = True
comptime GRAD_CLIP: Scalar[DT] = 1.0

comptime GPT_MODEL = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
    UInt64(0xC0FFEE), USE_MAX_ATTN,
]
# `-D NN_TRAIN_GRAPH`: the training step captured in a CUDA graph (NVIDIA;
# eager elsewhere). `-D NN_HOST_BATCH`: build each batch on the host (sample +
# one-hot + upload) instead of on the device — the pre-device-batch baseline.
# `-D NN_ITERS=N` shortens the run. Defines need `mojo build -D ...`, then
# run the binary.
comptime USE_CUDA_GRAPH = is_defined["NN_TRAIN_GRAPH"]()
comptime DEVICE_BATCH = not is_defined["NN_HOST_BATCH"]()
comptime GPT_AR = AutoregressiveTrainer[
    GPT_MODEL, AdamW, VOCAB, SEQ, BATCH, target="gpu",
    USE_TRAIN_CUDA_GRAPH=USE_CUDA_GRAPH, DEVICE_BATCH=DEVICE_BATCH,
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("TinyShakespeare GPT training — nn GPU (AutoregressiveTrainer)")
    print("=" * 70)
    print(
        "  vocab=" + String(VOCAB) + " seq=" + String(SEQ)
        + " embed=" + String(EMBED) + " heads=" + String(HEADS)
        + " layers=" + String(LAYERS) + " batch=" + String(BATCH)
    )
    print(
        "  base_lr=" + String(BASE_LR) + " wd=" + String(WD)
        + " dropout_p=" + String(DROPOUT_P)
        + " | tie=structural(TiedLinear) scaled_init=" + String(SCALED_INIT)
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
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    print(
        "  tokens=" + String(len(ids)) + " train=" + String(len(split.train))
        + " val=" + String(len(split.val))
    )

    # ── Build model + optimizer ──
    var ctx = DeviceContext()
    print("[init] building nn GPT on GPU...")
    var net = GPT_MODEL.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    # AdamW = storage Adam with decoupled weight decay (`wd`); nanoGPT's
    # grad-norm clip is applied per step by the driver (grad_clip arg).
    var optim = AdamW(lr=BASE_LR, beta2=BETA2, wd=WD)

    # ── Wrap in the autoregressive driver (owns corpus + sampling + LR; engages
    # the optimizer's arena via opt.adopt once the net is in its final home) ──
    var artr = GPT_AR.make_from(
        net^, optim^, tok^, split^, ctx,
        BASE_LR, WARMUP_ITERS, TOTAL_ITERS, MIN_LR_SCALE, GRAD_CLIP,
    )

    # ── Model-construction surgery on the net in its final home ──
    comptime if SCALED_INIT:
        gpt_scale_residual_proj[
            "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
            UInt64(0xC0FFEE), USE_MAX_ATTN,
        ](artr.net, Optional(ctx))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        UInt64(0xC0FFEE), USE_MAX_ATTN,
    ](artr.net)
    ctx.synchronize()

    var val_init = artr.val_loss(N_VAL_WINDOWS)
    print(
        "\n[iter 0] val_loss=" + String(val_init)
        + "  (random ≈ ln(V)=" + String(log(Float64(VOCAB))) + ")"
    )

    # ── Train: the whole run is one call ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var val_final = artr.fit(
        eval_every=EVAL_INTERVAL, n_val_windows=N_VAL_WINDOWS
    )
    var t1 = perf_counter_ns()
    print("\n  training time: " + fit(String(Float64(t1 - t0) / 1e9), 6) + " s")

    print("\n[final] val_loss=" + String(val_final) + " (start " + String(val_init) + ")")
    if val_final < val_init - 0.1:
        print("  PASS: val loss decreased by > 0.1 nats")
    else:
        print("  WARN: val loss did not improve substantially")

    # ── Diagnostic: per-token top-1 (is the loss consistent with prediction?) ──
    var acc = artr.val_top1(N_VAL_WINDOWS)
    print(
        "[diagnostic] val per-token top-1=" + fit(String(acc * 100.0), 5)
        + "%  (random ≈ " + fit(String(100.0 / Float64(VOCAB)), 4)
        + "%, from loss≈" + fit(String(val_final), 5)
        + " expect ~" + fit(String(exp(-val_final) * 100.0), 5) + "%)"
    )

    # ── Sampling ──
    var prompt = String("ROMEO:")
    print("\n[sample] prompt = " + repr(prompt))
    print("\n[sample] greedy (T=0.0):")
    print(prompt + artr.generate(prompt, 200, temperature=0.0))
    print("\n[sample] temperature (T=0.8, no top-k):")
    print(prompt + artr.generate(prompt, 200, temperature=0.8))

    var long_prompt = String(text[byte=0:250])
    print(
        "\n[sample] long-prompt diagnostic (250 real chars):\n"
        + "---- prompt ----\n" + long_prompt + "\n---- continuation (greedy) ----"
    )
    print(artr.generate(long_prompt, 200, temperature=0.0))
    print("---- continuation (T=0.8) ----")
    print(artr.generate(long_prompt, 200, temperature=0.8))
    print("\n" + "=" * 70)
