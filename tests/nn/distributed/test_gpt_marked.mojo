"""GPTMarked == GPTDropTied, bit for bit.

`gpt_marked.mojo` restates `gpt.mojo`'s construction ops for the marked
type (one `.inner` per block). This gate builds both models from the same
seed, applies each file's own ops, and checks:

  1. the initial parameter arenas are identical (the scaled residual init and
     the tie landed on the same weights);
  2. K training steps on a fixed batch give identical losses and arenas, with
     the marks compiled out (`ACTIVE = False`) and compiled in (`True`, no
     registry open, so `GradReady` only checks the registry and returns).

GPT 2 x 64, seq 64, dropout off, AdamW + clip, one rank, simulator backend.

Run:  pixi run -e apple mojo run -I . tests/nn/distributed/test_gpt_marked.mojo
"""

from std.random import seed
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Normal
from noeira.nn.core.module import Module
from noeira.nn.loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from noeira.nn.models.gpt import GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie
from noeira.nn.distributed.process_group import ProcessGroup
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.gpt_marked import (
    GPTMarked,
    gpt_marked_scale_residual_proj,
    gpt_marked_wire_tie,
)


comptime VOCAB = 65
comptime SEQ = 64
comptime EMBED = 64
comptime HEADS = 4
comptime LAYERS = 2
comptime FF = 4
comptime DROP: Float64 = 0.0
comptime SB = UInt64(0xC0FFEE)
comptime BL = 4
comptime K = 8
comptime ROW = SEQ * VOCAB
comptime PLAIN = GPTDropTied[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True]
comptime LOSS = SequenceCrossEntropyLoss[SEQ, VOCAB]


struct _Out(Movable):
    var init: List[Scalar[DT]]
    var final: List[Scalar[DT]]
    var losses: List[Float64]

    def __init__(out self):
        self.init = List[Scalar[DT]]()
        self.final = List[Scalar[DT]]()
        self.losses = List[Float64]()


def _batch(c: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    """A fixed one-hot batch: token (b, t) = (7b + 3t + t^2) mod VOCAB, the
    target the next one."""
    var x = Tensor.alloc(BL * ROW)
    var y = Tensor.alloc(BL * ROW)
    for i in range(BL * ROW):
        x.data[i] = 0.0
        y.data[i] = 0.0
    for b in range(BL):
        for t in range(SEQ):
            var tok = (7 * b + 3 * t + t * t) % VOCAB
            var nxt = (7 * b + 3 * (t + 1) + (t + 1) * (t + 1)) % VOCAB
            x.data[b * ROW + t * VOCAB + tok] = 1.0
            y.data[b * ROW + t * VOCAB + nxt] = 1.0
    x.upload(c)
    y.upload(c)
    return (x^, y^)


def _loop[M: Module](mut dp: DataParallel[M, 1], ctx: DeviceContext) raises -> _Out:
    comptime A = M.ACT_DT
    dp.synchronize()
    var out = _Out()
    out.init = dp.download_params(0)
    var xy = _batch(ctx)
    var logits = Tensor.alloc_gpu(ctx, BL * ROW)
    var grad = Tensor.alloc_gpu(ctx, BL * ROW)
    var gi = Tensor.alloc_gpu(ctx, BL * ROW)
    var loss = LOSS.make_gpu(ctx)
    var co = Optional(ctx)
    for _ in range(K):
        dp.zero_grad()
        dp.nets[0].forward["gpu", BL](
            child_refs[M.ARITY, A](rebind[TensorImpl[A]](xy[0])),
            rebind[TensorImpl[A]](logits),
            co,
        )
        out.losses.append(Float64(loss.forward["gpu", BL](logits, xy[1], co)))
        loss.vjp["gpu", BL](logits, xy[1], grad, co)
        dp.nets[0].vjp["gpu", BL](
            child_refs[M.ARITY, A](rebind[TensorImpl[A]](xy[0])),
            rebind[TensorImpl[A]](grad),
            child_refs[M.ARITY, A](rebind[TensorImpl[A]](gi)),
            co,
        )
        dp.clip_grads_device(1.0)
        dp.step()
    dp.synchronize()
    out.final = dp.download_params(0)
    return out^


def _plain(ctx: DeviceContext) raises -> _Out:
    seed(42)
    var dp = DataParallel[PLAIN, 1].make[Normal[0.0, 0.02]](
        ProcessGroup[1].shared(ctx), lr=1e-3, beta2=0.99, wd=0.1
    )
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True
    ](dp.nets[0], Optional(ctx))
    gpt_wire_tie["gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True](
        dp.nets[0]
    )
    return _loop(dp, ctx)


def _marked[ACTIVE: Bool](ctx: DeviceContext) raises -> _Out:
    comptime M = GPTMarked[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, ACTIVE
    ]
    seed(42)
    var dp = DataParallel[M, 1].make[Normal[0.0, 0.02]](
        ProcessGroup[1].shared(ctx), lr=1e-3, beta2=0.99, wd=0.1
    )
    gpt_marked_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, ACTIVE
    ](dp.nets[0], Optional(ctx))
    gpt_marked_wire_tie[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, ACTIVE
    ](dp.nets[0])
    return _loop(dp, ctx)


def _diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _gate(name: String, got: _Out, ref_: _Out) raises:
    var dl = 0.0
    for i in range(K):
        dl = max(dl, abs(got.losses[i] - ref_.losses[i]))
    var di = _diff(got.init, ref_.init)
    var df = _diff(got.final, ref_.final)
    print("   ", name, ": init", di, " losses", dl, " final params", df)
    assert_true(len(got.init) == len(ref_.init), name + ": arena sizes differ")
    assert_true(di == 0.0, name + ": initial weights differ (construction ops drifted)")
    assert_true(dl == 0.0 and df == 0.0, name + ": training differs")


def main() raises:
    print("GPTMarked vs GPTDropTied:", LAYERS, "x", EMBED, " seq", SEQ, " K =", K)
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    var plain = _plain(ctx)
    print("  plain loss", plain.losses[0], "->", plain.losses[K - 1])
    assert_true(
        plain.losses[K - 1] < plain.losses[0] - 0.1, "gate vacuous: no training"
    )
    _gate("marks compiled out", _marked[False](ctx), plain)
    _gate("marks compiled in ", _marked[True](ctx), plain)
    print("GPT MARKED GATES OK")
