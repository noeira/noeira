"""DistributedAutoregressiveTrainer under the four strategies, against DDP.

GPT 2 x 64, seq 64 (`GPTMarked`: every block a unit), dropout 0.2, AdamW
(wd 0.1), N = 2 ranks of BL windows each on the simulator, K steps of the
cosine schedule, no clip. DDP, ZeRO-1, ZeRO-2 and ZeRO-3 each train through
the same trainer code; only the strategy type differs. Checked against DDP:

  - the train CE of every step, the final weights, the validation CE and a
    greedy continuation, all bit for bit. Each rank computes its gradient
    with the same kernels on the same windows and dropout masks, every
    reduction sums the ranks in the same order, and Adam is elementwise.
  - DDP's two replicas end equal, the run trains (non-vacuous), and each
    rank's replica is on its own dropout stream.

And, on the model alone: `set_attr["rng_stream"](1)` reaches the GPT's
dropouts and moves their masks, while stream 0 is the default sequence.

Run (Mac):  pixi run -e apple mojo build -I . tests/nn/distributed/test_dist_trainer_strategies.mojo -o $B/tdts && $B/tdts
"""

from std.random import seed
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Normal
from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.distributed.process_group import ProcessGroup
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


comptime VOCAB = 65
comptime SEQ = 64
comptime EMBED = 64
comptime HEADS = 4
comptime LAYERS = 2
comptime FF = 4
comptime DROP: Float64 = 0.2
comptime SB = UInt64(0xC0FFEE)
comptime N = 2
comptime BL = 4
comptime K = 6
comptime N_VAL = 2 * N * BL
"""Validation windows: two batches per rank."""
comptime GEN = 24
comptime NET = GPTMarked[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True
]


def _surgery[M: Module](mut net: M, c: DeviceContext) raises:
    """nanoGPT's scaled residual init and the tied head (M is NET)."""
    ref gpt = rebind[NET](net)
    gpt_marked_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True
    ](gpt, Optional(c))
    gpt_marked_wire_tie[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True
    ](gpt)


struct _Run(Movable):
    var losses: List[Float64]
    var params: List[List[Scalar[DT]]]
    var val: Float64
    var text: String
    var bytes: Int

    def __init__(out self):
        self.losses = List[Float64]()
        self.params = List[List[Scalar[DT]]]()
        self.val = 0.0
        self.text = String()
        self.bytes = 0


def _train[S: ShardingStrategy](ctx: DeviceContext, text: String) raises -> _Run:
    var tok = CharTokenizer(text)
    var split = train_val_split(tok.encode(text), 0.1)
    seed(42)
    var dist = S.make[Normal[0.0, 0.02]](
        ProcessGroup[S.RANKS].shared(ctx), lr=1e-3, beta2=0.99, wd=0.1
    )
    for r in range(S.RANKS):
        var c = dist.ctx(r)
        _surgery(dist.net(r), c)
    var tr = DistributedAutoregressiveTrainer[S, VOCAB, SEQ, BL].make_from(
        dist^, tok^, split^, base_lr=1e-3, warmup_iters=2, total_iters=K
    )
    for r in range(S.RANKS):
        # The trainer put rank r on dropout stream r (the input dropout,
        # `children[2]`, stands for every dropout of the replica).
        ref gpt = rebind[NET](tr.dist.net(r))
        assert_true(
            gpt.children[2].stream_base == UInt64(r) << 48,
            "rank " + String(r) + " is not on its own dropout stream",
        )
    var out = _Run()
    for it in range(K):
        tr.train_step(it)
        out.losses.append(tr.train_loss(reset=True))
    out.val = tr.val_loss(N_VAL)
    out.text = tr.generate("ROMEO:", GEN)
    tr.dist.synchronize()
    for r in range(S.RANKS):
        out.params.append(tr.dist.download_params(r))
    out.bytes = tr.dist.state_bytes_per_rank(0)
    print(
        "   ", S.name(), ": CE", out.losses[0], "->", out.losses[K - 1],
        " val", out.val, " state bytes/rank", out.bytes,
    )
    return out^


def _diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) raises -> Float64:
    assert_true(len(a) == len(b), "layouts differ")
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _cmp(name: String, got: _Run, ref_: _Run) raises:
    var dl = 0.0
    for i in range(K):
        dl = max(dl, abs(got.losses[i] - ref_.losses[i]))
    var dp = _diff(got.params[0], ref_.params[0])
    var dv = abs(got.val - ref_.val)
    print(
        "   ", name, "vs DDP: CE", dl, " weights", dp, " val", dv,
        " text", "same" if got.text == ref_.text else "DIFFERENT",
    )
    assert_true(
        dl == 0.0 and dp == 0.0 and dv == 0.0 and got.text == ref_.text,
        name + ": not bit-identical to DDP",
    )


def _streams(ctx: DeviceContext) raises:
    """The same weights and input, one training-mode forward each: default
    stream, stream 0 set explicitly, stream 1."""
    comptime TOTAL = BL * SEQ * VOCAB
    var x = Tensor.alloc(TOTAL)
    for b in range(BL):
        for t in range(SEQ):
            x.data[(b * SEQ + t) * VOCAB + (3 * b + 5 * t) % VOCAB] = 1.0
    x.upload(ctx)
    var outs = List[List[Scalar[DT]]]()
    for stream in range(-1, 2):
        seed(42)
        var net = NET.make["gpu", Normal[0.0, 0.02]](Optional(ctx))
        _surgery(net, ctx)
        if stream >= 0:
            net.set_attr["rng_stream"](Scalar[DT](stream))
        var y = Tensor.alloc_gpu(ctx, TOTAL)
        net.forward["gpu", BL](child_refs[NET.ARITY, DT](x), y, Optional(ctx))
        y.download(ctx)
        outs.append(y.data.copy())
    var d0 = _diff(outs[0], outs[1])
    var d1 = _diff(outs[0], outs[2])
    print("   rng_stream: |stream 0 - default|", d0, " |stream 1 - default|", d1)
    assert_true(d0 == 0.0, "stream 0 is not the default sequence")
    assert_true(d1 > 0.0, "rng_stream did not reach the dropouts")


def main() raises:
    print(
        "DistributedAutoregressiveTrainer, 4 strategies: GPT", LAYERS, "x",
        EMBED, " seq", SEQ, " N", N, " BL", BL, " K", K, " dropout", DROP,
    )
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    _streams(ctx)
    var text = load_text()
    var d = _train[DataParallel[NET, N]](ctx, text)
    assert_true(
        d.losses[K - 1] < d.losses[0] - 0.1, "gate vacuous: no training"
    )
    var drep = _diff(d.params[0], d.params[1])
    print("    DDP replicas: max|rank 0 - rank 1| =", drep)
    assert_true(drep == 0.0, "DDP replicas drifted apart")
    _cmp("ZeRO-1", _train[Zero1[NET, N]](ctx, text), d)
    _cmp("ZeRO-2", _train[ZeroSharded[NET, N, 2]](ctx, text), d)
    _cmp("ZeRO-3", _train[ZeroSharded[NET, N, 3]](ctx, text), d)
    print("DIST TRAINER STRATEGIES: all four bit-identical")
