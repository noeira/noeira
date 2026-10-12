"""DistributedAutoregressiveTrainer at N = 1 trains as AutoregressiveTrainer.

The same GPT (`GPTDropTied` 2 x 64, seq 64, dropout 0.2), seed, AdamW (wd
0.1), schedule (warmup WARMUP, cosine to 0.1 over K), clip CLIP and batch B,
K steps by each driver. The distributed one runs `DataParallel[NET, 1]` on
the simulator, where no collective runs. Checked:

  - the LR of every step and the final weights, bit for bit;
  - the train CE of every step, within 1e-5: `AutoregressiveTrainer` sums
    the per-row losses on the host, this driver on the device;
  - a greedy and a T = 0.8 continuation of one prompt, identical.

The two drivers share no code for the schedule, the batch sampler or the
generation window (`lm_helpers.mojo` holds copies), so this gate is what
keeps them equal.

Run (Mac):  pixi run -e apple mojo build -I . tests/nn/distributed/test_dist_trainer_single.mojo -o $B/tdt1 && $B/tdt1
"""

from std.random import seed
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Normal
from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.models.gpt import (
    GPTDropTied,
    gpt_scale_residual_proj,
    gpt_wire_tie,
)
from noeira.nn.optimizer.adam import AdamW
from noeira.nn.training.autoregressive_trainer import AutoregressiveTrainer
from noeira.nn.distributed.process_group import ProcessGroup
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.autoregressive_trainer import (
    DistributedAutoregressiveTrainer,
)


comptime VOCAB = 65
comptime SEQ = 64
comptime EMBED = 64
comptime HEADS = 4
comptime LAYERS = 2
comptime FF = 4
comptime DROP: Float64 = 0.2
comptime SB = UInt64(0xC0FFEE)
comptime B = 8
comptime K = 8
comptime WARMUP = 3
comptime LR: Scalar[DT] = 1e-3
comptime CLIP: Scalar[DT] = 1.0
comptime GEN = 32
comptime NET = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True
]
comptime AR = AutoregressiveTrainer[NET, AdamW, VOCAB, SEQ, B, target="gpu"]
comptime DIST = DataParallel[NET, 1]


def _surgery(mut net: NET, c: DeviceContext) raises:
    gpt_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True
    ](net, Optional(c))
    gpt_wire_tie[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True
    ](net)


def main() raises:
    print(
        "DistributedAutoregressiveTrainer N=1 vs AutoregressiveTrainer: GPT",
        LAYERS, "x", EMBED, " seq", SEQ, " B", B, " K", K, " dropout", DROP,
        " clip", CLIP,
    )
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    var text = load_text()

    # ── AutoregressiveTrainer ──
    var tok_a = CharTokenizer(text)
    var split_a = train_val_split(tok_a.encode(text), 0.1)
    seed(42)
    var net = NET.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var artr = AR.make_from(
        net^, AdamW(lr=LR, beta2=0.99, wd=0.1), tok_a^, split_a^, ctx,
        LR, WARMUP, K, 0.1, CLIP,
    )
    _surgery(artr.net, ctx)
    ctx.synchronize()
    var loss_a = List[Float64]()
    var lr_a = List[Scalar[DT]]()
    for it in range(K):
        loss_a.append(artr._train_step(it))
        lr_a.append(artr.opt.get_lr())
    var pa = Tensor.alloc(artr.opt.arena.total)
    ctx.enqueue_copy(pa.data.unsafe_ptr(), artr.opt.arena.val.dev.value())
    ctx.synchronize()

    # ── the distributed driver, N = 1 ──
    var tok_d = CharTokenizer(text)
    var split_d = train_val_split(tok_d.encode(text), 0.1)
    seed(42)
    var dist = DIST.make[Normal[0.0, 0.02]](
        ProcessGroup[1].shared(ctx), lr=LR, beta2=0.99, wd=0.1
    )
    var c0 = dist.ctx(0)
    _surgery(dist.net(0), c0)
    var tr = DistributedAutoregressiveTrainer[DIST, VOCAB, SEQ, B].make_from(
        dist^, tok_d^, split_d^, LR, WARMUP, K, 0.1, CLIP
    )
    var loss_d = List[Float64]()
    var lr_d = List[Scalar[DT]]()
    for it in range(K):
        tr.train_step(it)
        loss_d.append(tr.train_loss(reset=True))
        lr_d.append(tr.dist.opts[0].get_lr())
    var pd = tr.dist.download_params(0)

    # ── compare ──
    var dl = 0.0
    var dlr = 0.0
    for it in range(K):
        dl = max(dl, abs(loss_a[it] - loss_d[it]))
        dlr = max(dlr, Float64(abs(lr_a[it] - lr_d[it])))
    assert_true(len(pa.data) == len(pd), "arena layouts differ")
    var dp = 0.0
    for i in range(len(pd)):
        dp = max(dp, Float64(abs(pa.data[i] - pd[i])))
    print("  CE", loss_a[0], "->", loss_a[K - 1])
    print("  max|CE diff|", dl, " max|LR diff|", dlr, " max|weight diff|", dp)
    assert_true(loss_a[K - 1] < loss_a[0] - 0.1, "gate vacuous: no training")
    assert_true(dlr == 0.0, "the LR schedules differ")
    assert_true(dp == 0.0, "the weights differ")
    assert_true(dl < 1e-5, "the train CE differs beyond the summation order")

    seed(7)
    var greedy_a = artr.generate("ROMEO:", GEN, temperature=0.0)
    seed(7)
    var greedy_d = tr.generate("ROMEO:", GEN, temperature=0.0)
    seed(7)
    var sample_a = artr.generate("ROMEO:", GEN, temperature=0.8)
    seed(7)
    var sample_d = tr.generate("ROMEO:", GEN, temperature=0.8)
    print("  greedy:", repr(greedy_d))
    print("  T=0.8: ", repr(sample_d))
    assert_true(greedy_a == greedy_d, "greedy continuations differ")
    assert_true(sample_a == sample_d, "sampled continuations differ")
    print("DIST TRAINER N=1: identical to AutoregressiveTrainer")
