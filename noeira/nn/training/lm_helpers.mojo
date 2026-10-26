"""Rules shared by next-token LM drivers: nanoGPT's LR schedule and sampling.

Copied from `AutoregressiveTrainer` (`_lr_scale`, `_sample_token`, the
one-hot upload), which keeps its own copies for now. The distributed driver
(`noeira/nn/distributed/autoregressive_trainer.mojo`) uses these, and
`tests/nn/distributed/test_dist_trainer_single.mojo` holds the two drivers to
the same LR every step and the same generated text, so a change to one copy
that is not made to the other fails there.
"""

from std.math import exp, cos
from std.random import random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


def cosine_warmup_lr_scale(
    it: Int, warmup_iters: Int, total_iters: Int, min_lr_scale: Float64
) -> Scalar[DT]:
    """Linear warmup over `warmup_iters`, then cosine decay to
    `min_lr_scale` at `total_iters` (a factor on the base LR)."""
    if it < warmup_iters:
        return Scalar[DT](Float64(it + 1) / Float64(warmup_iters))
    var denom = total_iters - warmup_iters
    if denom < 1:
        denom = 1
    var prog = Float64(it - warmup_iters) / Float64(denom)
    if prog > 1.0:
        prog = 1.0
    var c = 0.5 * (1.0 + cos(3.14159265358979 * prog))
    return Scalar[DT](min_lr_scale + (1.0 - min_lr_scale) * c)


def upload_onehot[
    ADT: DType, SEQ: Int, VOCAB: Int
](
    ctx: DeviceContext, mut t: TensorImpl[ADT], ids: List[Int], n_rows: Int
) raises:
    """Host one-hot `[n_rows, SEQ*VOCAB]` of token ids `[n_rows*SEQ]`, then
    an H2D copy into `t`'s device buffer, which keeps its address."""
    comptime ROW = SEQ * VOCAB
    var total = n_rows * ROW
    t.ensure(total)
    for i in range(total):
        t.data[i] = Scalar[ADT](0)
    for r in range(n_rows):
        for tt in range(SEQ):
            var tid = ids[r * SEQ + tt]
            if tid >= 0 and tid < VOCAB:
                t.data[r * ROW + tt * VOCAB + tid] = Scalar[ADT](1)
    t.n = total
    t.upload_resident(ctx)


def sample_next_token[
    ADT: DType, VOCAB: Int
](
    ref logits: TensorImpl[ADT], base: Int, temperature: Float64, top_k: Int
) -> Int:
    """NanoGPT's rule on `logits.data[base : base + VOCAB]`: the argmax when
    `temperature <= 0`, else a draw from the softmax of logits / temperature,
    restricted to the `top_k` largest when `0 < top_k < VOCAB`."""
    if temperature <= 0.0:
        var bv = Float64(logits.data[base])
        var bi = 0
        for v in range(1, VOCAB):
            if Float64(logits.data[base + v]) > bv:
                bv = Float64(logits.data[base + v])
                bi = v
        return bi
    var inv_t = 1.0 / temperature
    var scaled = List[Float64](capacity=VOCAB)
    for v in range(VOCAB):
        scaled.append(Float64(logits.data[base + v]) * inv_t)
    var keep = List[Bool](capacity=VOCAB)
    if top_k > 0 and top_k < VOCAB:
        for _ in range(VOCAB):
            keep.append(False)
        var work = List[Float64](capacity=VOCAB)
        for v in range(VOCAB):
            work.append(scaled[v])
        for _ in range(top_k):
            var bv: Float64 = -1e30
            var bi = 0
            for v in range(VOCAB):
                if work[v] > bv:
                    bv = work[v]
                    bi = v
            keep[bi] = True
            work[bi] = -1e30
    else:
        for _ in range(VOCAB):
            keep.append(True)
    var m: Float64 = -1e30
    for v in range(VOCAB):
        if keep[v] and scaled[v] > m:
            m = scaled[v]
    var se: Float64 = 0.0
    var exps = List[Float64](capacity=VOCAB)
    for v in range(VOCAB):
        if keep[v]:
            var e = exp(scaled[v] - m)
            exps.append(e)
            se += e
        else:
            exps.append(0.0)
    var u = random_float64(0.0, 1.0) * se
    var acc = 0.0
    for v in range(VOCAB):
        acc += exps[v]
        if u < acc:
            return v
    return VOCAB - 1
