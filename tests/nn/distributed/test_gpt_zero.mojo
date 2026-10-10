"""ZeRO-2 / ZeRO-3 on the GPT: `ZeroSharded[GPTMarked]` against DDP.

GPT 2 x 64, seq 64, dropout off, AdamW (wd 0.1), N ranks of 4 sequences each
(a different fixed batch per rank). Every transformer block is a unit; the
embedding, the positional bias and the final LayerNorm are the root, and the
LM head reads the embedding's cells (the tie), so the root's gradient takes
contributions at both ends of the backward. Checked, per step and at the end:

  - rank 0's loss, every step, equal to DDP's (bit for bit, no clip);
  - the final weights equal DDP's, bit for bit (no clip), with 1 and 2 slots;
  - with clip CLIP (fires on every step, asserted): the weights within 1e-5
    of DDP's (the norm is summed per shard).

Run (Mac):  pixi run -e apple mojo build -I . tests/nn/distributed/test_gpt_zero.mojo -o $B/tgz && $B/tgz
NVIDIA, 2 GPUs: add `-D DDP_DEVICES` (MAX comm, N = 2 only).
"""

from std.random import seed
from std.sys import is_defined, has_accelerator
from std.testing import assert_true
from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Normal
from noeira.nn.loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from noeira.nn.distributed.process_group import ProcessGroup
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero_sharded import ZeroSharded
from noeira.nn.distributed.fibers import RankStep
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
comptime CLIP: Scalar[DT] = 0.05
comptime NET = GPTMarked[VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True]
comptime LOSS = SequenceCrossEntropyLoss[SEQ, VOCAB]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()


def _pg[N: Int](ctx: DeviceContext) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES and N >= 2:
        return ProcessGroup[N].devices(1 << 22)
    else:
        return ProcessGroup[N].shared(ctx)


def _surgery(mut net: NET, c: DeviceContext) raises:
    gpt_marked_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True
    ](net, Optional(c))
    gpt_marked_wire_tie[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF, True, DROP, SB, True, True
    ](net)


struct _Rank(Movable):
    """Rank r's fixed batch and step buffers: token (b, t) = (7b + 3t + t^2
    + 11r) mod VOCAB, the target the next one."""

    var x: Tensor
    var y: Tensor
    var logits: Tensor
    var grad: Tensor
    var gi: Tensor
    var loss: LOSS

    def __init__(out self, c: DeviceContext, r: Int) raises:
        self.x = Tensor.alloc(BL * ROW)
        self.y = Tensor.alloc(BL * ROW)
        for b in range(BL):
            for t in range(SEQ):
                var tok = (7 * b + 3 * t + t * t + 11 * r) % VOCAB
                var nxt = (7 * b + 3 * (t + 1) + (t + 1) * (t + 1) + 11 * r) % VOCAB
                self.x.data[b * ROW + t * VOCAB + tok] = 1.0
                self.y.data[b * ROW + t * VOCAB + nxt] = 1.0
        self.x.upload(c)
        self.y.upload(c)
        self.logits = Tensor.alloc_gpu(c, BL * ROW)
        self.grad = Tensor.alloc_gpu(c, BL * ROW)
        self.gi = Tensor.alloc_gpu(c, BL * ROW)
        self.loss = LOSS.make_gpu(c)


def _fwd_bwd(mut net: NET, mut d: _Rank, c: DeviceContext) raises -> Float64:
    comptime A = NET.ACT_DT
    var co = Optional(c)
    var l: Float64
    with c.push_context():
        net.forward["gpu", BL](
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.x)),
            rebind[TensorImpl[A]](d.logits),
            co,
        )
        l = Float64(d.loss.forward["gpu", BL](d.logits, d.y, co))
        d.loss.vjp["gpu", BL](d.logits, d.y, d.grad, co)
        net.vjp["gpu", BL](
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.x)),
            rebind[TensorImpl[A]](d.grad),
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.gi)),
            co,
        )
    return l


struct _Out(Movable):
    var losses: List[Float64]
    var params: List[Scalar[DT]]
    var bytes: Int
    var fired: Int

    def __init__(out self):
        self.losses = List[Float64]()
        self.params = List[Scalar[DT]]()
        self.bytes = 0
        self.fired = 0


def _ddp[N: Int](ctx: DeviceContext, clip: Bool) raises -> _Out:
    seed(42)
    var dp = DataParallel[NET, N].make[Normal[0.0, 0.02]](
        _pg[N](ctx), lr=1e-3, beta2=0.99, wd=0.1
    )
    for r in range(N):
        _surgery(dp.nets[r], dp.ctx(r))
    dp.sync_params()
    var data = List[_Rank]()
    for r in range(N):
        data.append(_Rank(dp.ctx(r), r))
    var out = _Out()
    for _ in range(K):
        dp.zero_grad()
        for r in range(N):
            var l = _fwd_bwd(dp.nets[r], data[r], dp.ctx(r))
            if r == 0:
                out.losses.append(l)
        dp.allreduce_grads()
        if clip:
            dp.clip_grads_device(CLIP)
            if dp.opts[0].read_clip_norm(dp.ctx(0)) > CLIP:
                out.fired += 1
        dp.step()
    dp.synchronize()
    out.params = dp.download_params(0)
    out.bytes = dp.state_bytes_per_rank()
    return out^


struct _ZJob[N: Int, STAGE: Int](RankStep):
    """Rank r's step, on rank r's fiber. One per rank: it owns its batch and
    its loss log, and reaches its replica through the driver's address."""

    var z: Int
    var d: _Rank
    var losses: List[Float64]

    def __init__(out self, z: Int, var d: _Rank):
        self.z = z
        self.d = d^
        self.losses = List[Float64]()

    def run_rank(mut self, r: Int) raises:
        var zp = Pointer[
            ZeroSharded[NET, Self.N, Self.STAGE], MutUntrackedOrigin
        ](unsafe_from_address=self.z)
        self.losses.append(_fwd_bwd(zp[].nets[r], self.d, zp[].ctx(r)))


def _zero[N: Int, STAGE: Int](
    ctx: DeviceContext, clip: Bool, slots: Int
) raises -> _Out:
    seed(42)
    var z = ZeroSharded[NET, N, STAGE].make[Normal[0.0, 0.02]](
        _pg[N](ctx), lr=1e-3, beta2=0.99, wd=0.1, slots=slots
    )
    for r in range(N):
        _surgery(z.nets[r], z.ctx(r))
    z.sync_params()
    var jobs = List[_ZJob[N, STAGE]]()
    for r in range(N):
        jobs.append(_ZJob[N, STAGE](Int(Pointer(to=z)), _Rank(z.ctx(r), r)))
    for _ in range(K):
        z.forward_backward(jobs)
        if clip:
            z.clip_grads_device(CLIP)
        z.step()
    z.synchronize()
    print("     ", z.layout_summary())
    var out = _Out()
    out.losses = jobs[0].losses.copy()
    for r in range(1, N):
        assert_true(len(jobs[r].losses) == K, "a rank lost its loss log")
    out.params = z.download_params(0)
    out.bytes = z.state_bytes_per_rank(0)
    comptime if STAGE == 2:
        for r in range(1, N):
            var pr = z.download_params(r)
            for i in range(len(pr)):
                assert_true(pr[i] == out.params[i], "ZeRO-2 replicas drifted apart")
    return out^


def _diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _cmp(name: String, got: _Out, ref_: _Out, exact: Bool) raises:
    assert_true(len(got.params) == len(ref_.params), name + ": layouts differ")
    var dl = 0.0
    for i in range(K):
        dl = max(dl, abs(got.losses[i] - ref_.losses[i]))
    var dp = _diff(got.params, ref_.params)
    print("   ", name, ": losses", dl, " final params", dp)
    if exact:
        assert_true(dl == 0.0 and dp == 0.0, name + ": not bit-identical to DDP")
    else:
        assert_true(dp < 1e-5, name + ": drifted from DDP")


def _gate[N: Int](ctx: DeviceContext) raises:
    print("[N =", N, "]")
    var d = _ddp[N](ctx, False)
    print("    DDP loss", d.losses[0], "->", d.losses[K - 1])
    assert_true(d.losses[K - 1] < d.losses[0] - 0.1, "gate vacuous: no training")
    var z2 = _zero[N, 2](ctx, False, 1)
    _cmp("ZeRO-2 slots 1", z2, d, True)
    _cmp("ZeRO-2 slots 2", _zero[N, 2](ctx, False, 2), d, True)
    var z3 = _zero[N, 3](ctx, False, 1)
    _cmp("ZeRO-3 slots 1", z3, d, True)
    _cmp("ZeRO-3 slots 2", _zero[N, 3](ctx, False, 2), d, True)
    var dc = _ddp[N](ctx, True)
    print("    clip", CLIP, "fired on", dc.fired, "of", K, "steps")
    assert_true(dc.fired == K, "clip gate vacuous")
    _cmp("ZeRO-2 clip", _zero[N, 2](ctx, True, 1), dc, False)
    _cmp("ZeRO-3 clip", _zero[N, 3](ctx, True, 1), dc, False)
    print("    state bytes/rank: DDP", d.bytes, " ZeRO-2", z2.bytes, " ZeRO-3", z3.bytes)


def main() raises:
    print("ZeRO-2 / ZeRO-3 on GPTMarked:", LAYERS, "x", EMBED, " seq", SEQ,
          " BL", BL, " K", K)
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    comptime if USE_DEVICES:
        _gate[2](ctx)
    else:
        _gate[2](ctx)
        _gate[3](ctx)
    print("GPT ZERO-2/3 GATES OK")
