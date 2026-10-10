"""ZeRO-2 / ZeRO-3 gates: `ZeroSharded` against DDP.

MLP 16 -> 64 -> 64 -> 8. The first two layers are units (`GradReady`), the
last one is left unwrapped, so it lands in the root unit. Units are 1088,
4160 and 520 elements (34, 130 and 17 rows of 32): no row count divides by
3 or 4, so shards are unequal and cut parameters mid-tensor, and with 1 slot
both units share one weight slot and one gradient slot.

  A. No clip: after K AdamW steps, ZeRO-2 and ZeRO-3 weights equal DDP's BIT
     FOR BIT, for N = 1..4 and 1 or 2 slots; ZeRO-2's replicas agree.
  B. Clip: the norm is summed per shard (ZeRO-1's rule), so it matches DDP's
     to ~1e-7 relative and the weights to a few ULP; every rank reads the
     same norm.
  C. State bytes per rank: DDP vs ZeRO-2 vs ZeRO-3.

Run (Mac):  pixi run -e apple mojo build -I . tests/nn/distributed/test_zero_sharded.mojo -o $B/tz && $B/tz
NVIDIA, 2 GPUs: add `-D DDP_DEVICES` (MAX comm, N = 2 only).
"""

from std.random import seed
from std.sys import is_defined, has_accelerator
from std.testing import assert_true
from std.memory import Pointer, bitcast
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Xavier
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse_loss import MSELoss
from noeira.nn.distributed.process_group import ProcessGroup
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.grad_marks import GradReady
from noeira.nn.distributed.zero_sharded import ZeroSharded
from noeira.nn.distributed.fibers import RankStep


comptime D = 16
comptime H = 64
comptime O = 8
comptime B = 48
comptime K = 30
comptime LR: Scalar[DT] = 3e-3
comptime WD: Scalar[DT] = 0.05
comptime CLIP: Scalar[DT] = 0.05
comptime NET = Sequential[
    GradReady[LinearReLU[D, H]],
    GradReady[LinearReLU[H, H]],
    Linear[H, O],
]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()


def _pg[N: Int](ctx: DeviceContext) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES and N >= 2:
        return ProcessGroup[N].devices(1 << 20)
    else:
        return ProcessGroup[N].shared(ctx)


struct _RankData[BL: Int](Movable):
    """Rank r's fixed batch (BL rows) and step buffers."""

    var x: Tensor
    var y: Tensor
    var out: Tensor
    var go: Tensor
    var gi: Tensor
    var loss: MSELoss[O]

    def __init__(out self, c: DeviceContext, r: Int) raises:
        comptime L = Self.BL
        self.x = Tensor.alloc(L * D)
        self.y = Tensor.alloc(L * O)
        for i in range(L * D):
            self.x.data[i] = Scalar[DT]((((r * L * D + i) * 7) % 23) - 11) * 0.09
        for i in range(L * O):
            self.y.data[i] = Scalar[DT]((((r * L * O + i) * 5) % 17) - 8) * 0.11
        self.x.upload(c)
        self.y.upload(c)
        self.out = Tensor.alloc_gpu(c, L * O)
        self.go = Tensor.alloc_gpu(c, L * O)
        self.gi = Tensor.alloc_gpu(c, L * D)
        self.loss = MSELoss[O].make_gpu(c)


def _fwd_bwd[BL: Int](mut net: NET, mut d: _RankData[BL], c: DeviceContext) raises:
    comptime A = NET.ACT_DT
    var co = Optional(c)
    with c.push_context():
        net.forward["gpu", BL](
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.x)),
            rebind[TensorImpl[A]](d.out),
            co,
        )
        d.loss.vjp["gpu", BL](d.out, d.y, d.go, co)
        net.vjp["gpu", BL](
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.x)),
            rebind[TensorImpl[A]](d.go),
            child_refs[NET.ARITY, A](rebind[TensorImpl[A]](d.gi)),
            co,
        )


struct _Out(Movable):
    var params: List[List[Scalar[DT]]]
    var norms: List[Float64]
    var bytes: Int

    def __init__(out self):
        self.params = List[List[Scalar[DT]]]()
        self.norms = List[Float64]()
        self.bytes = 0


def _ddp[N: Int](ctx: DeviceContext, clip: Bool) raises -> _Out:
    seed(7)
    var dp = DataParallel[NET, N].make[Xavier](_pg[N](ctx), lr=LR, wd=WD)
    dp.sync_params()
    var d = List[_RankData[B // N]]()
    for r in range(N):
        d.append(_RankData[B // N](dp.ctx(r), r))
    var res = _Out()
    for _ in range(K):
        dp.zero_grad()
        for r in range(N):
            _fwd_bwd[B // N](dp.nets[r], d[r], dp.ctx(r))
        dp.allreduce_grads()
        if clip:
            dp.clip_grads_device(CLIP)
            res.norms.append(Float64(dp.opts[0].read_clip_norm(dp.ctx(0))))
        dp.step()
    for r in range(N):
        res.params.append(dp.download_params(r))
    res.bytes = dp.state_bytes_per_rank()
    return res^


struct _ZJob[N: Int, STAGE: Int](RankStep):
    """Rank r's forward + backward, on rank r's fiber. One per rank; it owns
    its batch and reaches its replica through the driver's address."""

    var z: Int
    var d: _RankData[B // Self.N]

    def __init__(out self, z: Int, var d: _RankData[B // Self.N]):
        self.z = z
        self.d = d^

    def run_rank(mut self, r: Int) raises:
        var zp = Pointer[
            ZeroSharded[NET, Self.N, Self.STAGE], MutUntrackedOrigin
        ](unsafe_from_address=self.z)
        _fwd_bwd[B // Self.N](zp[].nets[r], self.d, zp[].ctx(r))


def _zero[N: Int, STAGE: Int](
    ctx: DeviceContext, clip: Bool, slots: Int
) raises -> _Out:
    seed(7)
    var z = ZeroSharded[NET, N, STAGE].make[Xavier](
        _pg[N](ctx), lr=LR, wd=WD, slots=slots
    )
    z.sync_params()
    var jobs = List[_ZJob[N, STAGE]]()
    for r in range(N):
        jobs.append(
            _ZJob[N, STAGE](Int(Pointer(to=z)), _RankData[B // N](z.ctx(r), r))
        )
    var res = _Out()
    for _ in range(K):
        z.forward_backward(jobs)
        if clip:
            z.clip_grads_device(CLIP)
            res.norms.append(Float64(z.read_clip_norm(0)))
            for r in range(1, N):
                assert_true(
                    z.read_clip_norm(r) == z.read_clip_norm(0),
                    "ranks disagree on the global clip norm",
                )
        z.step()
    z.synchronize()
    comptime if STAGE == 2:
        for r in range(N):
            res.params.append(z.download_params(r))
    else:
        res.params.append(z.download_params())
    res.bytes = z.state_bytes_per_rank(0)
    print("     ", z.layout_summary())
    return res^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _ulps(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Int:
    var m = 0
    for i in range(len(a)):
        var ia = Int(bitcast[DType.int32](a[i]))
        var ib = Int(bitcast[DType.int32](b[i]))
        m = max(m, abs(ia - ib))
    return m


def _exact[N: Int, STAGE: Int](ctx: DeviceContext, d: _Out, slots: Int) raises -> Int:
    var z = _zero[N, STAGE](ctx, False, slots)
    assert_true(len(z.params[0]) == len(d.params[0]), "layouts differ")
    var dz = _max_abs_diff(z.params[0], d.params[0])
    print("    ZeRO-" + String(STAGE), " slots", slots, " max|DDP - ZeRO| =", dz)
    assert_true(dz == 0.0, "ZeRO-" + String(STAGE) + " is not bit-identical to DDP")
    for r in range(1, len(z.params)):
        assert_true(
            _max_abs_diff(z.params[r], z.params[0]) == 0.0,
            "ZeRO-2 replicas drifted apart",
        )
    return z.bytes


def _clip[N: Int, STAGE: Int](ctx: DeviceContext, dc: _Out) raises:
    var zc = _zero[N, STAGE](ctx, True, 1)
    var dn = 0.0
    var fired = 0
    for i in range(K):
        dn = max(dn, abs(dc.norms[i] - zc.norms[i]) / dc.norms[i])
        if dc.norms[i] > Float64(CLIP):
            fired += 1
    var dp = _max_abs_diff(dc.params[0], zc.params[0])
    print("    ZeRO-" + String(STAGE), "clip: fired", fired, "/", K,
          " max rel norm diff =", dn, " max|DDP - ZeRO| =", dp,
          "(", _ulps(dc.params[0], zc.params[0]), "ulp )")
    assert_true(fired > K // 2, "clip gate vacuous")
    assert_true(dn < 1e-6, "clip norm differs from DDP's")
    assert_true(dp < 1e-5, "clipped run drifted from DDP")


def _gate[N: Int](ctx: DeviceContext, init: List[Scalar[DT]]) raises:
    print("[N =", N, "]")
    var d = _ddp[N](ctx, False)
    var moved = _max_abs_diff(d.params[0], init)
    assert_true(moved > 1e-3, "gate vacuous: the parameters barely moved")
    var b2 = _exact[N, 2](ctx, d, 1)
    _ = _exact[N, 2](ctx, d, 2)
    var b3 = _exact[N, 3](ctx, d, 1)
    _ = _exact[N, 3](ctx, d, 2)
    var dc = _ddp[N](ctx, True)
    _clip[N, 2](ctx, dc)
    _clip[N, 3](ctx, dc)
    print("    state bytes/rank: DDP", d.bytes, " ZeRO-2", b2, " ZeRO-3", b3)


def main() raises:
    print("ZeRO-2 / ZeRO-3 vs DDP: MLP", D, "->", H, "->", H, "->", O,
          " B =", B, " K =", K, " AdamW wd =", WD)
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    seed(7)
    var dp1 = DataParallel[NET, 1].make[Xavier](ProcessGroup[1].shared(ctx), lr=LR)
    var init = dp1.download_params(0)
    comptime if USE_DEVICES:
        _gate[2](ctx, init)
    else:
        _gate[1](ctx, init)
        _gate[2](ctx, init)
        _gate[3](ctx, init)
        _gate[4](ctx, init)
    print("ZERO-2/3 GATES OK")
