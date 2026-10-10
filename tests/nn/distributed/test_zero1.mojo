"""ZeRO-1 gates (M3): sharded AdamW against DDP.

  A. No clip: ZeRO-1 at N ranks is BIT-IDENTICAL to DDP at N ranks, every rank
     (same reduction order, same Adam kernels on offset sub-buffers).
  B. Clip: the global norm comes from per-shard sums + one scalar allreduce, a
     different summation order than DDP's whole-arena reduction; the norm must
     match to ~1e-6 relative and the parameters to a few ULP (reported).
  C. Replica agreement under ZeRO-1: bit for bit after the all-gather.
  D. State memory per rank, DDP vs ZeRO-1 (weights + grads + Adam state).

N = 1, 2, 3, 4 on the shared-context simulator. The arena (1608 elements = 51
rows of 32) does not divide by 2, 3 or 4, so the shards are unequal, the last
one carries the padded tail, and every N cuts parameters mid-tensor. `-D
DDP_DEVICES` runs N = 2 on GPUs 0..1 with MAX comm instead.

Run (Mac):  pixi run -e apple mojo run -I . tests/nn/distributed/test_zero1.mojo
"""

from std.random import seed
from std.sys import is_defined, has_accelerator
from std.testing import assert_true
from std.memory import bitcast
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.distributed.process_group import ProcessGroup, shard_rows
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero import Zero1


comptime D = 16
comptime H = 64
comptime O = 8
comptime B = 48  # divisible by 1, 2, 3, 4
comptime K = 40
comptime LR: Scalar[DT] = 3e-3
comptime WD: Scalar[DT] = 0.05
comptime CLIP: Scalar[DT] = 0.05
comptime NET = Sequential[LinearReLU[D, H], Linear[H, O]]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()


def _batch(step: Int, mut x: List[Scalar[DT]], mut y: List[Scalar[DT]]):
    x = List[Scalar[DT]](length=B * D, fill=0)
    y = List[Scalar[DT]](length=B * O, fill=0)
    for i in range(B * D):
        x[i] = Scalar[DT](((i * 7 + step * 13) % 23) - 11) * 0.09
    for i in range(B * O):
        y[i] = Scalar[DT](((i * 5 + step * 3) % 17) - 8) * 0.11


def _fwd_bwd[BL: Int](
    mut net: NET, c: DeviceContext, r: Int, x: List[Scalar[DT]], y: List[Scalar[DT]]
) raises:
    """Forward, MSE gradient (mean over the BL local rows), vjp — rank r."""
    var xi = Tensor.alloc(BL * D)
    for i in range(BL * D):
        xi.data[i] = x[r * BL * D + i]
    xi.upload(c)
    var out = Tensor.alloc(BL * O)
    var go = Tensor.alloc(BL * O)
    var gi = Tensor.alloc(BL * D)
    with c.push_context():
        net.forward["gpu", BL](TensorRefs[1](xi), out, Optional(c))
        out.download(c)
        var s = Scalar[DT](2.0) / Scalar[DT](BL * O)
        for i in range(BL * O):
            go.data[i] = (out.data[i] - y[r * BL * O + i]) * s
        go.upload(c)
        net.vjp["gpu", BL](TensorRefs[1](xi), go, TensorRefs[1](gi), Optional(c))


def _pg[N: Int](ctx: DeviceContext) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES and N >= 2:
        return ProcessGroup[N].devices(1 << 20)
    else:
        return ProcessGroup[N].shared(ctx)


struct _Out(Movable):
    var params: List[List[Scalar[DT]]]
    var norms: List[Float64]
    var bytes: Int

    def __init__(out self):
        self.params = List[List[Scalar[DT]]]()
        self.norms = List[Float64]()
        self.bytes = 0


def _ddp[N: Int](ctx: DeviceContext, clip: Bool) raises -> _Out:
    comptime BL = B // N
    seed(7)
    var dp = DataParallel[NET, N].make[Xavier](_pg[N](ctx), lr=LR, wd=WD)
    dp.sync_params()
    var res = _Out()
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for step in range(K):
        _batch(step, x, y)
        dp.zero_grad()
        for r in range(N):
            _fwd_bwd[BL](dp.nets[r], dp.ctx(r), r, x, y)
        dp.allreduce_grads()
        if clip:
            dp.clip_grads_device(CLIP)
            res.norms.append(Float64(dp.opts[0].read_clip_norm(dp.ctx(0))))
        dp.step()
    for r in range(N):
        res.params.append(dp.download_params(r))
    res.bytes = 5 * dp.total * 4
    return res^


def _zero[N: Int](ctx: DeviceContext, clip: Bool) raises -> _Out:
    comptime BL = B // N
    seed(7)
    var z = Zero1[NET, N].make[Xavier](_pg[N](ctx), lr=LR, wd=WD)
    z.sync_params()
    var res = _Out()
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for step in range(K):
        _batch(step, x, y)
        z.zero_grad()
        for r in range(N):
            _fwd_bwd[BL](z.nets[r], z.ctx(r), r, x, y)
        z.reduce_scatter_grads()
        if clip:
            z.clip_grads_device(CLIP)
            res.norms.append(Float64(z.read_clip_norm(0)))
            for r in range(1, N):
                assert_true(
                    z.read_clip_norm(r) == z.read_clip_norm(0),
                    "ranks disagree on the global clip norm",
                )
        z.step()
    for r in range(N):
        res.params.append(z.download_params(r))
    res.bytes = z.state_bytes_per_rank(0)
    var shards = String("    N = ") + String(N) + " rows = " + String(z.rows) + " shards:"
    for r in range(N):
        shards += " [" + String(z.shards[r].off) + "+" + String(z.shards[r].n) + ")"
    print(shards, " arena =", z.total)
    return res^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _ulps(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Int:
    """Max distance in units of the last place (same-sign floats)."""
    var m = 0
    for i in range(len(a)):
        var ia = Int(bitcast[DType.int32](a[i]))
        var ib = Int(bitcast[DType.int32](b[i]))
        m = max(m, abs(ia - ib))
    return m


def _gate[N: Int](ctx: DeviceContext) raises:
    print("[N =", N, "]")
    # A + C: no clip -> bit-identical to DDP, replicas agree.
    var d = _ddp[N](ctx, False)
    var z = _zero[N](ctx, False)
    var moved = _max_abs_diff(d.params[0], _ddp_init(ctx))
    for r in range(N):
        var dz = _max_abs_diff(d.params[r], z.params[r])
        var drep = _max_abs_diff(z.params[0], z.params[r])
        print("    rank", r, " max|DDP - ZeRO1| =", dz, " max|rank0 - rank| =", drep)
        assert_true(dz == 0.0, "ZeRO-1 is not bit-identical to DDP (no clip)")
        assert_true(drep == 0.0, "ZeRO-1 replicas drifted apart")
    assert_true(moved > 1e-3, "gate vacuous: the parameters barely moved")
    # B: clip.
    var dc = _ddp[N](ctx, True)
    var zc = _zero[N](ctx, True)
    var dn = 0.0
    var fired = 0
    for i in range(K):
        dn = max(dn, abs(dc.norms[i] - zc.norms[i]) / dc.norms[i])
        if dc.norms[i] > Float64(CLIP):
            fired += 1
    var dpc = _max_abs_diff(dc.params[0], zc.params[0])
    print("    clip: fired", fired, "/", K, " max rel norm diff =", dn,
          " max|DDP - ZeRO1| =", dpc, "(", _ulps(dc.params[0], zc.params[0]), "ulp )")
    assert_true(fired > K // 2, "clip gate vacuous")
    assert_true(dn < 1e-6, "ZeRO-1 clip norm differs from DDP's")
    assert_true(dpc < 1e-5, "ZeRO-1 with clip drifted from DDP")
    for r in range(1, N):
        assert_true(_max_abs_diff(zc.params[0], zc.params[r]) == 0.0, "replicas drifted (clip)")
    # D: memory.
    print("    state bytes/rank: DDP", d.bytes, " ZeRO-1", z.bytes,
          " ratio", Float64(z.bytes) / Float64(d.bytes))


def _ddp_init(ctx: DeviceContext) raises -> List[Scalar[DT]]:
    """The weights both runs start from (rank 0 after the broadcast)."""
    seed(7)
    var dp = DataParallel[NET, 1].make[Xavier](ProcessGroup[1].shared(ctx), lr=LR)
    return dp.download_params(0)


def main() raises:
    print("ZeRO-1 vs DDP (MLP", D, "->", H, "->", O, ", B =", B, ", K =", K, ", AdamW wd =", WD, ")")
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    # The partition is MAX's: 51 rows over 4 ranks -> 13, 13, 13, 12.
    var p = shard_rows(51, 4, 3)
    assert_true(p[0] == 39 and p[1] == 12, "shard_rows does not match MAX's partition")
    comptime if USE_DEVICES:
        _gate[2](ctx)
    else:
        _gate[1](ctx)
        _gate[2](ctx)
        _gate[3](ctx)
        _gate[4](ctx)
    print("ZERO-1 GATES OK")
