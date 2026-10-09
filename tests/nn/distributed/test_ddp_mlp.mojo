"""DDP correctness gates on an MLP (M1, gates 1-4).

Runs on any GPU through the shared-context simulator (`ProcessGroup.shared`):
every rank on one context, the allreduce a local kernel. That exercises the
whole data-parallel data flow — replica build, weight broadcast, batch split,
gradient mean, clip, update — without a second GPU. `-D DDP_DEVICES` runs the
same gates on GPUs 0..N-1 with MAX's `comm` allreduce instead.

  1. N = 1 identity: `DataParallel[NET, 1]` is bit-identical to a plain
     model + arena-Adam loop.
  2. Equivalence: N = 2 with B/2 per rank tracks N = 1 with B (parameter
     difference reported; the reduction order differs, so not bit-exact).
  3. Replica agreement: after K steps every rank's arena is bit-identical.
     Also checks the broadcast is not vacuous: before `sync_params` the
     replicas DIFFER (Xavier draws on the host RNG per replica).
  4. Clip: with clipping on, the N = 2 pre-clip norm matches N = 1.

The loss is MSE, with its gradient built on the host: a mean over the
LOCAL batch, so the 1/N in the allreduce turns it into the global mean.

Run (Mac):   pixi run -e apple mojo run -I . tests/nn/distributed/test_ddp_mlp.mojo
Run (GPUs):  pixi run -e nvidia mojo build -D DDP_DEVICES -I . tests/nn/distributed/test_ddp_mlp.mojo -o /tmp/t && /tmp/t
"""

from std.random import seed
from std.sys import is_defined
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.distributed.process_group import ProcessGroup, backend_name
from noeira.nn.distributed.data_parallel import DataParallel


comptime D = 16
comptime H = 64
comptime O = 8
comptime B = 32
comptime K = 50
comptime LR: Scalar[DT] = 3e-3
comptime CLIP: Scalar[DT] = 0.05  # small enough to fire on most steps
comptime NET = Sequential[LinearReLU[D, H], Linear[H, O]]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()


def _global_batch(step: Int, mut x: List[Scalar[DT]], mut y: List[Scalar[DT]]):
    """A deterministic batch of B rows for `step` (same for every N)."""
    x = List[Scalar[DT]](length=B * D, fill=0)
    y = List[Scalar[DT]](length=B * O, fill=0)
    for i in range(B * D):
        x[i] = Scalar[DT](((i * 7 + step * 13) % 23) - 11) * 0.09
    for i in range(B * O):
        y[i] = Scalar[DT](((i * 5 + step * 3) % 17) - 8) * 0.11


def _mse_grad[BL: Int](
    yhat: List[Scalar[DT]], y: List[Scalar[DT]], row0: Int, mut go: Tensor
) -> Float64:
    """d/d(out) of mean((out - y)^2) over the BL local rows; returns the sum
    of squared errors (for the loss)."""
    var sse = 0.0
    var scale = Scalar[DT](2.0) / Scalar[DT](BL * O)
    for i in range(BL * O):
        var e = yhat[i] - y[row0 * O + i]
        go.data[i] = e * scale
        sse += Float64(e * e)
    return sse


def _rank_fwd_bwd[N: Int, BL: Int](
    mut dp: DataParallel[NET, N],
    r: Int,
    x: List[Scalar[DT]],
    y: List[Scalar[DT]],
) raises -> Float64:
    """Forward + MSE grad + vjp for rank r on rows [r*BL, (r+1)*BL)."""
    var c = dp.ctx(r)
    var xi = Tensor.alloc(BL * D)
    for i in range(BL * D):
        xi.data[i] = x[r * BL * D + i]
    xi.upload(c)
    var out = Tensor.alloc(BL * O)
    var go = Tensor.alloc(BL * O)
    var gi = Tensor.alloc(BL * D)
    # ⚠ Rank r's driver context must be CURRENT around its forward/vjp:
    # noeira's cuBLAS / cuBLASLt / cuDNN calls run against whatever context is
    # current on this thread (MAX's own vendor GEMM pushes it; ours do not).
    # One host thread driving N GPUs makes that rank 0's context otherwise.
    var sse: Float64
    with c.push_context():
        dp.nets[r].forward["gpu", BL](TensorRefs[1](xi), out, Optional(c))
        out.download(c)
        sse = _mse_grad[BL](out.data, y, r * BL, go)
        go.upload(c)
        dp.nets[r].vjp["gpu", BL](
            TensorRefs[1](xi), go, TensorRefs[1](gi), Optional(c)
        )
    return sse


struct _Run(Movable):
    var losses: List[Float64]
    var norms: List[Float64]
    var params: List[List[Scalar[DT]]]  # per rank, final arena

    def __init__(out self):
        self.losses = List[Float64]()
        self.norms = List[Float64]()
        self.params = List[List[Scalar[DT]]]()


def _make_pg[N: Int](ctx: DeviceContext, max_elems: Int) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES and N >= 2:
        return ProcessGroup[N].devices(max_elems)
    else:
        return ProcessGroup[N].shared(ctx)


def _train_ddp[N: Int](ctx: DeviceContext, clip: Bool, check_broadcast: Bool) raises -> _Run:
    comptime BL = B // N
    seed(42)
    var dp = DataParallel[NET, N].make[Xavier](_make_pg[N](ctx, 1 << 20), lr=LR)
    print("    N =", N, "backend =", backend_name(dp.pg.backend), "arena =", dp.total)
    comptime if N >= 2:
        if check_broadcast:
            var p0 = dp.download_params(0)
            var p1 = dp.download_params(1)
            var differ = False
            for i in range(len(p0)):
                if p0[i] != p1[i]:
                    differ = True
                    break
            assert_true(differ, "replicas identical BEFORE sync_params: the broadcast gate is vacuous")
    dp.sync_params()
    var run = _Run()
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for step in range(K):
        _global_batch(step, x, y)
        dp.zero_grad()
        var sse = 0.0
        for r in range(N):
            sse += _rank_fwd_bwd[N, BL](dp, r, x, y)
        dp.allreduce_grads()
        if clip:
            run.norms.append(Float64(dp.clip_grads(CLIP)))
        dp.step()
        run.losses.append(sse / Float64(B * O))
    dp.synchronize()
    for r in range(N):
        run.params.append(dp.download_params(r))
    return run^


def _train_plain(ctx: DeviceContext) raises -> _Run:
    """Today's single-GPU loop: model + arena Adam, no DataParallel."""
    seed(42)
    var net = NET.make["gpu", Xavier](Optional(ctx))
    var opt = Adam(lr=LR)
    opt.adopt["gpu"](net, Optional(ctx))
    var run = _Run()
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for step in range(K):
        _global_batch(step, x, y)
        opt.zero_grad["gpu"](net, Optional(ctx))
        var xi = Tensor.alloc(B * D)
        for i in range(B * D):
            xi.data[i] = x[i]
        xi.upload(ctx)
        var out = Tensor.alloc(B * O)
        var go = Tensor.alloc(B * O)
        var gi = Tensor.alloc(B * D)
        net.forward["gpu", B](TensorRefs[1](xi), out, Optional(ctx))
        out.download(ctx)
        var sse = _mse_grad[B](out.data, y, 0, go)
        go.upload(ctx)
        net.vjp["gpu", B](TensorRefs[1](xi), go, TensorRefs[1](gi), Optional(ctx))
        opt.step["gpu"](net, Optional(ctx))
        run.losses.append(sse / Float64(B * O))
    ctx.synchronize()
    var t = Tensor.alloc(opt.arena.total)
    ctx.enqueue_copy(t.data.unsafe_ptr(), opt.arena.val.dev.value())
    ctx.synchronize()
    run.params.append(t.data.copy())
    return run^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _max_abs(a: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i])))
    return m


def main() raises:
    print("DDP correctness gates (MLP", D, "->", H, "->", O, ", B =", B, ", K =", K, ")")
    var ctx = DeviceContext()

    # ── Gate 1: N = 1 identity ────────────────────────────────────────────────
    print("[1] N = 1 identity vs the plain loop")
    var plain = _train_plain(ctx)
    var one = _train_ddp[1](ctx, False, False)
    var d1 = _max_abs_diff(plain.params[0], one.params[0])
    var same_loss = True
    for i in range(K):
        if plain.losses[i] != one.losses[i]:
            same_loss = False
    print("    max|plain - dp1| =", d1, " losses identical:", same_loss,
          " loss", plain.losses[0], "->", plain.losses[K - 1])
    assert_true(d1 == 0.0 and same_loss, "N = 1 is not bit-identical")
    assert_true(plain.losses[K - 1] < 0.8 * plain.losses[0], "the MLP did not train")

    # ── Gates 2 + 3: N = 2 vs N = 1, replica agreement, broadcast ─────────────
    print("[2,3] N = 2 (B/2 per rank) vs N = 1 (B)")
    var two = _train_ddp[2](ctx, False, True)
    var drep = _max_abs_diff(two.params[0], two.params[1])
    print("    replica agreement max|rank0 - rank1| =", drep)
    assert_true(drep == 0.0, "replicas drifted apart")
    var d2 = _max_abs_diff(one.params[0], two.params[0])
    var scale = _max_abs(one.params[0])
    var dl = 0.0
    for i in range(K):
        dl = max(dl, abs(one.losses[i] - two.losses[i]) / one.losses[i])
    print("    max|N1 - N2| params =", d2, "(max|p| =", scale, ") max rel loss diff =", dl)
    assert_true(d2 < 1e-4 * scale + 1e-6, "N = 2 diverged from N = 1 (params)")
    assert_true(dl < 1e-4, "N = 2 diverged from N = 1 (loss)")

    # ── Gate 4: clip ─────────────────────────────────────────────────────────
    print("[4] clip (max_norm =", CLIP, ")")
    var c1 = _train_ddp[1](ctx, True, False)
    var c2 = _train_ddp[2](ctx, True, False)
    var dn = 0.0
    var fired = 0
    for i in range(K):
        dn = max(dn, abs(c1.norms[i] - c2.norms[i]) / c1.norms[i])
        if c1.norms[i] > Float64(CLIP):
            fired += 1
    var dclip = _max_abs_diff(c1.params[0], c2.params[0])
    print("    max rel norm diff =", dn, " clipped steps:", fired, "/", K,
          " max|N1 - N2| params =", dclip)
    assert_true(fired > K // 2, "clip gate vacuous: clipping rarely fired")
    assert_true(dn < 1e-5, "N = 2 clip norm differs from N = 1")
    assert_true(_max_abs_diff(c2.params[0], c2.params[1]) == 0.0, "replicas drifted apart under clip")
    print("DDP MLP GATES OK")
