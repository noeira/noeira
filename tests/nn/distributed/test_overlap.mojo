"""Gates for the overlapped gradient allreduce (buckets on a comm stream).

1. `plan_buckets` on a hand case: readiness per parameter, segments, merges,
   the end-of-backward tail, and the launch order.
2. The same MLP DDP step, K steps from the same start, five ways:

     plain        Sequential of the bare layers, one allreduce after the vjp
     wrapped      every layer in GradReady[..., True], overlap OFF
     per-module   overlap, a 1-element bucket target (one bucket per module)
     one-bucket   overlap, a target larger than the arena
     captured     per-module overlap inside RankGraphs (where the backend's
                  collectives can be captured: simulator, P2P)

   Every one must equal `plain` BIT FOR BIT on every rank: the buckets sum
   the same two (or N) values per element as the whole-arena allreduce, so
   any difference is a scheduling bug (a bucket reduced before its gradients
   were final, a main stream overwriting a gradient a peer still reads).

Where it means something:
  Mac          simulator; the comm context is a second Metal queue
  1 NVIDIA GPU simulator with a real second stream, and the captured mode
               (fork/join of the comm stream inside one CUDA graph)
  2+ GPUs      `-D DDP_DEVICES`: MAX comm on a comm stream per GPU

Run:  pixi run -e apple mojo run -I . tests/nn/distributed/test_overlap.mojo
NVIDIA: build with `mojo build`, run the binary under `pixi run -e nvidia`
(the captured mode needs the interceptor's LD_PRELOAD).
"""

from std.random import seed
from std.sys import is_defined, has_accelerator
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Xavier
from noeira.nn.core.module import Module
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse_loss import MSELoss
from noeira.nn.distributed.process_group import ProcessGroup, backend_name
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.rank_graphs import RankGraphs
from noeira.nn.distributed.grad_marks import GradReady
from noeira.nn.distributed.buckets import plan_buckets


comptime D = 16
comptime H = 64
comptime O = 8
comptime N = 2
comptime B = 32
comptime BL = B // N
comptime K = 30
comptime LR: Scalar[DT] = 3e-3
comptime WD: Scalar[DT] = 0.05
comptime CLIP: Scalar[DT] = 0.05
comptime PLAIN = Sequential[LinearReLU[D, H], LinearReLU[H, H], Linear[H, O]]
comptime MARKED = Sequential[
    GradReady[LinearReLU[D, H]],
    GradReady[LinearReLU[H, H]],
    GradReady[Linear[H, O]],
]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()


def _pg(ctx: DeviceContext) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES:
        return ProcessGroup[N].devices(1 << 20)
    else:
        return ProcessGroup[N].shared(ctx)


# ── 1. the planner ───────────────────────────────────────────────────────────


def _test_plan() raises:
    # Arena of 100 elements, five params: [0,10) [10,30) [32,50) [50,80) [80,96)
    # Marks in backward order: m0 = [80,96), m1 = [32,80) (two params), and
    # [0,30) unmarked (ready at the end, k = 2).
    var starts: List[Int] = [0, 10, 32, 50, 80]
    var sizes: List[Int] = [10, 20, 18, 30, 16]
    var lo: List[Int] = [80, 32]
    var hi: List[Int] = [96, 80]
    # Target 1: no merges. Segments [80,100)@0, [32,80)@1, [0,32)@end.
    var b1 = plan_buckets(starts, sizes, 100, lo, hi, 1)
    assert_equal(len(b1), 3)
    assert_equal(b1[0].off, 80); assert_equal(b1[0].n, 20); assert_equal(b1[0].ready, 0)
    assert_equal(b1[1].off, 32); assert_equal(b1[1].n, 48); assert_equal(b1[1].ready, 1)
    assert_equal(b1[2].off, 0); assert_equal(b1[2].n, 32); assert_equal(b1[2].ready, 2)
    # Target 70: m0+m1 merge (68), the tail does not fit.
    var b2 = plan_buckets(starts, sizes, 100, lo, hi, 70)
    assert_equal(len(b2), 2)
    assert_equal(b2[0].off, 32); assert_equal(b2[0].n, 68); assert_equal(b2[0].ready, 1)
    assert_equal(b2[1].off, 0); assert_equal(b2[1].ready, 2)
    # Target 1000: one bucket, ready at the end.
    var b3 = plan_buckets(starts, sizes, 100, lo, hi, 1000)
    assert_equal(len(b3), 1)
    assert_equal(b3[0].n, 100); assert_equal(b3[0].ready, 2)
    # A mark that cuts a parameter in two is refused.
    var bad_lo: List[Int] = [40]
    var bad_hi: List[Int] = [96]
    var raised = False
    try:
        _ = plan_buckets(starts, sizes, 100, bad_lo, bad_hi, 1)
    except:
        raised = True
    assert_true(raised, "a mark straddling a parameter must be refused")
    print("  plan_buckets: hand cases OK")


# ── 2. the step ──────────────────────────────────────────────────────────────


struct _Data(Movable):
    var x: List[Tensor]
    var y: List[Tensor]
    var out: List[Tensor]
    var go: List[Tensor]
    var gi: List[Tensor]
    var loss: List[MSELoss[O]]

    def __init__(out self, ctxs: List[DeviceContext]) raises:
        self.x = List[Tensor]()
        self.y = List[Tensor]()
        self.out = List[Tensor]()
        self.go = List[Tensor]()
        self.gi = List[Tensor]()
        self.loss = List[MSELoss[O]]()
        for r in range(N):
            var c = ctxs[r]
            var x = Tensor.alloc(BL * D)
            var y = Tensor.alloc(BL * O)
            for i in range(BL * D):
                x.data[i] = Scalar[DT]((((r * BL * D + i) * 7) % 23) - 11) * 0.09
            for i in range(BL * O):
                y.data[i] = Scalar[DT]((((r * BL * O + i) * 5) % 17) - 8) * 0.11
            x.upload(c)
            y.upload(c)
            self.x.append(x^)
            self.y.append(y^)
            self.out.append(Tensor.alloc_gpu(c, BL * O))
            self.go.append(Tensor.alloc_gpu(c, BL * O))
            self.gi.append(Tensor.alloc_gpu(c, BL * D))
            self.loss.append(MSELoss[O].make_gpu(c))


struct _Job[M: Module](Movable):
    var dp: DataParallel[Self.M, N]
    var data: _Data

    def __init__(out self, ctx: DeviceContext, bucket: Int) raises:
        """`bucket` > 0: overlap with that bucket target; 0: no overlap."""
        seed(7)
        self.dp = DataParallel[Self.M, N].make[Xavier](_pg(ctx), lr=LR, wd=WD)
        if bucket > 0:
            self.dp.enable_overlap(bucket)
        self.dp.sync_params()
        self.data = _Data(self.dp.pg.ctxs)
        self.dp.synchronize()

    def step(mut self) raises:
        self.dp.zero_grad()
        self.dp.begin_backward()
        for r in range(N):
            var c = self.dp.ctx(r)
            var co = Optional(c)
            comptime A = Self.M.ACT_DT
            with c.push_context():
                self.dp.nets[r].forward["gpu", BL](
                    child_refs[Self.M.ARITY, A](
                        rebind[TensorImpl[A]](self.data.x[r])
                    ),
                    rebind[TensorImpl[A]](self.data.out[r]),
                    co,
                )
                self.data.loss[r].vjp["gpu", BL](
                    self.data.out[r], self.data.y[r], self.data.go[r], co
                )
                self.dp.nets[r].vjp["gpu", BL](
                    child_refs[Self.M.ARITY, A](
                        rebind[TensorImpl[A]](self.data.x[r])
                    ),
                    rebind[TensorImpl[A]](self.data.go[r]),
                    child_refs[Self.M.ARITY, A](
                        rebind[TensorImpl[A]](self.data.gi[r])
                    ),
                    co,
                )
        self.dp.reduce_grads()
        self.dp.clip_grads_device(CLIP)
        self.dp.step()


def _run[M: Module, CAPTURE: Bool](
    ctx: DeviceContext, bucket: Int, label: String
) raises -> List[List[Scalar[DT]]]:
    var job = _Job[M](ctx, bucket)
    var g = RankGraphs(job.dp.pg.graph_ctxs())
    for _ in range(K):
        comptime if CAPTURE:

            def _step() capturing raises -> None:
                job.step()

            g.run[_step]()
        else:
            job.step()
    job.dp.synchronize()
    if bucket > 0:
        print("   ", label, ":", job.dp.bucket_summary())
    var out = List[List[Scalar[DT]]]()
    for r in range(N):
        out.append(job.dp.download_params(r))
    return out^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def _gate(name: String, got: List[List[Scalar[DT]]], ref_: List[List[Scalar[DT]]]) raises:
    for r in range(N):
        var d = _max_abs_diff(got[r], ref_[r])
        var rep = _max_abs_diff(got[0], got[r])
        print("   ", name, "rank", r, " max|got - plain| =", d,
              " max|rank0 - rank| =", rep)
        assert_true(d == 0.0, name + ": differs from the plain DDP step")
        assert_true(rep == 0.0, name + ": replicas drifted apart")


def main() raises:
    print("Overlap gates: MLP", D, "->", H, "->", H, "->", O, " N =", N,
          " B =", B, " K =", K,
          " (devices)" if USE_DEVICES else " (shared-context simulator)")
    _test_plan()
    comptime if not has_accelerator():
        print("No accelerator — skipping the training gates (they need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    var plain = _run[PLAIN, False](ctx, 0, "plain")
    var fresh = _Job[PLAIN](ctx, 0)
    var init = fresh.dp.download_params(0)
    var moved = _max_abs_diff(plain[0], init)
    print("  plain run moved the weights by", moved)
    assert_true(moved > 1e-3, "gate vacuous: the parameters barely moved")
    _gate("wrapped, no overlap", _run[MARKED, False](ctx, 0, "wrapped"), plain)
    _gate("overlap, per-module", _run[MARKED, False](ctx, 1, "per-module"), plain)
    _gate("overlap, one bucket", _run[MARKED, False](ctx, 1 << 30, "one-bucket"), plain)
    if fresh.dp.pg.capturable_collectives():
        _gate("overlap, captured", _run[MARKED, True](ctx, 1, "captured"), plain)
    else:
        print("  overlap, captured: SKIPPED, backend",
              backend_name(fresh.dp.pg.backend), "cannot capture its collectives")
    print("OVERLAP GATES OK")
