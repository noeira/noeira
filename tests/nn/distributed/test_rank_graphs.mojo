"""M2 gates: a data-parallel step captured as one CUDA graph per GPU.

The same MLP step (forward, device MSE vjp, net vjp, gradient collective,
device clip, AdamW) runs K times in four ways, from the same start:

  eager       enqueued every step (the M1 loop)
  whole       `RankGraphs`: every rank's step, collective included, captured
              once and replayed
  split       compute captured, the collective EAGER between two graphs (the
              no-P2P fallback, where MAX's naive allreduce allocates per call)
  zero-whole  ZeRO-1 (reduce-scatter, sharded AdamW, all-gather) captured

Gates: whole == eager and split == eager BIT FOR BIT on every rank (a replay
runs the same kernels on the same buffers, so any difference is a capture
bug: a stale kernel argument, a host value baked at capture time, a missed
node); zero-whole == ZeRO-1 eager bit for bit; replicas agree; the run moved.

Where it means something:
  Mac          CUDAGraph is a compile-time no-op, so every mode runs eagerly
               and the gates pass by construction. Checks the plumbing only.
  1 NVIDIA GPU N = 2 on the shared-context simulator: ONE graph on one
               context, simulator collectives inside it. Checks the
               per-context stream capture (`stream_of_context`) and the
               harness. Also prints MAX's stream vs the interceptor's.
  2+ GPUs      `-D DDP_DEVICES`: one graph per GPU, MAX comm inside them.

Run:  pixi run -e apple mojo run -I . tests/nn/distributed/test_rank_graphs.mojo
NVIDIA (the interceptor needs the pixi activation's LD_PRELOAD):
      pixi run -e nvidia mojo build -I . [-D DDP_DEVICES] \
          tests/nn/distributed/test_rank_graphs.mojo -o /tmp/trg
      pixi run -e nvidia /tmp/trg
"""

from std.random import seed
from std.sys import is_defined, has_nvidia_gpu_accelerator, has_accelerator
from std.testing import assert_true
from std.ffi import OwnedDLHandle
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse_loss import MSELoss
from noeira.nn.distributed.process_group import ProcessGroup, scale_copy
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero import Zero1
from noeira.nn.distributed.rank_graphs import RankGraphs


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
comptime NET = Sequential[LinearReLU[D, H], Linear[H, O]]
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()

comptime EAGER = 0
comptime WHOLE = 1
comptime SPLIT = 2


def _pg(ctx: DeviceContext) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES:
        return ProcessGroup[N].devices(1 << 20)
    else:
        return ProcessGroup[N].shared(ctx)


struct _Data(Movable):
    """Per-rank device buffers. The batch is fixed (uploaded once): the gate
    compares capture against eager, not learning."""

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


struct _DdpJob(Movable):
    """Owns the wrapper and every buffer the step touches, so a capturing
    closure mentions this one struct and nothing it could outlive."""

    var dp: DataParallel[NET, N]
    var data: _Data

    def __init__(out self, ctx: DeviceContext) raises:
        seed(7)
        self.dp = DataParallel[NET, N].make[Xavier](_pg(ctx), lr=LR, wd=WD)
        self.dp.sync_params()
        self.data = _Data(self.dp.pg.ctxs)
        self.dp.synchronize()

    def compute(mut self) raises:
        self.dp.zero_grad()
        for r in range(N):
            var c = self.dp.ctx(r)
            var co = Optional(c)
            with c.push_context():
                self.dp.nets[r].forward["gpu", BL](
                    TensorRefs[1](self.data.x[r]), self.data.out[r], co
                )
                self.data.loss[r].vjp["gpu", BL](
                    self.data.out[r], self.data.y[r], self.data.go[r], co
                )
                self.dp.nets[r].vjp["gpu", BL](
                    TensorRefs[1](self.data.x[r]),
                    self.data.go[r],
                    TensorRefs[1](self.data.gi[r]),
                    co,
                )

    def comm(mut self) raises:
        self.dp.allreduce_grads()

    def update(mut self) raises:
        self.dp.clip_grads_device(CLIP)
        self.dp.step()

    def step(mut self) raises:
        self.compute()
        self.comm()
        self.update()


struct _ZeroJob(Movable):
    var z: Zero1[NET, N]
    var data: _Data

    def __init__(out self, ctx: DeviceContext) raises:
        seed(7)
        self.z = Zero1[NET, N].make[Xavier](_pg(ctx), lr=LR, wd=WD)
        self.z.sync_params()
        self.data = _Data(self.z.pg.ctxs)
        self.z.synchronize()

    def step(mut self) raises:
        self.z.zero_grad()
        for r in range(N):
            var c = self.z.ctx(r)
            var co = Optional(c)
            with c.push_context():
                self.z.nets[r].forward["gpu", BL](
                    TensorRefs[1](self.data.x[r]), self.data.out[r], co
                )
                self.data.loss[r].vjp["gpu", BL](
                    self.data.out[r], self.data.y[r], self.data.go[r], co
                )
                self.z.nets[r].vjp["gpu", BL](
                    TensorRefs[1](self.data.x[r]),
                    self.data.go[r],
                    TensorRefs[1](self.data.gi[r]),
                    co,
                )
        self.z.reduce_scatter_grads()
        self.z.clip_grads_device(CLIP)
        self.z.step()


def _ddp[MODE: Int](ctx: DeviceContext) raises -> List[List[Scalar[DT]]]:
    var job = _DdpJob(ctx)
    var whole = RankGraphs(job.dp.pg.graph_ctxs())
    var pre = RankGraphs(job.dp.pg.graph_ctxs())
    var post = RankGraphs(job.dp.pg.graph_ctxs())
    for _ in range(K):
        comptime if MODE == EAGER:
            job.step()
        elif MODE == WHOLE:

            def _step() capturing raises -> None:
                job.step()

            whole.run[_step]()
        else:

            def _compute() capturing raises -> None:
                job.compute()

            def _update() capturing raises -> None:
                job.update()

            pre.run[_compute]()
            job.comm()
            post.run[_update]()
    job.dp.synchronize()
    var out = List[List[Scalar[DT]]]()
    for r in range(N):
        out.append(job.dp.download_params(r))
    return out^


def _zero[CAPTURE: Bool](ctx: DeviceContext) raises -> List[List[Scalar[DT]]]:
    var job = _ZeroJob(ctx)
    var g = RankGraphs(job.z.pg.graph_ctxs())
    for _ in range(K):
        comptime if CAPTURE:

            def _step() capturing raises -> None:
                job.step()

            g.run[_step]()
        else:
            job.step()
    job.z.synchronize()
    var out = List[List[Scalar[DT]]]()
    for r in range(N):
        out.append(job.z.download_params(r))
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
        print("   ", name, "rank", r, " max|captured - eager| =", d,
              " max|rank0 - rank| =", rep)
        assert_true(d == 0.0, name + ": captured step differs from eager")
        assert_true(rep == 0.0, name + ": replicas drifted apart")


def _stream_probe(ctx: DeviceContext) raises:
    """NVIDIA only: after a launch on `ctx`, the interceptor's process-wide
    stream must be the one MAX reports for `ctx` — the premise of
    `CUDAGraph(stream_of_context=True)`. A mismatch means MAX's kernels run on
    a stream `ctx.stream()` does not name, and per-context capture records
    nothing."""
    comptime if has_nvidia_gpu_accelerator():
        var t = Tensor.alloc_gpu(ctx, 4)
        var u = Tensor.alloc_gpu(ctx, 4)
        # A KERNEL: the interceptor records the stream at `cuLaunchKernelEx`,
        # and a fill or a copy is not one.
        scale_copy(u.dev.value(), t.dev.value(), Scalar[DT](1), 4, ctx)
        ctx.synchronize()
        var lib = OwnedDLHandle("./noeira/cuda/libcuda_intercept.so")
        var get = lib.get_function[Int]("intercept_get_mojo_stream")
        var seen = get()
        var own = CUDA(ctx.stream())
        var mine = Int(own.value()) if own else 0
        print("  stream probe: interceptor", hex(seen), " MAX ctx.stream()", hex(mine))
        if seen == 0:
            print("  (interceptor saw no launch: not preloaded; capture will disable)")
        else:
            assert_true(seen == mine, "MAX's ctx.stream() is not the stream kernels launch on")


def main() raises:
    print("RankGraphs gates: MLP", D, "->", H, "->", O, " N =", N, " B =", B,
          " K =", K, " (devices)" if USE_DEVICES else " (shared-context simulator)")
    comptime if not has_accelerator():
        print("No accelerator — skipping (the distributed gates need a GPU, or Metal for the simulator)")
        return
    var ctx = DeviceContext()
    _stream_probe(ctx)
    var eager = _ddp[EAGER](ctx)
    var fresh = _DdpJob(ctx)
    var init = fresh.dp.download_params(0)
    var moved = _max_abs_diff(eager[0], init)
    print("  eager run moved the weights by", moved)
    assert_true(moved > 1e-3, "gate vacuous: the parameters barely moved")
    # Without P2P (MAX's host-staged collectives allocate per call) only the
    # split mode can capture: the collective runs eagerly between two graphs.
    var whole = fresh.dp.pg.capturable_collectives()
    if whole:
        _gate("DDP whole", _ddp[WHOLE](ctx), eager)
    else:
        print("  DDP whole: SKIPPED, backend", fresh.dp.pg.backend,
              "cannot capture its collectives")
    _gate("DDP split", _ddp[SPLIT](ctx), eager)
    if whole:
        var z_eager = _zero[False](ctx)
        _gate("ZeRO-1 whole", _zero[True](ctx), z_eager)
    else:
        print("  ZeRO-1 whole: SKIPPED (its update runs a collective)")
    print("RANK GRAPH GATES OK")
