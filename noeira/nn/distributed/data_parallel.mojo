"""DataParallel — DDP over the optimizer's parameter arena (GPU only).

`ParamArena` packs every gradient of a model into ONE contiguous device buffer
(`arena.grd`), so data parallelism needs no hook in any module: after each
replica's backward, one allreduce over the whole arena gives every replica the
mean gradient, and the unchanged grouped Adam kernel applies the same update on
every rank.

    var dp = DataParallel[NET, N].make[INIT](pg^, lr=...)
    # per-replica surgery (tying, scaled init) on dp.nets[r], then:
    dp.sync_params()                       # rank 0's weights -> every rank
    for step in ...:
        dp.zero_grad()
        for r in range(N):                 # forward + loss + vjp, rank by rank
            dp.nets[r].forward[...](..., Optional(dp.ctx(r)))
            ...
        dp.allreduce_grads()               # grd_r = mean over ranks
        _ = dp.clip_grads(max_norm)        # optional, identical on every rank
        dp.step()

Everything is enqueued from one host thread and nothing here blocks it, which
is what MAX's collectives require: every rank's allreduce is launched before
any can finish.

The reduced gradient goes through a per-rank scratch arena (`red`) and comes
back into `grd` scaled by 1/N in one kernel, because MAX's 1-stage allreduce
writes its output while peers still read its input (no in-place). That costs
one arena of memory per GPU and one elementwise pass; M0 measures whether the
2-stage path (the one large arenas take) can run in place.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.optimizer.adam import Adam

from .process_group import ProcessGroup, scale_copy


struct DataParallel[M: Module, N: Int](Movable):
    var pg: ProcessGroup[Self.N]
    var nets: List[Self.M]
    var opts: List[Adam]
    var red: List[Tensor]
    """Per-rank allreduce output, arena-sized."""
    var total: Int
    """Arena length in elements (identical on every rank)."""

    def __init__(
        out self,
        var pg: ProcessGroup[Self.N],
        var nets: List[Self.M],
        var opts: List[Adam],
        var red: List[Tensor],
        total: Int,
    ):
        self.pg = pg^
        self.nets = nets^
        self.opts = opts^
        self.red = red^
        self.total = total

    @staticmethod
    def make[
        INIT: Initializer
    ](
        var pg: ProcessGroup[Self.N],
        lr: Scalar[DT],
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ) raises -> Self:
        """Build one replica per rank on that rank's context and adopt its
        parameters into an arena. The replicas are NOT synchronized yet: do
        any per-replica surgery, then call `sync_params`."""
        var nets = List[Self.M](capacity=Self.N)
        var opts = List[Adam](capacity=Self.N)
        var red = List[Tensor](capacity=Self.N)
        var total = -1
        for r in range(Self.N):
            var c = pg.ctx(r)
            nets.append(Self.M.make["gpu", INIT](Optional(c)))
            opts.append(Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps, wd=wd))
            opts[r].adopt["gpu"](nets[r], Optional(c))
            var t = opts[r].arena.total
            if total >= 0 and t != total:
                raise Error(
                    "DataParallel.make: rank "
                    + String(r)
                    + " arena has "
                    + String(t)
                    + " elements, rank 0 has "
                    + String(total)
                )
            total = t
            red.append(Tensor.alloc_gpu(c, t))
        return Self(pg^, nets^, opts^, red^, total)

    def ctx(self, r: Int) -> DeviceContext:
        return self.pg.ctx(r)

    def sync_params(mut self) raises:
        """Copy rank 0's parameter arena to every rank.

        Seeding every replica identically is not enough: initializers that
        draw on the host advance the process RNG, so replica r is built from
        a different stream than replica 0."""
        var vals = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            vals.append(self.opts[r].arena.val.dev.value())
        self.pg.broadcast(0, vals, self.total)

    def zero_grad(mut self) raises:
        for r in range(Self.N):
            self.opts[r].arena.zero_grad(self.pg.ctx(r))

    def allreduce_grads(mut self) raises:
        """`grd_r = (1/N) * sum_k grd_k` on every rank."""
        if Self.N == 1:
            return
        var ins = List[DeviceBuffer[DT]](capacity=Self.N)
        var outs = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            ins.append(self.opts[r].arena.grd.dev.value())
            outs.append(self.red[r].dev.value())
        self.pg.allreduce_sum(ins, outs, self.total)
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Self.N)
        for r in range(Self.N):
            scale_copy(ins[r], outs[r], inv_n, self.total, self.pg.ctx(r))

    def clip_grads(mut self, max_norm: Scalar[DT]) raises -> Scalar[DT]:
        """Global grad-norm clip on every rank; returns rank 0's pre-clip norm.

        After `allreduce_grads` every rank holds the same gradient, so each
        rank computing the norm of its own copy is redundant but consistent:
        no extra collective. ⚠ Downloads the norm (a host sync per rank) —
        a measurement path, not the hot loop's."""
        var norm0 = Scalar[DT](0.0)
        for r in range(Self.N):
            var nr = self.opts[r].arena_clip(max_norm, self.pg.ctx(r))
            if r == 0:
                norm0 = nr
        return norm0

    def clip_grads_device(mut self, max_norm: Scalar[DT]) raises:
        """`clip_grads` without the host read: device kernels only, so it
        neither blocks the host nor breaks a capture. The pre-clip norm stays
        in each optimizer's device buffer (`opts[r].read_clip_norm`)."""
        for r in range(Self.N):
            self.opts[r].clip_grads_device["gpu"](
                self.nets[r], max_norm, Optional(self.pg.ctx(r))
            )

    def set_lr(mut self, lr: Scalar[DT]):
        for r in range(Self.N):
            self.opts[r].set_lr(lr)

    def step(mut self) raises:
        """One Adam update on every rank (identical inputs, identical update)."""
        for r in range(Self.N):
            self.opts[r].step["gpu"](self.nets[r], Optional(self.pg.ctx(r)))

    def synchronize(self) raises:
        self.pg.synchronize()

    def state_bytes_per_rank(self) -> Int:
        """Bytes of weights + gradients + optimizer state per rank: val, grd,
        the allreduce scratch `red`, Adam's m and v (no activations, no Signal
        payload)."""
        return 5 * self.total * 4

    def download_params(mut self, r: Int) raises -> List[Scalar[DT]]:
        """Rank r's whole parameter arena on the host (gaps included)."""
        var c = self.pg.ctx(r)
        var t = Tensor.alloc(self.total)
        c.enqueue_copy(
            t.data.unsafe_ptr(), self.opts[r].arena.val.dev.value()
        )
        c.synchronize()
        return t.data.copy()
