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

Overlap (`enable_overlap`): instead of one allreduce after the backward,
buckets of the arena are reduced on a second stream per GPU while the backward
still runs. The model marks where its gradients are final by wrapping modules
in `GradReady[..., ACTIVE=True]` (`grad_marks.mojo`). The step becomes

        dp.zero_grad()
        dp.begin_backward()
        for r in range(N): forward + loss + vjp, rank by rank
        dp.reduce_grads()      # buckets on the comm streams, joined, then 1/N
        ...

`reduce_grads` without overlap is `allreduce_grads`.
"""

from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor
from noeira.nn.optimizer.adam import Adam

from .process_group import ProcessGroup, scale_copy, BACKEND_P2P
from .grad_marks import grad_marks, GradSpan
from .buckets import Bucket, plan_buckets


struct _ParamSlices(ParamVisitor):
    """Every parameter's gradient address and length, in walk order."""

    var addr: List[Int]
    var n: List[Int]

    def __init__(out self):
        self.addr = List[Int]()
        self.n = List[Int]()

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.dev:
            self.addr.append(Int(grad.dev.value().unsafe_ptr()))
            self.n.append(N)


struct DataParallel[M: Module, N: Int](Movable):
    var pg: ProcessGroup[Self.N]
    var nets: List[Self.M]
    var opts: List[Adam]
    var red: List[Tensor]
    """Per-rank allreduce output, arena-sized."""
    var total: Int
    """Arena length in elements (identical on every rank)."""
    var overlap: Bool
    var bucket_elems: Int
    var buckets: List[Bucket]
    """Planned on the first overlapped step, from rank 0's marks."""
    var n_marks: Int
    """Marks per rank per backward; -1 until planned."""

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
        self.overlap = False
        self.bucket_elems = 0
        self.buckets = List[Bucket]()
        self.n_marks = -1

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

    # ── overlap ─────────────────────────────────────────────────────────────

    def enable_overlap(mut self, bucket_elems: Int) raises:
        """Reduce gradient buckets of up to `bucket_elems` elements on a
        second stream per GPU, each as soon as the backward has finished it.
        Needs the model to mark its gradients (`GradReady[..., True]`)."""
        self.pg.enable_comm_streams()
        _ = grad_marks()
        self.overlap = True
        self.bucket_elems = bucket_elems

    def begin_backward(mut self) raises:
        """Call before the first rank's vjp of a step (no-op without overlap)."""
        if self.overlap and Self.N > 1:
            grad_marks()[].begin()

    def reduce_grads(mut self) raises:
        """Call once every rank's vjp is enqueued: `grd_r` becomes the mean
        over ranks, as `allreduce_grads`, with the buckets overlapped."""
        if not self.overlap or Self.N == 1:
            self.allreduce_grads()
            return
        var reg = grad_marks()
        reg[].end()
        var count = len(reg[].marks)
        if self.n_marks < 0:
            self._plan(count)
        elif count != Self.N * self.n_marks:
            raise Error(
                "DataParallel.reduce_grads: " + String(count)
                + " marks this step, planned for " + String(Self.N * self.n_marks)
            )
        var k_end = self.n_marks
        # On P2P the kernels synchronize the ranks themselves, so each comm
        # stream waits only for its own rank. The naive and shared backends
        # read peers' buffers with no ordering of their own: wait for all.
        var all_ranks = self.pg.backend != BACKEND_P2P
        for b in range(len(self.buckets)):
            ref bk = self.buckets[b]
            for r in range(Self.N):
                var cc = self.pg.comm_ctx(r)
                for k in range(Self.N):
                    if not all_ranks and k != r:
                        continue
                    var slot = -1
                    if bk.ready < k_end:
                        slot = reg[].marks[k * k_end + bk.ready].slot
                    if slot >= 0:
                        cc.stream().enqueue_wait_for(reg[].event(slot))
                    else:
                        cc.enqueue_wait_for(self.pg.ctx(k))
            var ins = List[DeviceBuffer[DT]](capacity=Self.N)
            var outs = List[DeviceBuffer[DT]](capacity=Self.N)
            for r in range(Self.N):
                ins.append(
                    self.opts[r].arena.grd.dev.value().create_sub_buffer[DT](
                        bk.off, bk.n
                    )
                )
                outs.append(
                    self.red[r].dev.value().create_sub_buffer[DT](bk.off, bk.n)
                )
            self.pg.allreduce_sum_comm(ins, outs, bk.n)
        # Join: no main stream touches grd or red before every comm stream
        # that reads them is done.
        for r in range(Self.N):
            for k in range(Self.N):
                if not all_ranks and k != r:
                    continue
                self.pg.ctx(r).enqueue_wait_for(self.pg.comm_ctx(k))
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Self.N)
        for r in range(Self.N):
            scale_copy(
                self.opts[r].arena.grd.dev.value(),
                self.red[r].dev.value(),
                inv_n,
                self.total,
                self.pg.ctx(r),
            )

    def _plan(mut self, count: Int) raises:
        """Bucket plan from rank 0's marks; checks every rank marked the same
        arena slices."""
        if count == 0 or count % Self.N != 0:
            raise Error(
                "DataParallel overlap: " + String(count) + " marks for "
                + String(Self.N) + " ranks (wrap modules in GradReady[..., True])"
            )
        var k_end = count // Self.N
        var reg = grad_marks()
        comptime ES = size_of[Scalar[DT]]()
        var base0 = Int(self.opts[0].arena.grd.dev.value().unsafe_ptr())
        var lo = List[Int](capacity=k_end)
        var hi = List[Int](capacity=k_end)
        for i in range(k_end):
            lo.append((reg[].marks[i].lo - base0) // ES)
            hi.append((reg[].marks[i].hi - base0) // ES)
        for r in range(1, Self.N):
            var base = Int(self.opts[r].arena.grd.dev.value().unsafe_ptr())
            for i in range(k_end):
                ref m = reg[].marks[r * k_end + i]
                if (m.lo - base) // ES != lo[i] or (m.hi - base) // ES != hi[i]:
                    raise Error(
                        "DataParallel overlap: rank " + String(r) + " mark "
                        + String(i) + " covers another arena slice than rank 0's"
                    )
        var sl = _ParamSlices()
        self.nets[0].for_each_param["gpu"](sl, Optional(self.pg.ctx(0)))
        var starts = List[Int](capacity=len(sl.addr))
        var sizes = List[Int](capacity=len(sl.addr))
        for i in range(len(sl.addr)):
            starts.append((sl.addr[i] - base0) // ES)
            sizes.append(sl.n[i])
        self.buckets = plan_buckets(
            starts, sizes, self.total, lo, hi, self.bucket_elems
        )
        self.n_marks = k_end

    def bucket_summary(self) -> String:
        """`n buckets: [off+n @ready] ...` (after the first overlapped step)."""
        var s = String(len(self.buckets), " buckets of <= ", self.bucket_elems,
                       " elems, ", self.n_marks, " marks/rank:")
        for b in range(len(self.buckets)):
            ref bk = self.buckets[b]
            s += String(" [", bk.off, "+", bk.n, " @",
                        "end" if bk.ready == self.n_marks else String(bk.ready), "]")
        return s

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
