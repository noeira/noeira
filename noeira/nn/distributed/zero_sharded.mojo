"""ZeroSharded — ZeRO-2 and ZeRO-3 over per-unit shards (FSDP's SHARD_GRAD_OP
and FULL_SHARD).

Units. A unit is a wrapped module (`GradReady[..., True]`, `grad_marks.mojo`),
plus the ROOT: every parameter outside a wrapper (for the GPT: the embedding,
the positional bias and the final LayerNorm; the tied head reads the
embedding's cells). A unit's parameters are packed into one flat buffer, each
on a `PARAM_ALIGN` boundary, viewed as `[rows, PARAM_ALIGN]`; rank r owns
`shard_rows(rows, N, r)` rows of EVERY unit, so each gather or scatter of a
unit moves an equal part through every rank (FSDP's flat-parameter sharding,
where ZeRO-1 here cut the whole arena once).

Per rank, persistent: the shard arena, five fp32 values per owned element —
master weight, reduced gradient, Adam m and v, decay mask. The step:

    forward_backward(bodies) # bodies[r].run_rank(r) on RankFibers (fibers.mojo)
      [ZeRO-3] gather the root's weights (kept for the whole step)
      every unit, in forward order, on every rank:
          [ZeRO-3] all-gather its weights into its slot (rendezvous)
          forward
      loss, then every unit in backward order:
          [ZeRO-3] all-gather its weights again, unless the slot still holds them
          zero its gradient slot; vjp
          reduce-scatter the slot into each rank's gradient shard, x 1/N
          (rendezvous: every rank's vjp of the unit is enqueued before it)
      reduce-scatter the root's gradient
    clip_grads_device(c)     # global norm: per-shard sums, one scalar allreduce
    step()                   # Adam on the shard arena
      [ZeRO-2] all-gather every unit into the replicated weights

ZeRO-2 keeps the weights replicated (the unit-major `full` buffer) and shards
gradients and optimizer state. ZeRO-3 shards the weights too: they exist in
full only in `slots` unit-sized slots and the root's buffer. Gradients, in
both, live in `slots` unit-sized slots (unit u in slot u mod slots) plus the
root's buffer: a unit's slot is free again once its reduce-scatter ran. Every
parameter is bound to its slot ONCE (`sync_params`), so the pointers the
kernels see never change and a captured graph would replay the same
addresses.

Memory per rank, fp32 values (Psi = parameters, U = largest unit, R = root):

    DDP     val + grd + red + m + v                      = 5 Psi
    ZeRO-1  val + grd + (g, m, v) / N                    = 2 Psi + 3 Psi/N
    ZeRO-2  full + slots.U + R + (val, g, m, v) / N      = Psi + s.U + R + 4 Psi/N
    ZeRO-3  2 (slots.U + R) + (val, g, m, v) / N         = 2 s.U + 2 R + 4 Psi/N
            (+ the decay mask: Psi in DDP and ZeRO-1, Psi/N here)

The master weights are a separate shard in ZeRO-2 (Psi/N more than updating
the replica in place) so both stages share one optimizer path; with bf16
working weights that split is what DeepSpeed does anyway.

Exactness. Each rank's gradients are computed by the same kernels on the
same values as in DDP; the reduce-scatter sums ranks in the order DDP's
allreduce does (rank 0, then += rank k) and then scales by 1/N; Adam is
elementwise. So without clipping, ZeroSharded's weights equal DDP's bit for
bit. The clip norm is summed per shard, as in ZeRO-1, so a clipped run
differs in the last bits of the norm.

Ordering. The shared simulator runs every rank on one stream. On P2P, MAX's
reduce-scatter and all-gather open and close with a cross-GPU barrier, so a
rank's collective has started only when every peer's inputs are ready and
ends only when no peer still reads its buffers. The naive path brackets its
collectives with `_stream_barrier`. `forward_backward` and `step` add one
barrier each at the boundary where a rank's shard is read by peers (gathers)
and then overwritten (Adam).

Limits: a tied parameter must sit in one unit (the GPT's sit in the root);
one vjp per unit per step (no micro-batch accumulation: the slot is zeroed
and scattered each time); collectives run on the main streams (no comm-stream
prefetch yet); no checkpoint save/load.
"""

from std.memory import Pointer
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.param import ParamVisitor, ParamVersionBump, ParamVisitorRef
from noeira.nn.optimizer.param_arena import PARAM_ALIGN, align_param_off
from noeira.nn.optimizer.adam import (
    _grouped_adam_kernel,
    _adam_advance_pow_kernel,
)
from noeira.nn.optimizer.grad_clip import (
    _arena_sumsq_kernel,
    _arena_scale_kernel,
)
from noeira.core.concurrent.block import ControlBlock

from .process_group import (
    ProcessGroup,
    BACKEND_NAIVE,
    shard_rows,
    scale_copy,
)
from .zero import _sum_partials_kernel, _scale_from_sumsq_kernel, _SUM_TPB
from .grad_marks import (
    unit_hooks,
    HOOK_PRE_FORWARD,
    HOOK_PRE_VJP,
    HOOK_POST_VJP,
)
from .fibers import (
    RankFibers,
    RankStep,
    fiber_arrive,
    fiber_depart,
    fibers_active,
    fiber_current,
)


# ── walks ────────────────────────────────────────────────────────────────────


struct _Collect(ParamVisitor):
    """Assigning walk: every parameter's unit (-1 = root), size and decay."""

    var unit: List[Int]
    var n: List[Int]
    var decay: List[Bool]

    def __init__(out self):
        self.unit = List[Int]()
        self.n = List[Int]()
        self.decay = List[Bool]()

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
        self.unit.append(unit_hooks()[].cur_unit)
        self.n.append(N)
        self.decay.append(apply_decay)


struct _Bind(ParamVisitor):
    """Binding walk (one rank): copy each parameter's value into the staging
    arena, then point its value and gradient at the driver's storage."""

    var i: Int
    var c: DeviceContext
    var stage: List[DeviceBuffer[DT]]
    var val: List[DeviceBuffer[DT]]
    var grd: List[DeviceBuffer[DT]]

    def __init__(out self, c: DeviceContext):
        self.i = 0
        self.c = c
        self.stage = List[DeviceBuffer[DT]]()
        self.val = List[DeviceBuffer[DT]]()
        self.grd = List[DeviceBuffer[DT]]()

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
        if not param.dev:
            raise Error("ZeroSharded: parameter " + name + " has no device buffer")
        self.c.enqueue_copy(self.stage[self.i], param.dev.value())
        param.dev = Optional(self.val[self.i])
        param.n = N
        grad.dev = Optional(self.grd[self.i])
        grad.n = N
        self.i += 1


# ── per-rank storage ─────────────────────────────────────────────────────────


struct _RankState(Movable):
    var full: Tensor
    """ZeRO-2: the replicated weights, unit-major. ZeRO-3: empty after
    `sync_params` (the staging arena of the initial weights)."""
    var pslots: Tensor
    """ZeRO-3: `slots` unit-sized weight slots."""
    var proot: Tensor
    """ZeRO-3: the root's weights."""
    var gslots: Tensor
    var groot: Tensor
    var val: Tensor
    """Master weights: this rank's rows of every unit."""
    var g: Tensor
    var m: Tensor
    var v: Tensor
    var decay: Tensor
    var red: Tensor
    """Naive backend only: one unit of allreduce scratch (no reduce-scatter
    without P2P)."""
    var pow: Tensor
    var partials: Tensor
    var sq: Tensor
    var sq_tot: Tensor
    var scale: Tensor
    var norm: Tensor
    var off: List[Int]
    """Per unit: first element of this rank's rows in the shard arena."""
    var shard_n: Int

    def __init__(out self):
        self.full = Tensor()
        self.pslots = Tensor()
        self.proot = Tensor()
        self.gslots = Tensor()
        self.groot = Tensor()
        self.val = Tensor()
        self.g = Tensor()
        self.m = Tensor()
        self.v = Tensor()
        self.decay = Tensor()
        self.red = Tensor()
        self.pow = Tensor()
        self.partials = Tensor()
        self.sq = Tensor()
        self.sq_tot = Tensor()
        self.scale = Tensor()
        self.norm = Tensor()
        self.off = List[Int]()
        self.shard_n = 0


def _zero_hook[
    M: Module, N: Int, STAGE: Int
](state: Int, event: Int, unit: Int, rank: Int, ctx: DeviceContext) raises:
    """A unit boundary on `rank`'s fiber. No borrow of the driver is held
    across a rendezvous (fibers.mojo's contract): each `z[].` call starts and
    ends between two of them, and none of them reaches one."""
    var z = Pointer[ZeroSharded[M, N, STAGE], MutUntrackedOrigin](
        unsafe_from_address=state
    )
    z[]._check_fiber(rank)
    if event == HOOK_POST_VJP:
        # Every rank's vjp of the unit is enqueued once the last one arrives.
        if fiber_arrive(rank):
            z[]._reduce_scatter_unit(unit)
            fiber_depart(rank)
        return
    comptime if STAGE == 3:
        # Every rank reads the slot table between the same two rendezvous, so
        # they all take the same branch.
        if not z[]._holds(unit):
            if fiber_arrive(rank):
                z[]._gather_unit(unit)
                z[]._set_holds(unit)
                fiber_depart(rank)
    if event == HOOK_PRE_VJP:
        z[]._zero_gslot(unit, rank)


struct ZeroSharded[M: Module, N: Int, STAGE: Int](Movable):
    var pg: ProcessGroup[Self.N]
    var nets: List[Self.M]
    var rs: List[_RankState]
    var slots: Int
    var bound: Bool
    var n_wrapped: Int
    """Wrapped units; the root is unit `n_wrapped`."""
    var u_n: List[Int]
    var u_rows: List[Int]
    var u_goff: List[Int]
    """Per unit: first element in the unit-major full layout."""
    var p_unit: List[Int]
    var p_off: List[Int]
    """Per parameter (walk order): offset inside its unit."""
    var p_n: List[Int]
    var p_decay: List[Bool]
    var full_total: Int
    var slot_elems: Int
    var cells: ControlBlock
    """Per step, written from the fibers: which unit each weight slot holds
    (-1: none yet this step)."""
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]

    def __init__(
        out self,
        var pg: ProcessGroup[Self.N],
        var nets: List[Self.M],
        slots: Int,
        lr: Scalar[DT],
        beta1: Scalar[DT],
        beta2: Scalar[DT],
        eps: Scalar[DT],
        wd: Scalar[DT],
    ) raises:
        self.pg = pg^
        self.nets = nets^
        self.rs = List[_RankState]()
        self.slots = slots
        self.bound = False
        self.n_wrapped = 0
        self.u_n = List[Int]()
        self.u_rows = List[Int]()
        self.u_goff = List[Int]()
        self.p_unit = List[Int]()
        self.p_off = List[Int]()
        self.p_n = List[Int]()
        self.p_decay = List[Bool]()
        self.full_total = 0
        self.slot_elems = 0
        self.cells = ControlBlock(max(slots, 1))
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd

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
        slots: Int = 1,
    ) raises -> Self:
        """One replica per rank, with its own standalone parameters. Do any
        per-replica surgery (scaled init, weight tying), then `sync_params`,
        which packs, shards and binds them."""
        comptime assert (
            Self.STAGE == 2 or Self.STAGE == 3
        ), "ZeroSharded: STAGE is 2 or 3 (ZeRO-1 is `Zero1`)"
        if slots < 1:
            raise Error("ZeroSharded: slots must be >= 1")
        var nets = List[Self.M](capacity=Self.N)
        for r in range(Self.N):
            nets.append(Self.M.make["gpu", INIT](Optional(pg.ctx(r))))
        return Self(pg^, nets^, slots, lr, beta1, beta2, eps, wd)

    def ctx(self, r: Int) -> DeviceContext:
        return self.pg.ctx(r)

    # ── setup ────────────────────────────────────────────────────────────────

    def _collect(mut self, r: Int) raises -> Tuple[_Collect, Int]:
        var h = unit_hooks()
        h[].begin_assign(r)
        var col = _Collect()
        try:
            self.nets[r].for_each_param["gpu"](col, Optional(self.pg.ctx(r)))
        except e:
            h[].end_assign()
            raise e^
        var n_units = h[].next_unit
        h[].end_assign()
        return (col^, n_units)

    def sync_params(mut self) raises:
        """Pack every replica's parameters by unit, copy rank 0's values to
        every rank, keep each rank's shard, and bind the parameters to their
        slots (ZeRO-3) or to the replicated arena (ZeRO-2). Once, after any
        per-replica surgery."""
        if self.bound:
            raise Error("ZeroSharded.sync_params: already bound")
        # 1. units, from an assigning walk of every replica
        var c0 = self._collect(0)
        ref col = c0[0]
        var n_units = c0[1]
        for r in range(1, Self.N):
            var cr = self._collect(r)
            if cr[1] != n_units or len(cr[0].n) != len(col.n):
                raise Error("ZeroSharded: replicas have different units")
            for i in range(len(col.n)):
                if cr[0].n[i] != col.n[i] or cr[0].unit[i] != col.unit[i]:
                    raise Error("ZeroSharded: replicas have different units")
        self.n_wrapped = n_units
        var root = n_units
        var nu = n_units + 1
        # 2. layout: each unit packed like an arena, units laid out by rows
        var cursor = List[Int](length=nu, fill=0)
        for i in range(len(col.n)):
            var u = col.unit[i] if col.unit[i] >= 0 else root
            var off = align_param_off(cursor[u])
            self.p_unit.append(u)
            self.p_off.append(off)
            self.p_n.append(col.n[i])
            self.p_decay.append(col.decay[i])
            cursor[u] = off + col.n[i]
        var goff = 0
        for u in range(nu):
            var rows = (cursor[u] + PARAM_ALIGN - 1) // PARAM_ALIGN
            self.u_n.append(cursor[u])
            self.u_rows.append(rows)
            self.u_goff.append(goff)
            goff += rows * PARAM_ALIGN
            if u < root:
                self.slot_elems = max(self.slot_elems, rows * PARAM_ALIGN)
        self.full_total = goff
        var root_elems = max(self.u_rows[root] * PARAM_ALIGN, 1)
        var max_unit = 1
        for u in range(nu):
            max_unit = max(max_unit, self.u_rows[u] * PARAM_ALIGN)
        # host decay mask, unit-major
        var dmask = List[Scalar[DT]](length=self.full_total, fill=Scalar[DT](0))
        for i in range(len(self.p_n)):
            if self.p_decay[i]:
                var base = self.u_goff[self.p_unit[i]] + self.p_off[i]
                for k in range(self.p_n[i]):
                    dmask[base + k] = Scalar[DT](1.0)
        # 3. per rank: storage, then bind
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            var st = _RankState()
            st.full = Tensor.alloc_gpu(c, max(self.full_total, 1))
            comptime if Self.STAGE == 3:
                st.pslots = Tensor.alloc_gpu(
                    c, max(self.slots * self.slot_elems, 1)
                )
                st.proot = Tensor.alloc_gpu(c, root_elems)
            st.gslots = Tensor.alloc_gpu(c, max(self.slots * self.slot_elems, 1))
            st.groot = Tensor.alloc_gpu(c, root_elems)
            var sn = 0
            for u in range(nu):
                st.off.append(sn)
                sn += shard_rows(self.u_rows[u], Self.N, r)[1] * PARAM_ALIGN
            st.shard_n = sn
            var cap = max(sn, 1)
            st.val = Tensor.alloc_gpu(c, cap)
            st.g = Tensor.alloc_gpu(c, cap)
            st.m = Tensor.alloc_gpu(c, cap)
            st.v = Tensor.alloc_gpu(c, cap)
            # decay: this rank's rows of the host mask
            var dm = Tensor.alloc(cap)
            for u in range(nu):
                var sh = shard_rows(self.u_rows[u], Self.N, r)
                var src = self.u_goff[u] + sh[0] * PARAM_ALIGN
                for k in range(sh[1] * PARAM_ALIGN):
                    dm.data[st.off[u] + k] = dmask[src + k]
            dm.upload(c)
            st.decay = dm^
            if self.pg.backend == BACKEND_NAIVE:
                st.red = Tensor.alloc_gpu(c, max_unit)
            st.pow = Tensor.alloc_gpu(c, 2)
            var one = List[Scalar[DT]](length=2, fill=Scalar[DT](1.0))
            c.enqueue_copy(st.pow.dev.value(), one.unsafe_ptr())
            c.synchronize()
            st.partials = Tensor.alloc_gpu(c, max((sn + TPB - 1) // TPB, 1))
            st.sq = Tensor.alloc_gpu(c, 1)
            st.sq_tot = Tensor.alloc_gpu(c, 1)
            st.scale = Tensor.alloc_gpu(c, 1)
            st.norm = Tensor.alloc_gpu(c, 1)
            self.rs.append(st^)
            var b = _Bind(c)
            for i in range(len(self.p_n)):
                var u = self.p_unit[i]
                var n = self.p_n[i]
                var fo = self.u_goff[u] + self.p_off[i]
                ref sr = self.rs[r]
                b.stage.append(sr.full.dev.value().create_sub_buffer[DT](fo, n))
                comptime if Self.STAGE == 2:
                    b.val.append(sr.full.dev.value().create_sub_buffer[DT](fo, n))
                else:
                    b.val.append(self._wslot(r, u).create_sub_buffer[DT](self.p_off[i], n))
                b.grd.append(self._gslot(r, u).create_sub_buffer[DT](self.p_off[i], n))
            self.nets[r].for_each_param["gpu"](b, Optional(c))
            if b.i != len(self.p_n):
                raise Error("ZeroSharded: the binding walk saw another model")
        # 4. rank 0's initial weights everywhere, 5. each rank keeps its rows
        var fulls = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            fulls.append(self.rs[r].full.dev.value())
        self.pg.broadcast(0, fulls, self.full_total)
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref st = self.rs[r]
            for u in range(nu):
                var sh = shard_rows(self.u_rows[u], Self.N, r)
                var n = sh[1] * PARAM_ALIGN
                if n == 0:
                    continue
                c.enqueue_copy(
                    st.val.dev.value().create_sub_buffer[DT](st.off[u], n),
                    st.full.dev.value().create_sub_buffer[DT](
                        self.u_goff[u] + sh[0] * PARAM_ALIGN, n
                    ),
                )
        self.pg.synchronize()
        comptime if Self.STAGE == 3:
            for r in range(Self.N):
                self.rs[r].full = Tensor()
        self.bound = True

    def _wslot(self, r: Int, u: Int) raises -> DeviceBuffer[DT]:
        """Unit u's full weights on rank r."""
        var n = self.u_rows[u] * PARAM_ALIGN
        comptime if Self.STAGE == 2:
            return self.rs[r].full.dev.value().create_sub_buffer[DT](
                self.u_goff[u], max(n, 1)
            )
        else:
            if u == self.n_wrapped:
                return self.rs[r].proot.dev.value()
            return self.rs[r].pslots.dev.value().create_sub_buffer[DT](
                (u % self.slots) * self.slot_elems, max(n, 1)
            )

    def _gslot(self, r: Int, u: Int) raises -> DeviceBuffer[DT]:
        """Unit u's full gradient on rank r."""
        if u == self.n_wrapped:
            return self.rs[r].groot.dev.value()
        var n = self.u_rows[u] * PARAM_ALIGN
        return self.rs[r].gslots.dev.value().create_sub_buffer[DT](
            (u % self.slots) * self.slot_elems, max(n, 1)
        )

    def _shard(self, r: Int, u: Int, buf: DeviceBuffer[DT]) raises -> DeviceBuffer[DT]:
        """Rank r's rows of unit u inside one of its shard buffers (the whole
        buffer when it owns none: nothing reads it then)."""
        var n = shard_rows(self.u_rows[u], Self.N, r)[1] * PARAM_ALIGN
        if n == 0:
            return buf
        return buf.create_sub_buffer[DT](self.rs[r].off[u], n)

    # ── collectives on a unit, for every rank ────────────────────────────────

    def _gather_unit(mut self, u: Int) raises:
        var srcs = List[DeviceBuffer[DT]](capacity=Self.N)
        var dsts = List[DeviceBuffer[DT]](capacity=Self.N)
        for k in range(Self.N):
            srcs.append(self._shard(k, u, self.rs[k].val.dev.value()))
            dsts.append(self._wslot(k, u))
        self.pg.all_gather_into(srcs, dsts, self.u_rows[u], PARAM_ALIGN)

    def _reduce_scatter_unit(mut self, u: Int) raises:
        """`g_r[unit u's rows of r] = (1/N) sum_k gslot_k(u)`, every rank."""
        var rows = self.u_rows[u]
        if rows == 0:
            return
        var n_all = rows * PARAM_ALIGN
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var ins = List[DeviceBuffer[DT]](capacity=Self.N)
        for k in range(Self.N):
            ins.append(self._gslot(k, u))
        if self.pg.backend == BACKEND_NAIVE:
            # No reduce-scatter without P2P: allreduce one unit, keep the rows.
            var reds = List[DeviceBuffer[DT]](capacity=Self.N)
            for k in range(Self.N):
                reds.append(
                    self.rs[k].red.dev.value().create_sub_buffer[DT](0, n_all)
                )
            self.pg.allreduce_sum(ins, reds, n_all)
            for r in range(Self.N):
                var sh = shard_rows(rows, Self.N, r)
                var n = sh[1] * PARAM_ALIGN
                if n > 0:
                    scale_copy(
                        self._shard(r, u, self.rs[r].g.dev.value()),
                        reds[r].create_sub_buffer[DT](sh[0] * PARAM_ALIGN, n),
                        inv_n, n, self.pg.ctx(r),
                    )
            return
        var outs = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            outs.append(self._shard(r, u, self.rs[r].g.dev.value()))
        self.pg.reduce_scatter_sum(ins, outs, rows, PARAM_ALIGN)
        for r in range(Self.N):
            var n = shard_rows(rows, Self.N, r)[1] * PARAM_ALIGN
            if n > 0:
                scale_copy(outs[r], outs[r], inv_n, n, self.pg.ctx(r))

    # ── hook helpers, on the fiber of `rank` (`_zero_hook`) ─────────────────

    def _check_fiber(self, rank: Int) raises:
        comptime if Self.N > 1:
            if not fibers_active():
                raise Error(
                    "ZeroSharded: run the step through forward_backward(bodies)"
                )
            if fiber_current() != rank:
                raise Error(
                    "ZeroSharded: rank " + String(rank)
                    + "'s unit ran on rank " + String(fiber_current())
                    + "'s fiber"
                )

    def _holds(self, u: Int) -> Bool:
        """ZeRO-3: whether u's weight slot already holds u this step."""
        return Int(self.cells.view().acquire_load(u % self.slots)) == u

    def _set_holds(self, u: Int):
        self.cells.view().release_store(u % self.slots, Int64(u))

    def _zero_gslot(self, u: Int, rank: Int) raises:
        var n = self.u_rows[u] * PARAM_ALIGN
        if n > 0:
            fill_dev(self._gslot(rank, u), n, Scalar[DT](0), self.pg.ctx(rank))

    # ── the step ─────────────────────────────────────────────────────────────

    def forward_backward[B: RankStep](mut self, mut bodies: List[B]) raises:
        """`bodies[r].run_rank(r)` (forward, loss, vjp of rank r) on rank r's
        fiber, with the unit collectives in between; on return each rank
        holds its shard of the mean gradient. One body per rank; a body
        reaches its replica through the driver's address
        (`Pointer(unsafe_from_address=...)[].nets[r]`), not a `mut` borrow
        held across the step (fibers.mojo)."""
        if not self.bound:
            raise Error("ZeroSharded: call sync_params() first")
        if len(bodies) != Self.N:
            raise Error(
                "ZeroSharded.forward_backward: " + String(len(bodies))
                + " bodies for " + String(Self.N) + " ranks"
            )
        var cv = self.cells.view()
        for s in range(self.slots):
            cv.release_store(s, -1)
        var root = self.n_wrapped
        var root_n = self.u_rows[root] * PARAM_ALIGN
        for r in range(Self.N):
            if root_n > 0:
                fill_dev(self.rs[r].groot.dev.value(), root_n, Scalar[DT](0), self.pg.ctx(r))
        # Peers' Adam (last step) wrote the shards every gather reads.
        self.pg.stream_barrier()
        var h = unit_hooks()
        h[].register(Int(Pointer(to=self)), _zero_hook[Self.M, Self.N, Self.STAGE])
        try:
            comptime if Self.STAGE == 3:
                self._gather_unit(root)
            RankFibers.run(bodies)
            self._reduce_scatter_unit(root)
        except e:
            h[].unregister()
            raise e^
        h[].unregister()

    def clip_grads_device(mut self, max_norm: Scalar[DT]) raises:
        """Global grad-norm clip over the gradient shards (`Zero1`'s rule):
        per-shard sums of squares, one scalar allreduce, the same scale on
        every rank. Device-only."""
        var sq_in = List[DeviceBuffer[DT]](capacity=Self.N)
        var sq_out = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref st = self.rs[r]
            var nblk = max((st.shard_n + TPB - 1) // TPB, 1)
            c.enqueue_function[_arena_sumsq_kernel](
                st.g.dev.value(), Int64(st.shard_n), st.partials.dev.value(),
                grid_dim=nblk, block_dim=TPB,
            )
            c.enqueue_function[_sum_partials_kernel](
                st.partials.dev.value(), Int64(nblk), st.sq.dev.value(),
                grid_dim=1, block_dim=_SUM_TPB,
            )
            sq_in.append(st.sq.dev.value())
            sq_out.append(st.sq_tot.dev.value())
        self.pg.allreduce_sum(sq_in, sq_out, 1)
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref st = self.rs[r]
            c.enqueue_function[_scale_from_sumsq_kernel](
                st.sq_tot.dev.value(), st.scale.dev.value(),
                st.norm.dev.value(), max_norm, Scalar[DT](1e-6),
                grid_dim=1, block_dim=1,
            )
            if st.shard_n > 0:
                c.enqueue_function[_arena_scale_kernel](
                    st.g.dev.value(), Int64(st.shard_n), st.scale.dev.value(),
                    grid_dim=(st.shard_n + TPB - 1) // TPB, block_dim=TPB,
                )

    def read_clip_norm(mut self, r: Int) raises -> Scalar[DT]:
        self.rs[r].norm.download(self.pg.ctx(r))
        return self.rs[r].norm.data[0]

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def step(mut self) raises:
        """Adam on every rank's shard arena; ZeRO-2 then refreshes every
        replica, one all-gather per unit."""
        # Every gather that read this rank's shard is done before Adam
        # overwrites it.
        self.pg.stream_barrier()
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref st = self.rs[r]
            c.enqueue_function[_adam_advance_pow_kernel](
                st.pow.dev.value(), self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            if st.shard_n == 0:
                continue
            c.enqueue_function[_grouped_adam_kernel](
                st.val.dev.value(),
                st.g.dev.value(),
                st.m.dev.value(),
                st.v.dev.value(),
                st.decay.dev.value(),
                Int64(st.shard_n),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                st.pow.dev.value(),
                self.wd,
                grid_dim=(st.shard_n + TPB - 1) // TPB,
                block_dim=TPB,
            )
        comptime if Self.STAGE == 2:
            for u in range(self.n_wrapped + 1):
                self._gather_unit(u)
        for r in range(Self.N):
            var bump = ParamVersionBump()
            var bref = ParamVisitorRef.of[ParamVersionBump, "gpu"](bump)
            self.nets[r].for_each_param["gpu"](bref, Optional(self.pg.ctx(r)))

    def synchronize(self) raises:
        self.pg.synchronize()

    # ── inspection ───────────────────────────────────────────────────────────

    def _unit_major(mut self, r: Int) raises -> List[Scalar[DT]]:
        """Every unit's weights, unit-major: rank r's replica (ZeRO-2) or
        every rank's master shards (ZeRO-3, `r` ignored)."""
        var out = List[Scalar[DT]](length=self.full_total, fill=Scalar[DT](0))
        comptime if Self.STAGE == 2:
            var c = self.pg.ctx(r)
            c.enqueue_copy(
                out.unsafe_ptr(),
                self.rs[r].full.dev.value().create_sub_buffer[DT](
                    0, self.full_total
                ),
            )
            c.synchronize()
        else:
            for k in range(Self.N):
                var c = self.pg.ctx(k)
                ref st = self.rs[k]
                if st.shard_n == 0:
                    continue
                var h = List[Scalar[DT]](length=st.shard_n, fill=Scalar[DT](0))
                c.enqueue_copy(
                    h.unsafe_ptr(),
                    st.val.dev.value().create_sub_buffer[DT](0, st.shard_n),
                )
                c.synchronize()
                for u in range(self.n_wrapped + 1):
                    var sh = shard_rows(self.u_rows[u], Self.N, k)
                    var dst = self.u_goff[u] + sh[0] * PARAM_ALIGN
                    for i in range(sh[1] * PARAM_ALIGN):
                        out[dst + i] = h[st.off[u] + i]
        return out^

    def arena_total(self) -> Int:
        """`DataParallel`'s arena length for the same model (walk order,
        every parameter on a `PARAM_ALIGN` boundary)."""
        var total = 0
        for i in range(len(self.p_n)):
            total = align_param_off(total) + self.p_n[i]
        return total

    def download_params(mut self, r: Int = 0) raises -> List[Scalar[DT]]:
        """The weights in `DataParallel.download_params`'s layout (walk
        order, every parameter on a `PARAM_ALIGN` boundary), for the gates:
        rank r's replica (ZeRO-2) or the gathered shards (ZeRO-3)."""
        var um = self._unit_major(r)
        var out = List[Scalar[DT]](length=self.arena_total(), fill=Scalar[DT](0))
        var off = 0
        for i in range(len(self.p_n)):
            off = align_param_off(off)
            var src = self.u_goff[self.p_unit[i]] + self.p_off[i]
            for k in range(self.p_n[i]):
                out[off + k] = um[src + k]
            off += self.p_n[i]
        return out^

    def state_bytes_per_rank(self, r: Int) -> Int:
        """Bytes of weights, gradients, optimizer state, decay mask and
        collective scratch held on rank r between steps (no activations, no
        Signal payload)."""
        ref st = self.rs[r]
        var e = st.full.n + st.pslots.n + st.proot.n + st.gslots.n + st.groot.n
        e += st.val.n + st.g.n + st.m.n + st.v.n + st.decay.n + st.red.n
        return e * 4

    def layout_summary(self) -> String:
        var root = self.n_wrapped
        var s = String(
            "ZeRO-", Self.STAGE, ": ", self.n_wrapped, " units + root, slots ",
            self.slots, " x ", self.slot_elems, " elems, root ",
            self.u_rows[root] * PARAM_ALIGN, ", full ", self.full_total,
            "; shard elems per rank:",
        )
        for r in range(Self.N):
            s += String(" ", self.rs[r].shard_n)
        return s
