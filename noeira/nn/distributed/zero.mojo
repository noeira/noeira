"""Zero1 — ZeRO stage 1 over the parameter arena: optimizer state sharded.

Every rank keeps a full replica of the weights (`val`) and of the gradients
(`grd`), as in DDP, but owns the Adam moments of ONE shard of the arena only:

    reduce-scatter grd -> gshard_r            (rank r's rows of the mean gradient)
    sq_r = sum(gshard_r^2); allreduce(sq)     (global clip norm: one scalar)
    gshard_r *= clip scale
    Adam on val_r[shard r] with m_r, v_r      (shard-sized moments)
    all-gather val                            (every rank gets every shard)

Because Adam is elementwise over the arena and `decay_mask` is per element, a
shard can cut through the middle of a parameter: no module is aware of it. The
update runs the SAME kernels as the unsharded `Adam` (`_grouped_adam_kernel`,
`_adam_advance_pow_kernel`) on offset sub-buffers, so with the same reduction
order the result is bit-identical to `DataParallel` — that is the M3 gate.

Shards are rows of the arena viewed as `[rows, PARAM_ALIGN]` (MAX's 2D
reduce-scatter partition, `shard_rows`), so every shard starts on the 128-byte
boundary the GEMMs need. The arena is allocated padded to whole rows
(`ParamArena.adopt(pad_to=PARAM_ALIGN)`); the tail past `total` is zero and the
optimizer never updates it.

Memory per GPU (fp32, Ψ = arena elements):
    DDP:    val Ψ + grd Ψ + red Ψ + m Ψ + v Ψ                 = 5Ψ (+ signal payload)
    ZeRO-1: val Ψ + grd Ψ + gshard Ψ/N + m Ψ/N + v Ψ/N        = 2Ψ + 3Ψ/N

Out of scope here: checkpoints (per-parameter moment views no longer exist;
`_MomentPlacer` is not run), the device LR schedule, and AMP's bf16 weight
caches beyond the version bump.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVersionBump, ParamVisitorRef
from noeira.nn.optimizer.param_arena import ParamArena, PARAM_ALIGN
from noeira.nn.optimizer.adam import (
    _grouped_adam_kernel,
    _adam_advance_pow_kernel,
)
from noeira.nn.optimizer.grad_clip import (
    _arena_sumsq_kernel,
    _arena_scale_kernel,
)

from .process_group import (
    ProcessGroup,
    BACKEND_NAIVE,
    shard_rows,
    scale_copy,
)

from std.math import sqrt
from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block


comptime _SUM_TPB = 256


def _sum_partials_kernel(
    partials: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """One block: `out[0] = sum(partials[0:n])`."""
    var n = Int(n_arg)
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n:
        my_sum += partials[unsafe_offset=k]
        k += _SUM_TPB
    var s = block.sum[block_size=_SUM_TPB, broadcast=False](val=my_sum)
    if t == 0:
        dst[unsafe_offset=0] = s[0]


def _scale_from_sumsq_kernel(
    sumsq: Pointer[Scalar[DT], MutAnyOrigin],
    scale_buf: Pointer[Scalar[DT], MutAnyOrigin],
    norm_buf: Pointer[Scalar[DT], MutAnyOrigin],
    max_norm: Scalar[DT],
    eps: Scalar[DT],
):
    """`_arena_finalize_kernel`'s rule from a global sum of squares:
    norm = sqrt(sumsq), scale = min(1, max_norm / max(norm, eps)), 0 if the
    norm is not finite, 1 if `max_norm <= 0`."""
    if Int(global_idx.x) != 0:
        return
    var norm = sqrt(sumsq[unsafe_offset=0])
    norm_buf[unsafe_offset=0] = norm
    if norm - norm != Scalar[DT](0.0):
        scale_buf[unsafe_offset=0] = Scalar[DT](0.0)
    elif max_norm <= Scalar[DT](0.0):
        scale_buf[unsafe_offset=0] = Scalar[DT](1.0)
    else:
        var denom = norm if norm > eps else eps
        var ratio = max_norm / denom
        scale_buf[unsafe_offset=0] = (
            ratio if ratio < Scalar[DT](1.0) else Scalar[DT](1.0)
        )


struct _Shard(Movable):
    """One rank's slice of the optimizer: its rows, gradient and moments."""

    var row0: Int
    var nrows: Int
    var off: Int
    """First arena element of the shard (`row0 * PARAM_ALIGN`)."""
    var n: Int
    """Elements in the shard (`nrows * PARAM_ALIGN`)."""
    var n_upd: Int
    """Elements the optimizer updates: the shard clipped to `[0, total)`."""
    var g: Tensor
    var m: Tensor
    var v: Tensor
    var pow: Tensor
    var partials: Tensor
    var sq: Tensor
    var sq_tot: Tensor
    var scale: Tensor
    var norm: Tensor

    def __init__(
        out self, c: DeviceContext, row0: Int, nrows: Int, total: Int
    ) raises:
        self.row0 = row0
        self.nrows = nrows
        self.off = row0 * PARAM_ALIGN
        self.n = nrows * PARAM_ALIGN
        self.n_upd = max(0, min(self.off + self.n, total) - self.off)
        var cap = max(self.n, 1)
        self.g = Tensor.alloc_gpu(c, cap)
        self.m = Tensor.alloc_gpu(c, cap)
        self.v = Tensor.alloc_gpu(c, cap)
        self.pow = Tensor.alloc_gpu(c, 2)
        var one = List[Scalar[DT]](length=2, fill=Scalar[DT](1.0))
        c.enqueue_copy(self.pow.dev.value(), one.unsafe_ptr())
        c.synchronize()
        self.partials = Tensor.alloc_gpu(c, max((self.n + TPB - 1) // TPB, 1))
        self.sq = Tensor.alloc_gpu(c, 1)
        self.sq_tot = Tensor.alloc_gpu(c, 1)
        self.scale = Tensor.alloc_gpu(c, 1)
        self.norm = Tensor.alloc_gpu(c, 1)


struct Zero1[M: Module, N: Int](Movable):
    var pg: ProcessGroup[Self.N]
    var nets: List[Self.M]
    var arenas: List[ParamArena]
    var shards: List[_Shard]
    var total: Int
    var rows: Int
    """Arena rows of `PARAM_ALIGN` elements (`capacity / PARAM_ALIGN`)."""
    var red: List[Tensor]
    """Full-size allreduce scratch — ONLY on the no-P2P backend, where MAX has
    no reduce-scatter (empty otherwise)."""
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]

    def __init__(
        out self,
        var pg: ProcessGroup[Self.N],
        var nets: List[Self.M],
        var arenas: List[ParamArena],
        var shards: List[_Shard],
        total: Int,
        rows: Int,
        var red: List[Tensor],
        lr: Scalar[DT],
        beta1: Scalar[DT],
        beta2: Scalar[DT],
        eps: Scalar[DT],
        wd: Scalar[DT],
    ):
        self.pg = pg^
        self.nets = nets^
        self.arenas = arenas^
        self.shards = shards^
        self.total = total
        self.rows = rows
        self.red = red^
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
    ) raises -> Self:
        """One replica per rank, arena padded to whole rows, one shard of
        optimizer state per rank. Replicas are NOT synchronized: do any
        per-replica surgery, then `sync_params`."""
        var nets = List[Self.M](capacity=Self.N)
        var arenas = List[ParamArena](capacity=Self.N)
        var total = -1
        for r in range(Self.N):
            var c = pg.ctx(r)
            nets.append(Self.M.make["gpu", INIT](Optional(c)))
            arenas.append(ParamArena())
            arenas[r].adopt["gpu"](nets[r], Optional(c), pad_to=PARAM_ALIGN)
            if total >= 0 and arenas[r].total != total:
                raise Error("Zero1.make: replicas have different arenas")
            total = arenas[r].total
        var rows = arenas[0].capacity // PARAM_ALIGN
        var shards = List[_Shard](capacity=Self.N)
        for r in range(Self.N):
            var sh = shard_rows(rows, Self.N, r)
            shards.append(_Shard(pg.ctx(r), sh[0], sh[1], total))
        var red = List[Tensor]()
        if pg.backend == BACKEND_NAIVE:
            for r in range(Self.N):
                red.append(Tensor.alloc_gpu(pg.ctx(r), rows * PARAM_ALIGN))
        return Self(
            pg^, nets^, arenas^, shards^, total, rows, red^,
            lr, beta1, beta2, eps, wd,
        )

    def ctx(self, r: Int) -> DeviceContext:
        return self.pg.ctx(r)

    def sync_params(mut self) raises:
        """Copy rank 0's whole parameter arena to every rank."""
        var vals = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            vals.append(self.arenas[r].val.dev.value())
        self.pg.broadcast(0, vals, self.rows * PARAM_ALIGN)

    def zero_grad(mut self) raises:
        for r in range(Self.N):
            self.arenas[r].zero_grad(self.pg.ctx(r))

    def reduce_scatter_grads(mut self) raises:
        """`gshard_r = (1/N) * sum_k grd_k[shard r]` on every rank.

        Without P2P (MAX has no reduce-scatter there) this is a full allreduce
        into `red` and a local slice: same result, DDP's traffic, and one
        extra arena of memory per GPU — the plan's documented fallback."""
        var n_all = self.rows * PARAM_ALIGN
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Self.N)
        var ins = List[DeviceBuffer[DT]](capacity=Self.N)
        var outs = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            ins.append(self.arenas[r].grd.dev.value())
        if self.pg.backend == BACKEND_NAIVE:
            for r in range(Self.N):
                outs.append(self.red[r].dev.value())
            self.pg.allreduce_sum(ins, outs, n_all)
            for r in range(Self.N):
                ref sh = self.shards[r]
                if sh.n > 0:
                    scale_copy(
                        sh.g.dev.value(),
                        outs[r].create_sub_buffer[DT](sh.off, sh.n),
                        inv_n, sh.n, self.pg.ctx(r),
                    )
            return
        for r in range(Self.N):
            outs.append(self.shards[r].g.dev.value())
        self.pg.reduce_scatter_sum(ins, outs, self.rows, PARAM_ALIGN)
        for r in range(Self.N):
            ref sh = self.shards[r]
            if sh.n > 0:
                var g = sh.g.dev.value()
                scale_copy(g, g, inv_n, sh.n, self.pg.ctx(r))

    def clip_grads_device(mut self, max_norm: Scalar[DT]) raises:
        """GLOBAL grad-norm clip over the sharded gradient: each rank sums the
        squares of its shard, one scalar allreduce gives the total, every rank
        derives the same scale and applies it to its shard. Device-only."""
        var sq_in = List[DeviceBuffer[DT]](capacity=Self.N)
        var sq_out = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref sh = self.shards[r]
            var nblk = max((sh.n + TPB - 1) // TPB, 1)
            c.enqueue_function[_arena_sumsq_kernel](
                sh.g.dev.value(), Int64(sh.n), sh.partials.dev.value(),
                grid_dim=nblk, block_dim=TPB,
            )
            c.enqueue_function[_sum_partials_kernel](
                sh.partials.dev.value(), Int64(nblk), sh.sq.dev.value(),
                grid_dim=1, block_dim=_SUM_TPB,
            )
            sq_in.append(sh.sq.dev.value())
            sq_out.append(sh.sq_tot.dev.value())
        self.pg.allreduce_sum(sq_in, sq_out, 1)
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref sh = self.shards[r]
            c.enqueue_function[_scale_from_sumsq_kernel](
                sh.sq_tot.dev.value(), sh.scale.dev.value(),
                sh.norm.dev.value(), max_norm, Scalar[DT](1e-6),
                grid_dim=1, block_dim=1,
            )
            if sh.n > 0:
                c.enqueue_function[_arena_scale_kernel](
                    sh.g.dev.value(), Int64(sh.n), sh.scale.dev.value(),
                    grid_dim=(sh.n + TPB - 1) // TPB, block_dim=TPB,
                )

    def read_clip_norm(mut self, r: Int) raises -> Scalar[DT]:
        """Rank r's copy of the last global pre-clip norm (host read)."""
        self.shards[r].norm.download(self.pg.ctx(r))
        return self.shards[r].norm.data[0]

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def step(mut self) raises:
        """Adam on each rank's shard, then the all-gather of the weights."""
        for r in range(Self.N):
            var c = self.pg.ctx(r)
            ref sh = self.shards[r]
            c.enqueue_function[_adam_advance_pow_kernel](
                sh.pow.dev.value(), self.beta1, self.beta2,
                grid_dim=1, block_dim=1,
            )
            if sh.n_upd == 0:
                continue
            ref ar = self.arenas[r]
            c.enqueue_function[_grouped_adam_kernel](
                ar.val.dev.value().create_sub_buffer[DT](sh.off, sh.n_upd),
                sh.g.dev.value(),
                sh.m.dev.value(),
                sh.v.dev.value(),
                ar.decay_mask.dev.value().create_sub_buffer[DT](
                    sh.off, sh.n_upd
                ),
                Int64(sh.n_upd),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                sh.pow.dev.value(),
                self.wd,
                grid_dim=(sh.n_upd + TPB - 1) // TPB,
                block_dim=TPB,
            )
        var vals = List[DeviceBuffer[DT]](capacity=Self.N)
        for r in range(Self.N):
            vals.append(self.arenas[r].val.dev.value())
        self.pg.all_gather_rows(vals, self.rows, PARAM_ALIGN)
        # AMP: weights changed under every bf16 weight cache (host walk, as in
        # `Adam.step`).
        for r in range(Self.N):
            var bump = ParamVersionBump()
            var bref = ParamVisitorRef.of[ParamVersionBump, "gpu"](bump)
            self.nets[r].for_each_param["gpu"](
                bref, Optional(self.pg.ctx(r))
            )

    def synchronize(self) raises:
        self.pg.synchronize()

    def download_params(mut self, r: Int) raises -> List[Scalar[DT]]:
        """Rank r's parameter arena on the host (`total` elements)."""
        var c = self.pg.ctx(r)
        var t = Tensor.alloc(self.total)
        c.enqueue_copy(
            t.data.unsafe_ptr(),
            self.arenas[r].val.dev.value().create_sub_buffer[DT](
                0, self.total
            ),
        )
        c.synchronize()
        return t.data.copy()

    def state_bytes_per_rank(self, r: Int) -> Int:
        """Bytes of weights + gradients + optimizer state on rank r (no
        activations, no Signal payload)."""
        ref sh = self.shards[r]
        var full = self.rows * PARAM_ALIGN
        var b = 2 * full + 3 * sh.n  # val, grd; gshard, m, v
        if len(self.red) > 0:
            b += full
        return b * 4
