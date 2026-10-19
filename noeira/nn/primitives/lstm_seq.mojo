"""LSTMSeq[VOCAB, HIDDEN, SEQ] — a fixed-length LSTM encoder as a standard
`Module` (storage surface).

Transformed from legacy `nn.primitives.lstm_seq` (surface-only change): the
unroll + BPTT loop, the BATCH-major↔timestep-major transposes, and the three
GPU kernels are carried over VERBATIM; only the surface (`TargetStorage` +
`Cache` + `TileTensor` view args → owning `Tensor` scratch + `TensorRefs` /
`mut Tensor` I/O) changes.

`LSTMCell` is recurrent and exposes a step API (`step_forward`/`step_backward`);
its `Module.forward`/`vjp` raise. `LSTMSeq` wraps the unroll + BPTT behind the
standard `Module` interface so an LSTM LM is a plain `Module` and trains through
the generic Trainer with no hand-written loop:

    Sequential[LSTMSeq[VOCAB, HIDDEN, SEQ], Tokenwise[SEQ, Linear[HIDDEN, VOCAB]]]

I/O matches the rest of the sequence stack (BATCH-major, SEQ folded into the
feature dim, per-token sub-vectors contiguous):
    forward : input  [BATCH, SEQ·VOCAB]  → output [BATCH, SEQ·HIDDEN]
    vjp     : grad_output [BATCH, SEQ·HIDDEN] → grad_input [BATCH, SEQ·VOCAB]

The cell wants a contiguous `[BATCH, ·]` per timestep, but the I/O is
BATCH-major, so each forward/backward transposes the sequence to timestep-major
`[SEQ, BATCH, ·]` once at the boundary (CPU: host loop; GPU: one kernel), runs
the contiguous-slice step loop over shared timestep-major buffers (the cell's
step methods take per-tensor element offsets), and transposes back. The
per-timestep h/c states + caches live in owned `Tensor` scratch (lazy by BATCH)
so BPTT can read them. `for_each_param` / `zero_grad` recurse into the cell.

CPU + GPU. The GPU path is an *unfused* RNN — SEQ sequential per-step kernel
launches — so it's correct but slow for long SEQ; keep SEQ modest.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from .lstm_cell import LSTMCell


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — sequence transpose (BATCH-major ↔ timestep-major) + add.
# Carried VERBATIM from legacy (args MutAnyOrigin = the GPU kernel ABI).
# ──────────────────────────────────────────────────────────────────────


def _bsd_to_sbd_kernel[
    BATCH: Int, SEQ: Int, D: Int
](
    src: Tile[DT, BATCH * SEQ * D],
    dst: Tile[DT, BATCH * SEQ * D],
):
    """`dst[t,b,d] = src[b,t,d]`. src `[BATCH, SEQ·D]`, dst `[SEQ, BATCH, D]`."""
    var idx = Int(global_idx.x)
    comptime total = BATCH * SEQ * D
    if idx >= total:
        return
    var b = idx // (SEQ * D)
    var rem = idx % (SEQ * D)
    var t = rem // D
    var d = rem % D
    dst[t * BATCH * D + b * D + d] = src[idx]


def _sbd_to_bsd_kernel[
    BATCH: Int, SEQ: Int, D: Int
](
    src: Tile[DT, BATCH * SEQ * D],
    dst: Tile[DT, BATCH * SEQ * D],
):
    """`dst[b,t,d] = src[t,b,d]`. src `[SEQ, BATCH, D]`, dst `[BATCH, SEQ·D]`."""
    var idx = Int(global_idx.x)
    comptime total = BATCH * SEQ * D
    if idx >= total:
        return
    var b = idx // (SEQ * D)
    var rem = idx % (SEQ * D)
    var t = rem // D
    var d = rem % D
    dst[idx] = src[t * BATCH * D + b * D + d]


def _add_kernel[
    N: Int
](
    a: Tile[DT, N],
    b: Tile[DT, N],
    dst: Tile[DT, N],
):
    """`dst[i] = a[i] + b[i]`."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = a[i] + b[i]


struct LSTMSeq[VOCAB: Int, HIDDEN: Int, SEQ: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ * Self.VOCAB)
    comptime OUT_DIM = Self.SEQ * Self.HIDDEN
    comptime Cell = LSTMCell[Self.VOCAB, Self.HIDDEN]
    comptime CACHE_SIZE = Self.Cell.CACHE_SIZE

    var cell: Self.Cell
    # Forward scratch (lazy by BATCH): timestep-major input + per-step h/c +
    # the cell's per-step backward cache.
    var x_seq: Tensor          # [SEQ, BATCH, VOCAB]
    var h_buf: Tensor          # [SEQ+1, BATCH, HIDDEN]
    var c_buf: Tensor          # [SEQ+1, BATCH, HIDDEN]
    var cache_buf: Tensor      # [SEQ, BATCH, CACHE_SIZE]
    # Backward scratch.
    var dh_seq: Tensor         # [SEQ, BATCH, HIDDEN]
    var dx_seq: Tensor         # [SEQ, BATCH, VOCAB]
    var dh_recur: Tensor       # [BATCH, HIDDEN] carry
    var dc_recur: Tensor
    var dh_prev: Tensor
    var dc_prev: Tensor
    var dh_t: Tensor           # [BATCH, HIDDEN] working

    def __init__(out self):
        self.cell = Self.Cell()
        self.x_seq = Tensor()
        self.h_buf = Tensor()
        self.c_buf = Tensor()
        self.cache_buf = Tensor()
        self.dh_seq = Tensor()
        self.dx_seq = Tensor()
        self.dh_recur = Tensor()
        self.dc_recur = Tensor()
        self.dh_prev = Tensor()
        self.dc_prev = Tensor()
        self.dh_t = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "LSTMSeq: target must be 'cpu' or 'gpu'"
        var m = Self()
        m.cell = Self.Cell.make[target, INIT](ctx)
        return m^

    # ----- scratch sizing -------------------------------------------------

    def _ensure[target: StaticString](
        mut self, batch: Int, ctx: Optional[DeviceContext]
    ) raises:
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        comptime if target == "cpu":
            self.x_seq.ensure(Self.SEQ * batch * V)
            self.h_buf.ensure((Self.SEQ + 1) * batch * H)
            self.c_buf.ensure((Self.SEQ + 1) * batch * H)
            self.cache_buf.ensure(Self.SEQ * batch * C)
            self.dh_seq.ensure(Self.SEQ * batch * H)
            self.dx_seq.ensure(Self.SEQ * batch * V)
            self.dh_recur.ensure(batch * H)
            self.dc_recur.ensure(batch * H)
            self.dh_prev.ensure(batch * H)
            self.dc_prev.ensure(batch * H)
            self.dh_t.ensure(batch * H)
        else:
            var c = ctx.value()
            self.x_seq.ensure_gpu(c, Self.SEQ * batch * V)
            self.h_buf.ensure_gpu(c, (Self.SEQ + 1) * batch * H)
            self.c_buf.ensure_gpu(c, (Self.SEQ + 1) * batch * H)
            self.cache_buf.ensure_gpu(c, Self.SEQ * batch * C)
            self.dh_seq.ensure_gpu(c, Self.SEQ * batch * H)
            self.dx_seq.ensure_gpu(c, Self.SEQ * batch * V)
            self.dh_recur.ensure_gpu(c, batch * H)
            self.dc_recur.ensure_gpu(c, batch * H)
            self.dh_prev.ensure_gpu(c, batch * H)
            self.dc_prev.ensure_gpu(c, batch * H)
            self.dh_t.ensure_gpu(c, batch * H)

    # ----- Forward: unroll the cell over SEQ -----------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        ref in0 = inputs[0]
        self._ensure[target](B, ctx)

        comptime if target == "cpu":
            out.ensure(B * Self.SEQ * H)
            var ip = in0.data.unsafe_ptr()
            var op = out.data.unsafe_ptr()
            var xs = self.x_seq.data.unsafe_ptr()
            var hb = self.h_buf.data.unsafe_ptr()
            var cb = self.c_buf.data.unsafe_ptr()
            for i in range(B * H):  # zero initial h_0 / c_0
                hb[unsafe_offset=i] = 0.0
                cb[unsafe_offset=i] = 0.0
            for b in range(B):  # transpose input BSD → SBD
                for t in range(Self.SEQ):
                    for d in range(V):
                        xs[unsafe_offset=t * B * V + b * V + d] = ip[
                            unsafe_offset=b * Self.SEQ * V + t * V + d
                        ]
            # unroll cell over SEQ via shared timestep-major buffers + offsets.
            # h / c are ONE Tensor each (read slab t, write slab t+1) — the
            # merged-state form avoids the same-buffer-twice exclusivity error.
            for t in range(Self.SEQ):
                self.cell.step_forward["cpu", B](
                    self.x_seq, self.h_buf, self.c_buf, self.cache_buf,
                    ctx,
                    x_off=t * B * V,
                    h_prev_off=t * B * H,
                    c_prev_off=t * B * H,
                    h_t_off=(t + 1) * B * H,
                    c_t_off=(t + 1) * B * H,
                    cache_off=t * B * C,
                )
            # transpose h_buf[1:] (SBD) → output (BSD)
            for b in range(B):
                for t in range(Self.SEQ):
                    for d in range(H):
                        op[unsafe_offset=b * Self.SEQ * H + t * H + d] = hb[
                            unsafe_offset=(t + 1) * B * H + b * H + d
                        ]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.SEQ * H)
            self.h_buf.dev.value().enqueue_fill(0.0)
            self.c_buf.dev.value().enqueue_fill(0.0)
            comptime nb = (B * Self.SEQ * V + TPB - 1) // TPB
            c.enqueue_function[_bsd_to_sbd_kernel[B, Self.SEQ, V]](
                in0.tt["gpu", B * Self.SEQ * V](),
                self.x_seq.tt["gpu", B * Self.SEQ * V](),
                grid_dim=nb, block_dim=TPB,
            )
            for t in range(Self.SEQ):
                self.cell.step_forward["gpu", B](
                    self.x_seq, self.h_buf, self.c_buf, self.cache_buf,
                    c,
                    x_off=t * B * V,
                    h_prev_off=t * B * H,
                    c_prev_off=t * B * H,
                    h_t_off=(t + 1) * B * H,
                    c_t_off=(t + 1) * B * H,
                    cache_off=t * B * C,
                )
            # transpose h_buf[1:] (SBD) → output (BSD). Source view skips the
            # h_0 slab (offset B·H) via a sub-buffer.
            comptime nbo = (B * Self.SEQ * H + TPB - 1) // TPB
            var hb_sub = self.h_buf.dev.value().create_sub_buffer[DT](
                B * H, B * Self.SEQ * H
            )
            c.enqueue_function[_sbd_to_bsd_kernel[B, Self.SEQ, H]](
                tile[B * Self.SEQ * H](hb_sub),
                out.tt["gpu", B * Self.SEQ * H](),
                grid_dim=nbo, block_dim=TPB,
            )

    # ----- Backward: BPTT over SEQ ---------------------------------------

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        ref gin = grad_inputs[0]
        self._ensure[target](B, ctx)

        comptime if target == "cpu":
            gin.ensure(B * Self.SEQ * V)
            var gop = grad_output.data.unsafe_ptr()
            var gip = gin.data.unsafe_ptr()
            var dhs = self.dh_seq.data.unsafe_ptr()
            var dxs = self.dx_seq.data.unsafe_ptr()
            var dh_rec = self.dh_recur.data.unsafe_ptr()
            var dc_rec = self.dc_recur.data.unsafe_ptr()
            var dh_pr = self.dh_prev.data.unsafe_ptr()
            var dc_pr = self.dc_prev.data.unsafe_ptr()
            var dh_tp = self.dh_t.data.unsafe_ptr()
            # transpose grad_output (BSD) → dh_seq (SBD); zero the recur carries.
            for b in range(B):
                for t in range(Self.SEQ):
                    for d in range(H):
                        dhs[unsafe_offset=t * B * H + b * H + d] = gop[
                            unsafe_offset=b * Self.SEQ * H + t * H + d
                        ]
            for i in range(B * H):
                dh_rec[unsafe_offset=i] = 0.0
                dc_rec[unsafe_offset=i] = 0.0

            for tt in range(Self.SEQ):
                var t = Self.SEQ - 1 - tt
                # dh_t = dh_seq[t] + dh_recur ; dc_t = dc_recur (passed directly).
                for i in range(B * H):
                    dh_tp[unsafe_offset=i] = dhs[unsafe_offset=t * B * H + i] + dh_rec[unsafe_offset=i]
                self.cell.step_backward["cpu", B](
                    self.dh_t, self.dc_recur, self.x_seq, self.h_buf,
                    self.c_buf, self.cache_buf, self.dx_seq, self.dh_prev,
                    self.dc_prev,
                    ctx,
                    dh_off=0,
                    dc_off=0,
                    x_off=t * B * V,
                    h_prev_off=t * B * H,
                    c_prev_off=t * B * H,
                    cache_off=t * B * C,
                    dx_off=t * B * V,
                    dh_prev_off=0,
                    dc_prev_off=0,
                )
                # carry: dh_recur = dh_prev, dc_recur = dc_prev.
                for i in range(B * H):
                    dh_rec[unsafe_offset=i] = dh_pr[unsafe_offset=i]
                    dc_rec[unsafe_offset=i] = dc_pr[unsafe_offset=i]

            # transpose dx_seq (SBD) → grad_input (BSD)
            for b in range(B):
                for t in range(Self.SEQ):
                    for d in range(V):
                        gip[unsafe_offset=b * Self.SEQ * V + t * V + d] = dxs[
                            unsafe_offset=t * B * V + b * V + d
                        ]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.SEQ * V)
            comptime nb = (B * Self.SEQ * H + TPB - 1) // TPB
            c.enqueue_function[_bsd_to_sbd_kernel[B, Self.SEQ, H]](
                grad_output.tt["gpu", B * Self.SEQ * H](),
                self.dh_seq.tt["gpu", B * Self.SEQ * H](),
                grid_dim=nb, block_dim=TPB,
            )
            self.dh_recur.dev.value().enqueue_fill(0.0)
            self.dc_recur.dev.value().enqueue_fill(0.0)

            for tt in range(Self.SEQ):
                var t = Self.SEQ - 1 - tt
                # dh_t = dh_seq[t] + dh_recur.
                comptime nadd = (B * H + TPB - 1) // TPB
                var dhs_sub = self.dh_seq.dev.value().create_sub_buffer[DT](
                    t * B * H, B * H
                )
                c.enqueue_function[_add_kernel[B * H]](
                    tile[B * H](dhs_sub),
                    self.dh_recur.tt["gpu", B * H](),
                    self.dh_t.tt["gpu", B * H](),
                    grid_dim=nadd, block_dim=TPB,
                )
                self.cell.step_backward["gpu", B](
                    self.dh_t, self.dc_recur, self.x_seq, self.h_buf,
                    self.c_buf, self.cache_buf, self.dx_seq, self.dh_prev,
                    self.dc_prev,
                    c,
                    dh_off=0,
                    dc_off=0,
                    x_off=t * B * V,
                    h_prev_off=t * B * H,
                    c_prev_off=t * B * H,
                    cache_off=t * B * C,
                    dx_off=t * B * V,
                    dh_prev_off=0,
                    dc_prev_off=0,
                )
                # carry: dh_recur = dh_prev, dc_recur = dc_prev.
                c.enqueue_copy(
                    self.dh_recur.dev.value(), self.dh_prev.dev.value()
                )
                c.enqueue_copy(
                    self.dc_recur.dev.value(), self.dc_prev.dev.value()
                )

            # transpose dx_seq (SBD) → grad_input (BSD)
            comptime nbv = (B * Self.SEQ * V + TPB - 1) // TPB
            c.enqueue_function[_sbd_to_bsd_kernel[B, Self.SEQ, V]](
                self.dx_seq.tt["gpu", B * Self.SEQ * V](),
                gin.tt["gpu", B * Self.SEQ * V](),
                grid_dim=nbv, block_dim=TPB,
            )

    # ----- Param walkers (recurse into the cell) -------------------------

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        var sep = "." if prefix.byte_length() > 0 else ""
        self.cell.for_each_param[target, V](
            visitor, ctx, prefix + sep + "cell"
        )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        self.cell.zero_grad[target](ctx)
