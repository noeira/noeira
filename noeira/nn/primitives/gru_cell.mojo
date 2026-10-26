"""GRUCell[IN_, HIDDEN] — PyTorch-equivalent GRU cell on the storage surface.

Transformed from legacy `nn.primitives.gru_cell.GRUCell` (surface-only change).
Binary `Module` (ARITY=2): inputs = (x_t [B,IN], h_prev [B,H]) → out = new
hidden [B,H]. Gate math / kernels / SIMD carried VERBATIM from legacy.

Math (PyTorch convention):
  r = σ(x · W_ir + b_ir + h · W_hr + b_hr)         reset gate
  z = σ(x · W_iz + b_iz + h · W_hz + b_hz)         update gate
  n = tanh(x · W_in + b_in + r ⊙ (h · W_hn + b_hn)) new candidate
  h' = (1 − z) ⊙ n + z ⊙ h

Storage convention (row-major):
  W_ih [IN, 3·H]  — columns 0..H = r, H..2H = z, 2H..3H = n
  W_hh [H,  3·H]  — same column split
  b_ih [3·H] ; b_hh [3·H]

Caches (BATCH-sized, lazy): r, z, n [B,H] + hn_pre [B,H] (= W_hn·h + b_hn).

TWO-PHASE → SINGLE-PHASE FOLD (the recurrent leaf gotcha)
---------------------------------------------------------
Legacy split the backward into `vjp_param_grads` (phase 1, reads cached
x/h) then `vjp_grad_input` (phase 2, RECOMPUTES the gate grads). The split
existed ONLY because legacy cached the forward x/h as raw pointers into the
caller's input slabs, which `grad_inputs` could ALIAS — so param grads
(which read x/h) had to run before dx/dh writes clobbered them.

The storage `Module.vjp` is single-phase AND receives `forward_input`
(the original x/h Tensors) as a separate pack from `grad_inputs`. There is
no aliasing between the two packs, so we FOLD both phases into one pass:
recompute the per-(b,col) gate gradients ONCE into d_ix/d_hx, accumulate
the bias + weight grads, then write dx/dh. The forward no longer caches x/h
pointers at all (vjp reads them from `forward_input`). On GPU the gate
kernel fills `_dcomb`, then the dW/db kernels, then the input kernel — all
in one `vjp`; the kernel order still keeps the dW reads before any dx/dh
writes (defensive, though the packs no longer alias).
"""

from noeira.nn.core.mm import mm, bmm
from std.math import exp, tanh
from linalg.matmul import matmul as max_matmul
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .linear import _transpose_tiled_kernel, _T_TILE, _T_BR


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels: gate pre-activations (ix = x·W_ih, hx = h·W_hh) and the
# dx/dh/dW GEMMs go through max_matmul (mirrors Linear/LSTM); only the
# elementwise gate (fwd) / gate-grad (bwd) math + the column-sum bias/dh
# reductions remain hand-rolled.
#   cache [B, 4H] = [r | z | n | hn_pre]
#   d_ix  [B, 3H] = [d_ir | d_iz | d_in]   d_hx [B, 3H] = [d_hr | d_hz | d_hn]
# ──────────────────────────────────────────────────────────────────────


def _gru_gate_fwd_kernel[
    BATCH: Int, H: Int,
](
    ix: Tile[DT, BATCH, 3 * H],
    hx: Tile[DT, BATCH, 3 * H],
    b_ih: Tile[DT, 3 * H],
    b_hh: Tile[DT, 3 * H],
    h_prev: Tile[DT, BATCH, H],
    out_buf: Tile[DT, BATCH, H],
    cache: Tile[DT, BATCH, 4 * H],
):
    """Elementwise gates from the two GEMM pre-activations. One thread per
    (sample, hidden unit). Reset gate couples in here (ng = tanh(in + r·hn))."""
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var rg = _sigmoid(
        rebind[Scalar[DT]](ix[bi, j]) + rebind[Scalar[DT]](b_ih[j])
        + rebind[Scalar[DT]](hx[bi, j]) + rebind[Scalar[DT]](b_hh[j])
    )
    var zg = _sigmoid(
        rebind[Scalar[DT]](ix[bi, H + j]) + rebind[Scalar[DT]](b_ih[H + j])
        + rebind[Scalar[DT]](hx[bi, H + j]) + rebind[Scalar[DT]](b_hh[H + j])
    )
    var in_pre = rebind[Scalar[DT]](ix[bi, 2 * H + j]) + rebind[Scalar[DT]](
        b_ih[2 * H + j]
    )
    var hn = rebind[Scalar[DT]](hx[bi, 2 * H + j]) + rebind[Scalar[DT]](
        b_hh[2 * H + j]
    )
    var ng = tanh(in_pre + rg * hn)
    out_buf[bi, j] = (
        (Scalar[DT](1.0) - zg) * ng + zg * rebind[Scalar[DT]](h_prev[bi, j])
    )
    cache[bi, j] = rg
    cache[bi, H + j] = zg
    cache[bi, 2 * H + j] = ng
    cache[bi, 3 * H + j] = hn


def _gru_gate_grad_kernel[
    BATCH: Int, H: Int,
](
    go: Tile[DT, BATCH, H],
    cache: Tile[DT, BATCH, 4 * H],
    h_prev: Tile[DT, BATCH, H],
    d_ix: Tile[DT, BATCH, 3 * H],
    d_hx: Tile[DT, BATCH, 3 * H],
):
    """Per-gate pre-activation grads → contiguous d_ix / d_hx [B, 3H]."""
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var one = Scalar[DT](1.0)
    var rg = rebind[Scalar[DT]](cache[bi, j])
    var zg = rebind[Scalar[DT]](cache[bi, H + j])
    var ng = rebind[Scalar[DT]](cache[bi, 2 * H + j])
    var hn = rebind[Scalar[DT]](cache[bi, 3 * H + j])
    var hval = rebind[Scalar[DT]](h_prev[bi, j])
    var dh_now = rebind[Scalar[DT]](go[bi, j])

    var dz = dh_now * (hval - ng)
    var dn = dh_now * (one - zg)
    var d_pre_n = dn * (one - ng * ng)        # tanh'
    var dr = d_pre_n * hn
    var d_hn = d_pre_n * rg
    var d_pre_r = dr * rg * (one - rg)        # sigmoid'
    var d_pre_z = dz * zg * (one - zg)        # sigmoid'

    d_ix[bi, j] = d_pre_r
    d_ix[bi, H + j] = d_pre_z
    d_ix[bi, 2 * H + j] = d_pre_n
    d_hx[bi, j] = d_pre_r
    d_hx[bi, H + j] = d_pre_z
    d_hx[bi, 2 * H + j] = d_hn


def _gru_accum_kernel[
    N: Int,
](
    dst: Tile[DT, N],
    src: Tile[DT, N],
):
    """dst += src (accumulate dW_tmp into the persistent param grad)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


def _gru_dbsum_kernel[
    BATCH: Int, N: Int,
](
    src: Tile[DT, BATCH, N],
    dst: Tile[DT, N],
):
    """dst[c] += Σ_b src[b, c]. Grid (N,), block-reduce over BATCH."""
    var c = Int(block_idx.x)
    if c >= N:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](src[b, c])
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        dst[c] = rebind[Scalar[DT]](dst[c]) + total[0]


def _gru_dh_add_zh_kernel[
    BATCH: Int, H: Int,
](
    go: Tile[DT, BATCH, H],
    cache: Tile[DT, BATCH, 4 * H],
    dh: Tile[DT, BATCH, H],
):
    """dh += go ⊙ z (the direct ∂h'/∂h path through z·h, added after the GEMM)."""
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    dh[bi, j] += rebind[Scalar[DT]](go[bi, j]) * rebind[Scalar[DT]](
        cache[bi, H + j]
    )


# ──────────────────────────────────────────────────────────────────────
# GRUCell.
# ──────────────────────────────────────────────────────────────────────


struct GRUCell[IN_: Int, HIDDEN: Int](Module):
    comptime ARITY = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM = Self.IN_
    comptime OUT_DIM = Self.HIDDEN

    @staticmethod
    def _build_in_dims() -> Array[Int, 2]:
        var d = Array[Int, 2](fill=0)
        d[0] = Self.IN_
        d[1] = Self.HIDDEN
        return d^

    comptime W_IH_SIZE = Self.IN_ * (3 * Self.HIDDEN)
    comptime W_HH_SIZE = Self.HIDDEN * (3 * Self.HIDDEN)
    comptime B_IH_SIZE = 3 * Self.HIDDEN

    # Parameters — Param fields are walked by reflection (for_each_param_auto).
    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b_ih: Param["b_ih", False, Self.B_IH_SIZE]
    var b_hh: Param["b_hh", False, Self.B_IH_SIZE]

    # Forward caches (CPU path): r/z/n [B,H] + hn_pre [B,H].
    var _r_cache: Tensor
    var _z_cache: Tensor
    var _n_cache: Tensor
    var _hn_pre:  Tensor

    # GPU scratch (device-only, lazy). cache [B,4H]; fwd GEMM ix/hx [B,3H];
    # bwd gate grads d_ix/d_hx [B,3H]; bwd dW transpose/temp.
    var _cache: Tensor   # [B, 4H]
    var _ix: Tensor      # [B, 3H]  x @ W_ih
    var _hx: Tensor      # [B, 3H]  h_prev @ W_hh
    var _dix: Tensor     # [B, 3H]  d_pre x-side
    var _dhx: Tensor     # [B, 3H]  d_pre h-side
    var _xT: Tensor      # [IN_, B]
    var _hT: Tensor      # [H, B]
    var _dWih_tmp: Tensor  # [IN_, 3H]
    var _dWhh_tmp: Tensor  # [H, 3H]

    def __init__(out self):
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b_ih = Param["b_ih", False, Self.B_IH_SIZE]()
        self.b_hh = Param["b_hh", False, Self.B_IH_SIZE]()
        self._r_cache = Tensor()
        self._z_cache = Tensor()
        self._n_cache = Tensor()
        self._hn_pre  = Tensor()
        self._cache = Tensor()
        self._ix = Tensor()
        self._hx = Tensor()
        self._dix = Tensor()
        self._dhx = Tensor()
        self._xT = Tensor()
        self._hT = Tensor()
        self._dWih_tmp = Tensor()
        self._dWhh_tmp = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var g = Self()
        g.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make[target](ctx)
        g.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make[target](ctx)
        g.b_ih = Param["b_ih", False, Self.B_IH_SIZE].make[target](ctx)
        g.b_hh = Param["b_hh", False, Self.B_IH_SIZE].make[target](ctx)
        INIT.init_weight[target](
            g.W_ih.val, Self.W_IH_SIZE, Self.IN_, 3 * Self.HIDDEN, ctx
        )
        INIT.init_weight[target](
            g.W_hh.val, Self.W_HH_SIZE, Self.HIDDEN, 3 * Self.HIDDEN, ctx
        )
        INIT.init_bias[target](g.b_ih.val, Self.B_IH_SIZE, ctx)
        INIT.init_bias[target](g.b_hh.val, Self.B_IH_SIZE, ctx)
        return g^

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x_in = inputs[0]
        ref h_in = inputs[1]

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN

        comptime if target == "gpu":
            var c = ctx.value()
            out.ensure_gpu(c, B * H)
            self._cache.ensure_gpu(c, B * 4 * H)
            self._ix.ensure_gpu(c, B * THREE_H)
            self._hx.ensure_gpu(c, B * THREE_H)
            # ix = x @ W_ih ; hx = h_prev @ W_hh  (two GEMMs; was a hand-rolled
            # per-thread inner product over IN_+H for all 3 gates).
            mm[A0=B, A1=Self.IN0_DIM, B0=Self.IN0_DIM, B1=THREE_H, O0=B, O1=THREE_H](
                self._ix.dev.value(), x_in.dev.value(), self.W_ih.val.dev.value(), c
            )
            mm[A0=B, A1=H, B0=H, B1=THREE_H, O0=B, O1=THREE_H](
                self._hx.dev.value(), h_in.dev.value(), self.W_hh.val.dev.value(), c
            )
            # elementwise gates + reset coupling + output + cache.
            comptime gk = _gru_gate_fwd_kernel[B, H]
            comptime nblk = (B * H + TPB - 1) // TPB
            c.enqueue_function[gk](
                self._ix.tt["gpu", B, THREE_H](),
                self._hx.tt["gpu", B, THREE_H](),
                self.b_ih.val.tt["gpu", THREE_H](),
                self.b_hh.val.tt["gpu", THREE_H](),
                h_in.tt["gpu", B, H](),
                out.tt["gpu", B, H](),
                self._cache.tt["gpu", B, 4 * H](),
                grid_dim=(nblk,), block_dim=(TPB,),
            )
            return

        out.ensure(B * H)
        self._r_cache.ensure(B * H)
        self._z_cache.ensure(B * H)
        self._n_cache.ensure(B * H)
        self._hn_pre.ensure(B * H)

        var h_p = h_in.data.unsafe_ptr()
        var out_p = out.data.unsafe_ptr()

        var b_ih_p = self.b_ih.val.data.unsafe_ptr()
        var b_hh_p = self.b_hh.val.data.unsafe_ptr()
        var r_c = self._r_cache.data.unsafe_ptr()
        var z_c = self._z_cache.data.unsafe_ptr()
        var n_c = self._n_cache.data.unsafe_ptr()
        var hn_c = self._hn_pre.data.unsafe_ptr()

        # Gate pre-activations via BLAS (carried VERBATIM from legacy):
        #   ix = x @ W_ih → [B, 3H] ; hx = h @ W_hh → [B, 3H] (bias added below)
        var x_tt = TileTensor(x_in.data, row_major[B, Self.IN0_DIM]())
        var h_tt = TileTensor(h_in.data, row_major[B, H]())
        var W_ih_tt = TileTensor(
            self.W_ih.val.data, row_major[Self.IN0_DIM, THREE_H]()
        )
        var W_hh_tt = TileTensor(self.W_hh.val.data, row_major[H, THREE_H]())
        var ix_buf = List[Scalar[DT]](length=B * THREE_H, fill=Scalar[DT](0))
        var hx_buf = List[Scalar[DT]](length=B * THREE_H, fill=Scalar[DT](0))
        var ix_tt = TileTensor(ix_buf, row_major[B, THREE_H]())
        var hx_tt = TileTensor(hx_buf, row_major[B, THREE_H]())
        max_matmul[target="cpu"](ix_tt, x_tt, W_ih_tt, None)
        max_matmul[target="cpu"](hx_tt, h_tt, W_hh_tt, None)

        for b in range(B):
            var g_off = b * THREE_H
            var h_off = b * H
            var out_off = b * H
            var c_off = b * H
            for col in range(H):
                var rg = _sigmoid(
                    ix_buf[g_off + col] + b_ih_p[unsafe_offset=col]
                    + hx_buf[g_off + col] + b_hh_p[unsafe_offset=col]
                )
                r_c[unsafe_offset=c_off + col] = rg

                var zg = _sigmoid(
                    ix_buf[g_off + H + col] + b_ih_p[unsafe_offset=H + col]
                    + hx_buf[g_off + H + col] + b_hh_p[unsafe_offset=H + col]
                )
                z_c[unsafe_offset=c_off + col] = zg

                var in_pre = ix_buf[g_off + 2 * H + col] + b_ih_p[unsafe_offset=2 * H + col]
                var hn_p = hx_buf[g_off + 2 * H + col] + b_hh_p[unsafe_offset=2 * H + col]
                hn_c[unsafe_offset=c_off + col] = hn_p

                var ng = tanh(in_pre + rg * hn_p)
                n_c[unsafe_offset=c_off + col] = ng

                out_p[unsafe_offset=out_off + col] = (
                    (Scalar[DT](1.0) - zg) * ng + zg * h_p[unsafe_offset=h_off + col]
                )

    # ------------------------------------------------------------------
    # Backward — two legacy phases FOLDED into one single-phase vjp.
    # ------------------------------------------------------------------

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x_in = forward_input[0]
        ref h_in = forward_input[1]
        ref dx_in = grad_inputs[0]
        ref dh_in = grad_inputs[1]

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN

        comptime if target == "gpu":
            var c = ctx.value()
            dx_in.ensure_gpu(c, B * Self.IN0_DIM)
            dh_in.ensure_gpu(c, B * H)
            self._dix.ensure_gpu(c, B * THREE_H)
            self._dhx.ensure_gpu(c, B * THREE_H)
            self._xT.ensure_gpu(c, Self.IN0_DIM * B)
            self._hT.ensure_gpu(c, H * B)
            self._dWih_tmp.ensure_gpu(c, Self.W_IH_SIZE)
            self._dWhh_tmp.ensure_gpu(c, Self.W_HH_SIZE)

            var cc = self._cache.tt["gpu", B, 4 * H]()
            var go_lt = grad_output.tt["gpu", B, H]()
            var h_lt = h_in.tt["gpu", B, H]()

            # Gate-grad math → contiguous d_ix / d_hx [B, 3H].
            comptime nblk = (B * H + TPB - 1) // TPB
            comptime gk = _gru_gate_grad_kernel[B, H]
            c.enqueue_function[gk](
                go_lt, cc, h_lt,
                self._dix.tt["gpu", B, THREE_H](),
                self._dhx.tt["gpu", B, THREE_H](),
                grid_dim=(nblk,), block_dim=(TPB,),
            )

            # Capture xᵀ / h_prevᵀ before dx/dh writes (defensive; packs don't
            # alias under the storage surface, but mirrors LSTM/Linear).
            c.enqueue_function[_transpose_tiled_kernel[DT]](
                x_in.dev.value(),
                self._xT.dev.value(),
                Int64(B),
                Int64(Self.IN0_DIM),
                grid_dim=(
                    (Self.IN0_DIM + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            c.enqueue_function[_transpose_tiled_kernel[DT]](
                h_in.dev.value(),
                self._hT.dev.value(),
                Int64(B),
                Int64(H),
                grid_dim=(
                    (H + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )

            # dx = d_ix @ W_ihᵀ ; dh = d_hx @ W_hhᵀ (then += go⊙z).
            var dx_tt = TileTensor(dx_in.dev.value(), row_major(B, Self.IN0_DIM))
            var dh_tt = TileTensor(dh_in.dev.value(), row_major(B, H))
            mm[transpose_b=True, A0=B, A1=THREE_H, B0=Self.IN0_DIM, B1=THREE_H, O0=B, O1=Self.IN0_DIM](
                dx_in.dev.value(), self._dix.dev.value(), self.W_ih.val.dev.value(), c
            )
            mm[transpose_b=True, A0=B, A1=THREE_H, B0=H, B1=THREE_H, O0=B, O1=H](
                dh_in.dev.value(), self._dhx.dev.value(), self.W_hh.val.dev.value(), c
            )
            comptime zk = _gru_dh_add_zh_kernel[B, H]
            c.enqueue_function[zk](
                go_lt, cc, dh_in.tt["gpu", B, H](),
                grid_dim=(nblk,), block_dim=(TPB,),
            )

            # dW_ih += xᵀ @ d_ix ; dW_hh += h_prevᵀ @ d_hx (temp + accumulate).
            var xT_tt = TileTensor(self._xT.dev.value(), row_major(Self.IN0_DIM, B))
            var hT_tt = TileTensor(self._hT.dev.value(), row_major(H, B))
            mm[A0=Self.IN0_DIM, A1=B, B0=B, B1=THREE_H, O0=Self.IN0_DIM, O1=THREE_H](
                self._dWih_tmp.dev.value(), self._xT.dev.value(), self._dix.dev.value(), c
            )
            mm[A0=H, A1=B, B0=B, B1=THREE_H, O0=H, O1=THREE_H](
                self._dWhh_tmp.dev.value(), self._hT.dev.value(), self._dhx.dev.value(), c
            )
            comptime aih = _gru_accum_kernel[Self.W_IH_SIZE]
            c.enqueue_function[aih](
                self.W_ih.grd.tt["gpu", Self.W_IH_SIZE](),
                self._dWih_tmp.tt["gpu", Self.W_IH_SIZE](),
                grid_dim=((Self.W_IH_SIZE + TPB - 1) // TPB,), block_dim=(TPB,),
            )
            comptime ahh = _gru_accum_kernel[Self.W_HH_SIZE]
            c.enqueue_function[ahh](
                self.W_hh.grd.tt["gpu", Self.W_HH_SIZE](),
                self._dWhh_tmp.tt["gpu", Self.W_HH_SIZE](),
                grid_dim=((Self.W_HH_SIZE + TPB - 1) // TPB,), block_dim=(TPB,),
            )

            # db_ih += Σ_b d_ix ; db_hh += Σ_b d_hx.
            comptime dbk = _gru_dbsum_kernel[B, THREE_H]
            c.enqueue_function[dbk](
                self._dix.tt["gpu", B, THREE_H](),
                self.b_ih.grd.tt["gpu", THREE_H](),
                grid_dim=(THREE_H,), block_dim=(TPB,),
            )
            c.enqueue_function[dbk](
                self._dhx.tt["gpu", B, THREE_H](),
                self.b_hh.grd.tt["gpu", THREE_H](),
                grid_dim=(THREE_H,), block_dim=(TPB,),
            )
            return

        # ----- CPU: single fused pass -----
        dx_in.ensure(B * Self.IN0_DIM)
        dh_in.ensure(B * H)

        var x_p = x_in.data.unsafe_ptr()
        var h_p = h_in.data.unsafe_ptr()
        var dW_ih_p = self.W_ih.grd.data.unsafe_ptr()
        var dW_hh_p = self.W_hh.grd.data.unsafe_ptr()
        var db_ih_p = self.b_ih.grd.data.unsafe_ptr()
        var db_hh_p = self.b_hh.grd.data.unsafe_ptr()
        var r_c = self._r_cache.data.unsafe_ptr()
        var z_c = self._z_cache.data.unsafe_ptr()
        var n_c = self._n_cache.data.unsafe_ptr()
        var hn_c = self._hn_pre.data.unsafe_ptr()
        var go_p = grad_output.data.unsafe_ptr()
        var dh_p = dh_in.data.unsafe_ptr()

        # Per-(b,col) gate gradients (scalar, O(B·H)) → d_ix/d_hx [B, 3H] +
        # bias accumulate (the FOLDED phase 1 + phase 2 gate recompute — one
        # pass since the two packs don't alias). Carried from legacy.
        var d_ix_buf = List[Scalar[DT]](
            length=B * THREE_H, fill=Scalar[DT](0)
        )
        var d_hx_buf = List[Scalar[DT]](
            length=B * THREE_H, fill=Scalar[DT](0)
        )
        for b in range(B):
            var c_off = b * H
            var g_off = b * THREE_H
            for col in range(H):
                var dh_now = go_p[unsafe_offset=c_off + col]
                var rg = r_c[unsafe_offset=c_off + col]
                var zg = z_c[unsafe_offset=c_off + col]
                var ng = n_c[unsafe_offset=c_off + col]
                var hn_v = hn_c[unsafe_offset=c_off + col]
                var h_val = h_p[unsafe_offset=c_off + col]

                var dz = dh_now * (h_val - ng)
                var dn = dh_now * (Scalar[DT](1.0) - zg)
                var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)  # tanh'
                var d_in_n = d_pre_n
                var dr = d_pre_n * hn_v          # gradient on r
                var d_hn = d_pre_n * rg          # gradient on hn_pre
                var d_pre_r = dr * rg * (Scalar[DT](1.0) - rg)  # sigmoid'
                var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)  # sigmoid'

                d_ix_buf[g_off + col]         = d_pre_r
                d_ix_buf[g_off + H + col]     = d_pre_z
                d_ix_buf[g_off + 2 * H + col] = d_in_n
                d_hx_buf[g_off + col]         = d_pre_r
                d_hx_buf[g_off + H + col]     = d_pre_z
                d_hx_buf[g_off + 2 * H + col] = d_hn

                db_ih_p[unsafe_offset=col]         += d_pre_r
                db_ih_p[unsafe_offset=H + col]     += d_pre_z
                db_ih_p[unsafe_offset=2 * H + col] += d_in_n
                db_hh_p[unsafe_offset=col]         += d_pre_r
                db_hh_p[unsafe_offset=H + col]     += d_pre_z
                db_hh_p[unsafe_offset=2 * H + col] += d_hn

        var d_ix_tt = TileTensor(d_ix_buf, row_major[B, THREE_H]())
        var d_hx_tt = TileTensor(d_hx_buf, row_major[B, THREE_H]())

        # dW_ih += xᵀ @ d_ix, dW_hh += hᵀ @ d_hx via BLAS (transpose x/h
        # first → temp). Carried VERBATIM from legacy.
        var xT_buf = List[Scalar[DT]](
            length=Self.IN0_DIM * B, fill=Scalar[DT](0)
        )
        var hT_buf = List[Scalar[DT]](length=H * B, fill=Scalar[DT](0))
        for b in range(B):
            for k in range(Self.IN0_DIM):
                xT_buf[k * B + b] = x_p[unsafe_offset=b * Self.IN0_DIM + k]
            for k in range(H):
                hT_buf[k * B + b] = h_p[unsafe_offset=b * H + k]
        var xT_tt = TileTensor(xT_buf, row_major[Self.IN0_DIM, B]())
        var hT_tt = TileTensor(hT_buf, row_major[H, B]())
        var dWih_buf = List[Scalar[DT]](
            length=Self.IN0_DIM * THREE_H, fill=Scalar[DT](0)
        )
        var dWhh_buf = List[Scalar[DT]](
            length=H * THREE_H, fill=Scalar[DT](0)
        )
        var dWih_tt = TileTensor(dWih_buf, row_major[Self.IN0_DIM, THREE_H]())
        var dWhh_tt = TileTensor(dWhh_buf, row_major[H, THREE_H]())
        max_matmul[target="cpu"](dWih_tt, xT_tt, d_ix_tt, None)
        max_matmul[target="cpu"](dWhh_tt, hT_tt, d_hx_tt, None)
        for i in range(Self.IN0_DIM * THREE_H):
            dW_ih_p[unsafe_offset=i] += dWih_buf[i]
        for i in range(H * THREE_H):
            dW_hh_p[unsafe_offset=i] += dWhh_buf[i]

        # Input grads: dx = d_ix @ W_ihᵀ ; dh = d_hx @ W_hhᵀ + go⊙z.
        var W_ih_tt = TileTensor(
            self.W_ih.val.data, row_major[Self.IN0_DIM, THREE_H]()
        )
        var W_hh_tt = TileTensor(self.W_hh.val.data, row_major[H, THREE_H]())
        var dx_tt = TileTensor(dx_in.data, row_major[B, Self.IN0_DIM]())
        var dh_tt = TileTensor(dh_in.data, row_major[B, H]())
        max_matmul[transpose_b=True, target="cpu"](dx_tt, d_ix_tt, W_ih_tt, None)
        max_matmul[transpose_b=True, target="cpu"](dh_tt, d_hx_tt, W_hh_tt, None)
        for b in range(B):
            var c_off = b * H
            for col in range(H):
                dh_p[unsafe_offset=c_off + col] += go_p[unsafe_offset=c_off + col] * z_c[unsafe_offset=c_off + col]

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers W_ih/W_hh/b_ih/b_hh).

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_IH_SIZE](
            self.W_ih.val, src.W_ih.val, tau, ctx
        )
        polyak_tensor[target, Self.W_HH_SIZE](
            self.W_hh.val, src.W_hh.val, tau, ctx
        )
        polyak_tensor[target, Self.B_IH_SIZE](
            self.b_ih.val, src.b_ih.val, tau, ctx
        )
        polyak_tensor[target, Self.B_IH_SIZE](
            self.b_hh.val, src.b_hh.val, tau, ctx
        )
