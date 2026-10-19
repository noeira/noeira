"""Conv2DTranspose[IC, OC, K, S, P, H, W, OP] — transposed (fractionally-strided)
2D convolution on the storage surface. The upsampling dual of `Conv2D`; used by
the DreamerV3 pixel decoder (latent → image reconstruction).

A transposed conv is the ADJOINT of a forward conv. Concretely:

  Conv2DTranspose.forward    ≡ Conv2D's col2im (the backward-data scatter)
  Conv2DTranspose.vjp(input) ≡ Conv2D's im2col + GEMM (the forward gather)

so the GPU path REUSES the LAYOUT-parametrized free kernels in `conv2d.mojo`
(`_im2col_kernel`, `_dx_col2im_kernel`, `_go_transpose_kernel`,
`_wT_transpose_kernel`, `_accum_kernel`, `_backward_db_kernel`) with a
channel/spatial substitution (IC↔OC, H↔OH_t), plus two tiny scatter/bias
kernels. The CPU path is direct nested loops (the obvious-correctness ground
truth the GPU is gated against).

Spatial (PyTorch ConvTranspose2d convention, output_padding OP):
    OH_t = (H − 1)·S − 2·P + K + OP
    OW_t = (W − 1)·S − 2·P + K + OP

Layouts (flat trait order, LAYOUT-aware via the shared `_in_off`/`_col_off`):
    input  ("small")  [BATCH, IC·H·W]                 pos (ic,ih,iw)
    weight            [IC,    OC·K·K]    col index = _col_off[LAYOUT, OC, K](oc,kh,kw)
    output ("big")    [BATCH, OC·OH_t·OW_t]           pos (oc,oh,ow)
    bias              [OC]                             (per BIG output channel)

  forward:  μ[oc,oh,ow] = Σ_{ic,kh,kw} W[ic,(oc,kh,kw)]·x[ic,ih,iw] + b[oc]
            with oh = ih·S − P + kh,  ow = iw·S − P + kw   (scatter / col2im)
  vjp:      d_x[ic,ih,iw] = Σ_{oc,kh,kw} W[ic,(oc,kh,kw)]·d_μ[oc,oh,ow]   (gather)
            d_W[ic,(oc,kh,kw)] += Σ x[ic,ih,iw]·d_μ[oc,oh,ow]
            d_b[oc] += Σ d_μ[oc,·,·]

fp32-flow only for now (no bf16/AMP path — `ACT_DT` inherits the Module default
DT). bf16-flow is a follow-up mirroring Conv2D's `ADT` branch.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile
from linalg.matmul import matmul as max_matmul
from noeira.nn.core.splitk_gemm import (
    splitk_path_applies,
    decide_partitions,
    dispatch_splitk_gemm,
)

from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .conv2d import (
    _in_off,
    _col_off,
    _out_off,
    _out_decode,
    _im2col_kernel,
    _dx_col2im_kernel,
    _go_transpose_kernel,
    _wT_transpose_kernel,
    _accum_kernel,
    _backward_db_kernel,
    CONV_TPB,
    CONV_DW_TPB,
)


# ── two small kernels the conv2d set doesn't already cover ────────────────────
def _scatter_small_kernel[
    BATCH: Int,
    IC: Int,
    SI: Int,
    IN_FLAT: Int,
    BS: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    packed: Tile[DT, BS, IC],
    out_small: Tile[DT, BATCH, IN_FLAT],
):
    """Scatter the [BS, IC] GEMM result into the small ("input"-shaped) tensor
    `[BATCH, IC·H·W]`. `packed[b·SI + s, ic] → out_small[b, off(ic,s)]`
    (overwrite — fresh grad_x)."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var pos = idx % IN_FLAT
    var ic, s = _out_decode[LAYOUT, IC, SI](pos)
    out_small[b, pos] = rebind[Scalar[DT]](packed[b * SI + s, ic])


def _addbias_big_kernel[
    BATCH: Int,
    OC: Int,
    SBIG: Int,
    OUT_FLAT: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    out_big: Tile[DT, BATCH, OUT_FLAT],
    bias: Tile[DT, OC],
):
    """Add the per-(big)-channel bias to every spatial position of the output."""
    var idx = Int(global_idx.x)
    if idx >= BATCH * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var pos = idx % OUT_FLAT
    var oc, _s = _out_decode[LAYOUT, OC, SBIG](pos)
    out_big[b, pos] = rebind[Scalar[DT]](out_big[b, pos]) + rebind[Scalar[DT]](
        bias[oc]
    )


# ── Conv2DTranspose ──────────────────────────────────────────────────────────
struct Conv2DTranspose[
    IC_: Int,
    OC_: Int,
    K_: Int,
    S_: Int,
    P_: Int,
    H_: Int,
    W_: Int,
    OP_: Int = 0,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY = 1
    comptime OHt = (Self.H_ - 1) * Self.S_ - 2 * Self.P_ + Self.K_ + Self.OP_
    comptime OWt = (Self.W_ - 1) * Self.S_ - 2 * Self.P_ + Self.K_ + Self.OP_
    comptime SI = Self.H_ * Self.W_  # small spatial
    comptime SBIG = Self.OHt * Self.OWt  # big spatial
    comptime COLT = Self.OC_ * Self.K_ * Self.K_
    comptime IN_FLAT = Self.IC_ * Self.H_ * Self.W_
    comptime OUT_FLAT = Self.OC_ * Self.OHt * Self.OWt
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_FLAT)
    comptime OUT_DIM = Self.OUT_FLAT
    comptime W_SIZE = Self.IC_ * Self.COLT
    comptime B_SIZE = Self.OC_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # GPU scratch (lazy, reused — capture-safe).
    var xT_t: Tensor  # [IC, BS]   transpose of the small input
    var wTT_t: Tensor  # [COLT, IC] transpose of weight (forward GEMM)
    var dcolT_t: Tensor  # [COLT, BS] forward col2im source
    var ecol_t: Tensor  # [BS, COLT] im2col of grad_output (vjp)
    var gxp_t: Tensor  # [BS, IC]   packed grad_x (vjp)
    var dW_tmp: Tensor  # [IC, COLT] dW temp (accumulated into grad)
    # Split-K reduction workspace for the dW GEMM, `[P, IC_, COLT]`, plus the
    # cached partition count (-1 = undecided, 1 = do not split). Owned here so
    # the GEMM does not hit `linalg.matmul`'s per-call cuMemAlloc/cuMemFree —
    # which is also what makes a step containing it capturable. See
    # `Linear.sk_ws` for the full rationale.
    #
    # Same regime as `Conv2D`'s dW and for the same reason: K is `B * SI`
    # (batch times the SMALL spatial map), which for a DreamerV3 decoder is
    # tens of thousands, while M is an in-channel count and N is `OC * K * K`.
    # Tiny grid, enormous contraction.
    #
    # ⚠ There is no `CPAD` equivalent here — `COLT` is whatever `OC * K * K`
    # comes out to. `multi_gemm_cond` wants `N % 128 == 0`, i.e. `OC * K² % 128
    # == 0` (a 4x4 kernel needs `OC % 8`), and `decide_partitions` returns 1
    # when it does not hold. That is not a missed optimisation to paper over:
    # MAX routes those shapes to cuBLAS, and substituting the multistage
    # kernel would be a wrong gradient. Padding N the way `Conv2D` does would
    # widen the eligible set, but it changes the forward's operand too and
    # needs its own measurement.
    var sk_ws: Tensor
    var _sk_p: Int

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.xT_t = Tensor()
        self.wTT_t = Tensor()
        self.dcolT_t = Tensor()
        self.ecol_t = Tensor()
        self.gxp_t = Tensor()
        self.dW_tmp = Tensor()
        self.sk_ws = Tensor()
        self._sk_p = -1

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        m.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        # Transposed-conv fans: fan_in = IC·K², fan_out = OC·K² (PyTorch).
        comptime fan_in = Self.IC_ * Self.K_ * Self.K_
        comptime fan_out = Self.OC_ * Self.K_ * Self.K_
        INIT.init_weight[target](m.weight.val, Self.W_SIZE, fan_in, fan_out, ctx)
        INIT.init_bias[target](m.bias.val, Self.B_SIZE, ctx)
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_FLAT)
            for k in range(B * Self.OUT_FLAT):
                out.data[k] = Scalar[DT](0)
            for b in range(B):
                var ibase = b * Self.IN_FLAT
                var obase = b * Self.OUT_FLAT
                for ih in range(Self.H_):
                    for iw in range(Self.W_):
                        for ic in range(Self.IC_):
                            var av = in0.data[
                                ibase
                                + _in_off[Self.LAYOUT, Self.IC_, Self.H_, Self.W_](
                                    ic, ih, iw
                                )
                            ]
                            var wrow = ic * Self.COLT
                            for oc in range(Self.OC_):
                                for kh in range(Self.K_):
                                    var oh = ih * Self.S_ - Self.P_ + kh
                                    if oh < 0 or oh >= Self.OHt:
                                        continue
                                    for kw in range(Self.K_):
                                        var ow = iw * Self.S_ - Self.P_ + kw
                                        if ow < 0 or ow >= Self.OWt:
                                            continue
                                        var wv = self.weight.val.data[
                                            wrow
                                            + _col_off[Self.LAYOUT, Self.OC_, Self.K_](
                                                oc, kh, kw
                                            )
                                        ]
                                        out.data[
                                            obase
                                            + _in_off[
                                                Self.LAYOUT, Self.OC_, Self.OHt, Self.OWt
                                            ](oc, oh, ow)
                                        ] += wv * av
                # + bias per big output channel
                for oc in range(Self.OC_):
                    var bv = self.bias.val.data[oc]
                    for oh in range(Self.OHt):
                        for ow in range(Self.OWt):
                            out.data[
                                obase
                                + _in_off[
                                    Self.LAYOUT, Self.OC_, Self.OHt, Self.OWt
                                ](oc, oh, ow)
                            ] += bv
        else:
            var c = ctx.value()
            comptime BS = B * Self.SI
            out.ensure_gpu(c, B * Self.OUT_FLAT)
            self.xT_t.ensure_gpu(c, Self.IC_ * BS)
            self.wTT_t.ensure_gpu(c, Self.COLT * Self.IC_)
            self.dcolT_t.ensure_gpu(c, Self.COLT * BS)
            # (1) xT[IC, BS] = transpose(x[B, IN_FLAT])
            comptime nb_xt = (Self.IC_ * BS + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B, Self.IC_, Self.SI, Self.IN_FLAT, BS, DT, Self.LAYOUT
                ]
            ](
                in0.tt["gpu", B, Self.IN_FLAT](),
                self.xT_t.tt["gpu", Self.IC_, BS](),
                grid_dim=nb_xt,
                block_dim=CONV_TPB,
            )
            # (2) wTT[COLT, IC] = transpose(weight[IC, COLT])
            comptime nb_wt = (Self.IC_ * Self.COLT + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[_wT_transpose_kernel[Self.IC_, Self.COLT]](
                self.weight.val.tt["gpu", Self.IC_, Self.COLT](),
                self.wTT_t.tt["gpu", Self.COLT, Self.IC_](),
                grid_dim=nb_wt,
                block_dim=CONV_TPB,
            )
            # (3) dcolT[COLT, BS] = wTT[COLT, IC] @ xT[IC, BS]
            var wTT_tt = TileTensor(
                self.wTT_t.dev.value(), row_major[Self.COLT, Self.IC_]()
            )
            var xT_tt = TileTensor(
                self.xT_t.dev.value(), row_major[Self.IC_, BS]()
            )
            var dcolT_tt = TileTensor(
                self.dcolT_t.dev.value(), row_major[Self.COLT, BS]()
            )
            max_matmul[target="gpu"](dcolT_tt, wTT_tt, xT_tt, c)
            # (4) col2im scatter → out (overwrite); reuses Conv2D's input-grad
            # col2im with the IC↔OC / H↔OHt substitution.
            comptime nb_c2i = (
                B * Self.OUT_FLAT + CONV_DW_TPB - 1
            ) // CONV_DW_TPB
            c.enqueue_function[
                _dx_col2im_kernel[
                    B,
                    Self.OC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.OHt,
                    Self.OWt,
                    Self.H_,
                    Self.W_,
                    Self.OUT_FLAT,
                    Self.COLT,
                    Self.SI,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                self.dcolT_t.tt["gpu", Self.COLT, BS](),
                out.tt["gpu", B, Self.OUT_FLAT](),
                grid_dim=nb_c2i,
                block_dim=CONV_DW_TPB,
            )
            # (5) + bias per big output channel
            comptime nb_bb = (B * Self.OUT_FLAT + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _addbias_big_kernel[
                    B, Self.OC_, Self.SBIG, Self.OUT_FLAT, Self.LAYOUT
                ]
            ](
                out.tt["gpu", B, Self.OUT_FLAT](),
                self.bias.val.tt["gpu", Self.OC_](),
                grid_dim=nb_bb,
                block_dim=CONV_TPB,
            )

    def _decide_sk_p(mut self, BS: Int, ctx: DeviceContext) raises:
        """Decide the dW GEMM's partition count, once, and cache it.

        The dW is `[IC_, BS] @ [BS, COLT]` with `BS = B * SI`. Decided on the
        first backward — an EAGER step, before any capture — and never again:
        P sets `grid_dim`, which is baked into a captured graph.
        """
        self._sk_p = decide_partitions(Self.IC_, Self.COLT, BS, ctx)
        if self._sk_p > 1:
            self.sk_ws.ensure_gpu(ctx, self._sk_p * Self.W_SIZE)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_FLAT)
            for k in range(B * Self.IN_FLAT):
                gin.data[k] = Scalar[DT](0)
            for b in range(B):
                var ibase = b * Self.IN_FLAT
                var obase = b * Self.OUT_FLAT
                # d_bias[oc] += Σ_{big spatial} grad_output
                for oc in range(Self.OC_):
                    var acc = Scalar[DT](0)
                    for oh in range(Self.OHt):
                        for ow in range(Self.OWt):
                            acc += grad_output.data[
                                obase
                                + _in_off[
                                    Self.LAYOUT, Self.OC_, Self.OHt, Self.OWt
                                ](oc, oh, ow)
                            ]
                    self.bias.grd.data[oc] += acc
                # d_x (gather) + d_W (outer) over the scatter mapping
                for ih in range(Self.H_):
                    for iw in range(Self.W_):
                        for ic in range(Self.IC_):
                            var xoff = ibase + _in_off[
                                Self.LAYOUT, Self.IC_, Self.H_, Self.W_
                            ](ic, ih, iw)
                            var av = fin.data[xoff]
                            var wrow = ic * Self.COLT
                            var gxacc = Scalar[DT](0)
                            for oc in range(Self.OC_):
                                for kh in range(Self.K_):
                                    var oh = ih * Self.S_ - Self.P_ + kh
                                    if oh < 0 or oh >= Self.OHt:
                                        continue
                                    for kw in range(Self.K_):
                                        var ow = iw * Self.S_ - Self.P_ + kw
                                        if ow < 0 or ow >= Self.OWt:
                                            continue
                                        var widx = wrow + _col_off[
                                            Self.LAYOUT, Self.OC_, Self.K_
                                        ](oc, kh, kw)
                                        var gy = grad_output.data[
                                            obase
                                            + _in_off[
                                                Self.LAYOUT,
                                                Self.OC_,
                                                Self.OHt,
                                                Self.OWt,
                                            ](oc, oh, ow)
                                        ]
                                        gxacc += self.weight.val.data[widx] * gy
                                        self.weight.grd.data[widx] += av * gy
                            gin.data[xoff] = gxacc
        else:
            var c = ctx.value()
            comptime BS = B * Self.SI
            gin.ensure_gpu(c, B * Self.IN_FLAT)
            self.ecol_t.ensure_gpu(c, BS * Self.COLT)
            self.gxp_t.ensure_gpu(c, BS * Self.IC_)
            self.xT_t.ensure_gpu(c, Self.IC_ * BS)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # (1) ecol[BS, COLT] = im2col(grad_output as a "big" [OC,OHt,OWt] img)
            comptime nb_ec = (BS * Self.COLT + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _im2col_kernel[
                    B,
                    Self.OC_,
                    Self.K_,
                    Self.S_,
                    Self.P_,
                    Self.OHt,
                    Self.OWt,
                    Self.H_,
                    Self.W_,
                    Self.OUT_FLAT,
                    Self.COLT,
                    Self.COLT,  # DCOL: no K pad on the transpose path yet
                    Self.SI,
                    BS,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_FLAT](),
                self.ecol_t.tt["gpu", BS, Self.COLT](),
                grid_dim=nb_ec,
                block_dim=CONV_TPB,
            )
            # (2) grad_x packed[BS, IC] = ecol[BS, COLT] @ weight[IC, COLT]ᵀ
            var ecol_tt = TileTensor(
                self.ecol_t.dev.value(), row_major[BS, Self.COLT]()
            )
            var w_tt = TileTensor(
                self.weight.val.dev.value(), row_major[Self.IC_, Self.COLT]()
            )
            var gxp_tt = TileTensor(
                self.gxp_t.dev.value(), row_major[BS, Self.IC_]()
            )
            max_matmul[transpose_b=True, target="gpu"](gxp_tt, ecol_tt, w_tt, c)
            comptime nb_sc = (B * Self.IN_FLAT + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _scatter_small_kernel[
                    B, Self.IC_, Self.SI, Self.IN_FLAT, BS, Self.LAYOUT
                ]
            ](
                self.gxp_t.tt["gpu", BS, Self.IC_](),
                gin.tt["gpu", B, Self.IN_FLAT](),
                grid_dim=nb_sc,
                block_dim=CONV_TPB,
            )
            # (3) xT[IC, BS] = transpose(forward_input)
            comptime nb_xt = (Self.IC_ * BS + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[
                _go_transpose_kernel[
                    B, Self.IC_, Self.SI, Self.IN_FLAT, BS, DT, Self.LAYOUT
                ]
            ](
                fin.tt["gpu", B, Self.IN_FLAT](),
                self.xT_t.tt["gpu", Self.IC_, BS](),
                grid_dim=nb_xt,
                block_dim=CONV_TPB,
            )
            # (4) dW[IC, COLT] = xT[IC, BS] @ ecol[BS, COLT] → accumulate
            var xT_tt = TileTensor(
                self.xT_t.dev.value(), row_major[Self.IC_, BS]()
            )
            var ecol2_tt = TileTensor(
                self.ecol_t.dev.value(), row_major[BS, Self.COLT]()
            )
            var dW_tmp_tt = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.IC_, Self.COLT]()
            )
            comptime if splitk_path_applies[c.default_device_info]():
                if self._sk_p < 0:
                    self._decide_sk_p(BS, c)
                if self._sk_p > 1:
                    dispatch_splitk_gemm(
                        dW_tmp_tt, xT_tt, ecol2_tt,
                        Self.IC_, Self.COLT, BS,
                        self._sk_p, self.sk_ws, c,
                    )
                else:
                    max_matmul[target="gpu"](dW_tmp_tt, xT_tt, ecol2_tt, c)
            else:
                max_matmul[target="gpu"](dW_tmp_tt, xT_tt, ecol2_tt, c)
            comptime nb_acc = (Self.W_SIZE + CONV_TPB - 1) // CONV_TPB
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                self.weight.grd.tt["gpu", Self.W_SIZE](),
                self.dW_tmp.tt["gpu", Self.W_SIZE](),
                grid_dim=nb_acc,
                block_dim=CONV_TPB,
            )
            # (5) d_bias[OC] += Σ grad_output over big spatial
            c.enqueue_function[
                _backward_db_kernel[
                    B,
                    Self.OC_,
                    Self.OHt,
                    Self.OWt,
                    Self.OUT_FLAT,
                    DT,
                    Self.LAYOUT,
                ]
            ](
                grad_output.tt["gpu", B, Self.OUT_FLAT](),
                self.bias.grd.tt["gpu", Self.OC_](),
                grid_dim=Self.OC_,
                block_dim=CONV_DW_TPB,
            )

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )

    # for_each_param / zero_grad inherit the Module reflection defaults.
