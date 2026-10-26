"""LayerNormNoAffine[DIM] — LayerNorm without learnable scale/bias (storage).

    y = (x - mean) / sqrt(var + eps)

Transformed from legacy `nn.primitives.LayerNormNoAffine` (surface-only change).
PARAM-free (AdaLN-zero supplies the affine externally via a `Modulate` step, so
the normalization itself must be affine-free). Mirrors the storage `LayerNorm`
template with the γ/β terms dropped. The CPU per-row reduction and the two GPU
kernels (forward / backward, 1 block per row) are carried over verbatim.

Cache (leaf-owned, output-caching — no input aliasing):
  * `cache_xhat`    [BATCH, DIM]  normalized x (= output)
  * `cache_inv_std` [BATCH]       1/sqrt(var+eps)

Backward (no gamma): grad_in = inv_std·(g − mean(g) − x̂·mean(g·x̂)), g = grad_out.
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.utils.numerics import get_accum_type
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime LNNA_EPS: Scalar[DT] = 1e-6
comptime LNNA_TPB: Int = 128
# Single-pass register-cache (mirrors storage LayerNorm; γ/β dropped). Read the
# thread's feature slice ONCE, raw-moments mean/var, normalize from registers
# when ELEMS≤cap, else 2-read fallback. accum_type = f32 for bf16 (identity f32).
comptime LNNA_ACC = get_accum_type[DT]()
comptime LNNA_REG_CAP = 8


# ── GPU kernels (single-pass register-cached; block-per-row) ────────────
def _lnna_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + LNNA_TPB - 1) // LNNA_TPB
    comptime REG_CACHE = ELEMS <= LNNA_REG_CAP
    var inv_dim = Scalar[LNNA_ACC](1.0) / Scalar[LNNA_ACC](DIM)
    var my_sum = Scalar[LNNA_ACC](0)
    var my_sumsq = Scalar[LNNA_ACC](0)

    comptime if REG_CACHE:
        var slice = Array[Scalar[LNNA_ACC], ELEMS](
            fill=Scalar[LNNA_ACC](0)
        )

        comptime for e in range(ELEMS):
            var col = t + e * LNNA_TPB
            if col < DIM:
                var x = rebind[Scalar[DT]](input[b, col]).cast[LNNA_ACC]()
                slice[e] = x
                my_sum += x
                my_sumsq += x * x
        var mean_val = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_sumsq)
            * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[LNNA_ACC](0):
            var_val = Scalar[LNNA_ACC](0)
        var inv_std = Scalar[LNNA_ACC](1.0) / sqrt(
            var_val + LNNA_EPS.cast[LNNA_ACC]()
        )
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()

        comptime for e in range(ELEMS):
            var col = t + e * LNNA_TPB
            if col < DIM:
                var x_hat = (slice[e] - mean_val) * inv_std
                cache_xhat[b, col] = x_hat.cast[DT]()
                output[b, col] = x_hat.cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[LNNA_ACC]()
            my_sum += x
            my_sumsq += x * x
            idx += LNNA_TPB
        var mean_val = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_sum) * inv_dim
        )
        var ex2 = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_sumsq)
            * inv_dim
        )
        var var_val = ex2 - mean_val * mean_val
        if var_val < Scalar[LNNA_ACC](0):
            var_val = Scalar[LNNA_ACC](0)
        var inv_std = Scalar[LNNA_ACC](1.0) / sqrt(
            var_val + LNNA_EPS.cast[LNNA_ACC]()
        )
        if t == 0:
            cache_inv_std[b] = inv_std.cast[DT]()
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx]).cast[LNNA_ACC]()
            var x_hat = (x - mean_val) * inv_std
            cache_xhat[b, idx] = x_hat.cast[DT]()
            output[b, idx] = x_hat.cast[DT]()
            idx += LNNA_TPB


def _lnna_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, BATCH],
    grad_input: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + LNNA_TPB - 1) // LNNA_TPB
    comptime REG_CACHE = ELEMS <= LNNA_REG_CAP
    var inv_dim = Scalar[LNNA_ACC](1.0) / Scalar[LNNA_ACC](DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b]).cast[LNNA_ACC]()
    var my_g = Scalar[LNNA_ACC](0)
    var my_g_xhat = Scalar[LNNA_ACC](0)

    comptime if REG_CACHE:
        var g_s = Array[Scalar[LNNA_ACC], ELEMS](fill=Scalar[LNNA_ACC](0))
        var xh_s = Array[Scalar[LNNA_ACC], ELEMS](
            fill=Scalar[LNNA_ACC](0)
        )

        comptime for e in range(ELEMS):
            var col = t + e * LNNA_TPB
            if col < DIM:
                var g = rebind[Scalar[DT]](grad_output[b, col]).cast[LNNA_ACC]()
                var xh = rebind[Scalar[DT]](cache_xhat[b, col]).cast[LNNA_ACC]()
                g_s[e] = g
                xh_s[e] = xh
                my_g += g
                my_g_xhat += g * xh
        var mean_g = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g_xhat)
            * inv_dim
        )

        comptime for e in range(ELEMS):
            var col = t + e * LNNA_TPB
            if col < DIM:
                grad_input[b, col] = (
                    inv_std * (g_s[e] - mean_g - xh_s[e] * mean_g_xhat)
                ).cast[DT]()
    else:
        var idx = t
        while idx < DIM:
            var g = rebind[Scalar[DT]](grad_output[b, idx]).cast[LNNA_ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[LNNA_ACC]()
            my_g += g
            my_g_xhat += g * xh
            idx += LNNA_TPB
        var mean_g = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g) * inv_dim
        )
        var mean_g_xhat = (
            block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g_xhat)
            * inv_dim
        )
        idx = t
        while idx < DIM:
            var g = rebind[Scalar[DT]](grad_output[b, idx]).cast[LNNA_ACC]()
            var xh = rebind[Scalar[DT]](cache_xhat[b, idx]).cast[LNNA_ACC]()
            grad_input[b, idx] = (
                inv_std * (g - mean_g - xh * mean_g_xhat)
            ).cast[DT]()
            idx += LNNA_TPB


struct LayerNormNoAffine[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var cache_xhat: Tensor  # [BATCH, DIM]
    var cache_inv_std: Tensor  # [BATCH]

    def __init__(out self):
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        return Self()

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
            out.ensure(B * Self.DIM_)
            self.cache_xhat.ensure(B * Self.DIM_)
            self.cache_inv_std.ensure(B)
            var in_t = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var out_t = TileTensor(out.data, row_major[B, Self.DIM_]())
            var xhat_t = TileTensor(
                self.cache_xhat.data, row_major[B, Self.DIM_]()
            )
            var inv_v = TileTensor(self.cache_inv_std.data, row_major[B]())
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var s: Scalar[DT] = 0.0
                for d in range(Self.DIM_):
                    s += in_t[b, d]
                var mean = s * inv_dim
                var sv: Scalar[DT] = 0.0
                for d in range(Self.DIM_):
                    var diff = in_t[b, d] - mean
                    sv += diff * diff
                var var_v = sv * inv_dim
                var inv_std = Scalar[DT](1.0) / sqrt(var_v + LNNA_EPS)
                inv_v[b] = inv_std
                for d in range(Self.DIM_):
                    var xh = (in_t[b, d] - mean) * inv_std
                    xhat_t[b, d] = xh
                    out_t[b, d] = xh
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
            self.cache_inv_std.ensure_gpu(c, B)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime lb = RowMajor[B]
            c.enqueue_function[_lnna_forward_kernel[B, Self.DIM_]](
                in0.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.cache_xhat.tt["gpu", B, Self.DIM_](),
                self.cache_inv_std.tt["gpu", B](),
                grid_dim=B,
                block_dim=LNNA_TPB,
            )

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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var xhat_t = TileTensor(
                self.cache_xhat.data, row_major[B, Self.DIM_]()
            )
            var inv_v = TileTensor(self.cache_inv_std.data, row_major[B]())
            var go_t = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_t = TileTensor(gin.data, row_major[B, Self.DIM_]())
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var inv_std = inv_v[b]
                var sum_g: Scalar[DT] = 0.0
                var sum_g_xhat: Scalar[DT] = 0.0
                for d in range(Self.DIM_):
                    var g = go_t[b, d]
                    sum_g += g
                    sum_g_xhat += g * xhat_t[b, d]
                var mean_g = sum_g * inv_dim
                var mean_g_xhat = sum_g_xhat * inv_dim
                for d in range(Self.DIM_):
                    var g = go_t[b, d]
                    var xh = xhat_t[b, d]
                    gi_t[b, d] = inv_std * (g - mean_g - xh * mean_g_xhat)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime lb = RowMajor[B]
            c.enqueue_function[_lnna_backward_kernel[B, Self.DIM_]](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.cache_xhat.tt["gpu", B, Self.DIM_](),
                self.cache_inv_std.tt["gpu", B](),
                gin.tt["gpu", B, Self.DIM_](),
                grid_dim=B,
                block_dim=LNNA_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
