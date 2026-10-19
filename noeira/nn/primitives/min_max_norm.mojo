"""MinMaxNorm[DIM] — per-sample (x - min) / (max - min) scaling (storage surface).

Transformed from legacy `nn.primitives.MinMaxNorm` (surface-only change). Param-
less; the per-sample copy of the input row is leaf-owned in a `Tensor` cache so
backward can re-derive min/max/argmin/argmax without indexing the orchestrator's
input slab. CPU loops + the two GPU kernels (one block per sample, threads stride
over DIM, block reductions for min/max/argmin/argmax/G/Gy) are carried verbatim.

Math (per sample of dim N):
    m = min(x), M = max(x), s = clamp(M - m, ≥ ε)
    y_j = (x_j - m) / s

Backward (given grad_y, compute grad_x):
    G  = Σ grad_y
    Gy = Σ grad_y · y
    grad_x[argmax] = (grad_y[argmax] - Gy) / s
    grad_x[argmin] = (Gy + grad_y[argmin] - G) / s
    grad_x[i ∉ {argmin, argmax}] = grad_y[i] / s
    grad_x = 0 in the degenerate (M - m < ε) case.
"""

from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime MMN_EPS: Scalar[DT] = 1e-5
comptime MMN_TPB: Int = 128
# Forward register-caches the thread's feature slice (read once → min/max →
# normalize from registers) when ELEMS≤cap; else the legacy 2-read path.
comptime MMN_REG_CAP = 8


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) —
# one block per sample, threads stride over DIM. Block reductions:
# min, max for stats; min over argmin/argmax sentinels for index
# resolution; sum for G = Σ grad_y and Gy = Σ grad_y · y.
# ──────────────────────────────────────────────────────────────────────


def _min_max_norm_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime ELEMS = (DIM + MMN_TPB - 1) // MMN_TPB
    comptime REG_CACHE = ELEMS <= MMN_REG_CAP

    var pos_inf = Scalar[DT](1e30)
    var neg_inf = Scalar[DT](-1e30)
    var my_min = pos_inf
    var my_max = neg_inf

    comptime if REG_CACHE:
        var slice = Array[Scalar[DT], ELEMS](fill=Scalar[DT](0))

        comptime for e in range(ELEMS):
            var col = t + e * MMN_TPB
            if col < DIM:
                var v = rebind[Scalar[DT]](input[b, col])
                slice[e] = v
                if v < my_min:
                    my_min = v
                if v > my_max:
                    my_max = v
        var min_val = block.min[block_size=MMN_TPB, broadcast=True](val=my_min)
        var max_val = block.max[block_size=MMN_TPB, broadcast=True](val=my_max)
        var s = max_val - min_val
        if s < MMN_EPS:
            s = MMN_EPS
        var inv_s = Scalar[DT](1.0) / s

        comptime for e in range(ELEMS):
            var col = t + e * MMN_TPB
            if col < DIM:
                cache[b, col] = slice[e]
                output[b, col] = (slice[e] - min_val) * inv_s
    else:
        var idx = t
        while idx < DIM:
            var v = rebind[Scalar[DT]](input[b, idx])
            if v < my_min:
                my_min = v
            if v > my_max:
                my_max = v
            idx += MMN_TPB
        var min_val = block.min[block_size=MMN_TPB, broadcast=True](val=my_min)
        var max_val = block.max[block_size=MMN_TPB, broadcast=True](val=my_max)
        var s = max_val - min_val
        if s < MMN_EPS:
            s = MMN_EPS
        var inv_s = Scalar[DT](1.0) / s
        idx = t
        while idx < DIM:
            var x = rebind[Scalar[DT]](input[b, idx])
            cache[b, idx] = x
            output[b, idx] = (x - min_val) * inv_s
            idx += MMN_TPB


def _min_max_norm_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    cache: Tile[DT, BATCH, DIM],
    grad_input: Tile[DT, BATCH, DIM],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return

    var pos_inf = Scalar[DT](1e30)
    var neg_inf = Scalar[DT](-1e30)
    var my_min = pos_inf
    var my_max = neg_inf
    var my_argmin: Int = 0
    var my_argmax: Int = 0
    var idx = t
    while idx < DIM:
        var v = rebind[Scalar[DT]](cache[b, idx])
        if v < my_min:
            my_min = v
            my_argmin = idx
        if v > my_max:
            my_max = v
            my_argmax = idx
        idx += MMN_TPB

    var min_val = block.min[block_size=MMN_TPB, broadcast=True](val=my_min)
    var max_val = block.max[block_size=MMN_TPB, broadcast=True](val=my_max)

    # Resolve argmin/argmax: threads that don't hold the block min/max
    # emit a sentinel (DIM), then block.min finds the smallest valid index.
    var SENTINEL = DIM
    var my_argmin_s: Int = my_argmin if my_min == min_val else SENTINEL
    var my_argmax_s: Int = my_argmax if my_max == max_val else SENTINEL
    var argmin = block.min[block_size=MMN_TPB, broadcast=True](
        val=Scalar[DType.int32](my_argmin_s)
    )
    var argmax = block.min[block_size=MMN_TPB, broadcast=True](
        val=Scalar[DType.int32](my_argmax_s)
    )

    var s = max_val - min_val
    var degenerate = s < MMN_EPS
    if degenerate:
        s = MMN_EPS
    var inv_s = Scalar[DT](1.0) / s

    if degenerate:
        idx = t
        while idx < DIM:
            grad_input[b, idx] = Scalar[DT](0.0)
            idx += MMN_TPB
        return

    var my_G = Scalar[DT](0.0)
    var my_Gy = Scalar[DT](0.0)
    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](cache[b, idx])
        var y = (x - min_val) * inv_s
        var dy = rebind[Scalar[DT]](grad_output[b, idx])
        my_G += dy
        my_Gy += dy * y
        idx += MMN_TPB
    var G = block.sum[block_size=MMN_TPB, broadcast=True](val=my_G)
    var Gy = block.sum[block_size=MMN_TPB, broadcast=True](val=my_Gy)

    var dy_argmin = rebind[Scalar[DT]](grad_output[b, Int(argmin)])
    var dy_argmax = rebind[Scalar[DT]](grad_output[b, Int(argmax)])

    idx = t
    while idx < DIM:
        var dy = rebind[Scalar[DT]](grad_output[b, idx])
        var dx: Scalar[DT]
        if Int32(idx) == argmin and Int32(idx) == argmax:
            dx = Scalar[DT](0.0)
        elif Int32(idx) == argmin:
            dx = (Gy + dy_argmin - G) * inv_s
        elif Int32(idx) == argmax:
            dx = (dy_argmax - Gy) * inv_s
        else:
            dx = dy * inv_s
        grad_input[b, idx] = dx
        idx += MMN_TPB


struct MinMaxNorm[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    # Cache: per-sample copy of x, re-scanned for min/max/argmin/argmax in
    # vjp. Cheaper than caching indices (no int-as-float fragility).
    var cache_x: Tensor  # [BATCH, DIM]

    def __init__(out self):
        self.cache_x = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert Self.DIM_ > 1, "MinMaxNorm: DIM must be > 1"
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
            self.cache_x.ensure(B * Self.DIM_)
            var in_t = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var out_t = TileTensor(out.data, row_major[B, Self.DIM_]())
            var cache_t = TileTensor(
                self.cache_x.data, row_major[B, Self.DIM_]()
            )
            for b in range(B):
                var x0: Scalar[DT] = in_t[b, 0]
                var min_val = x0
                var max_val = x0
                cache_t[b, 0] = x0
                for i in range(1, Self.DIM_):
                    var v: Scalar[DT] = in_t[b, i]
                    cache_t[b, i] = v
                    if v < min_val:
                        min_val = v
                    if v > max_val:
                        max_val = v
                var s = max_val - min_val
                if s < MMN_EPS:
                    s = MMN_EPS
                var inv_s = Scalar[DT](1.0) / s
                for i in range(Self.DIM_):
                    out_t[b, i] = (cache_t[b, i] - min_val) * inv_s
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_x.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            c.enqueue_function[_min_max_norm_forward_kernel[B, Self.DIM_]](
                in0.tt["gpu", B, Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
                self.cache_x.tt["gpu", B, Self.DIM_](),
                grid_dim=B,
                block_dim=MMN_TPB,
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
            var cache_t = TileTensor(
                self.cache_x.data, row_major[B, Self.DIM_]()
            )
            var go_t = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_t = TileTensor(gin.data, row_major[B, Self.DIM_]())
            for b in range(B):
                var x0: Scalar[DT] = cache_t[b, 0]
                var min_val = x0
                var max_val = x0
                var argmin = 0
                var argmax = 0
                for i in range(1, Self.DIM_):
                    var v: Scalar[DT] = cache_t[b, i]
                    if v < min_val:
                        min_val = v
                        argmin = i
                    if v > max_val:
                        max_val = v
                        argmax = i
                var raw_s = max_val - min_val
                var degenerate = raw_s < MMN_EPS
                if degenerate:
                    for i in range(Self.DIM_):
                        gi_t[b, i] = Scalar[DT](0.0)
                    continue
                var inv_s = Scalar[DT](1.0) / raw_s
                var g_sum: Scalar[DT] = 0.0
                var gy_sum: Scalar[DT] = 0.0
                for i in range(Self.DIM_):
                    var y = (cache_t[b, i] - min_val) * inv_s
                    var dy: Scalar[DT] = go_t[b, i]
                    g_sum += dy
                    gy_sum += dy * y
                var dy_argmin: Scalar[DT] = go_t[b, argmin]
                var dy_argmax: Scalar[DT] = go_t[b, argmax]
                for i in range(Self.DIM_):
                    var dy: Scalar[DT] = go_t[b, i]
                    var dx: Scalar[DT]
                    if i == argmin and i == argmax:
                        dx = Scalar[DT](0.0)
                    elif i == argmin:
                        dx = (gy_sum + dy_argmin - g_sum) * inv_s
                    elif i == argmax:
                        dx = (dy_argmax - gy_sum) * inv_s
                    else:
                        dx = dy * inv_s
                    gi_t[b, i] = dx
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            c.enqueue_function[_min_max_norm_backward_kernel[B, Self.DIM_]](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.cache_x.tt["gpu", B, Self.DIM_](),
                gin.tt["gpu", B, Self.DIM_](),
                grid_dim=B,
                block_dim=MMN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf → no-op). No polyak_from (no Params).
