"""Flatten[DIM] — identity passthrough Module (storage surface).

Transformed from legacy `nn.primitives.Flatten`. From the Module's standpoint a
flatten is the identity: `[BATCH, DIM]` in == `[BATCH, DIM]` out (the shape
change is purely in the caller's view layout). No params, no cache; backward is
the same identity copy. Lets `Sequential[Conv2D, ReLU, …, Flatten, Linear, …]`
compose without orchestrator glue.

Channels-last (NHWC) note: Flatten is LAYOUT-AGNOSTIC — it carries no LAYOUT
param. It's a pure identity copy, so an NHWC conv output flattens to an
`[h,w,c]`-ordered vector and an NCHW output to `[c,h,w]`-ordered, with no
reordering here. The downstream `Linear` simply learns its weights in whichever
order it's fed; the only consequence is that a checkpoint trained under one
layout has its first-dense weight columns permuted relative to the other (handled
by the channels_last checkpoint migration, not by this Module).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, CPU_SIMD_W, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


def _flatten_copy_kernel[
    N: Int, ADT: DType = DT
](
    src: Tile[ADT, N],
    dst: Tile[ADT, N],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[ADT]](src[idx])


struct Flatten[DIM_: Int, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype (AMP). Flatten is dtype-TRANSPARENT — a pure
    # identity copy with no math/cast — so it carries ACT_DT through unchanged.
    # ACT_DT == DT (default) → byte-identical to the fp32 path.
    comptime ACT_DT = Self.ADT

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            var sp = in0.data.unsafe_ptr()
            var dp = out.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= N:
                dp.unsafe_store(k, sp.unsafe_load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < N:
                dp[unsafe_offset=k] = sp[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_flatten_copy_kernel[N, Self.ACT_DT]](
                in0.tt["gpu", N](),
                out.tt["gpu", N](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            var sp = grad_output.data.unsafe_ptr()
            var dp = gin.data.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= N:
                dp.unsafe_store(k, sp.unsafe_load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < N:
                dp[unsafe_offset=k] = sp[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime nblk = (N + TPB - 1) // TPB
            c.enqueue_function[_flatten_copy_kernel[N, Self.ACT_DT]](
                grad_output.tt["gpu", N](),
                gin.tt["gpu", N](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
