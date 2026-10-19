"""Clamp[DIM] — element-wise hard clamp to [min_val, max_val] (storage surface).

Transformed from legacy `nn.primitives.Clamp` (surface-only change). The CPU SIMD
loops and the two GPU kernels (forward / backward) are carried over verbatim.

Forward:  out[b, d] = max(min_val, min(max_val, x[b, d]))
Backward: grad_in[b, d] = grad_out[b, d] if min_val < x < max_val else 0

`min_val` / `max_val` are runtime per-instance scalars (mirror `Scale`); set via
`set_min_max`. No cache — the backward re-reads `x` from `forward_input` (the
orchestrator-owned input storage, kept live across the call by `TensorRefs`).

Used by `DDPGTargetYBlock` (1 instance, action clamp) and `TD3TargetYBlock`
(2 instances: noise clip + smoothed-action clamp).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _clamp_forward_kernel[
    N: Int,
](
    input: Tile[DT, N],
    output: Tile[DT, N],
    min_val: Scalar[DT],
    max_val: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](input[idx])
        var y = x
        if y > max_val:
            y = max_val
        if y < min_val:
            y = min_val
        output[idx] = y


def _clamp_backward_kernel[
    N: Int,
](
    grad_output: Tile[DT, N],
    input: Tile[DT, N],
    grad_input: Tile[DT, N],
    min_val: Scalar[DT],
    max_val: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](input[idx])
        var zero: Scalar[DT] = 0.0
        if x < min_val or x > max_val:
            grad_input[idx] = zero
        else:
            grad_input[idx] = rebind[Scalar[DT]](grad_output[idx])


struct Clamp[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var min_val: Scalar[DT]
    var max_val: Scalar[DT]

    def __init__(out self):
        self.min_val = Scalar[DT](-1.0)
        self.max_val = Scalar[DT](1.0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        var cl = Self()
        return cl^

    def set_min_max(mut self, min_val: Scalar[DT], max_val: Scalar[DT]):
        self.min_val = min_val
        self.max_val = max_val

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Named-attribute setter so a `ComputeGraph` can configure the bounds
        via `set_node_attr["node", "min_val"/"max_val"]` (mirrors `Scale`).
        Needed by the TD3 target-y graph (noise clip + action clamp)."""
        comptime if ATTR == "min_val":
            self.min_val = value
        elif ATTR == "max_val":
            self.max_val = value

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(N)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var min_v = SIMD[DT, CPU_SIMD_W](self.min_val)
            var max_v = SIMD[DT, CPU_SIMD_W](self.max_val)
            var k = 0
            while k + CPU_SIMD_W <= N:
                var v = in_p.unsafe_load[width=CPU_SIMD_W](k)
                # min(max(v, min_v), max_v)
                v = v.gt(min_v).select(v, min_v)
                v = v.lt(max_v).select(v, max_v)
                out_p.unsafe_store(k, v)
                k += CPU_SIMD_W
            while k < N:
                var v = in_p[unsafe_offset=k]
                if v > self.max_val:
                    v = self.max_val
                if v < self.min_val:
                    v = self.min_val
                out_p[unsafe_offset=k] = v
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            c.enqueue_function[_clamp_forward_kernel[N]](
                in0.tt["gpu", N](),
                out.tt["gpu", N](),
                self.min_val,
                self.max_val,
                grid_dim=n_blocks,
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
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.DIM_
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var x_p = fin.data.unsafe_ptr()
            var min_v = SIMD[DT, CPU_SIMD_W](self.min_val)
            var max_v = SIMD[DT, CPU_SIMD_W](self.max_val)
            var zero = SIMD[DT, CPU_SIMD_W](0.0)
            var k = 0
            while k + CPU_SIMD_W <= N:
                var x = x_p.unsafe_load[width=CPU_SIMD_W](k)
                var go = go_p.unsafe_load[width=CPU_SIMD_W](k)
                # in_range = (x > min_v) AND (x < max_v); else zero.
                var in_range = x.gt(min_v) & x.lt(max_v)
                gi_p.unsafe_store(k, in_range.select(go, zero))
                k += CPU_SIMD_W
            while k < N:
                var x = x_p[unsafe_offset=k]
                if x < self.min_val or x > self.max_val:
                    gi_p[unsafe_offset=k] = Scalar[DT](0.0)
                else:
                    gi_p[unsafe_offset=k] = go_p[unsafe_offset=k]
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            c.enqueue_function[_clamp_backward_kernel[N]](
                grad_output.tt["gpu", N](),
                fin.tt["gpu", N](),
                gin.tt["gpu", N](),
                self.min_val,
                self.max_val,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
