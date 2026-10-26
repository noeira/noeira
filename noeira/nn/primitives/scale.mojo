"""Scale[DIM] — multiplies by a runtime scalar `multiplier` (storage surface).

Transformed from legacy `nn.primitives.Scale` (surface-only change). The CPU SIMD
loop and the GPU `_scale_kernel` are carried over verbatim.

Forward:  out = m·in
Backward: grad_in = m·grad_out

`multiplier` is a public mut field the caller updates per-step (SAC tracks the
moving α this way); `set_multiplier` mirrors the legacy `set_attr["multiplier"]`.
No params, no cache. Conforms to `Module`.

Device-resident multiplier (`multiplier_buf` / `_scale_dev_kernel`): when wired
via `set_attr_buf["multiplier"]`, the GPU forward/vjp read the scale factor from
a length-1 `DeviceBuffer` instead of a baked-scalar kernel arg. SAC uses this to
let its on-device α (a sub-buffer view of the ScalarAdam state) update each step
without breaking CUDA-graph capture. CPU always uses the host `multiplier`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile, ImmTile, tile

from noeira.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ── GPU kernel (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _scale_kernel[
    N: Int,
](
    input: Tile[DT, N],
    output: Tile[DT, N],
    multiplier: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * multiplier


def _scale_dev_kernel[
    N: Int,
](
    input: Tile[DT, N],
    output: Tile[DT, N],
    multiplier: ImmTile[DT, 1],
):
    # Device-resident multiplier variant — reads the scale factor from
    # `mptr[0]` instead of a baked scalar arg, so the value can be updated by
    # another GPU kernel (SAC's on-device alpha) without breaking CUDA-graph
    # capture. Every thread reads the same `mptr[0]`.
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](input[idx]) * multiplier[0]


struct Scale[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    @staticmethod
    def display_label() -> String:
        return String("Scale")

    var multiplier: Scalar[DT]
    # Optional device-resident multiplier source. When set (via `set_attr_buf`),
    # the GPU forward/vjp read the scale factor from `multiplier_ptr[0]` instead
    # of baking `multiplier` into the kernel args — required for CUDA-graph
    # capture when another GPU kernel (SAC's on-device alpha) updates the value
    # each step. None -> baked-scalar path (bit-identical). CPU always uses the
    # host `multiplier`.
    var multiplier_buf: Optional[DeviceBuffer[DT]]

    def __init__(out self):
        self.multiplier = Scalar[DT](1.0)
        self.multiplier_buf = None

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        var s = Self()
        return s^

    def set_multiplier(mut self, value: Scalar[DT]):
        self.multiplier = value

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "multiplier":
            self.multiplier = value

    def set_attr_buf[
        ATTR: StaticString
    ](mut self, buf: DeviceBuffer[DT]):
        comptime if ATTR == "multiplier":
            self.multiplier_buf = buf

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
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            var k = 0
            while k + CPU_SIMD_W <= N:
                out_p.unsafe_store(k, in_p.unsafe_load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                out_p[unsafe_offset=k] = in_p[unsafe_offset=k] * self.multiplier
                k += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            if self.multiplier_buf:
                var mptr = tile[1](self.multiplier_buf.value())
                c.enqueue_function[_scale_dev_kernel[N]](
                    in0.tt["gpu", N](),
                    out.tt["gpu", N](),
                    mptr,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
            else:
                c.enqueue_function[_scale_kernel[N]](
                    in0.tt["gpu", N](),
                    out.tt["gpu", N](),
                    self.multiplier,
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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(N)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var m_v = SIMD[DT, CPU_SIMD_W](self.multiplier)
            var k = 0
            while k + CPU_SIMD_W <= N:
                gi_p.unsafe_store(k, go_p.unsafe_load[width=CPU_SIMD_W](k) * m_v)
                k += CPU_SIMD_W
            while k < N:
                gi_p[unsafe_offset=k] = go_p[unsafe_offset=k] * self.multiplier
                k += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            if self.multiplier_buf:
                var mptr = tile[1](self.multiplier_buf.value())
                c.enqueue_function[_scale_dev_kernel[N]](
                    grad_output.tt["gpu", N](),
                    gin.tt["gpu", N](),
                    mptr,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
            else:
                c.enqueue_function[_scale_kernel[N]](
                    grad_output.tt["gpu", N](),
                    gin.tt["gpu", N](),
                    self.multiplier,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
