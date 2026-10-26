"""SwiGLU[HIDDEN] — gated-linear-unit activation core (storage surface).

Transformed from legacy `nn.primitives.swiglu` (surface-only change; the CPU
scalar math and the GPU fwd/bwd kernels are carried over VERBATIM).

The reference Dreamer 4 MLP is `fc_in: Linear(d, 2*hidden)` → split into
`(u, v)` → `u * silu(v)` → `fc_out: Linear(hidden, d)`. This leaf is the
middle (parameter-free) gate, so the full FFN composes as

    Sequential[Linear[d, 2*hidden], SwiGLU[hidden], Linear[hidden, d]]

Input is the concatenated projection `[u ‖ v]` (each `HIDDEN` wide), output is
`u · silu(v)`:

    IN_DIM = 2*HIDDEN, OUT_DIM = HIDDEN
    silu(v)  = v·σ(v),  σ(v) = 1/(1+e^-v)
    silu'(v) = σ(v)·(1 + v·(1-σ(v)))
    out[k]      = u[k] · silu(v[k])
    grad_u[k]   = grad_out[k] · silu(v[k])
    grad_v[k]   = grad_out[k] · u[k] · silu'(v[k])

Param-free, output-cached (`u`, `v` as leaf-owned `Tensor` fields, one buffer
per slab — the attention.mojo storage idiom; backward reads only the cache +
grad_output, not the forward input). Conforms to `Module`.
"""

from std.math import exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import RowMajor, Tile


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ───────
def _swiglu_forward_kernel[
    BATCH: Int, HIDDEN: Int
](
    input: Tile[DT, BATCH, 2 * HIDDEN],
    output: Tile[DT, BATCH, HIDDEN],
    cache_u: Tile[DT, BATCH, HIDDEN],
    cache_v: Tile[DT, BATCH, HIDDEN],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    var u = rebind[Scalar[DT]](input[b, k])
    var v = rebind[Scalar[DT]](input[b, HIDDEN + k])
    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
    cache_u[b, k] = u
    cache_v[b, k] = v
    output[b, k] = u * (v * s)


def _swiglu_backward_kernel[
    BATCH: Int, HIDDEN: Int
](
    grad_output: Tile[DT, BATCH, HIDDEN],
    cache_u: Tile[DT, BATCH, HIDDEN],
    cache_v: Tile[DT, BATCH, HIDDEN],
    grad_input: Tile[DT, BATCH, 2 * HIDDEN],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * HIDDEN:
        return
    var b = idx // HIDDEN
    var k = idx % HIDDEN
    var go = rebind[Scalar[DT]](grad_output[b, k])
    var u = rebind[Scalar[DT]](cache_u[b, k])
    var v = rebind[Scalar[DT]](cache_v[b, k])
    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
    var sv = v * s
    var sp = s * (Scalar[DT](1) + v * (Scalar[DT](1) - s))
    grad_input[b, k] = go * sv
    grad_input[b, HIDDEN + k] = go * u * sp


struct SwiGLU[HIDDEN: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=2 * Self.HIDDEN)
    # `Array` is not `ImplicitlyCopyable` (Mojo 1.0): indexing the comptime
    # `IN_DIMS` from a runtime context would materialize the whole array.
    comptime IN_DIM0 = 2 * Self.HIDDEN
    comptime OUT_DIM = Self.HIDDEN

    # Output-caching: leaf-owned Tensor slabs (one buffer each), lazily sized.
    var cache_u: Tensor
    var cache_v: Tensor

    def __init__(out self):
        self.cache_u = Tensor()
        self.cache_v = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "SwiGLU: target must be 'cpu' or 'gpu'"
        )
        comptime if target != "cpu":
            if not ctx:
                raise Error("SwiGLU.make[target='gpu']: ctx required")
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime CN = B * Self.HIDDEN
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            self.cache_u.ensure(CN)
            self.cache_v.ensure(CN)
            ref ip = in0.data
            ref op = out.data
            ref cu = self.cache_u.data
            ref cv = self.cache_v.data
            for b in range(B):
                for k in range(Self.HIDDEN):
                    var u = ip[b * Self.IN_DIM0 + k]
                    var v = ip[b * Self.IN_DIM0 + Self.HIDDEN + k]
                    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
                    cu[b * Self.HIDDEN + k] = u
                    cv[b * Self.HIDDEN + k] = v
                    op[b * Self.OUT_DIM + k] = u * (v * s)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            self.cache_u.ensure_gpu(c, CN)
            self.cache_v.ensure_gpu(c, CN)
            comptime lin = RowMajor[B, 2 * Self.HIDDEN]
            comptime lout = RowMajor[B, Self.HIDDEN]
            comptime n_blocks = (CN + TPB - 1) // TPB
            comptime kernel = _swiglu_forward_kernel[B, Self.HIDDEN]
            c.enqueue_function[kernel](
                in0.tt["gpu", B, 2 * Self.HIDDEN](),
                out.tt["gpu", B, Self.HIDDEN](),
                self.cache_u.tt["gpu", B, Self.HIDDEN](),
                self.cache_v.tt["gpu", B, Self.HIDDEN](),
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
        # forward_input unused — this leaf is output-caching (reads only the
        # cache + grad_output).
        comptime CN = B * Self.HIDDEN
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_DIM0)
            ref gop = grad_output.data
            ref gip = gin.data
            ref cu = self.cache_u.data
            ref cv = self.cache_v.data
            for b in range(B):
                for k in range(Self.HIDDEN):
                    var g = gop[b * Self.OUT_DIM + k]
                    var u = cu[b * Self.HIDDEN + k]
                    var v = cv[b * Self.HIDDEN + k]
                    var s = Scalar[DT](1) / (Scalar[DT](1) + exp(-v))
                    var sv = v * s
                    var sp = s * (Scalar[DT](1) + v * (Scalar[DT](1) - s))
                    gip[b * Self.IN_DIM0 + k] = g * sv
                    gip[b * Self.IN_DIM0 + Self.HIDDEN + k] = g * u * sp
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_DIM0)
            comptime lin = RowMajor[B, 2 * Self.HIDDEN]
            comptime lout = RowMajor[B, Self.HIDDEN]
            comptime n_blocks = (CN + TPB - 1) // TPB
            comptime kernel = _swiglu_backward_kernel[B, Self.HIDDEN]
            c.enqueue_function[kernel](
                grad_output.tt["gpu", B, Self.HIDDEN](),
                self.cache_u.tt["gpu", B, Self.HIDDEN](),
                self.cache_v.tt["gpu", B, Self.HIDDEN](),
                gin.tt["gpu", B, 2 * Self.HIDDEN](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults.
