"""SinusoidalPosAdd[T, S, D, SCALE] — additive 2D sinusoidal positions (storage).

Transformed from legacy `nn.primitives.SinusoidalPosAdd` (surface-only change).
The sinusoid math (`build_sinusoid_bias` / `_sinusoid_val`) and the two GPU
kernels are carried over verbatim.

Dreamer 4 adds separable time + space sinusoidal positions to every token of
the (T, S, D) grid:

    pos[t, s, j] = sinusoid(t)[j] + sinusoid(s)[j]      (optionally / sqrt(D))
    out = tokens + pos

The bias depends only on (T, S, D), so it is precomputed once at `make` into a
`T*S*D` owned `Tensor` (CPU `data` + device `dev`, uploaded on GPU) and added in
forward. Backward is the identity (the bias is a constant, no params):

    grad_in = grad_out

Per-sample flat layout is row-major `(T, S, D)`: token (t, s) at offset
`(t*S + s)*D`. `SCALE=True` divides the summed positions by sqrt(D).
No params. Conforms to `Module`.
"""

from std.math import exp, sin, cos, sqrt, log
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime SINUSOID_BASE: Float64 = 10000.0


def _sinusoid_val(pos: Int, j: Int, d: Int) -> Float64:
    """sinusoid_table(n, d)[pos, j] (single entry)."""
    var k = Float64(j // 2)
    var div = exp(-(2.0 * k) / Float64(d) * log(SINUSOID_BASE))
    var ang = Float64(pos) * div
    return sin(ang) if (j % 2) == 0 else cos(ang)


def build_sinusoid_bias[
    T: Int, S: Int, D: Int, SCALE: Bool
]() -> List[Scalar[DT]]:
    """Precompute the additive (T*S*D) position bias."""
    var scale = (1.0 / sqrt(Float64(D))) if SCALE else 1.0
    var bias = List[Scalar[DT]]()
    for t in range(T):
        for s in range(S):
            for j in range(D):
                var v = _sinusoid_val(t, j, D) + _sinusoid_val(s, j, D)
                bias.append(Scalar[DT](v * scale))
    return bias^


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _pos_add_kernel[
    BATCH: Int, N: Int
](
    input: Tile[DT, BATCH, N],
    bias: Tile[DT, N],
    output: Tile[DT, BATCH, N],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    var i = idx % N
    output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](input.ptr[unsafe_offset=idx]) + rebind[Scalar[DT]](
        bias.ptr[unsafe_offset=i]
    )


def _copy_kernel[
    BATCH: Int, N: Int
](
    src: Tile[DT, BATCH, N],
    dst: Tile[DT, BATCH, N],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * N:
        return
    dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](src.ptr[unsafe_offset=idx])


struct SinusoidalPosAdd[T_: Int, S_: Int, D_: Int, SCALE_: Bool = False](Module):
    comptime ARITY: Int = 1
    comptime N: Int = Self.T_ * Self.S_ * Self.D_
    comptime IN_DIMS = Array[Int, 1](fill=Self.N)
    comptime OUT_DIM = Self.N

    var bias: Tensor  # [T*S*D] (CPU data + device dev)

    def __init__(out self):
        self.bias = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SinusoidalPosAdd: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        var b = build_sinusoid_bias[Self.T_, Self.S_, Self.D_, Self.SCALE_]()
        m.bias = Tensor.alloc(Self.N)
        for i in range(Self.N):
            m.bias.data[i] = b[i]
        comptime if target != "cpu":
            m.bias.upload(ctx.value())
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
            out.ensure(B * Self.N)
            var inp = TileTensor(in0.data, row_major[B, Self.N]())
            var out_t = TileTensor(out.data, row_major[B, Self.N]())
            var bp = self.bias.data.unsafe_ptr()
            for b in range(B):
                for i in range(Self.N):
                    out_t[b, i] = inp[b, i] + bp[unsafe_offset=i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.N)
            comptime n_blocks = (B * Self.N + TPB - 1) // TPB
            c.enqueue_function[_pos_add_kernel[B, Self.N]](
                in0.tt["gpu", B, Self.N](),
                self.bias.tt["gpu", Self.N](),
                out.tt["gpu", B, Self.N](),
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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.N)
            var go = TileTensor(grad_output.data, row_major[B, Self.N]())
            var gi = TileTensor(gin.data, row_major[B, Self.N]())
            for b in range(B):
                for i in range(Self.N):
                    gi[b, i] = go[b, i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.N)
            comptime n_blocks = (B * Self.N + TPB - 1) // TPB
            c.enqueue_function[_copy_kernel[B, Self.N]](
                grad_output.tt["gpu", B, Self.N](),
                gin.tt["gpu", B, Self.N](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields; `bias` is a plain Tensor).
