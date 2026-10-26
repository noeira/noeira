"""SinusoidalPosAddBT[T, S, D, SCALE] — sinusoidal positions at B·T layout (storage).

Transformed from legacy `nn.primitives.SinusoidalPosAddBT` (surface-only change).
The bias table (`build_sinusoid_bias`) and the two GPU kernels are carried over
verbatim.

`SinusoidalPosAdd` assumes nn-BATCH = B and a per-sample (T·S·D) grid. The
Dreamer 4 encoder/decoder instead run at nn-BATCH = B·T (one frame per
sample, sequence S), where the additive position `pos_t[t] + pos_s[s]` varies
with `t = batch_index % T`. This leaf adds it at that layout:

    out[bt, s*D + j] = in[bt, s*D + j] + bias[(bt % T)*S*D + s*D + j]

where `bias` is the same precomputed `T*S*D` table as `SinusoidalPosAdd`,
held in an owned `Tensor` (CPU data + device dev). Param-free; identity vjp
(the bias is constant). Conforms to `Module`.
"""

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
from .sinusoidal_pos import build_sinusoid_bias


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _pos_bt_add_kernel[
    BATCH: Int, T: Int, SD: Int
](
    input: Tile[DT, BATCH, SD],
    bias: Tile[DT, T * SD],
    output: Tile[DT, BATCH, SD],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * SD:
        return
    var bt = idx // SD
    var local = idx % SD
    var t = bt % T
    output.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](input.ptr[unsafe_offset=idx]) + rebind[Scalar[DT]](
        bias.ptr[unsafe_offset=t * SD + local]
    )


def _pos_bt_copy_kernel[BATCH: Int, SD: Int](
    src: Tile[DT, BATCH, SD],
    dst: Tile[DT, BATCH, SD],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * SD:
        dst.ptr[unsafe_offset=idx] = rebind[Scalar[DT]](src.ptr[unsafe_offset=idx])


struct SinusoidalPosAddBT[
    T_: Int, S_: Int, D_: Int, SCALE_: Bool = False
](Module):
    comptime ARITY: Int = 1
    comptime SD: Int = Self.S_ * Self.D_
    comptime IN_DIMS = Array[Int, 1](fill=Self.SD)
    comptime OUT_DIM = Self.SD

    var bias: Tensor  # [T*S*D] (CPU data + device dev)

    def __init__(out self):
        self.bias = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SinusoidalPosAddBT: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime N = Self.T_ * Self.SD
        var b = build_sinusoid_bias[Self.T_, Self.S_, Self.D_, Self.SCALE_]()
        m.bias = Tensor.alloc(N)
        for i in range(N):
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
            out.ensure(B * Self.SD)
            var inp = TileTensor(in0.data, row_major[B, Self.SD]())
            var out_t = TileTensor(out.data, row_major[B, Self.SD]())
            var bp = self.bias.data.unsafe_ptr()
            for bt in range(B):
                var t = bt % Self.T_
                for i in range(Self.SD):
                    out_t[bt, i] = inp[bt, i] + bp[unsafe_offset=t * Self.SD + i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.SD)
            comptime n_blocks = (B * Self.SD + TPB - 1) // TPB
            c.enqueue_function[_pos_bt_add_kernel[B, Self.T_, Self.SD]](
                in0.tt["gpu", B, Self.SD](),
                self.bias.tt["gpu", Self.T_ * Self.SD](),
                out.tt["gpu", B, Self.SD](),
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
            gin.ensure(B * Self.SD)
            var go = TileTensor(grad_output.data, row_major[B, Self.SD]())
            var gi = TileTensor(gin.data, row_major[B, Self.SD]())
            for bt in range(B):
                for i in range(Self.SD):
                    gi[bt, i] = go[bt, i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.SD)
            comptime n_blocks = (B * Self.SD + TPB - 1) // TPB
            c.enqueue_function[_pos_bt_copy_kernel[B, Self.SD]](
                grad_output.tt["gpu", B, Self.SD](),
                gin.tt["gpu", B, Self.SD](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields; `bias` is a plain Tensor).
