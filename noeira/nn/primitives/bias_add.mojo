"""BiasAdd[DIM] — learnable broadcast bias (storage surface).

Transformed from legacy `nn.primitives.BiasAdd` (surface-only change). The CPU
loops + the three GPU kernels (forward / grad-input copy / grad-bias reduce) are
carried over verbatim.

    out[b, i] = in[b, i] + bias[i]

A single `DIM`-vector parameter added to every row. Used as a learnable
(position) embedding in GPT/ViT: instantiated at the full sequence width
(`seq_len*embed_dim`) so each position gets its own additive bias.

IN_DIM == OUT_DIM == DIM. No cache: backward needs neither the input nor the
output.

  * grad_in = grad_out (identity — bias add is +1 w.r.t. input)
  * grad_bias[i] += sum_b grad_out[b, i]   (reduce over batch)

`bias` is weight-decay-exempt (biases shouldn't decay). Init: β=0
(`INIT.init_bias` zero-fills, matching the legacy β handling).

bf16-FLOW (AMP "Step B"): `BiasAdd[DIM]` is fp32 (unchanged), while
`BiasAdd[DIM, DType.bfloat16]` flows ACTIVATIONS at bf16 (`ACT_DT == bfloat16`).
The owned `bias` Param stays fp32 (master); only a CACHED bf16 bias (`b_a`, cast
each forward — the bias is small) is low-precision. Forward = bf16 in + (fp32
bias→bf16) → bf16 out. Backward: grad_in = grad_out (bf16 identity); grad_bias +=
colsum(grad_out) accumulates the bf16 grad into the FP32 master grad (each element
cast to DT, fp32 accumulator). The fwd/dbias kernels are dtype-parametric (`ADT`).
The fp32 (ACT_DT == DT) path is byte-for-byte the legacy NoAMP path; the bf16 path
is GPU-only.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


comptime BA_TPB: Int = 128
comptime BF16 = DType.bfloat16


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
# `_bias_add_fwd_kernel` is dtype-parametric: the fp32 path runs at DT, the
# bf16-flow path at bfloat16 (input/output/cached-bias all at ADT).
def _bias_add_fwd_kernel[
    BATCH: Int, DIM: Int, ADT: DType = DT
](
    input: Tile[ADT, BATCH, DIM],
    bias: Tile[ADT, DIM],
    output: Tile[ADT, BATCH, DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * DIM
    if gid >= total:
        return
    var b = gid // DIM
    var i = gid % DIM
    output[b, i] = rebind[Scalar[ADT]](input[b, i]) + rebind[Scalar[ADT]](
        bias[i]
    )


# grad_in = grad_out: device→device flat copy. Dtype-parametric (`ADT`):
# bf16-flow copies a bf16 grad straight through (identity).
def _bias_add_copy_kernel[
    N: Int, ADT: DType = DT
](
    src: Tile[ADT, N],
    dst: Tile[ADT, N],
):
    var gid = Int(global_idx.x)
    if gid < N:
        dst[gid] = rebind[Scalar[ADT]](src[gid])


# grad_bias[i] += sum_b grad_out[b, i]. Dtype-parametric on the grad_output
# activation (`ADT`): the bf16 path reads a bf16 `grad_output` and accumulates
# into the FP32 master `grad_bias` (each element cast to DT before summing — the
# accumulator/reduction stays fp32).
def _bias_add_dbias_kernel[
    BATCH: Int, DIM: Int, ADT: DType = DT
](
    grad_output: Tile[ADT, BATCH, DIM],
    grad_bias: Tile[DT, DIM],
):
    # One block per DIM column; threads reduce over the batch.
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_db += rebind[Scalar[ADT]](grad_output[bi, col]).cast[DT]()
        bi += BA_TPB
    var total_db = block.sum[block_size=BA_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_bias[col] = rebind[Scalar[DT]](grad_bias[col]) + total_db[0]


def _cast_f2b_kernel[
    N: Int
](
    src: Tile[DT, N],
    dst: Tile[BF16, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[BF16]()


struct BiasAdd[DIM_: Int, ADT: DType = DT](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype. `BiasAdd[DIM]` = fp32 (ACT_DT == DT, the legacy
    # path); `BiasAdd[DIM, bfloat16]` flows activations at bf16.
    comptime ACT_DT = Self.ADT

    var bias: Param["bias", False, Self.DIM_]
    # bf16-flow cached bias (lazy; ACT_DT == bf16 && target == "gpu" only). The
    # master `bias` Param stays fp32; `b_a` is recast from `bias.val` each forward
    # (the bias is small — a per-forward cast is fine, no version-gating needed).
    var b_a: TensorImpl[Self.ADT]

    def __init__(out self):
        self.bias = Param["bias", False, Self.DIM_]()
        self.b_a = TensorImpl[Self.ADT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var ba = Self()
        ba.bias = Param["bias", False, Self.DIM_].make[target](ctx)
        INIT.init_bias[target](ba.bias.val, Self.DIM_, ctx)
        return ba^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here, but the checker won't collapse the opaque
            # `Self.ACT_DT` to `DT` — so rebind the activation refs (sound here).
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.DIM_)
                var in_v = TileTensor(in0d.data, row_major[B, Self.DIM_]())
                var out_v = TileTensor(outd.data, row_major[B, Self.DIM_]())
                var bias_v = TileTensor(
                    self.bias.val.data, row_major[Self.DIM_]()
                )
                for b in range(B):
                    for i in range(Self.DIM_):
                        out_v[b, i] = in_v[b, i] + bias_v[i]
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.DIM_)
                comptime l2d = RowMajor[B, Self.DIM_]
                comptime ld = RowMajor[Self.DIM_]
                comptime total = B * Self.DIM_
                comptime n_blocks = (total + TPB - 1) // TPB
                c.enqueue_function[_bias_add_fwd_kernel[B, Self.DIM_]](
                    in0d.tt["gpu", B, Self.DIM_](),
                    self.bias.val.tt["gpu", Self.DIM_](),
                    outd.tt["gpu", B, Self.DIM_](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow BiasAdd is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime ld = RowMajor[Self.DIM_]
            comptime total = B * Self.DIM_
            comptime n_blocks = (total + TPB - 1) // TPB
            # bias: cheap per-forward DT→bf16 cast (input ALREADY bf16).
            self.b_a.ensure_gpu(c, Self.DIM_)
            c.enqueue_function[_cast_f2b_kernel[Self.DIM_]](
                self.bias.val.tt["gpu", Self.DIM_](),
                self.b_a.tt["gpu", Self.DIM_](),
                grid_dim=(Self.DIM_ + 255) // 256,
                block_dim=256,
            )
            c.enqueue_function[_bias_add_fwd_kernel[B, Self.DIM_, Self.ADT]](
                in0.tt["gpu", B, Self.DIM_](),
                self.b_a.tt["gpu", Self.DIM_](),
                out.tt["gpu", B, Self.DIM_](),
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
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            comptime if target == "cpu":
                gind.ensure(B * Self.DIM_)
                var go_v = TileTensor(god.data, row_major[B, Self.DIM_]())
                var gi_v = TileTensor(gind.data, row_major[B, Self.DIM_]())
                # grad_in = grad_out (identity).
                for b in range(B):
                    for i in range(Self.DIM_):
                        gi_v[b, i] = go_v[b, i]
                # grad_bias[i] += sum_b grad_out[b, i]
                var gb_v = TileTensor(
                    self.bias.grd.data, row_major[Self.DIM_]()
                )
                for b in range(B):
                    for i in range(Self.DIM_):
                        gb_v[i] += go_v[b, i]
            else:
                var c = ctx.value()
                comptime l2d = RowMajor[B, Self.DIM_]
                comptime ld = RowMajor[Self.DIM_]
                comptime total = B * Self.DIM_
                comptime lflat = RowMajor[total]
                gind.ensure_gpu(c, B * Self.DIM_)
                # grad_in = grad_out: device→device copy via flat kernel.
                comptime n_blocks = (total + TPB - 1) // TPB
                c.enqueue_function[_bias_add_copy_kernel[total]](
                    god.tt["gpu", total](),
                    gind.tt["gpu", total](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                # grad_bias[i] += sum_b grad_out[b, i] (one block per column).
                c.enqueue_function[_bias_add_dbias_kernel[B, Self.DIM_]](
                    god.tt["gpu", B, Self.DIM_](),
                    self.bias.grd.tt["gpu", Self.DIM_](),
                    grid_dim=Self.DIM_,
                    block_dim=BA_TPB,
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow BiasAdd is GPU-only"
            var c = ctx.value()
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime ld = RowMajor[Self.DIM_]
            comptime total = B * Self.DIM_
            comptime lflat = RowMajor[total]
            gin.ensure_gpu(c, B * Self.DIM_)
            # grad_in = grad_out: bf16 identity copy (grad flows at bf16).
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_bias_add_copy_kernel[total, Self.ADT]](
                grad_output.tt["gpu", total](),
                gin.tt["gpu", total](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            # grad_bias[i] += sum_b grad_out[b, i]: bf16 grad → fp32 master grad
            # (each element cast to DT, fp32 accumulator).
            c.enqueue_function[_bias_add_dbias_kernel[B, Self.DIM_, Self.ADT]](
                grad_output.tt["gpu", B, Self.DIM_](),
                self.bias.grd.tt["gpu", Self.DIM_](),
                grid_dim=Self.DIM_,
                block_dim=BA_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `bias` Param).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.DIM_](
            self.bias.val, src.bias.val, tau, ctx
        )
