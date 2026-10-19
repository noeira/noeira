"""QKVToMajor[SEQ, DIM] — token-major QKV → qkv-major (storage surface).

Transformed from legacy `nn.primitives.qkv_to_major` (surface-only change; the
CPU permutation loops and the GPU fwd/bwd kernels are carried over VERBATIM).

A QKV projection `Tokenwise[Linear[DIM, 3*DIM]]` emits token-major output:
per sample `[tok0: q(DIM) k(DIM) v(DIM) | tok1: q k v | …]`, i.e. flat index
`t*3*DIM + g*DIM + d` for group g∈{q,k,v}. `ScaledDotProductAttention` expects
qkv-major: `[all-Q | all-K | all-V]`, flat index `g*SEQ*DIM + t*DIM + d`. This
op rearranges the former into the latter:

    out[g*SEQ*DIM + t*DIM + d] = in[t*3*DIM + g*DIM + d]

IN_DIM == OUT_DIM == 3*SEQ*DIM; no params, no cache. Backward is the inverse
permutation. Conforms to `Module` (param-less → reflection no-op walks).

bf16-FLOW (AMP "Step B"): `QKVToMajor[SEQ, DIM]` is fp32 (unchanged), while
`QKVToMajor[SEQ, DIM, DType.bfloat16]` flows ACTIVATIONS at bf16 (`ACT_DT ==
bfloat16`). This op is a PURE PERMUTE/reshape — no arithmetic, dtype-transparent
— so the bf16 path just moves bf16 elements (read bf16, write bf16, NO cast).
The fwd/bwd permute kernels are dtype-parametric (`ADT`). The fp32 (ACT_DT == DT)
path is byte-for-byte the legacy NoAMP path; the bf16 path is GPU-only.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import RowMajor, Tile


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ───────
# Dtype-parametric (`ADT`): the fp32 path permutes a DT activation, the bf16
# path permutes a bf16 activation. A pure element move — no cast either way.
def _qkv_to_major_fwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int, ADT: DType = DT
](
    src: Tile[ADT, BATCH, 3 * SEQ * DIM],
    dst: Tile[ADT, BATCH, 3 * SEQ * DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (3 * SEQ * DIM)
    var o = gid % (3 * SEQ * DIM)          # qkv-major out index
    var g = o // (SEQ * DIM)
    var rem = o % (SEQ * DIM)
    var t = rem // DIM
    var d = rem % DIM
    dst[b, o] = rebind[Scalar[ADT]](src[b, t * 3 * DIM + g * DIM + d])


def _qkv_to_major_bwd_kernel[
    BATCH: Int, SEQ: Int, DIM: Int, ADT: DType = DT
](
    grad_out: Tile[ADT, BATCH, 3 * SEQ * DIM],
    grad_in: Tile[ADT, BATCH, 3 * SEQ * DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (3 * SEQ * DIM)
    var o = gid % (3 * SEQ * DIM)          # qkv-major out index
    var g = o // (SEQ * DIM)
    var rem = o % (SEQ * DIM)
    var t = rem // DIM
    var d = rem % DIM
    grad_in[b, t * 3 * DIM + g * DIM + d] = rebind[Scalar[ADT]](grad_out[b, o])


struct QKVToMajor[SEQ: Int, DIM: Int, ADT: DType = DT](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=3 * Self.SEQ * Self.DIM)
    # `Array` is not `ImplicitlyCopyable` (Mojo 1.0): indexing the comptime
    # `IN_DIMS` from a runtime context would materialize the whole array.
    comptime IN_DIM0 = 3 * Self.SEQ * Self.DIM
    comptime OUT_DIM = 3 * Self.SEQ * Self.DIM
    # Activation-flow dtype. `QKVToMajor[SEQ, DIM]` = fp32 (ACT_DT == DT, the
    # legacy path); `QKVToMajor[SEQ, DIM, bfloat16]` flows activations at bf16.
    comptime ACT_DT = Self.ADT

    def __init__(out self):
        comptime assert Self.SEQ > 0 and Self.DIM > 0, (
            "QKVToMajor: SEQ, DIM must be > 0"
        )

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "QKVToMajor: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime N = B * Self.OUT_DIM
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here, but the checker won't collapse the opaque
            # `Self.ACT_DT` to `DT` — so rebind the activation refs (sound here).
            # `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(N)
                ref ip = in0d.data
                ref op = outd.data
                for b in range(B):
                    for g in range(3):
                        for t in range(Self.SEQ):
                            for d in range(Self.DIM):
                                op[b * Self.OUT_DIM + g * SD + t * Self.DIM + d] = (
                                    ip[b * Self.IN_DIM0 + t * D3 + g * Self.DIM + d]
                                )
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, N)
                comptime lay = RowMajor[B, 3 * Self.SEQ * Self.DIM]
                comptime total = B * 3 * Self.SEQ * Self.DIM
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime kernel = _qkv_to_major_fwd_kernel[B, Self.SEQ, Self.DIM]
                c.enqueue_function[kernel](
                    in0d.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                    outd.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only; pure bf16 element permute) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow QKVToMajor is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime lay = RowMajor[B, 3 * Self.SEQ * Self.DIM]
            comptime total = B * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_fwd_kernel[
                B, Self.SEQ, Self.DIM, Self.ADT
            ]
            c.enqueue_function[kernel](
                in0.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                out.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
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
        comptime N = B * Self.IN_DIM0
        comptime D3 = 3 * Self.DIM
        comptime SD = Self.SEQ * Self.DIM
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            comptime if target == "cpu":
                gind.ensure(N)
                ref gop = god.data
                ref gip = gind.data
                for b in range(B):
                    for g in range(3):
                        for t in range(Self.SEQ):
                            for d in range(Self.DIM):
                                gip[b * Self.IN_DIM0 + t * D3 + g * Self.DIM + d] = (
                                    gop[b * Self.OUT_DIM + g * SD + t * Self.DIM + d]
                                )
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, N)
                comptime lay = RowMajor[B, 3 * Self.SEQ * Self.DIM]
                comptime total = B * 3 * Self.SEQ * Self.DIM
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime kernel = _qkv_to_major_bwd_kernel[B, Self.SEQ, Self.DIM]
                c.enqueue_function[kernel](
                    god.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                    gind.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only; inverse permute, pure bf16) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow QKVToMajor is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime lay = RowMajor[B, 3 * Self.SEQ * Self.DIM]
            comptime total = B * 3 * Self.SEQ * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _qkv_to_major_bwd_kernel[
                B, Self.SEQ, Self.DIM, Self.ADT
            ]
            c.enqueue_function[kernel](
                grad_output.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                gin.tt["gpu", B, 3 * Self.SEQ * Self.DIM](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
