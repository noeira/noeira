"""Elementwise[DIM, OP] — generic elementwise activation on the storage surface.

The storage-surface twin of legacy `nn.primitives.Elementwise[DIM, OP]`. Reuses
the legacy `ElementOp` trait + `ops/` structs VERBATIM (they depend only on
`DT`), so the per-lane math is bit-identical. Every concrete activation is a
one-line alias:

    comptime ReLU    = Elementwise[DIM, ReLUOp]      # see activations.mojo
    comptime Tanh    = Elementwise[DIM, TanhOp]
    comptime Sigmoid = Elementwise[DIM, SigmoidOp]

KEY simplification vs legacy: the storage `vjp` receives `forward_input` (x)
explicitly (invariant §3.1), so there is NO cache field and NO
`_cached_input_ptr` alias. For output-cache ops (`owns_cache=True`, e.g. Tanh)
backward recomputes `y = OP.forward(x)` then `OP.backward(y, go)` — bit-
identical to having cached `y`, because `y` is a pure function of `x`. For
input-cache ops (`owns_cache=False`, e.g. ReLU) backward is `OP.backward(x, go)`.

CPU uses the SIMD ops over a tracked `.data.unsafe_ptr()` (origin = the list,
NOT the wildcard); GPU uses one kernel per direction parameterised on `OP`.

bf16-FLOW (AMP "Step B"): `Elementwise[DIM, OP]` is fp32 (ACT_DT == DT, the
legacy NoAMP path, byte-identical); `Elementwise[DIM, OP, DType.bfloat16]` is
fp32-INTERNAL but flows its I/O activations at bf16. The `ElementOp` math is
fp32-only (its API takes `Scalar[DT]`), so the GPU kernels cast each bf16
activation element UP to fp32 (`forward_input`/`grad_output` on read), call
`OP.forward_scalar`/`OP.backward_scalar`, then cast the result DOWN to bf16 on
write (`o`/`gi`) — elementwise activations (ReLU/Tanh/GELU/…) are numerically
fine across that round-trip (mirrors `LinearAct`). bf16-flow is GPU-only. The
default `ADT = DT` reproduces the legacy fp32 kernels byte-for-byte.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, CPU_SIMD_W, TPB
from noeira.nn.core.element_op import ElementOp
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


# ── GPU kernels (OP supplies the math via comptime) ─────────────────────
# Dtype-parametric on the ACTIVATION dtype (`ADT`): the fp32 path runs at DT
# (default `ADT = DT` → byte-identical legacy kernels); the bf16-flow path holds
# the I/O activations at bfloat16, casting UP to fp32 for the (fp32-only) OP math
# and back DOWN to bf16 on store.
def _ew_fwd_kernel[
    M: Int, OP: ElementOp, ADT: DType = DT
](
    x: Tile[ADT, M],
    o: Tile[ADT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        comptime if ADT == DT:
            o[i] = rebind[Scalar[ADT]](
                OP.forward_scalar(rebind[Scalar[DT]](x[i]))
            )
        else:
            # bf16-flow: lift to fp32 for the activation math, store back at bf16.
            var xv = rebind[Scalar[ADT]](x[i]).cast[DT]()
            o[i] = OP.forward_scalar(xv).cast[ADT]()


def _ew_bwd_kernel[
    M: Int, OP: ElementOp, ADT: DType = DT
](
    x: Tile[ADT, M],
    go: Tile[ADT, M],
    gi: Tile[ADT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        comptime if ADT == DT:
            var xv = rebind[Scalar[DT]](x[i])
            var gov = rebind[Scalar[DT]](go[i])
            comptime if OP.owns_cache:
                # output-cache op: recompute y = f(x), then gi = f'(y)·go.
                gi[i] = rebind[Scalar[ADT]](
                    OP.backward_scalar(OP.forward_scalar(xv), gov)
                )
            else:
                gi[i] = rebind[Scalar[ADT]](OP.backward_scalar(xv, gov))
        else:
            # bf16-flow: cast x + go UP to fp32 for the (fp32-only) OP math, then
            # cast the gated grad DOWN to bf16 on store.
            var xv = rebind[Scalar[ADT]](x[i]).cast[DT]()
            var gov = rebind[Scalar[ADT]](go[i]).cast[DT]()
            comptime if OP.owns_cache:
                gi[i] = OP.backward_scalar(OP.forward_scalar(xv), gov).cast[ADT]()
            else:
                gi[i] = OP.backward_scalar(xv, gov).cast[ADT]()


struct Elementwise[DIM_: Int, OP: ElementOp, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype. `Elementwise[DIM, OP]` = fp32 (ACT_DT == DT, the
    # legacy NoAMP path, byte-identical); `Elementwise[DIM, OP, bfloat16]` flows
    # its I/O activations at bf16 while computing the activation math fp32
    # INTERNALLY (the OP math is fp32-only — cast UP/DOWN at the boundary).
    # bf16-flow is GPU-only.
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return Self.OP.display_label()

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
        comptime M = B * Self.DIM_
        ref in0 = inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here — rebind the activation refs (sound; the dtypes
            # are equal, the checker just won't collapse the opaque `Self.ACT_DT`
            # for the CPU `.data.unsafe_ptr()` SIMD path). `TensorImpl[ACT_DT]` ≡
            # `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(M)
                var xp = in0d.data.unsafe_ptr()
                var op = outd.data.unsafe_ptr()
                var k = 0
                while k + CPU_SIMD_W <= M:
                    op.unsafe_store(
                        k,
                        Self.OP.forward_simd[CPU_SIMD_W](
                            xp.unsafe_load[width=CPU_SIMD_W](k)
                        ),
                    )
                    k += CPU_SIMD_W
                while k < M:
                    op[unsafe_offset=k] = Self.OP.forward_scalar(xp[unsafe_offset=k])
                    k += 1
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, M)
                comptime nblk = (M + TPB - 1) // TPB
                c.enqueue_function[_ew_fwd_kernel[M, Self.OP, Self.ADT]](
                    in0d.tt["gpu", M](),
                    outd.tt["gpu", M](),
                    grid_dim=nblk,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only). Activations cast at the I/O boundary;
            #    the OP math runs fp32 internally (fp32-internal leaf). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Elementwise is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_ew_fwd_kernel[M, Self.OP, Self.ADT]](
                in0.tt["gpu", M](),
                out.tt["gpu", M](),
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
        comptime M = B * Self.DIM_
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            ref find = rebind[Tensor](fin)
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            comptime if target == "cpu":
                gind.ensure(M)
                var xp = find.data.unsafe_ptr()
                var gp = god.data.unsafe_ptr()
                var ip = gind.data.unsafe_ptr()
                var k = 0
                while k + CPU_SIMD_W <= M:
                    var xv = xp.unsafe_load[width=CPU_SIMD_W](k)
                    var gv = gp.unsafe_load[width=CPU_SIMD_W](k)
                    comptime if Self.OP.owns_cache:
                        ip.unsafe_store(
                            k,
                            Self.OP.backward_simd[CPU_SIMD_W](
                                Self.OP.forward_simd[CPU_SIMD_W](xv), gv
                            ),
                        )
                    else:
                        ip.unsafe_store(k, Self.OP.backward_simd[CPU_SIMD_W](xv, gv))
                    k += CPU_SIMD_W
                while k < M:
                    comptime if Self.OP.owns_cache:
                        ip[unsafe_offset=k] = Self.OP.backward_scalar(
                            Self.OP.forward_scalar(xp[unsafe_offset=k]), gp[unsafe_offset=k]
                        )
                    else:
                        ip[unsafe_offset=k] = Self.OP.backward_scalar(xp[unsafe_offset=k], gp[unsafe_offset=k])
                    k += 1
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, M)
                comptime nblk = (M + TPB - 1) // TPB
                c.enqueue_function[_ew_bwd_kernel[M, Self.OP, Self.ADT]](
                    find.tt["gpu", M](),
                    god.tt["gpu", M](),
                    gind.tt["gpu", M](),
                    grid_dim=nblk,
                    block_dim=TPB,
                )
        else:
            # ── bf16-flow path (GPU-only). I/O activations cast at the boundary;
            #    the OP math runs fp32 internally (fp32-internal leaf). ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Elementwise is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, M)
            comptime nblk = (M + TPB - 1) // TPB
            c.enqueue_function[_ew_bwd_kernel[M, Self.OP, Self.ADT]](
                fin.tt["gpu", M](),
                grad_output.tt["gpu", M](),
                gin.tt["gpu", M](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection no-op defaults
    # (param-less: reflection finds no IsParam fields).
