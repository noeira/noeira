"""FeedForwardGELU[S, D, FF] — a transformer's feed-forward block as ONE module.

    out = Tok[Linear[FF, D]]( GELUTanh( Tok[Linear[D, FF]](x) ) )

the composition `TransformerFFN` spells as a Sequential of three children,
fused on NVIDIA (fp32) in MAX's GEMM epilogue (`elementwise_lambda_fn`, run
on each output tile before its store):

  forward   z = x·W1 + b1 and h = GELU(z) stored by ONE GEMM — no separate
            GELU pass (which read z back and wrote h); then fc2 (`Linear`).
  backward  fc2: dW2 += hᵀ·go, dh = go·W2ᵀ (cuBLAS), db2 += Σ go;
            dz = GELU'(z) ⊙ dh (the `act` child's kernel);
            fc1: db1 += Σ dz, dW1 += xᵀ·dz, dx = dz·W1ᵀ (cuBLAS).

Measured on an RTX 5090 at the GPT block (16,384 rows, 384 -> 1536 -> 384),
per layer, and why the backward is NOT fused:

  forward    MAX GEMM 243 us + GELU 181 us   ->  fused GEMM 261 us
  backward   cuBLAS dh 209 us + GELU' 120 us  vs  MAX GEMM (transpose_b) with
             a GELU' epilogue 495 us — MAX's multistage kernel is slow on
             that operand form, so fusing GELU' into it lost what the
             forward fusion gained (GPT step 20.93 -> 20.98 ms).
  cuBLASLt   its GELU epilogues are the tanh GELU (checked), but in fp32 it
             does not fuse them: the GEMM, then a generic
             `cublasLt::epilogue::impl::globalKernel` (446 us for
             DGELU_BGRAD). GPT step 20.9 -> 22.7 ms. Its in-kernel GELU
             epilogues are fp16 / bf16.

Everywhere else (CPU, Apple, bf16 activations) it runs its three children —
`fc1`, `act`, `fc2` — exactly as the Sequential would (rows = B·S, as under
`Tokenwise`), so it is a drop-in for `TransformerFFN` with the same weights.
Parameters walk as `0.*` (fc1) and `1.*` (fc2).
"""

from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.module import Module
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs, child_refs
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.cublas_gemm import cublas_gemm, CUBLAS_WS_BYTES
from .linear import Linear, enqueue_bias_grad
from .activations import GELUTanh
from .ops.gelu_tanh_op import GELUTanhOp
from max.gpu.host import DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.utils import elementwise_epilogue_type
from std.utils.index import IndexList


def _mm_bias_gelu[
    R: Int, D: Int, FF: Int
](
    mut h: DeviceBuffer[DT],
    mut z: DeviceBuffer[DT],
    x: DeviceBuffer[DT],
    w1: DeviceBuffer[DT],
    b1: DeviceBuffer[DT],
    c: DeviceContext,
) raises:
    """z[R, FF] = x[R, D] @ w1[D, FF] + b1, h = GELUTanh(z): one GEMM, both
    stored by its epilogue (NVIDIA: on Metal an epilogue-only buffer is not
    resident, `bench_matmul_epilogue_fusion.mojo`)."""
    var xv = TileTensor(x, row_major[R, D]())
    var wv = TileTensor(w1, row_major[D, FF]())
    var hv = TileTensor(h, row_major[R, FF]())
    var zv = TileTensor(z, row_major[R, FF]())
    var bv = TileTensor(b1, row_major[FF]())

    @__parameter
    @always_inline
    @__copy_capture(hv, zv, bv)
    def _bias_gelu[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        var zz = val.cast[DT]()
        var hh = zz
        comptime for i in range(width):
            zz[i] += rebind[Scalar[DT]](bv[coords[1] + i])
            hh[i] = GELUTanhOp.forward_scalar(zz[i])
        zv.store_linear[alignment=alignment](coords, zz)
        hv.store_linear[alignment=alignment](coords, hh)

    max_matmul[
        target="gpu",
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](_bias_gelu),
    ](hv, xv, wv, c)


struct FeedForwardGELU[S: Int, D: Int, FF: Int, ADT: DType = DT](Module):
    comptime ARITY: Int = 1
    comptime ACT_DT = Self.ADT
    comptime IN_DIMS = Array[Int, 1](fill=Self.S * Self.D)
    comptime OUT_DIM = Self.S * Self.D
    # The fused path: NVIDIA, fp32 activations.
    comptime FUSED = has_nvidia_gpu_accelerator() and Self.ADT == DT

    var fc1: Linear[Self.D, Self.FF, Self.ADT]
    var act: GELUTanh[Self.FF, Self.ADT]
    var fc2: Linear[Self.FF, Self.D, Self.ADT]
    var z: TensorImpl[Self.ADT]  # [B·S, FF] pre-activation
    var h: TensorImpl[Self.ADT]  # [B·S, FF] GELU(z) = fc2's input
    var gh: TensorImpl[Self.ADT]  # [B·S, FF] d h
    var gz: TensorImpl[Self.ADT]  # [B·S, FF] d z

    def __init__(out self):
        self.fc1 = Linear[Self.D, Self.FF, Self.ADT]()
        self.act = GELUTanh[Self.FF, Self.ADT]()
        self.fc2 = Linear[Self.FF, Self.D, Self.ADT]()
        self.z = TensorImpl[Self.ADT]()
        self.h = TensorImpl[Self.ADT]()
        self.gh = TensorImpl[Self.ADT]()
        self.gz = TensorImpl[Self.ADT]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.fc1 = Linear[Self.D, Self.FF, Self.ADT].make[target, INIT](ctx)
        m.act = GELUTanh[Self.FF, Self.ADT].make[target, INIT](ctx)
        m.fc2 = Linear[Self.FF, Self.D, Self.ADT].make[target, INIT](ctx)
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime R = B * Self.S
        comptime if target == "gpu" and Self.FUSED:
            ref x = inputs[0]
            var c = ctx.value()
            self.z.ensure_gpu(c, R * Self.FF)
            self.h.ensure_gpu(c, R * Self.FF)
            _mm_bias_gelu[R, Self.D, Self.FF](
                rebind[Tensor](self.h).dev.value(),
                rebind[Tensor](self.z).dev.value(),
                rebind[Tensor](x).dev.value(),
                self.fc1.weight.val.dev.value(), self.fc1.bias.val.dev.value(),
                c,
            )
        else:
            self.fc1.forward[target, R, POLICY=POLICY](inputs, self.z, ctx)
            self.act.forward[target, R, POLICY=POLICY](
                child_refs[1, Self.ADT](self.z), self.h, ctx
            )
        self.fc2.forward[target, R, POLICY=POLICY](
            child_refs[1, Self.ADT](self.h), out, ctx
        )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime R = B * Self.S
        comptime if target == "gpu" and Self.FUSED:
            ref x = rebind[Tensor](forward_input[0])
            ref gin = rebind[Tensor](grad_inputs[0])
            ref go = rebind[Tensor](grad_output)
            ref hd = rebind[Tensor](self.h)
            ref dh = rebind[Tensor](self.gh)
            ref dz = rebind[Tensor](self.gz)
            var c = ctx.value()
            dh.ensure_gpu(c, R * Self.FF)
            gin.ensure_gpu(c, R * Self.D)
            self.fc2.blas_ws.ensure_gpu(c, CUBLAS_WS_BYTES)
            var ws = self.fc2.blas_ws.dev.value()
            # fc2: db2 += Σ go, dW2 += hᵀ·go.
            enqueue_bias_grad[DT](
                c, go.dev.value(), self.fc2.bias.grd.dev.value(), R, Self.D,
                self.fc2.gb_part,
            )
            cublas_gemm[True, False](
                c, self.fc2.weight.grd.dev.value(), hd.dev.value(),
                go.dev.value(), ws, Self.FF, Self.D, R, 1.0,
            )
            # dh = go·W2ᵀ (cuBLAS), then dz = GELU'(z) ⊙ dh.
            cublas_gemm[False, True](
                c, dh.dev.value(), go.dev.value(),
                self.fc2.weight.val.dev.value(), ws, R, Self.FF, Self.D, 0.0,
            )
            self.act.vjp[target, R, POLICY=POLICY](
                child_refs[1, Self.ADT](self.z), self.gh,
                child_refs[1, Self.ADT](self.gz), ctx,
            )
            # fc1: db1 += Σ dz, dW1 += xᵀ·dz, dx = dz·W1ᵀ.
            enqueue_bias_grad[DT](
                c, dz.dev.value(), self.fc1.bias.grd.dev.value(), R, Self.FF,
                self.fc1.gb_part,
            )
            cublas_gemm[True, False](
                c, self.fc1.weight.grd.dev.value(), x.dev.value(),
                dz.dev.value(), ws, Self.D, Self.FF, R, 1.0,
            )
            cublas_gemm[False, True](
                c, gin.dev.value(), dz.dev.value(),
                self.fc1.weight.val.dev.value(), ws, R, Self.D, Self.FF, 0.0,
            )
        else:
            self.fc2.vjp[target, R, POLICY=POLICY](
                child_refs[1, Self.ADT](self.h), grad_output,
                child_refs[1, Self.ADT](self.gh), ctx,
            )
            self.act.vjp[target, R, POLICY=POLICY](
                child_refs[1, Self.ADT](self.z), self.gh,
                child_refs[1, Self.ADT](self.gz), ctx,
            )
            self.fc1.vjp[target, R, POLICY=POLICY](
                forward_input, self.gz, grad_inputs, ctx
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.fc1.for_each_param[target](visitor, ctx, join_name(prefix, "0"))
        self.fc2.for_each_param[target](visitor, ctx, join_name(prefix, "1"))

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.fc1.for_each_state[target](visitor, ctx, join_name(prefix, "0"))
        self.fc2.for_each_state[target](visitor, ctx, join_name(prefix, "1"))

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.fc1.zero_grad[target](ctx)
        self.fc2.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.fc1.polyak_from[target](src.fc1, tau, ctx)
        self.fc2.polyak_from[target](src.fc2, tau, ctx)

    def release_buffers(mut self):
        self.fc1.release_buffers()
        self.fc2.release_buffers()
        self.z.release()
        self.h.release()
        self.gh.release()
        self.gz.release()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.fc1.set_attr[ATTR](value)
        self.act.set_attr[ATTR](value)
        self.fc2.set_attr[ATTR](value)

    @staticmethod
    def display_label() -> String:
        return String("FeedForwardGELU")
