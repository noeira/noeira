"""`gemm_cpu`: a BLAS-shaped fp32 GEMM on the host, for the CPU paths of the
side modules that need transposes, alpha and beta (`nn/peft/lora.mojo`'s
adapter, `nn/quant/qlinear.mojo`'s dequantized weight).

On macOS it is one `cblas_sgemm` (Accelerate), whose alpha / beta carry a
scale and an accumulation for free. Elsewhere it is a plain triple loop:
correct, and slow at model sizes. The CPU paths it serves are gates and
small fixtures; nothing that runs a model on a Linux CPU goes through it.
"""

from std.sys import CompilationTarget
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from noeira.nn.constants import DT
from .tensor import Tensor


def gemm_cpu[
    TA: Bool, TB: Bool
](
    M: Int,
    N: Int,
    K: Int,
    alpha: Float32,
    a: Tensor,
    lda: Int,
    b: Tensor,
    ldb: Int,
    beta: Float32,
    mut c: Tensor,
    ldc: Int,
) raises:
    """`c[M, N] = alpha · op(a) @ op(b) + beta · c`, row-major fp32: op(a) is
    a[M, K] (row stride `lda`), or a[K, M]ᵀ when `TA`; op(b) is b[K, N], or
    b[N, K]ᵀ when `TB`. As in BLAS, `c` is not read when beta is 0."""
    comptime assert DT == DType.float32, "gemm_cpu is fp32"
    comptime if CompilationTarget.is_macos():
        var cblas = get_cblas_f32_function()
        cblas(
            _CBLASOrder.ROW_MAJOR,
            _CBLASTranspose.TRANSPOSE if TA else _CBLASTranspose.NO_TRANSPOSE,
            _CBLASTranspose.TRANSPOSE if TB else _CBLASTranspose.NO_TRANSPOSE,
            Int32(M),
            Int32(N),
            Int32(K),
            alpha,
            rebind[Pointer[Float32, ImmutAnyOrigin]](a.data.unsafe_ptr()),
            Int32(lda),
            rebind[Pointer[Float32, ImmutAnyOrigin]](b.data.unsafe_ptr()),
            Int32(ldb),
            beta,
            rebind[Pointer[Float32, MutAnyOrigin]](c.data.unsafe_ptr()),
            Int32(ldc),
        )
    else:
        for i in range(M):
            for j in range(N):
                var acc = Float32(0)
                for k in range(K):
                    var av = a.data[k * lda + i] if TA else a.data[i * lda + k]
                    var bv = b.data[j * ldb + k] if TB else b.data[k * ldb + j]
                    acc += av * bv
                var prev = Float32(0)
                if beta != 0:
                    prev = beta * c.data[i * ldc + j]
                c.data[i * ldc + j] = alpha * acc + prev
