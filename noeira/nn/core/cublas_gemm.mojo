"""`cublas_gemm` — row-major fp32 GEMMs through cuBLAS, for `Linear`'s
backward on NVIDIA:

    dw[K, N] += x[B, K]ᵀ @ go[B, N]     cublas_gemm[TA=True,  TB=False], β = 1
    gx[B, K]  = go[B, N] @ w[K, N]ᵀ     cublas_gemm[TA=False, TB=True],  β = 0

MAX's GPU matmul rejects `transpose_a` (linalg/matmul: "transpose_a not yet
supported"), so the weight gradient used to transpose the forward input into
`cacheT` (a full pass over [B, K]), GEMM into a `dW_tmp`, then add that into
the master gradient. cuBLAS takes the transpose and β = 1 natively: one call,
no transposed copy, no temporary, no accumulate kernel. And it takes ANY
shape without allocating, so neither GEMM needs the padded operands that
MAX's dispatch demands (`n % 128 == 0` or a per-call 32 MB vendor fallback).

MAX ships the bindings (`_cublas.cublas`) and its vendor `matmul` wraps them,
but that wrapper allocates and zeroes a fresh 32 MB workspace on EVERY call
(an allocation inside a captured graph, and 32 MB of memset per GEMM). This
calls `cublasGemmEx` directly on MAX's shared per-context handle with the
CALLER's persistent workspace — set on every call, because MAX's own vendor
calls re-point the shared handle at their temporary one.

Row-major → cuBLAS's column-major: a row-major X[r, c] is the column-major
Xᵀ, so C = AᵀB (row-major) is Cᵀ = Bᵀ·A (column-major) = go_col · x_colᵀ:
ops (N, T), dims (N, K, B), leading dims (N, K, N) — the call MAX's own
wrapper makes for `transpose_a=True, c_row_major=True`.

NVIDIA only. Deterministic (cuBLAS's default, atomics-free GEMMs), so an
eager step and its CUDA-graph replay stay bit-identical.
"""

from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from linalg.matmul.vendor.blas import _get_global_handle, Backend, _ffi_void_ptr
from _cublas.cublas import (
    Algorithm,
    ComputeType,
    _convert_to_cublas_datatype,
    _convert_to_cublas_transpose,
    check_cublas_error,
    cublasGemmEx,
    cublasMath_t,
    cublasSetMathMode,
    cublasSetWorkspace,
)

from noeira.nn.constants import DT


comptime CUBLAS_WS_BYTES = 4 * 1024 * 1024
"""Per-caller workspace. cuBLAS asks for 4 MiB on sm_90+ for its split-K
reductions; less makes it pick other kernels, never fail."""


def cublas_gemm[
    TA: Bool, TB: Bool
](
    c: DeviceContext,
    dst: DeviceBuffer[DT],
    a: DeviceBuffer[DT],
    b: DeviceBuffer[DT],
    ws: DeviceBuffer[DType.uint8],
    M: Int,
    N: Int,
    K: Int,
    beta: Float32,
) raises:
    """`dst[M, N] = op(a) @ op(b) + beta · dst`, all row-major fp32 (TF32
    math): op(a) is a[M, K], or a[K, M]ᵀ when `TA`; op(b) is b[K, N], or
    b[N, K]ᵀ when `TB`. `ws` must hold `CUBLAS_WS_BYTES` and outlive the call
    (and any graph that captured it).

    Row-major X with row stride ld is cuBLAS's column-major Xᵀ, so
    out = op(a)·op(b) is outᵀ = op(b)ᵀ·op(a)ᵀ: cuBLAS gets (b, a) with the
    same transpose flags, dims (N, M, K) and the operands' own row strides as
    leading dimensions — MAX's own `c_row_major` mapping."""
    comptime assert has_nvidia_gpu_accelerator(), "cuBLAS is NVIDIA-only"
    comptime assert DT == DType.float32, "cublas_gemm: fp32 only"
    var handle = _get_global_handle[DT, Backend.CUBLAS](c)._get_cublas()
    check_cublas_error(
        cublasSetWorkspace(handle, _ffi_void_ptr(ws.unsafe_ptr()), CUBLAS_WS_BYTES)
    )
    check_cublas_error(
        cublasSetMathMode(handle, cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH)
    )
    var alpha = Float32(1.0)
    var beta_v = beta
    var lda = M if TA else K
    var ldb = K if TB else N
    check_cublas_error(
        cublasGemmEx(
            handle,
            _convert_to_cublas_transpose(TB),
            _convert_to_cublas_transpose(TA),
            Int32(N),
            Int32(M),
            Int32(K),
            UnsafePointer(to=alpha)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            _ffi_void_ptr(b.unsafe_ptr()),
            _convert_to_cublas_datatype[DT](),
            Int32(ldb),
            _ffi_void_ptr(a.unsafe_ptr()),
            _convert_to_cublas_datatype[DT](),
            Int32(lda),
            UnsafePointer(to=beta_v)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            _ffi_void_ptr(dst.unsafe_ptr()),
            _convert_to_cublas_datatype[DT](),
            Int32(N),
            ComputeType.COMPUTE_32F_FAST_TF32,
            Algorithm.DEFAULT,
        ),
        msg=String("cublas_gemm: [", M, "x", N, "] K=", K, " TA=", TA, " TB=", TB),
    )
