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
calls `cublasGemmEx` directly on MAX's shared per-context handle with ONE
persistent workspace per context (`_shared_workspace`) — set on every call,
because MAX's own vendor calls re-point the shared handle at their temporary
one.

Row-major → cuBLAS's column-major: a row-major X[r, c] is the column-major
Xᵀ, so C = AᵀB (row-major) is Cᵀ = Bᵀ·A (column-major) = go_col · x_colᵀ:
ops (N, T), dims (N, K, B), leading dims (N, K, N) — the call MAX's own
wrapper makes for `transpose_a=True, c_row_major=True`.

NVIDIA only. Deterministic (cuBLAS's default, atomics-free GEMMs), so an
eager step and its CUDA-graph replay stay bit-identical.
"""

from std.sys import has_nvidia_gpu_accelerator
from std.sys.defines import get_defined_string, is_defined
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import Layout as AllocLayout
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


comptime GEMM_PATH = get_defined_string["NN_GEMM_PATH", "auto"]()
"""Which GEMM `Linear` / `LinearAct` use on NVIDIA (fp32), for A/B runs:

  - `auto` (default): the backward through `cublas_gemm`; the forward through
    it where `cublas_fwd` says so, MAX's `mm` / `mm_bias` elsewhere;
  - `max`: MAX's GEMMs everywhere, padded where MAX's dispatch needs it (the
    paths before cuBLAS);
  - `cublas`: `cublas_gemm` for every forward and backward GEMM;
  - `fwdmax` / `bwdmax`: MAX's GEMMs for the forward / the backward only, the
    other direction as `auto` (to bisect a difference between the two);
  - `linmax` / `convmax`: MAX's GEMMs for `Linear` / `LinearAct` only, or for
    `Conv2D` only (bisects a model-level difference between the two)."""

comptime CUBLAS_FP32 = is_defined["NN_CUBLAS_FP32"]()
"""`-D NN_CUBLAS_FP32`: `cublas_gemm` computes every GEMM in full fp32."""


def cublas_tf32(m: Int, n: Int, k: Int) -> Bool:
    """Whether a `cublas_gemm` of `m * n * k` multiply-adds runs in TF32
    (comptime). A layer's three GEMMs (forward, dW, dx) share the product
    B * IN * OUT, so a layer gets one precision.

    Full fp32 below 2^24 MACs: there it costs 0-20 % over TF32 on an RTX 5090
    (a few us), and it is what the small-batch layers ran before: MAX's
    vendor fallback computes fp32, and PPO on Hopper (minibatch 64) stalled
    in 7 of 20 seeds with TF32 GEMMs against 2 of 20 on the old path and 3 of
    15 in fp32. From 2^24 up, TF32 (fp32 costs 1.3-2.3x there), as MAX's
    multistage kernel always ran those shapes: SAC's 256-wide trunk at batch
    256, GPT, ViT."""
    if CUBLAS_FP32:
        return False
    return m * n * k >= (1 << 24)

comptime CUBLAS_BWD = has_nvidia_gpu_accelerator() and GEMM_PATH != "max" and GEMM_PATH != "bwdmax" and GEMM_PATH != "linmax"
"""The NVIDIA fp32 backward of `Linear` / `LinearAct` runs on `cublas_gemm`."""


def cublas_fwd(m: Int, n: Int, k: Int, padded: Bool) -> Bool:
    """Whether a forward GEMM `[m, k] @ [k, n]` runs on `cublas_gemm` (comptime).
    `padded`: MAX's dispatch would need the operands padded to take its
    multistage path.

    `auto` keeps MAX's GEMM in the three regimes where it measured faster
    (`benchmarks/bench_linear_gemm_paths_gpu.mojo`, RTX 5090, 119 shapes from
    noeira's agents and nn examples); everywhere else cuBLAS was 1.0-20x
    faster, most of all at the small and mid-size shapes RL nets run (2-3x
    typical; a padded 128-wide tile over an 8- or 64-wide layer):

      - m == 1 on an aligned shape: MAX's GEMV (SAC eval [1, 256] @ [256,
        256]: 2.5 vs 2.7 us);
      - m >= 16384, 96 <= k <= 1024, n >= 64: MAX's multistage kernel at
        large batch (GPT qkv [16384, 384] @ [384, 1152]: 170 vs 210 us;
        PPO GAE [24576, 99] @ [99, 512]: 67 vs 75 us);
      - m <= 32 with a weight of 2^25+ elements on an aligned shape: weight
        streaming (DreamerV3's [16, 3072] @ [3072, 13824]: 109 vs 144 us).

    With this rule every measured shape is within 4% of the faster path."""
    if not has_nvidia_gpu_accelerator() or GEMM_PATH == "max" or GEMM_PATH == "fwdmax" or GEMM_PATH == "linmax":
        return False
    if GEMM_PATH == "cublas":
        return True
    if m == 1 and not padded:
        return False
    if m >= 16384 and k >= 96 and k <= 1024 and n >= 64:
        return False
    if m <= 32 and k * n >= (1 << 25) and not padded:
        return False
    return True


comptime CUBLAS_WS_BYTES = 32 * 1024 * 1024
"""The shared workspace. With 4 MiB cuBLAS picked slower kernels for the
backward at large batch (dW over 6144 rows: 79 vs 35 us with 32 MiB, where
MAX's took 56); 32 MiB is NVIDIA's recommendation for Hopper and later.
Less only makes it pick other kernels, never fail."""


def _shared_workspace(c: DeviceContext) raises -> DeviceBuffer[DType.uint8]:
    """The context's cuBLAS workspace: ONE buffer per `DeviceContext`, created
    on first use and kept for the life of the process (a captured graph holds
    its raw pointer). Shared by every layer: GEMMs on one stream run in order,
    so they never use it at the same time.

    Created on the first call, which must be EAGER (an allocation cannot be
    captured); every trainer runs its step once before capturing it. Stored
    the way MAX keeps its per-context BLAS handle (`_get_global_handle`)."""
    var name = String("NOEIRA_CUBLAS_WS_", c.id())
    var g = _get_global_or_null(name)
    if g:
        return g.value().unsafe_bitcast[DeviceBuffer[DType.uint8]]()[]
    var p = alloc(AllocLayout[DeviceBuffer[DType.uint8]].single()).unsafe_leak()
    p.unsafe_write(c.enqueue_create_buffer[DType.uint8](CUBLAS_WS_BYTES))
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    return p[]


def cublas_gemm[
    TA: Bool, TB: Bool, TF32: Bool = True
](
    c: DeviceContext,
    dst: DeviceBuffer[DT],
    a: DeviceBuffer[DT],
    b: DeviceBuffer[DT],
    M: Int,
    N: Int,
    K: Int,
    beta: Float32,
) raises:
    """`dst[M, N] = op(a) @ op(b) + beta · dst`, all row-major fp32 (computed
    in TF32 when `TF32`, else full fp32; see `cublas_tf32`): op(a) is a[M, K], or a[K, M]ᵀ when `TA`; op(b) is b[K, N], or
    b[N, K]ᵀ when `TB`. The workspace is the context's shared one
    (`_shared_workspace`).

    Row-major X with row stride ld is cuBLAS's column-major Xᵀ, so
    out = op(a)·op(b) is outᵀ = op(b)ᵀ·op(a)ᵀ: cuBLAS gets (b, a) with the
    same transpose flags, dims (N, M, K) and the operands' own row strides as
    leading dimensions — MAX's own `c_row_major` mapping."""
    comptime assert DT == DType.float32, "cublas_gemm: fp32 only"
    _gemm_ex[TA, TB, DT, DT, TF32](c, dst, a, b, M, N, K, beta)


def cublas_gemm_lp[
    TA: Bool, TB: Bool, IN_DT: DType, OUT_DT: DType
](
    c: DeviceContext,
    dst: DeviceBuffer[OUT_DT],
    a: DeviceBuffer[IN_DT],
    b: DeviceBuffer[IN_DT],
    M: Int,
    N: Int,
    K: Int,
    beta: Float32,
) raises:
    """`cublas_gemm` with low-precision operands, for the bf16-flow `Linear`:
    `a` and `b` are `IN_DT` (bf16), `dst` is `OUT_DT` — bf16 for an
    activation (y, dx), fp32 for the master weight gradient, accumulated in
    place with `beta = 1`. Accumulation is fp32 (`COMPUTE_32F`) either way;
    bf16 operands run on the tensor cores without a TF32 mode."""
    _gemm_ex[TA, TB, IN_DT, OUT_DT, False](c, dst, a, b, M, N, K, beta)


def _gemm_ex[
    TA: Bool, TB: Bool, IN_DT: DType, OUT_DT: DType, TF32: Bool
](
    c: DeviceContext,
    dst: DeviceBuffer[OUT_DT],
    a: DeviceBuffer[IN_DT],
    b: DeviceBuffer[IN_DT],
    M: Int,
    N: Int,
    K: Int,
    beta: Float32,
) raises:
    """The one `cublasGemmEx` call behind `cublas_gemm` / `cublas_gemm_lp`
    (row-major mapping in `cublas_gemm`'s docstring)."""
    comptime assert has_nvidia_gpu_accelerator(), "cuBLAS is NVIDIA-only"
    var ws = _shared_workspace(c)
    var handle = _get_global_handle[DT, Backend.CUBLAS](c)._get_cublas()
    check_cublas_error(
        cublasSetWorkspace(handle, _ffi_void_ptr(ws.unsafe_ptr()), CUBLAS_WS_BYTES)
    )
    check_cublas_error(
        cublasSetMathMode(
            handle,
            cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH if TF32 else cublasMath_t.CUBLAS_DEFAULT_MATH,
        )
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
            _convert_to_cublas_datatype[IN_DT](),
            Int32(ldb),
            _ffi_void_ptr(a.unsafe_ptr()),
            _convert_to_cublas_datatype[IN_DT](),
            Int32(lda),
            UnsafePointer(to=beta_v)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            _ffi_void_ptr(dst.unsafe_ptr()),
            _convert_to_cublas_datatype[OUT_DT](),
            Int32(N),
            ComputeType.COMPUTE_32F_FAST_TF32 if TF32 else ComputeType.COMPUTE_32F,
            Algorithm.DEFAULT,
        ),
        msg=String("cublas_gemm: [", M, "x", N, "] K=", K, " TA=", TA, " TB=", TB, " ", IN_DT, "->", OUT_DT),
    )
