"""`cublaslt_gemm` — row-major bf16 GEMMs with a FUSED epilogue through
cuBLASLt, for the bf16-flow `Linear` / `FeedForwardGELU` on NVIDIA.

`cublas_gemm_lp` (cuBLAS) leaves the bias add and the GELU as separate
memory-bound passes over [rows, OUT] activations. cuBLASLt applies them in the
GEMM's epilogue, in registers, before the store:

  - `EPI_BIAS`:          D = A·B + bias
  - `EPI_GELU_AUX_BIAS`: D = gelu(A·B + bias), and AUX = A·B + bias (the
                         pre-activation the backward needs)
  - `EPI_DGELU`:         D = (A·B) * gelu'(AUX)

(In bf16 these run inside the GEMM kernel; in fp32 cuBLASLt does NOT fuse its
GELU epilogues — a separate generic epilogue kernel, see
`feed_forward_gelu.mojo` — which is why only the bf16 path uses this.)
cuBLASLt's GELU is the tanh approximation, as `GELUTanh`.

Row-major mapping as `cublas_gemm`: out[M, N] = op(a)·op(b) is computed as
the column-major outᵀ = op(b)ᵀ·op(a)ᵀ, so the per-output-feature bias (length
N) is cuBLASLt's per-ROW bias of D (length m = N), as its epilogue expects.

Bindings: minimal and raw, as `cudnn_conv.mojo` — every handle, descriptor
and device pointer a pointer-sized `Int`, C `int` / enums `Int32`, `size_t`
`Int` — through MAX's loader for libcublasLt. Enum values come from MAX's
typed enums (`_value`), not literals.

State, all created on the first, EAGER call (the heuristic and the descriptor
creation cannot run inside a CUDA-graph capture):
  - one cuBLASLt handle per `DeviceContext`;
  - one plan per (context, shape, transposes, dtypes, epilogue): the matmul
    descriptor, the three layouts and the algorithm cuBLASLt's heuristic
    ranks first (deterministic: the same choice every process, unlike a
    timed search). Per-call pointers (bias, aux) are set on the descriptor
    before each launch — a host-side write, legal during capture;
  - the workspace is `cublas_gemm`'s shared one (one stream: never used by
    two GEMMs at once).

NVIDIA only.
"""

from std.sys import has_nvidia_gpu_accelerator
from std.sys.defines import get_defined_string
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import Layout as AllocLayout
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from _cublas.cublaslt import (
    _get_dylib_function,
    cublasLtMatmulDescAttributes_t as Attr,
    Epilogue,
    Preference,
)
from _cublas.dtype import DataType
from _cublas.cublas import ComputeType

from noeira.nn.core.cublas_gemm import _shared_workspace, CUBLAS_WS_BYTES


comptime LT_BIAS = get_defined_string["NN_LT_BIAS", "1"]() == "1"
"""The fp32 `Linear` forward on cuBLAS takes its bias in a cuBLASLt epilogue
instead of a separate kernel (default). Measured on the 86 shapes whose
forward is on cuBLAS (`bench_linear_gemm_paths_gpu.mojo`, RTX 5090): faster
on every one, median 1.14x, 1.02-1.53x (one launch fewer: a 2 us head at
batch 1 takes 1.3). `-D NN_LT_BIAS=0`: the GEMM + bias kernel (A/B)."""

comptime EPI_NONE = 0
comptime EPI_BIAS = 1
comptime EPI_GELU_AUX_BIAS = 2
comptime EPI_DGELU = 3

comptime _HEUR_BYTES = 96
"""sizeof(cublasLtMatmulHeuristicResult_t): algo (64) + workspaceSize (8) +
state (4) + wavesCount (4) + reserved (16)."""


def _dt[d: DType]() -> Int32:
    comptime if d == DType.bfloat16:
        return Int32(DataType.R_16BF._value)
    elif d == DType.float16:
        return Int32(DataType.R_16F._value)
    else:
        return Int32(DataType.R_32F._value)


def _epi(e: Int) -> Int32:
    if e == EPI_BIAS:
        return Int32(Epilogue.BIAS._value)
    if e == EPI_GELU_AUX_BIAS:
        return Int32(Epilogue.GELU_AUX_BIAS._value)
    if e == EPI_DGELU:
        return Int32(Epilogue.DGELU._value)
    return Int32(Epilogue.DEFAULT._value)


def _check(rc: Int32, what: String) raises:
    if rc != 0:
        raise Error("cuBLASLt " + what + " failed: status " + String(rc))


def _new[fname: StaticString]() raises -> Int:
    """`fname(&out)` for a create function with one out-pointer argument."""
    var p = alloc[Int](1)
    p[0] = 0
    _check(
        _get_dylib_function[fname, def(Int) thin abi("C") -> Int32]()(Int(p)),
        String(fname),
    )
    var out = p[0]
    p.free()
    return out


def _set_desc(desc: Int, attr: Int32, buf: Int, size: Int) raises:
    _check(
        _get_dylib_function[
            "cublasLtMatmulDescSetAttribute",
            def(Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(desc, attr, buf, size),
        "cublasLtMatmulDescSetAttribute",
    )


def _set_i32(desc: Int, attr: Int32, v: Int32) raises:
    # The value goes through a HEAP buffer: a stack local whose address
    # reaches C only as an `Int` was not stored before the call (the
    # library read garbage — `EPILOGUE = DEFAULT` came back NOT_SUPPORTED).
    var p = alloc[Int32](1)
    p[0] = v
    _set_desc(desc, attr, Int(p), 4)
    p.free()


def _set_i64(desc: Int, attr: Int32, v: Int64) raises:
    var p = alloc[Int64](1)
    p[0] = v
    _set_desc(desc, attr, Int(p), 8)
    p.free()


def _layout(dt: Int32, rows: Int, cols: Int, ld: Int) raises -> Int:
    var p = alloc[Int](1)
    p[0] = 0
    _check(
        _get_dylib_function[
            "cublasLtMatrixLayoutCreate",
            def(Int, Int32, UInt64, UInt64, Int64) thin abi("C") -> Int32,
        ]()(Int(p), dt, UInt64(rows), UInt64(cols), Int64(ld)),
        "cublasLtMatrixLayoutCreate",
    )
    var out = p[0]
    p.free()
    return out


@fieldwise_init
struct _Handle(Copyable, Movable):
    var handle: Int


def _handle(c: DeviceContext) raises -> Int:
    var name = String("NOEIRA_CUBLASLT_HANDLE_", c.id())
    var g = _get_global_or_null(name)
    if g:
        return g.value().unsafe_bitcast[_Handle]()[].handle
    var h = _new["cublasLtCreate"]()
    var p = alloc(AllocLayout[_Handle].single()).unsafe_leak()
    p.unsafe_write(_Handle(h))
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    return h


def _alpha_beta() -> Int:
    """A process-lifetime host buffer holding alpha = 1, beta = 0 (fp32). A
    captured cublasLtMatmul keeps the host scalars' values, but an eager
    call reads them through the pointer — a heap buffer, for the reason in
    `_set_i32`."""
    var g = _get_global_or_null("NOEIRA_CUBLASLT_ALPHA_BETA")
    if g:
        return Int(g.value())
    var p = alloc[Float32](2)
    p[0] = 1.0
    p[1] = 0.0
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice("NOEIRA_CUBLASLT_ALPHA_BETA"), p.bitcast[NoneType]()
    )
    return Int(p)


@fieldwise_init
struct _Plan(Copyable, Movable):
    var desc: Int
    var la: Int
    var lb: Int
    var ld: Int
    var algo: Int
    """Device-independent host buffer holding the heuristic result (its
    first 64 bytes are the `cublasLtMatmulAlgo_t` passed to the launch)."""


def _plan[
    TA: Bool, TB: Bool, IN_DT: DType, OUT_DT: DType, BIAS_DT: DType,
    TF32: Bool = False,
](c: DeviceContext, M: Int, N: Int, K: Int, epi: Int, aux_ld: Int) raises -> _Plan:
    var name = String(
        "NOEIRA_CUBLASLT_PLAN_", c.id(), "_", M, "x", N, "x", K, "_", TA, TB,
        "_", IN_DT, OUT_DT, BIAS_DT, "_", epi, "_", aux_ld, "_", TF32,
    )
    var g = _get_global_or_null(name)
    if g:
        return g.value().unsafe_bitcast[_Plan]()[].copy()
    var h = _handle(c)
    var dp = alloc[Int](1)
    dp[0] = 0
    _check(
        _get_dylib_function[
            "cublasLtMatmulDescCreate", def(Int, Int32, Int32) thin abi("C") -> Int32
        ]()(
            Int(dp),
            Int32(
                ComputeType.COMPUTE_32F_FAST_TF32._value if TF32 else ComputeType.COMPUTE_32F._value
            ),
            Int32(DataType.R_32F._value),
        ),
        "cublasLtMatmulDescCreate",
    )
    var desc = dp[0]
    dp.free()
    # cuBLASLt's A is our b (op TB), its B our a (op TA): see the module doc.
    _set_i32(desc, Int32(Attr.CUBLASLT_MATMUL_DESC_TRANSA._value), Int32(1 if TB else 0))
    _set_i32(desc, Int32(Attr.CUBLASLT_MATMUL_DESC_TRANSB._value), Int32(1 if TA else 0))
    _set_i32(desc, Int32(Attr.CUBLASLT_MATMUL_DESC_EPILOGUE._value), _epi(epi))
    if epi == EPI_BIAS or epi == EPI_GELU_AUX_BIAS:
        _set_i32(desc, Int32(Attr.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE._value), _dt[BIAS_DT]())
    if epi == EPI_GELU_AUX_BIAS or epi == EPI_DGELU:
        _set_i64(desc, Int32(Attr.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD._value), Int64(aux_ld))
    # Column-major: A = b as stored ([K, N] row-major = N x K col... see doc),
    # B = a, D = outᵀ (N rows, M cols, ld N).
    var la = _layout(_dt[IN_DT](), K if TB else N, N if TB else K, K if TB else N)
    var lb = _layout(_dt[IN_DT](), M if TA else K, K if TA else M, M if TA else K)
    var ld = _layout(_dt[OUT_DT](), N, M, N)
    var pref = _new["cublasLtMatmulPreferenceCreate"]()
    var ws = alloc[Int64](1)
    ws[0] = Int64(CUBLAS_WS_BYTES)
    _check(
        _get_dylib_function[
            "cublasLtMatmulPreferenceSetAttribute",
            def(Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(pref, Int32(Preference.MAX_WORKSPACE_BYTES._value), Int(ws), 8),
        "cublasLtMatmulPreferenceSetAttribute",
    )
    ws.free()
    var algo = alloc[UInt8](_HEUR_BYTES)
    var n_found_p = alloc[Int32](1)
    n_found_p[0] = 0
    _check(
        _get_dylib_function[
            "cublasLtMatmulAlgoGetHeuristic",
            def(Int, Int, Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(h, desc, la, lb, ld, ld, pref, Int32(1), Int(algo), Int(n_found_p)),
        "cublasLtMatmulAlgoGetHeuristic",
    )
    var n_found = n_found_p[0]
    n_found_p.free()
    if n_found < 1:
        raise Error(
            "cuBLASLt: no algorithm for [" + String(M) + "x" + String(N) + "] K="
            + String(K) + " epilogue " + String(epi)
        )
    var plan = _Plan(desc, la, lb, ld, Int(algo))
    var p = alloc(AllocLayout[_Plan].single()).unsafe_leak()
    p.unsafe_write(plan.copy())
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    return plan^


def cublaslt_gemm[
    TA: Bool, TB: Bool, IN_DT: DType, OUT_DT: DType, TF32: Bool = False
](
    c: DeviceContext,
    dst: DeviceBuffer[OUT_DT],
    a: DeviceBuffer[IN_DT],
    b: DeviceBuffer[IN_DT],
    M: Int,
    N: Int,
    K: Int,
    epi: Int,
    bias: Int = 0,
    aux: Int = 0,
    aux_ld: Int = 0,
) raises:
    """`dst[M, N] = epilogue(op(a) @ op(b))`, row-major, fp32 accumulation
    (op as `cublas_gemm`). `bias`: device pointer to N `OUT_DT` values
    (`EPI_BIAS`, `EPI_GELU_AUX_BIAS`). `aux`: device pointer to the [M, N]
    pre-activation, row stride `aux_ld` — written by `EPI_GELU_AUX_BIAS`,
    read by `EPI_DGELU`. beta = 0. `TF32`: fp32 operands computed in TF32
    (`cublas_tf32`'s rule), else full fp32 / bf16 accumulation in fp32."""
    comptime assert has_nvidia_gpu_accelerator(), "cuBLASLt is NVIDIA-only"
    # The bias is OUT_DT: cuBLASLt rejects an fp32 bias with a bf16 output
    # (INVALID_VALUE from the heuristic on cuBLAS 12.9).
    var plan = _plan[TA, TB, IN_DT, OUT_DT, OUT_DT, TF32](c, M, N, K, epi, aux_ld)
    if epi == EPI_BIAS or epi == EPI_GELU_AUX_BIAS:
        _set_i64(plan.desc, Int32(Attr.CUBLASLT_MATMUL_DESC_BIAS_POINTER._value), Int64(bias))
    if epi == EPI_GELU_AUX_BIAS or epi == EPI_DGELU:
        _set_i64(plan.desc, Int32(Attr.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER._value), Int64(aux))
    var ws = _shared_workspace(c)
    var ab = _alpha_beta()
    var d = Int(dst.unsafe_ptr())
    _check(
        _get_dylib_function[
            "cublasLtMatmul",
            def(
                Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int,
                Int, Int, Int, type_of(CUDA(c.stream())),
            ) thin abi("C") -> Int32,
        ]()(
            _handle(c), plan.desc, ab,
            Int(b.unsafe_ptr()), plan.la, Int(a.unsafe_ptr()), plan.lb,
            ab + 4, d, plan.ld, d, plan.ld,
            plan.algo, Int(ws.unsafe_ptr()), CUBLAS_WS_BYTES, CUDA(c.stream()),
        ),
        "cublasLtMatmul [" + String(M) + "x" + String(N) + "] K=" + String(K),
    )
