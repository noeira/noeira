"""`cudnn_conv_*` — Conv2D forward / backward-data / backward-filter through
cuDNN on NVIDIA, fp32, for `Conv2D`'s NVIDIA path.

`Conv2D` otherwise lowers a convolution to im2col + one GEMM: an explicit
`[B·OH·OW, IC·K·K]` matrix written and read back on every forward and again
in the backward. At large spatial sizes that buffer is the cost (a 32x32
CIFAR layer at batch 100 materialises 118 MB); cuDNN's implicit-GEMM kernels
never build it. At small spatial sizes im2col + cuBLAS measured faster than
cuDNN, so `Conv2D` picks per shape (`Conv2D.use_cudnn`).

Bindings. MAX ships `_cudnn` bindings for the forward and backward-data
calls, but they type C `int` as `Int16` and its perf struct with 1-byte
enums (MAX's own conv notes the struct layout is wrong), and the
backward-filter calls are not bound at all. These are minimal bindings with
the C types: every handle, descriptor and device pointer is passed as a
pointer-sized `Int` (the same register in the C ABI), `int` as `Int32`,
`size_t` as `Int`. The library is MAX's loader (`libcudnn.so.9`, which
exports the whole legacy API).

State (all created on the first, EAGER call — cuDNN's algorithm search
allocates, so it cannot run inside a CUDA-graph capture):
  - one cuDNN handle per `DeviceContext` (runtime global, like MAX's), its
    stream set on every call;
  - one plan per (context, shape, layout, precision): tensor / filter /
    convolution descriptors and the three algorithms, the best HEURISTIC
    rank among the deterministic GEMM algorithms (`_allowed`; `cudnnFind*`
    timing picked a different algorithm from run to run, so two runs of one
    binary trained differently — `-D NN_CUDNN_ALGO=find` keeps it for
    benchmarking);
  - one workspace per context, grown to the largest plan's need; a replaced
    buffer is kept alive, since a captured graph may hold its pointer.

Precision follows `cublas_tf32`: TF32 allowed (`CUDNN_DEFAULT_MATH`) from
2^24 MACs per layer, plain fp32 FMA (`CUDNN_FMA_MATH`) below.
"""

from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import Layout as AllocLayout
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from _cudnn.cnn_infer import _get_dylib_function

from std.sys.defines import get_defined_string
from noeira.nn.constants import DT


comptime CUDNN_ALGO = get_defined_string["NN_CUDNN_ALGO", "heur"]()
"""How a plan picks its algorithms: `heur` (default) = cuDNN's heuristic
ranking (`cudnnGet*Algorithm_v7`), the same answer in every process; `find` =
`cudnnFind*` timing, which can pick a different algorithm from one run to the
next when two are close (seen: the OC = 1 case flipped between runs and once
landed on an algorithm outside the 1e-2 gate)."""

comptime CUDNN_VERBOSE = get_defined_string["NN_CUDNN_VERBOSE", "0"]() == "1"

comptime CONV_PATH = get_defined_string["NN_CONV_PATH", "auto"]()
"""How `Conv2D` runs on NVIDIA (fp32), for A/B runs: `auto` (the measured
per-shape rule, `Conv2D.use_cudnn`), `im2col` (im2col + GEMM everywhere) or
`cudnn` (cuDNN everywhere)."""


comptime _NCHW = Int32(0)
comptime _NHWC = Int32(1)
comptime _FLOAT = Int32(0)
comptime _CROSS_CORRELATION = Int32(1)
comptime _DEFAULT_MATH = Int32(0)
comptime _FMA_MATH = Int32(3)
comptime _DETERMINISTIC = Int32(1)
comptime _PERF_BYTES = 48
"""`cudnnConvolution{Fwd,BwdData,BwdFilter}AlgoPerf_t`: algo i32 @0, status
i32 @4, time f32 @8, memory size_t @16, determinism i32 @24, mathType i32
@28, reserved 3 x i32 — 48 bytes with alignment."""
comptime _N_ALGO = 8


def _check(status: Int32, what: StaticString) raises:
    if status != 0:
        raise Error("cuDNN ", what, " failed with status ", status)


# ── bindings (C ABI: pointers as Int, int as Int32, size_t as Int) ─────────

def _create(name: StaticString, out_addr: Int) raises -> Int32:
    """`cudnnCreate` / `cudnnCreate*Descriptor`: one `T*` out-argument."""
    if name == "cudnnCreate":
        return _get_dylib_function["cudnnCreate", def(Int) thin abi("C") -> Int32]()(out_addr)
    if name == "cudnnCreateTensorDescriptor":
        return _get_dylib_function["cudnnCreateTensorDescriptor", def(Int) thin abi("C") -> Int32]()(out_addr)
    if name == "cudnnCreateFilterDescriptor":
        return _get_dylib_function["cudnnCreateFilterDescriptor", def(Int) thin abi("C") -> Int32]()(out_addr)
    return _get_dylib_function["cudnnCreateConvolutionDescriptor", def(Int) thin abi("C") -> Int32]()(out_addr)


def _new(name: StaticString) raises -> Int:
    var h = 0
    _check(_create(name, Int(UnsafePointer(to=h))), name)
    return h


def _set_tensor(desc: Int, fmt: Int32, n: Int, c: Int, h: Int, w: Int) raises:
    _check(
        _get_dylib_function[
            "cudnnSetTensor4dDescriptor",
            def(Int, Int32, Int32, Int32, Int32, Int32, Int32) thin abi("C") -> Int32,
        ]()(desc, fmt, _FLOAT, Int32(n), Int32(c), Int32(h), Int32(w)),
        "cudnnSetTensor4dDescriptor",
    )


def _set_filter(desc: Int, fmt: Int32, k: Int, c: Int, h: Int, w: Int) raises:
    _check(
        _get_dylib_function[
            "cudnnSetFilter4dDescriptor",
            def(Int, Int32, Int32, Int32, Int32, Int32, Int32) thin abi("C") -> Int32,
        ]()(desc, _FLOAT, fmt, Int32(k), Int32(c), Int32(h), Int32(w)),
        "cudnnSetFilter4dDescriptor",
    )


def _set_conv(desc: Int, pad: Int, stride: Int, math: Int32) raises:
    _check(
        _get_dylib_function[
            "cudnnSetConvolution2dDescriptor",
            def(Int, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) thin abi("C") -> Int32,
        ]()(desc, Int32(pad), Int32(pad), Int32(stride), Int32(stride), 1, 1, _CROSS_CORRELATION, _FLOAT),
        "cudnnSetConvolution2dDescriptor",
    )
    _check(
        _get_dylib_function[
            "cudnnSetConvolutionMathType", def(Int, Int32) thin abi("C") -> Int32
        ]()(desc, math),
        "cudnnSetConvolutionMathType",
    )


def _find(
    name: StaticString, handle: Int, a: Int, b: Int, conv: Int, d: Int,
    perf: Int,
) raises -> Int:
    """`cudnnFindConvolution{Forward,BackwardData,BackwardFilter}Algorithm`:
    `(handle, desc, desc, conv, desc, requested, *returned, perf*)`."""
    var returned: Int32 = 0
    var ret_addr = Int(UnsafePointer(to=returned))
    var st: Int32
    comptime if CUDNN_ALGO == "heur":
        if name == "fwd":
            st = _get_dylib_function[
                "cudnnGetConvolutionForwardAlgorithm_v7",
                def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
            ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
        elif name == "bwd_data":
            st = _get_dylib_function[
                "cudnnGetConvolutionBackwardDataAlgorithm_v7",
                def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
            ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
        else:
            st = _get_dylib_function[
                "cudnnGetConvolutionBackwardFilterAlgorithm_v7",
                def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
            ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
        _check(st, "cudnnGetConvolution*Algorithm_v7")
        _ = returned
        return Int(returned)
    if name == "fwd":
        st = _get_dylib_function[
            "cudnnFindConvolutionForwardAlgorithm",
            def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
    elif name == "bwd_data":
        st = _get_dylib_function[
            "cudnnFindConvolutionBackwardDataAlgorithm",
            def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
    else:
        st = _get_dylib_function[
            "cudnnFindConvolutionBackwardFilterAlgorithm",
            def(Int, Int, Int, Int, Int, Int32, Int, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, Int32(_N_ALGO), ret_addr, perf)
    _check(st, "cudnnFindConvolution*Algorithm")
    _ = returned
    return Int(returned)


def _workspace_size(
    name: StaticString, handle: Int, a: Int, b: Int, conv: Int, d: Int, algo: Int32,
) raises -> Int:
    var size = 0
    var size_addr = Int(UnsafePointer(to=size))
    var st: Int32
    if name == "fwd":
        st = _get_dylib_function[
            "cudnnGetConvolutionForwardWorkspaceSize",
            def(Int, Int, Int, Int, Int, Int32, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, algo, size_addr)
    elif name == "bwd_data":
        st = _get_dylib_function[
            "cudnnGetConvolutionBackwardDataWorkspaceSize",
            def(Int, Int, Int, Int, Int, Int32, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, algo, size_addr)
    else:
        st = _get_dylib_function[
            "cudnnGetConvolutionBackwardFilterWorkspaceSize",
            def(Int, Int, Int, Int, Int, Int32, Int) thin abi("C") -> Int32,
        ]()(handle, a, b, conv, d, algo, size_addr)
    _check(st, "cudnnGetConvolution*WorkspaceSize")
    _ = size
    return size


def _run(
    name: StaticString, handle: Int, alpha: Int, d1: Int, p1: Int, d2: Int, p2: Int,
    conv: Int, algo: Int32, ws: Int, ws_size: Int, beta: Int, d3: Int, p3: Int,
) raises:
    """`cudnnConvolution{Forward,BackwardData,BackwardFilter}`: all three take
    `(handle, alpha*, desc, ptr, desc, ptr, conv, algo, ws, ws_size, beta*,
    desc, ptr)`."""
    var st: Int32
    if name == "fwd":
        st = _get_dylib_function[
            "cudnnConvolutionForward",
            def(Int, Int, Int, Int, Int, Int, Int, Int32, Int, Int, Int, Int, Int) thin abi("C") -> Int32,
        ]()(handle, alpha, d1, p1, d2, p2, conv, algo, ws, ws_size, beta, d3, p3)
    elif name == "bwd_data":
        st = _get_dylib_function[
            "cudnnConvolutionBackwardData",
            def(Int, Int, Int, Int, Int, Int, Int, Int32, Int, Int, Int, Int, Int) thin abi("C") -> Int32,
        ]()(handle, alpha, d1, p1, d2, p2, conv, algo, ws, ws_size, beta, d3, p3)
    else:
        st = _get_dylib_function[
            "cudnnConvolutionBackwardFilter",
            def(Int, Int, Int, Int, Int, Int, Int, Int32, Int, Int, Int, Int, Int) thin abi("C") -> Int32,
        ]()(handle, alpha, d1, p1, d2, p2, conv, algo, ws, ws_size, beta, d3, p3)
    _check(st, "cudnnConvolution*")


# ── per-context state ──────────────────────────────────────────────────────

@fieldwise_init
struct _Handle(TrivialRegisterPassable):
    var handle: Int


@fieldwise_init
struct _Workspace(Movable):
    var buf: DeviceBuffer[DType.uint8]
    var size: Int


@fieldwise_init
struct _Plan(TrivialRegisterPassable):
    var x: Int
    var y: Int
    var w: Int
    var conv_fwd: Int
    var conv_bd: Int
    var conv_bf: Int
    var algo_fwd: Int32
    var algo_bd: Int32
    var algo_bf: Int32
    var ws: Int


def _handle(c: DeviceContext) raises -> Int:
    var name = String("NOEIRA_CUDNN_HANDLE_", c.id())
    var g = _get_global_or_null(name)
    var h: Int
    if g:
        h = g.value().unsafe_bitcast[_Handle]()[].handle
    else:
        h = _new("cudnnCreate")
        var p = alloc(AllocLayout[_Handle].single()).unsafe_leak()
        p.unsafe_write(_Handle(h))
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(name), p.bitcast[NoneType]()
        )
    _check(
        _get_dylib_function["cudnnSetStream", def(Int, type_of(CUDA(c.stream()))) thin abi("C") -> Int32]()(
            h, CUDA(c.stream())
        ),
        "cudnnSetStream",
    )
    return h


def _workspace(c: DeviceContext, need: Int) raises -> Int:
    """The context's shared workspace, grown to at least `need` bytes. A
    replaced buffer is leaked on purpose: a captured graph may still hold
    its pointer."""
    var name = String("NOEIRA_CUDNN_WS_", c.id())
    var g = _get_global_or_null(name)
    if g:
        var p = g.value().unsafe_bitcast[_Workspace]()
        if p[].size >= need:
            return Int(p[].buf.unsafe_ptr())
        var bigger = c.enqueue_create_buffer[DType.uint8](need)
        # Keep the old buffer alive (leaked): a graph may hold its pointer.
        var keep = alloc(AllocLayout[DeviceBuffer[DType.uint8]].single()).unsafe_leak()
        keep.unsafe_write(p[].buf.copy())
        p[].buf = bigger^
        p[].size = need
        return Int(p[].buf.unsafe_ptr())
    var n = max(need, 1 << 20)
    var p = alloc(AllocLayout[_Workspace].single()).unsafe_leak()
    p.unsafe_write(_Workspace(c.enqueue_create_buffer[DType.uint8](n), n))
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    return Int(p[].buf.unsafe_ptr())


def _allowed(what: StaticString, algo: Int32) -> Bool:
    """The implicit / explicit GEMM algorithms only: no Winograd and no FFT
    (lower fp32 accuracy), and no atomic backward algorithms (non-
    deterministic). Forward: IMPLICIT_GEMM 0, IMPLICIT_PRECOMP_GEMM 1, GEMM 2.
    Backward data: ALGO_1. Backward filter: ALGO_1."""
    if what == "forward":
        return algo == 0 or algo == 1 or algo == 2
    return algo == 1


def _pick(perf: UnsafePointer[UInt8, _], n: Int, what: StaticString) raises -> Int32:
    """The first allowed, deterministic, successful algorithm in cuDNN's
    order (heuristic rank, or time with `find`)."""
    for i in range(n):
        var base = perf + i * _PERF_BYTES
        var algo = base.bitcast[Int32]()[0]
        var status = base.bitcast[Int32]()[1]
        var determinism = (base + 24).bitcast[Int32]()[0]
        if status == 0 and determinism == _DETERMINISTIC and _allowed(what, algo):
            return algo
    raise Error("cuDNN: no deterministic ", what, " algorithm")


def _plan[
    B: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    NHWC: Bool, TF32: Bool,
](c: DeviceContext, handle: Int) raises -> _Plan:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    var name = String(
        "NOEIRA_CUDNN_PLAN_", c.id(), "_", B, "_", IC, "_", OC, "_", K, "_", S,
        "_", P, "_", H, "_", W, "_", NHWC, "_", TF32,
    )
    var g = _get_global_or_null(name)
    if g:
        return g.value().unsafe_bitcast[_Plan]()[]
    comptime fmt = _NHWC if NHWC else _NCHW
    comptime math = _DEFAULT_MATH if TF32 else _FMA_MATH
    var x = _new("cudnnCreateTensorDescriptor")
    var y = _new("cudnnCreateTensorDescriptor")
    var w = _new("cudnnCreateFilterDescriptor")
    _set_tensor(x, fmt, B, IC, H, W)
    _set_tensor(y, fmt, B, OC, OH, OW)
    _set_filter(w, fmt, OC, IC, K, K)
    # One convolution descriptor per direction: the math type is a property
    # of the descriptor and each direction's search may settle on its own.
    var cf = _new("cudnnCreateConvolutionDescriptor")
    var cd = _new("cudnnCreateConvolutionDescriptor")
    var cb = _new("cudnnCreateConvolutionDescriptor")
    _set_conv(cf, P, S, math)
    _set_conv(cd, P, S, math)
    _set_conv(cb, P, S, math)
    # cudnnFind* runs the candidates on real buffers it allocates itself.
    c.synchronize()
    var perf_alloc = alloc[UInt8]({count = _N_ALGO * _PERF_BYTES}).into_managed()
    var perf = UnsafePointer(perf_alloc.unsafe_ptr())
    var pp = perf
    var nf = _find("fwd", handle, x, w, cf, y, Int(perf))
    var af = _pick(pp, nf, "forward")
    var nd = _find("bwd_data", handle, w, y, cd, x, Int(perf))
    var ad = _pick(pp, nd, "backward-data")
    var nb = _find("bwd_filter", handle, x, y, cb, w, Int(perf))
    var ab = _pick(pp, nb, "backward-filter")
    _ = perf_alloc^
    comptime if CUDNN_VERBOSE:
        print("[cudnn] plan", name, "fwd", af, "bwd_data", ad, "bwd_filter", ab)
    var ws = max(
        _workspace_size("fwd", handle, x, w, cf, y, af),
        max(
            _workspace_size("bwd_data", handle, w, y, cd, x, ad),
            _workspace_size("bwd_filter", handle, x, y, cb, w, ab),
        ),
    )
    var plan = _Plan(x, y, w, cf, cd, cb, af, ad, ab, ws)
    var p = alloc(AllocLayout[_Plan].single()).unsafe_leak()
    p.unsafe_write(plan)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), p.bitcast[NoneType]()
    )
    return plan


# ── public API ─────────────────────────────────────────────────────────────

def cudnn_conv_forward[
    B: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    NHWC: Bool, TF32: Bool,
](
    c: DeviceContext,
    y: DeviceBuffer[DT],
    x: DeviceBuffer[DT],
    w: DeviceBuffer[DT],
) raises:
    """`y = conv(x, w)` (no bias), β = 0."""
    var h = _handle(c)
    var p = _plan[B, IC, OC, K, S, P, H, W, NHWC, TF32](c, h)
    var ws = _workspace(c, p.ws)
    var alpha = Float32(1.0)
    var beta = Float32(0.0)
    _run(
        "fwd", h, Int(UnsafePointer(to=alpha)), p.x, Int(x.unsafe_ptr()),
        p.w, Int(w.unsafe_ptr()), p.conv_fwd, p.algo_fwd, ws, p.ws,
        Int(UnsafePointer(to=beta)), p.y, Int(y.unsafe_ptr()),
    )
    _ = alpha
    _ = beta


def cudnn_conv_backward_data[
    B: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    NHWC: Bool, TF32: Bool,
](
    c: DeviceContext,
    dx: DeviceBuffer[DT],
    w: DeviceBuffer[DT],
    dy: DeviceBuffer[DT],
) raises:
    """`dx = conv_transpose(dy, w)`, β = 0 (overwrites)."""
    var h = _handle(c)
    var p = _plan[B, IC, OC, K, S, P, H, W, NHWC, TF32](c, h)
    var ws = _workspace(c, p.ws)
    var alpha = Float32(1.0)
    var beta = Float32(0.0)
    _run(
        "bwd_data", h, Int(UnsafePointer(to=alpha)), p.w, Int(w.unsafe_ptr()),
        p.y, Int(dy.unsafe_ptr()), p.conv_bd, p.algo_bd, ws, p.ws,
        Int(UnsafePointer(to=beta)), p.x, Int(dx.unsafe_ptr()),
    )
    _ = alpha
    _ = beta


def cudnn_conv_backward_filter[
    B: Int, IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    NHWC: Bool, TF32: Bool,
](
    c: DeviceContext,
    dw: DeviceBuffer[DT],
    x: DeviceBuffer[DT],
    dy: DeviceBuffer[DT],
) raises:
    """`dw += xᵀ ⋆ dy` (β = 1: accumulates into the master gradient)."""
    var h = _handle(c)
    var p = _plan[B, IC, OC, K, S, P, H, W, NHWC, TF32](c, h)
    var ws = _workspace(c, p.ws)
    var alpha = Float32(1.0)
    var beta = Float32(1.0)
    _run(
        "bwd_filter", h, Int(UnsafePointer(to=alpha)), p.x, Int(x.unsafe_ptr()),
        p.y, Int(dy.unsafe_ptr()), p.conv_bf, p.algo_bf, ws, p.ws,
        Int(UnsafePointer(to=beta)), p.w, Int(dw.unsafe_ptr()),
    )
    _ = alpha
    _ = beta
