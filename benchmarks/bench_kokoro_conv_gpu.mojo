"""Kokoro's generator convolutions: im2col + cuBLAS against cuDNN (NVIDIA).

    pixi run -e nvidia mojo build -I . benchmarks/bench_kokoro_conv_gpu.mojo -o build/bench_kokoro_conv
    ./build/bench_kokoro_conv            # on the Orin: JetPack's cuBLAS first on LD_LIBRARY_PATH

The shapes are the generator's for one 84-frame sentence: stage 0 `[1680, 256]`,
stage 1 `[10081, 128]`, kernels 3 / 7 / 11 at dilations 1 / 3 / 5 ("same"
padding, stride 1). Activations channels-last, weights `[COUT, K, CIN]` as
`KokoroDevice` holds them — for cuDNN an NHWC filter `[K=COUT, R=1, S=K, C=CIN]`.

For each shape and precision (fp32 = cuBLAS fp32 / cuDNN FMA_MATH; TF32 =
cuBLAS TF32 / cuDNN DEFAULT_MATH) it prints the im2col path's time, every
algorithm `cudnnFindConvolutionForwardAlgorithm` returns, and the best one's
time and error against the im2col path's output."""

from std.collections import Optional
from std.time import perf_counter_ns
from std.random import random_float64, seed
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from _cudnn.infer import (
    cudnnContext,
    cudnnCreate,
    cudnnCreateFilterDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnDataType_t,
    cudnnFilterStruct,
    cudnnMathType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorStruct,
    cudnnConvolutionFwdAlgo_t,
)
from _cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionFwdAlgoPerfStruct,
    cudnnConvolutionMode_t,
    cudnnConvolutionStruct,
    cudnnCreateConvolutionDescriptor,
    cudnnFindConvolutionForwardAlgorithm,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionMathType,
)

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from noeira.audio.kokoro.device import _im2col_kernel, TPB, IM2COL_BUDGET

comptime REPS = 20


def ck(s: cudnnStatus_t, what: String) raises:
    if s != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        raise Error(String("cuDNN ", what, ": ", s))


def fill(ctx: DeviceContext, n: Int, scale: Float64) raises -> DeviceBuffer[DT]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    for i in range(n):
        h[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * scale)
    var d = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(d, h)
    return d


def im2col_conv[TF32: Bool](
    ctx: DeviceContext, x: DeviceBuffer[DT], w: DeviceBuffer[DT], y: DeviceBuffer[DT],
    cols: DeviceBuffer[DT], l: Int, c: Int, k: Int, dil: Int,
) raises:
    """`KokoroDevice.conv1d`'s path, stride 1, "same" padding, COUT = CIN."""
    var pad = (k * dil - dil) // 2
    var width = c * k
    var chunk = min(max(IM2COL_BUDGET // width, 1), l)
    var row0 = 0
    while row0 < l:
        var rows = chunk if row0 + chunk <= l else l - row0
        ctx.enqueue_function[_im2col_kernel](
            x, cols, Int64(l), Int64(c), Int64(k), Int64(row0), Int64(rows),
            Int64(1), Int64(pad), Int64(dil),
            grid_dim=((c + TPB - 1) // TPB, k, rows), block_dim=TPB,
        )
        cublas_gemm[False, True, TF32](
            ctx, y.create_sub_buffer[DT](row0 * c, rows * c), cols, w, rows, c, width, 0.0,
        )
        row0 += rows


def rel_err(ctx: DeviceContext, a: DeviceBuffer[DT], b: DeviceBuffer[DT], n: Int) raises -> Float64:
    var ha = ctx.enqueue_create_host_buffer[DT](n)
    var hb = ctx.enqueue_create_host_buffer[DT](n)
    ctx.enqueue_copy(ha, a)
    ctx.enqueue_copy(hb, b)
    ctx.synchronize()
    var d = 0.0
    var m = 0.0
    for i in range(n):
        d = max(d, abs(Float64(ha[i]) - Float64(hb[i])))
        m = max(m, abs(Float64(hb[i])))
    return d / m


def algo_name(a: cudnnConvolutionFwdAlgo_t) -> String:
    var names: List[String] = [
        "IMPLICIT_GEMM", "IMPLICIT_PRECOMP_GEMM", "GEMM", "DIRECT", "FFT", "FFT_TILING",
        "WINOGRAD", "WINOGRAD_NONFUSED",
    ]
    var i = Int(a._value)
    return names[i] if i >= 0 and i < len(names) else String("algo ", i)


def bench_shape[TF32: Bool](
    ctx: DeviceContext, h: UnsafePointer[cudnnContext, AnyOrigin[mut=True]],
    xd: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]],
    yd: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]],
    wd: UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]],
    cd: UnsafePointer[cudnnConvolutionStruct, AnyOrigin[mut=True]],
    l: Int, c: Int, k: Int, dil: Int,
) raises -> Tuple[Float64, Float64, Float64]:
    var pad = (k * dil - dil) // 2
    var x = fill(ctx, l * c, 1.0)
    var w = fill(ctx, c * k * c, 1.0 / Float64(c * k) ** 0.5)
    var y_ref = ctx.enqueue_create_buffer[DT](l * c)
    var y = ctx.enqueue_create_buffer[DT](l * c)
    var width = c * k
    var cols = ctx.enqueue_create_buffer[DT](min(max(IM2COL_BUDGET // width, 1), l) * width)

    # im2col + cuBLAS
    im2col_conv[TF32](ctx, x, w, y_ref, cols, l, c, k, dil)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        im2col_conv[TF32](ctx, x, w, y_ref, cols, l, c, k, dil)
    ctx.synchronize()
    var t_im2col = Float64(perf_counter_ns() - t0) / 1e3 / REPS

    # cuDNN
    ck(cudnnSetTensor4dDescriptor(xd, cudnnTensorFormat_t.CUDNN_TENSOR_NHWC, cudnnDataType_t.CUDNN_DATA_FLOAT,
        1, Int32(c), 1, Int32(l)), "x desc")
    ck(cudnnSetTensor4dDescriptor(yd, cudnnTensorFormat_t.CUDNN_TENSOR_NHWC, cudnnDataType_t.CUDNN_DATA_FLOAT,
        1, Int32(c), 1, Int32(l)), "y desc")
    ck(cudnnSetFilter4dDescriptor(wd, cudnnDataType_t.CUDNN_DATA_FLOAT, cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
        Int32(c), Int32(c), 1, Int32(k)), "w desc")
    ck(cudnnSetConvolution2dDescriptor(cd, 0, Int32(pad), 1, 1, 1, Int32(dil),
        cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudnnDataType_t.CUDNN_DATA_FLOAT), "conv desc")
    comptime math = cudnnMathType_t.CUDNN_DEFAULT_MATH if TF32 else cudnnMathType_t.CUDNN_FMA_MATH
    ck(cudnnSetConvolutionMathType(cd, math), "math type")

    var perf_alloc = alloc[cudnnConvolutionFwdAlgoPerfStruct]({count = 8}).into_managed()
    var perf = perf_alloc.unsafe_ptr()
    var got: Int32 = 0
    ck(cudnnFindConvolutionForwardAlgorithm(h, xd, wd, cd, yd, 8, UnsafePointer(to=got), perf), "find")
    ctx.synchronize()
    var best = -1
    for i in range(Int(got)):
        var p = perf[unsafe_offset=i]
        if p.status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
            print("       (", algo_name(p.algo), ": ", p.status, ")")
        if p.status == cudnnStatus_t.CUDNN_STATUS_SUCCESS:
            print("      ", algo_name(p.algo), " ", p.time, " ms  ws ", p.memory // 1024, " KiB  math ", Int(p.mathType._value))
            if best < 0:
                best = i
    if best < 0:
        raise Error("no cuDNN algorithm")
    var algo = perf[unsafe_offset=best].algo
    var ws_size = 0
    ck(cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, algo, UnsafePointer(to=ws_size)), "ws")
    var ws = ctx.enqueue_create_buffer[DType.uint8](max(ws_size, 1))
    var alpha: Float32 = 1.0
    var beta: Float32 = 0.0
    ck(cudnnConvolutionForward(h, UnsafePointer(to=alpha).bitcast[NoneType](), xd, x.unsafe_ptr().bitcast[NoneType](),
        wd, w.unsafe_ptr().bitcast[NoneType](), cd, algo, ws.unsafe_ptr().bitcast[NoneType](), ws_size,
        UnsafePointer(to=beta).bitcast[NoneType](), yd, y.unsafe_ptr().bitcast[NoneType]()), "forward")
    ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(REPS):
        ck(cudnnConvolutionForward(h, UnsafePointer(to=alpha).bitcast[NoneType](), xd, x.unsafe_ptr().bitcast[NoneType](),
            wd, w.unsafe_ptr().bitcast[NoneType](), cd, algo, ws.unsafe_ptr().bitcast[NoneType](), ws_size,
            UnsafePointer(to=beta).bitcast[NoneType](), yd, y.unsafe_ptr().bitcast[NoneType]()), "forward")
    ctx.synchronize()
    var t_cudnn = Float64(perf_counter_ns() - t0) / 1e3 / REPS
    var err = rel_err(ctx, y, y_ref, l * c)
    return (t_im2col, t_cudnn, err)


def main() raises:
    seed(7)
    var ctx = DeviceContext()
    with ctx.push_context() as c:
        var h_o: Optional[UnsafePointer[cudnnContext, AnyOrigin[mut=True]]] = None
        ck(cudnnCreate(UnsafePointer(to=h_o)), "create")
        var h = h_o.value()
        ck(cudnnSetStream(h, CUDA(c.stream())), "stream")
        var xd_o: Optional[UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]] = None
        var yd_o: Optional[UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]] = None
        var wd_o: Optional[UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]]] = None
        var cd_o: Optional[UnsafePointer[cudnnConvolutionStruct, AnyOrigin[mut=True]]] = None
        ck(cudnnCreateTensorDescriptor(UnsafePointer(to=xd_o)), "x")
        ck(cudnnCreateTensorDescriptor(UnsafePointer(to=yd_o)), "y")
        ck(cudnnCreateFilterDescriptor(UnsafePointer(to=wd_o)), "w")
        ck(cudnnCreateConvolutionDescriptor(UnsafePointer(to=cd_o)), "conv")
        var xd = xd_o.value()
        var yd = yd_o.value()
        var wd = wd_o.value()
        var cd = cd_o.value()

        var stages: List[Tuple[Int, Int]] = [(1680, 256), (10081, 128)]
        var kernels: List[Int] = [3, 7, 11]
        var dils: List[Int] = [1, 3, 5]
        var tot: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        for s in stages:
            for k in kernels:
                for d in dils:
                    var l = s[0]
                    var ch = s[1]
                    print("[", l, ", ", ch, "] k ", k, " dil ", d)
                    print("    fp32:")
                    var r32 = bench_shape[False](c, h, xd, yd, wd, cd, l, ch, k, d)
                    print("    -> im2col+cuBLAS ", r32[0], " us | cuDNN ", r32[1], " us | rel err ", r32[2])
                    print("    tf32:")
                    var r16 = bench_shape[True](c, h, xd, yd, wd, cd, l, ch, k, d)
                    print("    -> im2col+cuBLAS ", r16[0], " us | cuDNN ", r16[1], " us | rel err ", r16[2])
                    tot[0] += r32[0]
                    tot[1] += r32[1]
                    tot[2] += r16[0]
                    tot[3] += r16[1]
        print("SUM over the 18 shapes (one conv each; the generator runs 48): fp32 im2col ", tot[0] / 1e3,
            " ms, cuDNN ", tot[1] / 1e3, " ms | tf32 im2col ", tot[2] / 1e3, " ms, cuDNN ", tot[3] / 1e3, " ms")
