# +--------------------------------------------------------------------------+ #
# | Kokoro's Conv1d through cuDNN (NVIDIA, TF32 build)
# +--------------------------------------------------------------------------+ #
"""`KokoroDevice.conv1d` as one cuDNN implicit-GEMM call, for the TF32 build on
NVIDIA (`-D KOKORO_TF32=1`).

## Why

The im2col path writes every input window out (`[L_out, K CIN]`) and the GEMM
reads it back. For one 84-frame sentence that is ~1.3 GB written and read again
by the generator's 48 convolutions. On the Orin Nano (68 GB/s) that traffic is
most of their time. cuDNN's `IMPLICIT_PRECOMP_GEMM` gathers the windows inside
the GEMM. Measured on the Orin (`benchmarks/bench_kokoro_conv_gpu.mojo`, the
generator's 18 shapes, TF32): 9.6 ms against 18.7 for im2col + cuBLAS, 2-3x per
shape at stage 1, outputs within 3.5e-6.

## Only in TF32

With TF32 forbidden (`CUDNN_FMA_MATH`) the same cuDNN takes 57-205 ms per
DILATED shape: the fp32 build keeps im2col.

## Which cuDNN

The `jetson` pixi environment ships cuDNN 9.10.2, a server build: on the Orin
every convolution fails (`CUDNN_STATUS_EXECUTION_FAILED`). JetPack's (9.3,
`/usr/lib/aarch64-linux-gnu`) must be found first. `libcudnn.so.9` is opened by
name, so put a directory holding links to JetPack's `libcudnn*.so.9` FIRST on
`LD_LIBRARY_PATH` (all of `/usr/lib/aarch64-linux-gnu` first would shadow the
environment's other libraries). `CudnnConv1d.create` runs one small convolution
against a host reference and returns None when cuDNN is missing or unusable; the
caller then keeps im2col.

## Layout

Activations channels-last `[L, C]` = NHWC with N = H = 1, W = L. The device's
conv weights are `[COUT, K, CIN]` = an NHWC filter `[K=COUT, R=1, S=K, C=CIN]`.
"""

from std.collections import Optional
from std.random import random_float64
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from _cudnn.infer import (
    cudnnContext,
    cudnnCreate,
    cudnnCreateFilterDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnDataType_t,
    cudnnDestroy,
    cudnnDestroyFilterDescriptor,
    cudnnDestroyTensorDescriptor,
    cudnnGetVersion,
    cudnnMathType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnConvolutionFwdAlgo_t,
)
from _cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionMode_t,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionMathType,
)

from noeira.nn.constants import DT

comptime _CPtr = UnsafePointer[cudnnContext, AnyOrigin[mut=True]]
comptime _ALGO = cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM


def _ck(s: cudnnStatus_t, what: String) raises:
    if s != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        raise Error(String("cuDNN ", what, ": ", s))


struct CudnnConv1d(Movable):
    """A cuDNN handle on the context's stream and one set of descriptors, set
    again for each call (a few us of host time)."""

    @__allow_legacy_any_origin_fields
    var handle: _CPtr

    @__allow_legacy_any_origin_fields
    var xd: _CPtr

    @__allow_legacy_any_origin_fields
    var yd: _CPtr

    @__allow_legacy_any_origin_fields
    var wd: _CPtr

    @__allow_legacy_any_origin_fields
    var cd: _CPtr

    var ws: DeviceBuffer[DType.uint8]
    var ws_size: Int
    var version: Int

    def __init__(out self, ctx: DeviceContext) raises:
        with ctx.push_context() as c:
            var h: Optional[_CPtr] = None
            _ck(cudnnCreate(UnsafePointer(to=h)), "create")
            self.handle = h.value()
            _ck(cudnnSetStream(self.handle, CUDA(c.stream())), "set stream")
            var xd: Optional[_CPtr] = None
            var yd: Optional[_CPtr] = None
            var wd: Optional[_CPtr] = None
            var cd: Optional[_CPtr] = None
            _ck(cudnnCreateTensorDescriptor(UnsafePointer(to=xd)), "x descriptor")
            _ck(cudnnCreateTensorDescriptor(UnsafePointer(to=yd)), "y descriptor")
            _ck(cudnnCreateFilterDescriptor(UnsafePointer(to=wd)), "filter descriptor")
            _ck(cudnnCreateConvolutionDescriptor(UnsafePointer(to=cd)), "conv descriptor")
            self.xd = xd.value()
            self.yd = yd.value()
            self.wd = wd.value()
            self.cd = cd.value()
            self.ws = c.enqueue_create_buffer[DType.uint8](1)
            self.ws_size = 0
            self.version = cudnnGetVersion()

    def __deinit__(deinit self):
        try:
            _ = cudnnDestroyConvolutionDescriptor(self.cd)
            _ = cudnnDestroyFilterDescriptor(self.wd)
            _ = cudnnDestroyTensorDescriptor(self.yd)
            _ = cudnnDestroyTensorDescriptor(self.xd)
            _ = cudnnDestroy(self.handle)
        except:
            pass

    @staticmethod
    def create(ctx: DeviceContext) -> Optional[CudnnConv1d]:
        """A usable cuDNN, or None (missing library, or one whose kernels do
        not run here): one dilated, strided convolution against a host loop."""
        try:
            var cv = CudnnConv1d(ctx)
            comptime L = 37
            comptime CI = 8
            comptime CO = 4
            comptime K = 3
            comptime STRIDE = 2
            comptime PAD = 2
            comptime DIL = 2
            var lo = (L + 2 * PAD - DIL * (K - 1) - 1) // STRIDE + 1
            var hx = ctx.enqueue_create_host_buffer[DT](L * CI)
            var hw = ctx.enqueue_create_host_buffer[DT](CO * K * CI)
            for i in range(L * CI):
                hx[i] = Scalar[DT](random_float64() - 0.5)
            for i in range(CO * K * CI):
                hw[i] = Scalar[DT](random_float64() - 0.5)
            var x = ctx.enqueue_create_buffer[DT](L * CI)
            var w = ctx.enqueue_create_buffer[DT](CO * K * CI)
            var y = ctx.enqueue_create_buffer[DT](lo * CO)
            ctx.enqueue_copy(x, hx)
            ctx.enqueue_copy(w, hw)
            cv.forward(ctx, x, w, y, L, CI, CO, K, lo, STRIDE, PAD, DIL)
            var hy = ctx.enqueue_create_host_buffer[DT](lo * CO)
            ctx.enqueue_copy(hy, y)
            ctx.synchronize()
            var worst = Float64(0)
            for t in range(lo):
                for o in range(CO):
                    var acc = Float64(0)
                    for j in range(K):
                        var s = t * STRIDE - PAD + j * DIL
                        if s >= 0 and s < L:
                            for c in range(CI):
                                acc += Float64(hw[o * K * CI + j * CI + c]) * Float64(hx[s * CI + c])
                    worst = max(worst, abs(acc - Float64(hy[t * CO + o])))
            if worst > 1e-2:  # TF32 on 24 products of O(0.25): ~1e-3
                print("kokoro: cuDNN ", cv.version, " gives wrong convolutions here (", worst,
                    "): im2col instead")
                return None
            return cv^
        except e:
            print("kokoro: cuDNN unusable here (", e, "): im2col instead")
            return None

    def forward(
        mut self, ctx: DeviceContext,
        x: DeviceBuffer[DT], w: DeviceBuffer[DT], y: DeviceBuffer[DT],
        l_in: Int, c_in: Int, c_out: Int, k: Int, l_out: Int,
        stride: Int, pad: Int, dil: Int,
    ) raises:
        """`y[l_out, c_out] = conv(x[l_in, c_in], w[c_out, k, c_in])`, TF32."""
        with ctx.push_context():
            _ck(cudnnSetTensor4dDescriptor(self.xd, cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                cudnnDataType_t.CUDNN_DATA_FLOAT, 1, Int32(c_in), 1, Int32(l_in)), "x descriptor")
            _ck(cudnnSetTensor4dDescriptor(self.yd, cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                cudnnDataType_t.CUDNN_DATA_FLOAT, 1, Int32(c_out), 1, Int32(l_out)), "y descriptor")
            _ck(cudnnSetFilter4dDescriptor(self.wd, cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC, Int32(c_out), Int32(c_in), 1, Int32(k)), "filter descriptor")
            _ck(cudnnSetConvolution2dDescriptor(self.cd, 0, Int32(pad), 1, Int32(stride), 1, Int32(dil),
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudnnDataType_t.CUDNN_DATA_FLOAT), "conv descriptor")
            # DEFAULT_MATH lets an fp32 convolution run in TF32 on Ampere and later.
            _ck(cudnnSetConvolutionMathType(self.cd, cudnnMathType_t.CUDNN_DEFAULT_MATH), "math type")
            var need = 0
            _ck(cudnnGetConvolutionForwardWorkspaceSize(
                self.handle, self.xd, self.wd, self.cd, self.yd, _ALGO, UnsafePointer(to=need)), "workspace size")
            if need > self.ws_size:
                self.ws = ctx.enqueue_create_buffer[DType.uint8](need)
                self.ws_size = need
            var alpha: Float32 = 1.0
            var beta: Float32 = 0.0
            _ck(cudnnConvolutionForward(
                self.handle, UnsafePointer(to=alpha).bitcast[NoneType](),
                self.xd, x.unsafe_ptr().bitcast[NoneType](),
                self.wd, w.unsafe_ptr().bitcast[NoneType](),
                self.cd, _ALGO, self.ws.unsafe_ptr().bitcast[NoneType](), self.ws_size,
                UnsafePointer(to=beta).bitcast[NoneType](),
                self.yd, y.unsafe_ptr().bitcast[NoneType](),
            ), "forward")
