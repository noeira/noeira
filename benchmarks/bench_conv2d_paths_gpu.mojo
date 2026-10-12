"""Conv2D GPU time per call, MAX's padded GEMMs vs cuBLAS, at the conv shapes of
the ResNet-20 and CNN CIFAR examples (batch 100) — and the PyTorch / cuDNN
twin `tools/nn/bench_conv_torch.py` on the same shapes.

    pixi run mojo build -I . -D NN_GEMM_PATH=max  benchmarks/bench_conv2d_paths_gpu.mojo -o bc_max
    pixi run mojo build -I . -D NN_GEMM_PATH=auto benchmarks/bench_conv2d_paths_gpu.mojo -o bc_auto
    pixi run -e default ./bc_max ; pixi run -e default ./bc_auto

Per shape, the GPU time of ONE call inside a captured CUDA graph of R
back-to-back calls (no launch overhead), averaged over NREP replays: `fwd`
(im2col, GEMM, scatter + bias) and `fwd+bwd` (then the vjp: im2col reuse,
grad transpose, dW and d_col GEMMs, bias grad, col2im). Run through
`pixi run` so the CUDA interceptor is preloaded.
"""

from std.time import perf_counter_ns
from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.cuda.graph import CUDAGraph, maybe_capture_replay
from std.sys.defines import get_defined_int
from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.cublas_gemm import GEMM_PATH
from noeira.nn.core.cudnn_conv import CONV_PATH
from noeira.nn.primitives.conv2d import Conv2D


comptime R = 20
"""Calls per captured graph."""
comptime NREP = 20
"""Timed replays per measurement."""
comptime PART = get_defined_int["BC_PART", 0]()
"""0: the ResNet-20 / CNN examples (batch 100); 1-3: the rest of noeira's conv
shapes (RL nets; vision backbones; world models)."""


def _fill(mut t: Tensor, n: Int, ctx: DeviceContext) raises:
    t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64(-1, 1))
    t.upload(ctx)


def _replay_us[
    STEP: def () capturing raises -> None
](ctx: DeviceContext) raises -> Float64:
    var g = Optional[CUDAGraph](None)
    maybe_capture_replay[STEP, VERBOSE=False](g, ctx)  # runs, then captures
    if not g or not g.value().is_captured():
        print("  ! graph not captured: this row is timed EAGERLY (launch-bound)")
    for _ in range(3):
        maybe_capture_replay[STEP, VERBOSE=False](g, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(NREP):
        maybe_capture_replay[STEP, VERBOSE=False](g, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1e3 / Float64(NREP * R)


def bench_conv[IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int = 100, LAYOUT: Int = LAYOUT_NCHW](name: String, ctx: DeviceContext) raises:
    comptime Cv = Conv2D[IC, OC, K, S, P, H, W, DT, LAYOUT]
    seed(IC * 31 + OC)
    var m = Cv.make["gpu", Kaiming](Optional(ctx))
    var x = Tensor()
    var go = Tensor()
    _fill(x, B * Cv.IN_FLAT, ctx)
    _fill(go, B * Cv.OUT_FLAT, ctx)
    var y = Tensor()
    var gi = Tensor()
    m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
    m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))
    ctx.synchronize()

    def _f() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))

    def _fb() capturing raises -> None:
        for _ in range(R):
            m.forward["gpu", B](TensorRefs[1](x), y, Optional(ctx))
            m.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(ctx))

    var f = _replay_us[_f](ctx)
    var fb = _replay_us[_fb](ctx)
    print(
        "ROW path=", GEMM_PATH, "/", CONV_PATH, " name=", name, " ic=", IC, " oc=", OC, " k=", K,
        " s=", S, " p=", P, " h=", H, " w=", W, " b=", B, " nhwc=", LAYOUT == LAYOUT_NHWC,
        " cudnn=", Cv.use_cudnn[B](), " fwd_us=", f, " fwdbwd_us=", fb, sep="",
    )
    _ = m^
    _ = x^
    _ = go^
    _ = y^
    _ = gi^


def main() raises:
    var ctx = DeviceContext()
    print("Conv2D GPU us per call | NN_GEMM_PATH =", GEMM_PATH, "| NN_CONV_PATH =", CONV_PATH, "| part", PART)
    comptime if PART == 0:
        # ResNet-20 (CIFAR): stem, three stages, stride-2 blocks, 1x1 shortcuts
        bench_conv[3, 16, 3, 1, 1, 32, 32]("resnet_stem", ctx)
        bench_conv[16, 16, 3, 1, 1, 32, 32]("resnet_s1", ctx)
        bench_conv[16, 32, 3, 2, 1, 32, 32]("resnet_down2", ctx)
        bench_conv[16, 32, 1, 2, 0, 32, 32]("resnet_short2", ctx)
        bench_conv[32, 32, 3, 1, 1, 16, 16]("resnet_s2", ctx)
        bench_conv[32, 64, 3, 2, 1, 16, 16]("resnet_down3", ctx)
        bench_conv[32, 64, 1, 2, 0, 16, 16]("resnet_short3", ctx)
        bench_conv[64, 64, 3, 1, 1, 8, 8]("resnet_s3", ctx)
        # CNN CIFAR example
        bench_conv[3, 32, 3, 1, 1, 32, 32]("cnn_c1", ctx)
        bench_conv[32, 32, 3, 1, 1, 32, 32]("cnn_c2", ctx)
        bench_conv[32, 64, 3, 1, 1, 16, 16]("cnn_c3", ctx)
        bench_conv[64, 64, 3, 1, 1, 16, 16]("cnn_c4", ctx)
        bench_conv[64, 128, 3, 1, 1, 8, 8]("cnn_c5", ctx)
        bench_conv[128, 128, 3, 1, 1, 8, 8]("cnn_c6", ctx)
    comptime if PART == 1:
        # Nature CNN (Rainbow / C51 / MuZero Pong)
        bench_conv[4, 32, 8, 4, 0, 84, 84, 32]("nature_c1_b32", ctx)
        bench_conv[4, 32, 8, 4, 0, 84, 84, 256]("nature_c1_b256", ctx)
        bench_conv[4, 32, 8, 4, 0, 84, 84, 64]("nature_c1_b64", ctx)
        bench_conv[32, 64, 4, 2, 0, 20, 20, 32]("nature_c2_b32", ctx)
        bench_conv[32, 64, 4, 2, 0, 20, 20, 256]("nature_c2_b256", ctx)
        bench_conv[64, 64, 3, 1, 0, 9, 9, 32]("nature_c3_b32", ctx)
        bench_conv[64, 64, 3, 1, 0, 9, 9, 256]("nature_c3_b256", ctx)
        bench_conv[4, 32, 8, 4, 0, 84, 84, 64, LAYOUT_NHWC]("nature_c1_nhwc", ctx)
        # EfficientZero v2 Atari representation (NHWC) and 6x6 heads
        bench_conv[12, 32, 3, 2, 1, 96, 96, 256, LAYOUT_NHWC]("ez_stem", ctx)
        bench_conv[32, 32, 3, 1, 1, 48, 48, 256, LAYOUT_NHWC]("ez_48", ctx)
        bench_conv[32, 64, 3, 2, 1, 48, 48, 256, LAYOUT_NHWC]("ez_down48", ctx)
        bench_conv[64, 64, 3, 1, 1, 24, 24, 256, LAYOUT_NHWC]("ez_24", ctx)
        bench_conv[64, 64, 3, 1, 1, 12, 12, 256, LAYOUT_NHWC]("ez_12", ctx)
        bench_conv[64, 64, 3, 1, 1, 6, 6, 256, LAYOUT_NHWC]("ez_6_nhwc", ctx)
        bench_conv[64, 64, 3, 1, 1, 6, 6, 256]("ez_6", ctx)
        bench_conv[80, 64, 3, 1, 1, 6, 6, 256]("ez_dyn80", ctx)
        bench_conv[1, 16, 1, 1, 0, 6, 6, 256]("ez_act1x1", ctx)
        bench_conv[64, 16, 1, 1, 0, 6, 6, 256]("ez_head1x1", ctx)
        bench_conv[64, 64, 3, 1, 1, 6, 6, 4]("ez_6_act4", ctx)
        # Connect Four: MuZero spatial and AlphaZero
        bench_conv[3, 64, 3, 1, 1, 6, 7, 128]("c4_stem64", ctx)
        bench_conv[64, 64, 3, 1, 1, 6, 7, 128]("c4_res64", ctx)
        bench_conv[64, 64, 3, 1, 1, 6, 7, 64]("c4_res64_b64", ctx)
        bench_conv[80, 64, 3, 1, 1, 6, 7, 128]("c4_dyn80", ctx)
        bench_conv[64, 16, 1, 1, 0, 6, 7, 128]("c4_head1x1", ctx)
        bench_conv[3, 128, 3, 1, 1, 6, 7, 128]("az_stem", ctx)
        bench_conv[128, 128, 3, 1, 1, 6, 7, 128]("az_res", ctx)
        bench_conv[128, 128, 3, 1, 1, 6, 7, 64]("az_res_b64", ctx)
        # pixel student
        bench_conv[12, 32, 3, 1, 1, 16, 16, 1024]("pix_c1", ctx)
        bench_conv[32, 64, 3, 2, 1, 16, 16, 1024]("pix_c2", ctx)
        bench_conv[64, 64, 3, 2, 1, 8, 8, 1024]("pix_c3", ctx)
    comptime if PART == 2:
        # ACT ResNet-18, SO-101 240x320 (32 rows = batch 16 x 2 cams)
        bench_conv[3, 64, 7, 2, 3, 240, 320, 32]("act_stem", ctx)
        bench_conv[64, 64, 3, 1, 1, 60, 80, 32]("act_l1", ctx)
        bench_conv[64, 128, 3, 2, 1, 60, 80, 32]("act_down2", ctx)
        bench_conv[64, 128, 1, 2, 0, 60, 80, 32]("act_short2", ctx)
        bench_conv[128, 128, 3, 1, 1, 30, 40, 32]("act_l2", ctx)
        bench_conv[128, 256, 3, 2, 1, 30, 40, 32]("act_down3", ctx)
        bench_conv[128, 256, 1, 2, 0, 30, 40, 32]("act_short3", ctx)
        bench_conv[256, 256, 3, 1, 1, 15, 20, 32]("act_l3", ctx)
        bench_conv[256, 512, 3, 2, 1, 15, 20, 32]("act_down4", ctx)
        bench_conv[256, 512, 1, 2, 0, 15, 20, 32]("act_short4", ctx)
        bench_conv[512, 512, 3, 1, 1, 8, 10, 32]("act_l4", ctx)
        bench_conv[3, 64, 7, 2, 3, 240, 320, 2]("act_stem_deploy", ctx)
        bench_conv[128, 128, 3, 1, 1, 30, 40, 2]("act_l2_deploy", ctx)
        # ACT LIBERO 128x128
        bench_conv[3, 64, 7, 2, 3, 128, 128, 32]("libero_stem", ctx)
        bench_conv[64, 64, 3, 1, 1, 32, 32, 32]("libero_l1", ctx)
        bench_conv[128, 128, 3, 1, 1, 16, 16, 32]("libero_l2", ctx)
        bench_conv[256, 256, 3, 1, 1, 8, 8, 32]("libero_l3", ctx)
        # MNIST, ViT and LeWM patch stems
        bench_conv[1, 16, 5, 2, 0, 28, 28, 100]("mnist_c1", ctx)
        bench_conv[16, 32, 5, 2, 0, 12, 12, 100]("mnist_c2", ctx)
        bench_conv[3, 192, 4, 4, 0, 32, 32, 128]("vit_patch", ctx)
        bench_conv[3, 192, 14, 14, 0, 224, 224, 512]("lewm_patch", ctx)
    comptime if PART == 3:
        # DreamerV3 (per-timestep batch 16), pool and strided variants
        bench_conv[1, 64, 5, 1, 2, 96, 96, 16]("dv3_enc1", ctx)
        bench_conv[64, 96, 5, 1, 2, 48, 48, 16]("dv3_enc2", ctx)
        bench_conv[96, 128, 5, 1, 2, 24, 24, 16]("dv3_enc3", ctx)
        bench_conv[128, 128, 5, 1, 2, 12, 12, 16]("dv3_enc4", ctx)
        bench_conv[96, 64, 5, 1, 2, 48, 48, 16]("dv3_dec3", ctx)
        bench_conv[64, 1, 5, 1, 2, 96, 96, 16]("dv3_dec_out", ctx)
        bench_conv[4, 48, 4, 2, 1, 96, 96, 16]("dv3s_enc1", ctx)
        bench_conv[48, 96, 4, 2, 1, 48, 48, 16]("dv3s_enc2", ctx)
        bench_conv[96, 192, 4, 2, 1, 24, 24, 16]("dv3s_enc3", ctx)
        bench_conv[192, 384, 4, 2, 1, 12, 12, 16]("dv3s_enc4", ctx)
        bench_conv[1, 64, 5, 1, 2, 96, 96, 1]("dv3_enc1_act", ctx)
        # Dreamer4 perceptual backbone (BT = 80, 64x64)
        bench_conv[3, 16, 3, 1, 1, 64, 64, 80]("d4_stem", ctx)
        bench_conv[16, 16, 3, 1, 1, 64, 64, 80]("d4_s1", ctx)
        bench_conv[16, 32, 3, 2, 1, 64, 64, 80]("d4_down2", ctx)
        bench_conv[32, 32, 3, 1, 1, 32, 32, 80]("d4_s2", ctx)
