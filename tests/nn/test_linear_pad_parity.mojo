"""Does the K/N alignment padding change what `Linear` computes?

The padded columns are exactly 0, so the dot products must be unchanged up to
reduction-order noise. Checks UNALIGNED widths (padding active) and ALIGNED
ones (the untouched path) against the CPU forward, in std units of the CPU
output: a padding bug (a wrong stride, a column folded into the next row) is
O(1) there.

Band per backend: 1e-4 on Metal (fp32 both sides); 1e-2 on CUDA, where these
shapes run in TF32 (MAX's multistage kernel or `cublas_tf32`'s rule) — a
per-element relative band of 1e-4 read TF32 rounding of near-zero outputs as
a 10 % error. On NVIDIA the default routing sends every shape here to
`cublas_gemm` UNPADDED (`cublas_fwd`, printed per shape); the padded MAX path
runs with `-D NN_GEMM_PATH=max`:

    pixi run -e default mojo run -I . -D NN_GEMM_PATH=max tests/nn/test_linear_pad_parity.mojo
"""

from std.math import abs, sqrt
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear import Linear


comptime TOL = 1e-2 if has_nvidia_gpu_accelerator() else 1e-4


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    comptime L = Linear[IN, OUT]
    print(
        "  IN=", IN, " OUT=", OUT, " B=", B, "   K_PAD=", L.K_PAD,
        " (", L.NEEDS_PAD, ")  N_PAD=", L.N_PAD, " (", L.NEEDS_N_PAD, ")",
        "  cublas_fwd=", L.use_cublas_fwd[B](), sep="",
    )

    # Same weights on both devices: build on CPU, copy the slab, upload.
    var lc = L.make["cpu", INIT=Kaiming]()
    var lg = L.make["gpu", INIT=Kaiming](ctx=ctx)
    # Overwrite the GPU module's weights with the CPU module's, host-side, then
    # push them down — the two INITs draw different RNG otherwise.
    lg.weight.val.ensure_host(ctx, L.W_SIZE)
    lg.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lg.weight.val.data[i] = lc.weight.val.data[i]
    for i in range(L.B_SIZE):
        lg.bias.val.data[i] = lc.bias.val.data[i]
    lg.weight.val.upload_resident(ctx)
    lg.bias.val.upload_resident(ctx)

    var xc = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xc.data[i] = Scalar[DT](0.01) * Scalar[DT]((i % 37) - 18)
    var xg = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xg.data[i] = xc.data[i]
    xg.upload(ctx)

    var yc = Tensor.alloc(B * OUT)
    var yg = Tensor.alloc_gpu(ctx, B * OUT)
    lc.forward["cpu", B](TensorRefs[1](xc), yc, None)
    lg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    yg.download(ctx)
    ctx.synchronize()

    var max_abs = Float64(0)
    var mean = Float64(0)
    for i in range(B * OUT):
        mean += Float64(yc.data[i])
        max_abs = max(max_abs, abs(Float64(yc.data[i]) - Float64(yg.data[i])))
    mean /= Float64(B * OUT)
    var var_ = Float64(0)
    for i in range(B * OUT):
        var_ += (Float64(yc.data[i]) - mean) ** 2
    var sd = sqrt(var_ / Float64(B * OUT))
    var err = max_abs / (sd if sd > 0.0 else 1.0)
    # ⚠ NON-VACUITY: a comparison of two all-zero buffers also reports 0.0.
    var mag = Float64(0)
    for i in range(B * OUT):
        if abs(Float64(yg.data[i])) > mag:
            mag = abs(Float64(yg.data[i]))
    print(
        "     max_abs=", max_abs, "  err=", err, " std units",
        "   |gpu|max=", mag, "  cpu[0]=", yc.data[0], " gpu[0]=", yg.data[0],
        sep="",
    )
    if mag == 0.0:
        raise Error("VACUOUS: the GPU output is all zeros")
    if err > TOL:
        raise Error("PADDING CHANGED THE RESULT — " + String(err) + " std units")


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print()
    print("== unaligned widths (padding ACTIVE) ==")
    check[518, 512, 268](ctx)     # TD-MPC2 za = latent|act
    check[101, 512, 64](ctx)      # BINS as an input width
    check[30, 256, 64](ctx)       # SAC critic obs|act
    check[24, 256, 7](ctx)        # odd batch too
    print()
    print("== narrow/unaligned OUTPUT widths (N padding ACTIVE) ==")
    check[512, 101, 268](ctx)     # TD-MPC2 two-hot head: BINS=101
    check[512, 12, 268](ctx)      # policy head: 2*ACT
    check[512, 1, 268](ctx)       # termination head
    check[518, 101, 64](ctx)      # BOTH dims padded at once
    print()
    print("== aligned widths (padding INACTIVE — must be untouched) ==")
    check[512, 512, 268](ctx)
    check[256, 128, 64](ctx)
    print()
    print("all good")
