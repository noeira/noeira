"""`Linear.vjp` must produce the same gradients with K/N padding as without.

The backward is where the padding is easiest to get wrong, because its two
GEMMs are unaligned on the OPPOSITE axes from the forward's:

    grad_w     = xᵀ @ go      ->  N = OUT_
    grad_input = go @ Wᵀ      ->  K = OUT_,  N = IN_

and the padded `dW` comes back as `[K_PAD, N_PAD]`, whose ROW STRIDE differs
from the master grad's `[IN_, OUT_]`. A flat accumulate would fold the padded
columns into the next row's gradient — a wrong gradient that still trains, just
worse. That is the specific defect this gate exists to catch, so it compares
grad_input, grad_w AND grad_bias against the CPU backward, in std units of the
CPU value (that defect is O(1) there).

Band per backend: 1e-4 on Metal (fp32 both sides), 1e-2 on CUDA (TF32 GEMMs;
a per-element relative band read TF32 rounding of near-zero entries as an
error). On NVIDIA the backward runs on `cublas_gemm` unpadded (`CUBLAS_BWD`,
printed); the padded MAX backward runs with `-D NN_GEMM_PATH=max`.

    pixi run -e apple mojo run -I . tests/nn/test_linear_pad_vjp_parity.mojo
    pixi run -e default mojo run -I . -D NN_GEMM_PATH=max tests/nn/test_linear_pad_vjp_parity.mojo
"""

from std.math import abs, sqrt
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.cublas_gemm import CUBLAS_BWD
from noeira.nn.primitives.linear import Linear


comptime TOL = 1e-2 if has_nvidia_gpu_accelerator() else 1e-4


def _cmp(name: String, cpu: Tensor, gpu: Tensor, n: Int) raises -> Float64:
    """Max |gpu - cpu| in std units of the CPU tensor. (A per-element
    relative error read roundoff of entries that cancel to ~0 — grad_bias
    column sums, TF32 outputs — as a 10 % error; a wrong reduction or stride
    is off by O(the value).)"""
    var mag = Float64(0)
    var mean = Float64(0)
    var max_abs = Float64(0)
    for i in range(n):
        mag = max(mag, abs(Float64(cpu.data[i])))
        mean += Float64(cpu.data[i])
        max_abs = max(max_abs, abs(Float64(gpu.data[i]) - Float64(cpu.data[i])))
    mean /= Float64(n)
    var var_ = Float64(0)
    for i in range(n):
        var_ += (Float64(cpu.data[i]) - mean) ** 2
    var sd = sqrt(var_ / Float64(n))
    # ⚠ NON-VACUITY: an all-zero gradient would compare equal to anything.
    if mag == 0.0:
        raise Error("VACUOUS: " + name + " is identically zero")
    var err = max_abs / (sd if sd > 0.0 else mag)
    if err > TOL:
        raise Error(name + " mismatch: " + String(err) + " std units")
    return err


def check[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises:
    comptime L = Linear[IN, OUT]
    print(
        "  IN=", IN, " OUT=", OUT, " B=", B, "   K_PAD=", L.K_PAD, " (",
        L.NEEDS_PAD, ")  N_PAD=", L.N_PAD, " (", L.NEEDS_N_PAD, ")", sep="",
    )

    var lc = L.make["cpu", INIT=Kaiming]()
    var lg = L.make["gpu", INIT=Kaiming](ctx=ctx)
    lg.weight.val.ensure_host(ctx, L.W_SIZE)
    lg.bias.val.ensure_host(ctx, L.B_SIZE)
    for i in range(L.W_SIZE):
        lg.weight.val.data[i] = lc.weight.val.data[i]
    for i in range(L.B_SIZE):
        lg.bias.val.data[i] = lc.bias.val.data[i]
    lg.weight.val.upload_resident(ctx)
    lg.bias.val.upload_resident(ctx)

    # forward input + an upstream gradient, identical on both devices
    var xc = Tensor.alloc(B * IN)
    var gc = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        xc.data[i] = Scalar[DT](0.017) * Scalar[DT]((i % 31) - 15)
    for i in range(B * OUT):
        gc.data[i] = Scalar[DT](0.023) * Scalar[DT]((i % 23) - 11)
    var xg = Tensor.alloc(B * IN)
    var gg = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        xg.data[i] = xc.data[i]
    for i in range(B * OUT):
        gg.data[i] = gc.data[i]
    xg.upload(ctx)
    gg.upload(ctx)

    var yc = Tensor.alloc(B * OUT)
    var yg = Tensor.alloc_gpu(ctx, B * OUT)
    var gic = Tensor.alloc(B * IN)
    var gig = Tensor.alloc_gpu(ctx, B * IN)

    # a forward first — vjp reads caches the forward populates
    lc.forward["cpu", B](TensorRefs[1](xc), yc, None)
    lg.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    lc.vjp["cpu", B](TensorRefs[1](xc), gc, TensorRefs[1](gic), None)
    lg.vjp["gpu", B](TensorRefs[1](xg), gg, TensorRefs[1](gig), Optional(ctx))

    gig.download(ctx)
    lg.weight.grd.ensure_host(ctx, L.W_SIZE)
    lg.bias.grd.ensure_host(ctx, L.B_SIZE)
    lg.weight.grd.download(ctx)
    lg.bias.grd.download(ctx)
    ctx.synchronize()

    var r_gi = _cmp("grad_input", gic, gig, B * IN)
    var r_gw = _cmp("grad_w", lc.weight.grd, lg.weight.grd, L.W_SIZE)
    var r_gb = _cmp("grad_bias", lc.bias.grd, lg.bias.grd, L.B_SIZE)
    print(
        "     grad_input ", r_gi, "   grad_w ", r_gw, "   grad_bias ", r_gb,
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print(
        "Linear vjp K/N-padding parity —", ctx.name(), "| cuBLAS backward:",
        CUBLAS_BWD, "| tol", TOL, "std units",
    )
    print()
    print("== N padded (the two-hot / policy / termination heads) ==")
    check[512, 101, 256](ctx)     # BINS=101
    check[512, 12, 256](ctx)      # 2*ACT
    check[512, 1, 128](ctx)       # termination
    print()
    print("== K padded (the za = latent|act trunks) ==")
    check[518, 512, 256](ctx)
    check[30, 256, 128](ctx)      # SAC critic obs|act
    print()
    print("== BOTH padded ==")
    check[518, 101, 128](ctx)
    print()
    print("== neither (untouched path) ==")
    check[512, 512, 256](ctx)
    check[256, 128, 128](ctx)
    print()
    print("ALL PASSED")
