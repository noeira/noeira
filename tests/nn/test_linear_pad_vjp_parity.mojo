"""`Linear.vjp` must produce the same gradients with K/N padding as without.

The backward is where the padding is easiest to get wrong, because its two
GEMMs are unaligned on the OPPOSITE axes from the forward's:

    grad_w     = xᵀ @ go      ->  N = OUT_
    grad_input = go @ Wᵀ      ->  K = OUT_,  N = IN_

and the padded `dW` comes back as `[K_PAD, N_PAD]`, whose ROW STRIDE differs
from the master grad's `[IN_, OUT_]`. A flat accumulate would fold the padded
columns into the next row's gradient — a wrong gradient that still trains, just
worse. That is the specific defect this gate exists to catch, so it compares
grad_input, grad_w AND grad_bias element by element against the CPU backward.

    pixi run -e apple mojo run -I . tests/nn/test_linear_pad_vjp_parity.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear import Linear


def _cmp(
    name: String, cpu: Tensor, gpu: Tensor, n: Int, tol: Float64,
    floor_frac: Float64 = 0.0,
) raises -> Float64:
    """Per-element relative error; the denominator is floored at 1e-6, or at
    `floor_frac` x the tensor's largest |element| when that is larger.

    grad_bias uses floor_frac 0.1: it is a column SUM over the batch, and the
    upstream pattern here makes some columns cancel to exactly 0 — the CPU
    returns -4.5e-8 there and the GPU's chunked reduction (a different order)
    -1.5e-7, both roundoff of zero, which a 1e-6 floor reads as a 10 % error.
    A wrong reduction (a lost chunk, a wrong column) is off by O(the value)."""
    var max_rel = Float64(0)
    var mag = Float64(0)
    for i in range(n):
        mag = max(mag, abs(Float64(cpu.data[i])))
    for i in range(n):
        var a = Float64(cpu.data[i])
        var b = Float64(gpu.data[i])
        var denom = max(abs(a), max(1e-6, floor_frac * mag))
        var r = abs(a - b) / denom
        if r > max_rel:
            max_rel = r
    # ⚠ NON-VACUITY: an all-zero gradient would compare equal to anything.
    if mag == 0.0:
        raise Error("VACUOUS: " + name + " is identically zero")
    if max_rel > tol:
        raise Error(name + " mismatch: max_rel=" + String(max_rel))
    return max_rel


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

    var r_gi = _cmp("grad_input", gic, gig, B * IN, 1e-4)
    var r_gw = _cmp("grad_w", lc.weight.grd, lg.weight.grd, L.W_SIZE, 1e-4)
    var r_gb = _cmp("grad_bias", lc.bias.grd, lg.bias.grd, L.B_SIZE, 1e-4, 0.1)
    print(
        "     grad_input ", r_gi, "   grad_w ", r_gw, "   grad_bias ", r_gb,
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("Linear vjp K/N-padding parity —", ctx.name())
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
