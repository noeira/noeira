"""`LayerNorm[DIM, EPS=…]` honours EPS on the GPU, not only on the CPU.

The device forward kernel read the module constant `LN_EPS` (1e-5) instead of
the struct's `EPS`, so `LayerNorm[DIM, EPS=1e-12]` normalised with 1e-12 on CPU
and 1e-5 on GPU — no error, a wrong number. Nothing in the tree passed a
non-default EPS until the LeWM port, which needs 1e-12 (HF ViT) and 1e-5.

Each case is checked against the CLOSED FORM `gamma·(x-mean)/sqrt(var+EPS)+beta`
on both targets, on rows whose variance is small enough that EPS moves the
answer far beyond float32 rounding — so dropping EPS cannot pass by accident:

  * EPS = 0.25 on rows of variance ~0.01: EPS dominates (x̂ shrinks ~5x).
  * EPS = 1e-12 on the same rows: differs from 1e-5 by ~5e-4 relative,
    25x the tolerance.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_layer_norm_eps_gpu.mojo
"""

from std.math import abs, sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.layer_norm import LayerNorm


comptime DIM = 48
comptime B = 5


def _closed_form(
    ref x: Tensor, ref gamma: Tensor, ref beta: Tensor, eps: Float64,
) -> List[Float64]:
    var out = List[Float64](length=B * DIM, fill=0.0)
    for b in range(B):
        var mean = 0.0
        for d in range(DIM):
            mean += Float64(x.data[b * DIM + d])
        mean /= Float64(DIM)
        var var_ = 0.0
        for d in range(DIM):
            var dv = Float64(x.data[b * DIM + d]) - mean
            var_ += dv * dv
        var_ /= Float64(DIM)
        var inv = 1.0 / sqrt(var_ + eps)
        for d in range(DIM):
            out[b * DIM + d] = (
                Float64(gamma.data[d]) * (Float64(x.data[b * DIM + d]) - mean) * inv
                + Float64(beta.data[d])
            )
    return out^


def _check[EPS: Scalar[DT]](c: DeviceContext, label: String) raises:
    var cpu = LayerNorm[DIM, DT, EPS].make["cpu", Deterministic]()
    var gpu = LayerNorm[DIM, DT, EPS].make["gpu", Deterministic](Optional(c))
    var x = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        # std ~0.1 per row (x a per-row factor), mean ~0. NOT offset: the
        # GPU kernel takes var = E[x^2] - mean^2 in float32, and a row with
        # |mean|/std ~ 40 loses ~1e-4 of its variance to that cancellation —
        # a real property (watched by the LeWM parity gates), but not EPS.
        x.data[i] = Scalar[DT](
            0.1 * Float64((i * 37) % 23 - 11) / 6.6 * (1.0 + 0.3 * Float64(i // DIM))
        )
    var want = _closed_form(x, cpu.gamma.val, cpu.beta.val, Float64(EPS))

    var c_out = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](x), c_out, None)

    var gx = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = x.data[i]
    gx.upload(c)
    var g_out = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    g_out.download(c)

    var ec = 0.0
    var eg = 0.0
    var scale = 0.0
    for i in range(B * DIM):
        ec = max(ec, abs(Float64(c_out.data[i]) - want[i]))
        eg = max(eg, abs(Float64(g_out.data[i]) - want[i]))
        scale = max(scale, abs(want[i]))
    print("  ", label, " max|out| =", scale, " cpu err =", ec, " gpu err =", eg)
    comptime TOL = 2e-5
    if ec > TOL * scale:
        raise Error("FAIL " + label + ": CPU LayerNorm disagrees with the closed form")
    if eg > TOL * scale:
        raise Error(
            "FAIL " + label + ": GPU LayerNorm disagrees with the closed form"
            " — the device kernel is not using EPS"
        )


def main() raises:
    print("LayerNorm EPS honoured on CPU and GPU")
    var c = DeviceContext()
    _check[Scalar[DT](0.25)](c, String("EPS=0.25 "))
    _check[Scalar[DT](1e-12)](c, String("EPS=1e-12"))
    print("PASS")
