"""Shared by the QLinear gates (CPU and GPU): one SmolVLA text shape against a
`Linear` holding the same dequantized weight.

    y  = x  @ deq(W)ᵀ      QLinear[IN, OUT]  vs  Linear[IN, OUT] with W_lin = deq(W)ᵀ
    dx = dy @ deq(W)                                and a zero bias

The weight is a seeded Gaussian, quantized by RTN; x and dy are seeded too.
Both sides read the same deq(W), so the comparison isolates the GEMM path:
on the CPU and Metal both are fp32 GEMMs, on NVIDIA the QLinear forward is
MAX's int4 GEMM with bf16 activations.
"""

from std.math import abs
from std.random import random_float64, seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.quant.qlinear import QLinear


struct Verdict(Copyable, Movable):
    var fwd: Float64
    """max |y_q - y_lin| / max |y_lin|."""
    var dx: Float64
    """max |dx_q - dx_lin| / max |dx_lin|."""
    var fwd_vs_exact: Float64
    """max |y_q - x deq(W)ᵀ| / max |x deq(W)ᵀ|, the reference in fp64."""

    def __init__(out self, fwd: Float64, dx: Float64, fwd_vs_exact: Float64):
        self.fwd = fwd
        self.dx = dx
        self.fwd_vs_exact = fwd_vs_exact


def rel(ref a: List[Float32], ref b: List[Float32], n: Int) -> Float64:
    var d = Float64(0)
    var m = Float64(0)
    for i in range(n):
        d = max(d, abs(Float64(a[i]) - Float64(b[i])))
        m = max(m, abs(Float64(b[i])))
    return d / m if m > 0 else d


def gaussian(n: Int, scale: Float64) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        # Box-Muller would do; a sum of uniforms is plenty for a fixture.
        var s = Float64(0)
        for _ in range(4):
            s += random_float64() - 0.5
        out.append(Float32(s * scale * 1.7320508))
    return out^


def to_host(mut t: Tensor, n: Int, ctx: Optional[DeviceContext]) raises -> List[Float32]:
    """The first `n` values of `t`, downloaded first when it lives on a GPU."""
    if ctx:
        t.download(ctx.value())
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(t.data[i])
    return out^


def put(mut t: Tensor, ref vals: List[Float32], target: StaticString,
        ctx: Optional[DeviceContext]) raises:
    t.data = vals.copy()
    t.n = len(vals)
    if target == "gpu":
        t.upload(ctx.value())


def check_shape[
    target: StaticString, IN: Int, OUT: Int, T: Int
](ctx: Optional[DeviceContext], label: String) raises -> Verdict:
    seed(IN * 7 + OUT)
    var w = gaussian(OUT * IN, 0.05)        # [OUT, IN], torch's orientation
    var ql = QLinear[IN, OUT].make[target, Deterministic](ctx)
    ql.set_weight[target](w, ctx)
    var deq = ql.dequantized()              # [OUT, IN]

    # Linear holding deq(W)ᵀ ([IN, OUT]) and a zero bias.
    var lin = Linear[IN, OUT].make[target, Deterministic](ctx)
    var wl = List[Float32](length=IN * OUT, fill=0)
    for o in range(OUT):
        for i in range(IN):
            wl[i * OUT + o] = deq[o * IN + i]
    var zb = List[Float32](length=OUT, fill=0)
    put(lin.weight.val, wl, target, ctx)
    put(lin.bias.val, zb, target, ctx)
    lin.weight.val.version += 1
    lin.freeze()

    var xs = gaussian(T * IN, 1.0)
    var gs = gaussian(T * OUT, 1.0)
    var x = Tensor()
    var g = Tensor()
    put(x, xs, target, ctx)
    put(g, gs, target, ctx)
    var yq = Tensor()
    var yl = Tensor()
    ql.forward[target, T](TensorRefs[1](x), yq, ctx)
    lin.forward[target, T](TensorRefs[1](x), yl, ctx)
    var gq = Tensor()
    var gl = Tensor()
    ql.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gq), ctx)
    lin.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gl), ctx)

    var yq_h = to_host(yq, T * OUT, ctx)
    var yl_h = to_host(yl, T * OUT, ctx)
    var gq_h = to_host(gq, T * IN, ctx)
    var gl_h = to_host(gl, T * IN, ctx)
    # The exact product, in fp64.
    var ye = List[Float32](capacity=T * OUT)
    for t in range(T):
        for o in range(OUT):
            var acc = Float64(0)
            for i in range(IN):
                acc += Float64(xs[t * IN + i]) * Float64(deq[o * IN + i])
            ye.append(Float32(acc))
    var v = Verdict(
        rel(yq_h, yl_h, T * OUT), rel(gq_h, gl_h, T * IN),
        rel(yq_h, ye, T * OUT),
    )
    print("    ", label, "[", IN, "->", OUT, "] T =", T, ": forward",
          v.fwd, "| dx", v.dx, "| forward vs exact", v.fwd_vs_exact)
    return v^
