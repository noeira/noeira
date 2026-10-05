"""`ELU` = torch's `nn.ELU()` (alpha 1), forward and backward, CPU and GPU.

The reference values are float64 `math.expm1` / `math.exp` from Python,
printed once and pasted below. 13 points, so both the SIMD body and the
scalar tail of the CPU path run; points straddle 0 on both sides. `ReLU`
runs on the same points and MUST fail the tolerance (they agree for x > 0
only): a gate that cannot tell them apart checks nothing.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_elu.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.activations import ELU, ReLU


comptime N = 13
comptime TOL = 2e-6


def _xs() -> List[Float64]:
    return [
        -4.0, -2.5, -1.3,
        -0.7, -0.2, -0.001,
        0.0, 0.001, 0.3,
        0.9, 1.7, 2.4,
        3.6,
    ]


def _ys() -> List[Float64]:
    """x if x > 0 else expm1(x), float64."""
    return [
        -0.9816843611112658, -0.9179150013761012, -0.7274682069659875,
        -0.5034146962085905, -0.18126924692201815, -0.0009995001666250085,
        0.0, 0.001, 0.3,
        0.9, 1.7, 2.4,
        3.6,
    ]


def _gs() -> List[Float64]:
    """1 if x > 0 else exp(x), float64."""
    return [
        0.01831563888873418, 0.0820849986238988, 0.2725317930340126,
        0.4965853037914095, 0.8187307530779818, 0.999000499833375,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0,
    ]


def _run[target: StaticString, RELU: Bool](c: Optional[DeviceContext]) raises -> Tuple[Float64, Float64]:
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    var xs = _xs()
    for i in range(N):
        x.data[i] = Scalar[DT](xs[i])
        go.data[i] = Scalar[DT](1.0)
    if c:
        x.upload(c.value())
        go.upload(c.value())
    var y = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    comptime if RELU:
        var m = ReLU[N].make[target, Deterministic](c)
        m.forward[target, 1](TensorRefs[1](x), y, c)
        m.vjp[target, 1](TensorRefs[1](x), go, TensorRefs[1](gi), c)
    else:
        var m = ELU[N].make[target, Deterministic](c)
        m.forward[target, 1](TensorRefs[1](x), y, c)
        m.vjp[target, 1](TensorRefs[1](x), go, TensorRefs[1](gi), c)
    if c:
        y.download(c.value())
        gi.download(c.value())
    var ys = _ys()
    var gs = _gs()
    var ey = 0.0
    var eg = 0.0
    for i in range(N):
        ey = max(ey, abs(Float64(y.data[i]) - ys[i]))
        eg = max(eg, abs(Float64(gi.data[i]) - gs[i]))
    return (ey, eg)


def main() raises:
    print("ELU vs float64 closed form")
    var c = DeviceContext()
    var cpu = _run["cpu", False](None)
    var gpu = _run["gpu", False](Optional(c))
    var relu = _run["cpu", True](None)
    print("  ELU cpu: fwd", cpu[0], " bwd", cpu[1])
    print("  ELU gpu: fwd", gpu[0], " bwd", gpu[1])
    print("  ReLU cpu: fwd", relu[0], " bwd", relu[1], " (must FAIL)")
    if cpu[0] > TOL or cpu[1] > TOL:
        raise Error("FAIL: ELU on CPU")
    if gpu[0] > TOL or gpu[1] > TOL:
        raise Error("FAIL: ELU on GPU")
    if relu[0] <= TOL:
        raise Error("FAIL: ReLU passes too — the gate cannot tell them apart")
    print("PASS")
