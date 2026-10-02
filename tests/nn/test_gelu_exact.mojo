"""`GELUExact` = `torch.nn.GELU()` (erf), forward and backward, CPU and GPU.

The reference values are torch's own (float64, `F.gelu` + autograd, printed
once and pasted below) — not this file's `erf`, which would make the check
circular. The tanh `GELU` is run on the same points and MUST fail the same
tolerance: torch's exact and tanh forms differ by up to 4.4e-4 here, so a
gate that cannot tell them apart is not checking which one we have.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_gelu_exact.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.activations import GELU, GELUExact


comptime N = 11
comptime TOL = 2e-6

def _xs() -> List[Float64]:
    """torch.float64 inputs."""
    return [-4.0, -2.5, -1.3, -0.7, -0.2, 0.0, 0.3, 0.9, 1.7, 2.4, 3.6]


def _ys() -> List[Float64]:
    """F.gelu(x), torch float64."""
    return [
        -0.00012668496733247991, -0.015524163314440398, -0.12584062996129339,
        -0.16937455655615108, -0.084148058112179402, 0.0, 0.1853734266566858,
        0.73434588718791649, 1.6242387133104768, 2.3803259137809691,
        3.5994272090754329,
    ]


def _gs() -> List[Float64]:
    """d/dx F.gelu(x), torch autograd float64."""
    return [
        -0.00050364966122642149, -0.037611085908145193, -0.12597868507653925,
        0.023385898866340099, 0.34253175176580575, 0.5, 0.73232776682710987,
        1.0554165995621199, 1.1153179687821648, 1.0455493367830269,
        1.0020437383582521,
    ]


def _fill(mut x: Tensor, mut go: Tensor, c: Optional[DeviceContext]) raises:
    var xs = _xs()
    for i in range(N):
        x.data[i] = Scalar[DT](xs[i])
        go.data[i] = Scalar[DT](1.0)
    if c:
        x.upload(c.value())
        go.upload(c.value())


def _err(mut y: Tensor, mut gi: Tensor, c: Optional[DeviceContext]) raises -> Tuple[Float64, Float64]:
    """Max |y - torch| and |dy/dx - torch|."""
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


def _run_exact[target: StaticString](c: Optional[DeviceContext]) raises -> Tuple[Float64, Float64]:
    var m = GELUExact[N].make[target, Deterministic](c)
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    _fill(x, go, c)
    var y = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    m.forward[target, 1](TensorRefs[1](x), y, c)
    m.vjp[target, 1](TensorRefs[1](x), go, TensorRefs[1](gi), c)
    return _err(y, gi, c)


def _run_tanh(c: Optional[DeviceContext]) raises -> Tuple[Float64, Float64]:
    var m = GELU[N].make["cpu", Deterministic](c)
    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    _fill(x, go, c)
    var y = Tensor.alloc(N)
    var gi = Tensor.alloc(N)
    m.forward["cpu", 1](TensorRefs[1](x), y, c)
    m.vjp["cpu", 1](TensorRefs[1](x), go, TensorRefs[1](gi), c)
    return _err(y, gi, c)


def main() raises:
    print("GELUExact vs torch F.gelu (exact)")
    var c = DeviceContext()
    var cpu = _run_exact["cpu"](None)
    var gpu = _run_exact["gpu"](Optional(c))
    var tanh_cpu = _run_tanh(None)
    print("  GELUExact cpu: fwd", cpu[0], " bwd", cpu[1])
    print("  GELUExact gpu: fwd", gpu[0], " bwd", gpu[1])
    print("  GELU(tanh) cpu: fwd", tanh_cpu[0], " bwd", tanh_cpu[1], " (must FAIL)")
    if cpu[0] > TOL or cpu[1] > TOL:
        raise Error("FAIL: GELUExact on CPU is not torch's exact GELU")
    if gpu[0] > TOL or gpu[1] > TOL:
        raise Error("FAIL: GELUExact on GPU is not torch's exact GELU")
    if tanh_cpu[0] <= TOL:
        raise Error("FAIL: the tanh GELU passes too — the gate cannot tell them apart")
    print("PASS")
