"""`BatchNorm1D[..., UNBIASED_RUNNING]` — the running-variance update, CPU + GPU.

torch's `BatchNorm1d` normalises with the biased batch variance but feeds the
UNBIASED one (x B/(B-1)) to `running_var`. One train-mode step from the init
(running_var = 1, momentum 0.1) at B = 4, where the factor is 4/3:

    UNBIASED_RUNNING=False (default, unchanged):  0.9 + 0.1 * var_biased
    UNBIASED_RUNNING=True  (LeWM, = torch):       0.9 + 0.1 * var_biased * 4/3

Both settings on both targets; the two expectations differ by ~3 % of the
update, so a setting that is ignored cannot pass.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_batch_norm_1d_unbiased_running.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D, BN_DEFAULT_MOM, BN_DEFAULT_EPS


comptime DIM = 6
comptime B = 4


def _check[UNBIASED: Bool, target: StaticString](
    c: Optional[DeviceContext], label: String
) raises:
    var bn = BatchNorm1D[DIM, BN_DEFAULT_MOM, BN_DEFAULT_EPS, DT, UNBIASED].make[
        target, Kaiming
    ](c)
    var inp = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        inp.data[i] = Scalar[DT](Float64((i * 7) % 11 - 5) * 0.3 + Float64(i % DIM))
    var want = List[Float64](length=DIM, fill=0.0)
    for f in range(DIM):
        var m = 0.0
        for b in range(B):
            m += Float64(inp.data[b * DIM + f])
        m /= Float64(B)
        var v = 0.0
        for b in range(B):
            var d = Float64(inp.data[b * DIM + f]) - m
            v += d * d
        v /= Float64(B)
        if UNBIASED:
            v *= Float64(B) / Float64(B - 1)
        want[f] = 0.9 * 1.0 + 0.1 * v
    comptime if target == "gpu":
        inp.upload(c.value())
    var out = Tensor()
    bn.set_attr["training"](Scalar[DT](1.0))
    bn.forward[target, B](child_refs[1, DT](inp), out, c)
    comptime if target == "gpu":
        c.value().synchronize()
        bn.running_var.t.download(c.value())
    var err = 0.0
    for f in range(DIM):
        err = max(err, abs(Float64(bn.running_var.t.data[f]) - want[f]))
    print("  ", label, " max |running_var - expected| =", err)
    if err > 1e-5:
        raise Error("FAIL " + label + ": running_var update is not the expected one")


def main() raises:
    print("BatchNorm1D running-variance update (B = 4)")
    var c = DeviceContext()
    _check[False, "cpu"](None, String("biased   cpu"))
    _check[False, "gpu"](Optional(c), String("biased   gpu"))
    _check[True, "cpu"](None, String("unbiased cpu"))
    _check[True, "gpu"](Optional(c), String("unbiased gpu"))
    print("PASS")
