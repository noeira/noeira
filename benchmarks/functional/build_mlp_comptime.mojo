"""Build-time probe, comptime arm: a 1 -> W -> W -> 1 tanh MLP, MSE, nn's
Adam, CPU. The width is a compile-time define, so every width is a build:

    pixi run -e apple mojo build -I . -D WIDTH=64 benchmarks/functional/build_mlp_comptime.mojo -o /tmp/mlp_ct

Its functional twin is `build_mlp_functional.mojo` (width at run time).
"""

from std.math import sin
from std.random import seed, random_float64
from std.sys.defines import get_defined_int

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.loss.mse import mse_forward, mse_backward
from noeira.nn.optimizer.adam import Adam

comptime W = get_defined_int["WIDTH", 64]()
comptime B = 64
comptime NET = Sequential[Linear[1, W], Tanh[W], Linear[W, W], Tanh[W], Linear[W, 1]]


def main() raises:
    seed(0)
    var net = NET.make["cpu", Xavier]()
    var opt = Adam(lr=1e-2)
    var x = Tensor.alloc(B)
    var y = Tensor.alloc(B)
    var out = Tensor.alloc(B)
    var go = Tensor.alloc(B)
    var gx = Tensor.alloc(B)
    for step in range(2001):
        for k in range(B):
            var v = random_float64() * 2.0 - 1.0
            x.data[k] = Scalar[DT](v)
            y.data[k] = Scalar[DT](sin(3.0 * v))
        net.zero_grad["cpu"](None)
        net.forward["cpu", B](TensorRefs[1](x), out)
        mse_backward["cpu", B, 1](out, y, go)
        net.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx))
        opt.step["cpu"](net)
        if step % 1000 == 0:
            print("width", W, "step", step, "loss", mse_forward[B, 1](out, y))
