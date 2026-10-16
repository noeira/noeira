"""Build-time probe, functional arm: the same MLP as `build_mlp_comptime`,
its width read at RUN time — one build serves every width:

    pixi run -e apple mojo build -I . benchmarks/functional/build_mlp_functional.mojo -o /tmp/mlp_fn
    /tmp/mlp_fn 64 ; /tmp/mlp_fn 256
"""

from std.math import sin
from std.random import seed, random_float64
from std.sys import argv

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.functional import Tape, FTensor
import noeira.nn.functional.F as F


def main() raises:
    var args = argv()
    var w = Int(args[1]) if len(args) > 1 else 64
    comptime B = 64
    seed(0)
    var tape = Tape()
    var w1 = tape.param[Xavier](1, w)
    var b1 = tape.zeros(1, w)
    var w2 = tape.param[Xavier](w, w)
    var b2 = tape.zeros(1, w)
    var w3 = tape.param[Xavier](w, 1)
    var b3 = tape.zeros(1, 1)
    var opt = Adam(lr=1e-2)
    var xs = List[Scalar[DT]](length=B, fill=0)
    var ys = List[Scalar[DT]](length=B, fill=0)
    for step in range(2001):
        for k in range(B):
            var v = random_float64() * 2.0 - 1.0
            xs[k] = Scalar[DT](v)
            ys[k] = Scalar[DT](sin(3.0 * v))
        tape.reset()
        tape.zero_grad()
        var h = F.linear(tape.input(B, 1, xs), w1, b1).tanh()
        h = F.linear(h, w2, b2).tanh()
        var loss = F.mse(F.linear(h, w3, b3), tape.input(B, 1, ys))
        loss.backward()
        opt.step["cpu"](tape)
        if step % 1000 == 0:
            print("width", w, "step", step, "loss", loss.item())
