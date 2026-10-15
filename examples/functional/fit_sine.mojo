"""Functional mode: fit y = sin(3x) with an MLP whose DEPTH and WIDTH are
command-line arguments — one binary, any architecture, no recompile.

    pixi run -e apple mojo build -I . examples/functional/fit_sine.mojo -o /tmp/fit_sine
    /tmp/fit_sine            # depth 2, width 32, tanh
    /tmp/fit_sine 4 64 relu  # depth 4, width 64, relu

The network is a Python-style loop over a `List[FTensor]` of parameters.
"""

from std.math import sin
from std.random import seed, random_float64
from std.sys import argv

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.functional import Tape, FTensor
import noeira.nn.functional.F as F


def mlp(x: FTensor, weights: List[FTensor], biases: List[FTensor], act: String) raises -> FTensor:
    var h = x
    for i in range(len(weights)):
        h = h @ weights[i] + biases[i]
        if i + 1 < len(weights):
            h = h.relu() if act == "relu" else h.tanh()
    return h


def main() raises:
    var args = argv()
    var depth = Int(args[1]) if len(args) > 1 else 2
    var width = Int(args[2]) if len(args) > 2 else 32
    var act = String(args[3]) if len(args) > 3 else String("tanh")
    comptime BATCH = 64
    comptime STEPS = 3000
    var lr: Scalar[DT] = 0.05

    seed(0)
    var tape = Tape()
    var weights = List[FTensor]()
    var biases = List[FTensor]()
    var fan_in = 1
    for i in range(depth + 1):
        var fan_out = 1 if i == depth else width
        weights.append(tape.param[Xavier](fan_in, fan_out))
        biases.append(tape.zeros(1, fan_out))
        fan_in = fan_out
    print("MLP 1 ->", depth, "x", width, act, "-> 1;", len(tape.params()), "parameter tensors")

    var xs = List[Scalar[DT]](length=BATCH, fill=0)
    var ys = List[Scalar[DT]](length=BATCH, fill=0)
    for step in range(STEPS + 1):
        for k in range(BATCH):
            var v = random_float64() * 2.0 - 1.0
            xs[k] = Scalar[DT](v)
            ys[k] = Scalar[DT](sin(3.0 * v))
        tape.reset()
        tape.zero_grad()
        var x = tape.input(BATCH, 1, xs)
        var y = tape.input(BATCH, 1, ys)
        var loss = F.mse(mlp(x, weights, biases, act), y)
        loss.backward()
        tape.sgd_step(lr)
        if step % 500 == 0:
            print("  step", step, " loss", loss.item(), " nodes", tape.num_nodes())

    # Evaluate on a grid.
    tape.reset()
    var grid = List[Scalar[DT]]()
    for k in range(9):
        grid.append(Scalar[DT](-1.0 + 0.25 * Float64(k)))
    var pred = mlp(tape.input(9, 1, grid), weights, biases, act)
    print("  x      sin(3x)   mlp")
    for k in range(9):
        var v = Float64(grid[k])
        print("  ", grid[k], "  ", Scalar[DT](sin(3.0 * v)), "  ", pred.value(k))
