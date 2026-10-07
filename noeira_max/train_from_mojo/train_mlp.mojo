"""An MLP train step defined in Mojo, trained on MAX from Mojo, gated against
the same step built in Python.

The Mojo program builds the whole step with `max_graph_gen`: parameter and
Adam-moment buffers, the forward pass, the MSE loss, the gradient (`Tape`,
the one call into Python's transform), AdamW's update and its stores. MAX
compiles it once; `maxrt` then runs every step, with no Python in the loop,
updating the buffers this program owns in place.

The gate: `py_grad.reference` builds the same step with the Python prototype
(`build_train_step`, `mlp_step.py`'s MSE, `optim.AdamW`) from the same
weights and batch, and runs it as many steps. Checked:
- the two graphs' MLIR text, which should be identical;
- every step's loss, and the final parameters, moments and counter, bit for
  bit.
`--mutate` builds the Mojo step with twice the learning rate: the gate must
then fail (it is not vacuous).

    noeira_max/train_from_mojo/run.sh [--gpu]
    ./train_mlp OUT_DIR [--shape small|ppo|sac] [--steps N] [--gpu] [--capture] [--mutate]
"""

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from max_train import (
    Shape, build_mlp_step, check, host_copy, input_shape, lend, median_us, numbers, param_name,
    pipelined_us, read, shape_named, state_names, step_name, train,
)
from maxrt import HostBuffer, Runtime, Tensor

comptime GLUE = "noeira_max.train_from_mojo.py_grad"
comptime LR = 3e-4


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var shape = String("small")
    var steps = 20
    var gpu = False
    var mutate = False
    var capture = False
    var i = 2
    while i < len(args):
        if args[i] == "--shape":
            shape = String(args[i + 1])
            i += 1
        elif args[i] == "--steps":
            steps = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--gpu":
            gpu = True
        elif args[i] == "--capture":  # CUDA: replay the captured step
            gpu = True
            capture = True
        elif args[i] == "--mutate":
            mutate = True
        else:
            raise Error("unknown flag " + String(args[i]))
        i += 1
    var s = shape_named(shape)
    var device = String("gpu") if gpu else String("cpu")
    var n = 2 * (len(s.dims) - 1)
    var glue = Python.import_module(GLUE)
    var builtins = Python.import_module("builtins")
    var py_dims = builtins.list()
    for d in s.dims:
        _ = py_dims.append(PythonObject(d))

    # Build in Mojo; Python only differentiates, then compiles.
    var t0 = perf_counter_ns()
    var g = build_mlp_step(s, device, LR * 2.0 if mutate else LR)
    var build_s = Float64(perf_counter_ns() - t0) / 1e9
    var text = String(glue.graph_text(g.backend.graph))
    var rt = Runtime(accelerator=gpu)
    var model = g.compile(rt, dir + "/" + step_name(s) + ".mef", device)
    print("[mojo] built", step_name(s), "in", build_s, "s; compiled in", g.compile_seconds, "s on", device)

    # The buffers: this program's memory, updated in place by every step.
    var data = glue.problem(py_dims, s.batch, 0)
    var inputs = rt.tensor_map()
    var staging = rt.tensor_map()
    var on_device = List[Tensor]()

    for k in range(n):
        lend(inputs, staging, on_device, rt, gpu, k, host_copy(data["init"][param_name(k)]), input_shape(s, k))
    for k in range(n, 3 * n + 1):  # the counter and the moments: zeros
        lend(inputs, staging, on_device, rt, gpu, k, HostBuffer(4 * numbers(input_shape(s, k))), input_shape(s, k))
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 1, host_copy(data["x"]), input_shape(s, 3 * n + 1))
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 2, host_copy(data["y"]), input_shape(s, 3 * n + 2))

    # Train: C API only.
    var losses = List[Float32]()
    var times = List[Int]()
    var lent = List[Tensor]()
    var outputs = List[Tensor]()  # rewritten by every replay: alive while replaying
    train(model, inputs, 3 * n + 3, steps, gpu, capture, losses, times, lent, outputs)
    print("[mojo]", steps, "steps", "(captured from step 1)" if capture else "", "; loss", losses[0], "->",
          losses[len(losses) - 1], "; median step", median_us(times, 1), "us, loss copied back each step")

    # The same step from the Python prototype.
    var ref_out = glue.reference(py_dims, s.batch, 0, steps, step_name(s), s.act, LR, device)
    print("[python] reference compiled in", Float64(py=ref_out["compile_s"]), "s")

    var finals = List[PythonObject]()
    for k in range(3 * n + 1):
        var py_shape = builtins.list()
        for d in input_shape(s, k):
            _ = py_shape.append(PythonObject(d))
        finals.append(read(inputs, on_device, gpu, k, glue, py_shape))
    var failures = check(text, losses, finals, state_names(s), ref_out)

    if mutate:
        if failures == 0:
            raise Error("MUTANT NOT CAUGHT: twice the learning rate passed the gate")
        print("mutant caught (", failures, "checks failed), as it must be")
        return
    if failures > 0:
        raise Error("FAIL: the Mojo-built step differs from the Python one")
    print("PASS: the train step built in Mojo is the Python prototype's step, bit for bit")
    if gpu:
        # Further steps, back to back: the step's own cost, with no per-step copy.
        print("[mojo] pipelined, 1000 steps: executed", pipelined_us(model, inputs, lent, rt, 1000, False),
              "us per step" + (String("; replayed ") + String(pipelined_us(model, inputs, lent, rt, 1000, True))
              + " us per step" if capture else String("")))
    _ = outputs^  # replays write into it until here
