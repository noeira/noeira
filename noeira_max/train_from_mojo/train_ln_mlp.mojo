"""A train step with a Mojo kernel pair, built in Mojo: dense -> LayerNorm
(noeira's custom op) -> ReLU -> dense, MSE, AdamW.

The forward pass calls the autodiff prototype's `noeira_layer_norm_fwd`
kernel through `max_graph_gen.custom`. It returns the normalised rows and
their mean and rstd as residuals. The gradient comes from the same transform
as the MLP's: its rule for `mo.custom` dispatches on the kernel's symbol and
emits `noeira_layer_norm_bwd` with those residuals. So Mojo builds the graph,
Mojo kernels compute the layer both ways, and the one Python step is still
differentiation.

Gated like `train_mlp.mojo`, against the same model built in Python
(`py_grad.ln_reference`): graph text, every loss, the final buffers.

    ./train_ln_mlp OUT_DIR [--steps N] [--gpu]
"""

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from max_graph_gen import Dim, Graph, TensorType, Value, buffer_load, constant, custom, relu, reshape
from max_train import AdamW, Tape, check, host_copy, lend, mse, numbers, read
from maxrt import HostBuffer, Runtime, Tensor

comptime GLUE = "noeira_max.train_from_mojo.py_grad"
comptime LR = 3e-4
comptime IN = 8
comptime HID = 64
comptime OUT = 4
comptime B = 16


def param_name(k: Int) -> String:
    """The six parameters, in the step's input order."""
    var all: List[String] = ["w0", "b0", "g0", "beta0", "w1", "b1"]
    return all[k]


def param_shape(k: Int) -> List[Int]:
    if k == 0:
        return [IN, HID]
    if k == 4:
        return [HID, OUT]
    if k == 5:
        return [OUT]
    return [HID]


def input_shape(k: Int) -> List[Int]:
    """Inputs: the six parameters, the counter, each one's m and v, x, y."""
    if k < 6:
        return param_shape(k)
    if k == 6:
        return [1]
    if k <= 18:
        return param_shape((k - 7) // 2)
    if k == 19:
        return [B, IN]
    return [B, OUT]


def names() -> List[String]:
    var out = List[String]()
    for k in range(6):
        out.append(param_name(k))
    out.append("step")
    for k in range(6):
        out.append("m." + param_name(k))
        out.append("v." + param_name(k))
    return out^


def f32(shape: List[Int], device: String, buffer: Bool = False) -> TensorType:
    var dims = List[Dim]()
    for d in shape:
        dims.append(Dim(d))
    return TensorType(DType.float32, dims^, device, is_buffer=buffer)


def layer_norm(x: Value, gamma: Value, beta: Value, kernels: String) raises -> Value:
    """`models/common.py`'s `layer_norm_kernel`: the rows, an epsilon
    constant, the custom op, and the result in `x`'s shape."""
    var dev = x.type.device
    var rows = reshape(x, [Dim(B), Dim(HID)])
    var eps = constant([1e-5], DType.float32, [Dim(1)], dev)
    var outs = custom(
        kernels, "noeira_layer_norm_fwd", [rows^, gamma.copy(), beta.copy(), eps^],
        [f32([B, HID], dev), f32([B, 1], dev), f32([B, 1], dev)],
    )
    return reshape(outs[0], [Dim(B), Dim(HID)])


def build(device: String, kernels: String) raises -> Graph:
    var types = List[TensorType]()
    for k in range(21):
        types.append(f32(input_shape(k), device, buffer=k < 19))
    var g = Graph("ln_mlp_train_step", types)
    var inputs = g.inputs()
    var params = List[Value]()
    var m = List[Value]()
    var v = List[Value]()
    for k in range(6):
        params.append(inputs[k].copy())
        m.append(inputs[7 + 2 * k].copy())
        v.append(inputs[8 + 2 * k].copy())

    # Loaded in name order (b0, b1, beta0, g0, w0, w1), as the transform
    # loads a dict of buffers.
    var order = [1, 5, 3, 2, 0, 4]
    var loaded = List[Value]()
    for _ in range(6):
        loaded.append(Value(Python.none(), f32([], device)))
    var primals = List[Value]()
    for k in range(6):
        var value = buffer_load(params[order[k]])
        loaded[order[k]] = value.copy()
        primals.append(value^)

    var tape = Tape()
    var h = inputs[19] @ loaded[0] + loaded[1]
    h = layer_norm(h, loaded[2], loaded[3], kernels)
    h = relu(h)
    var loss = mse(h @ loaded[4] + loaded[5], inputs[20])
    var by_order = tape.gradients(primals, loss)
    var grads = List[Value]()
    for _ in range(6):
        grads.append(Value(Python.none(), f32([], device)))
    for k in range(6):
        grads[order[k]] = by_order[k].copy()
    AdamW(LR, 0.9, 0.999, 1e-8).apply(params, m, v, inputs[6], grads)
    g.output([loss^])
    return g^


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var steps = 50
    var gpu = False
    var i = 2
    while i < len(args):
        if args[i] == "--steps":
            steps = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--gpu":
            gpu = True
        else:
            raise Error("unknown flag " + String(args[i]))
        i += 1
    var device = String("gpu") if gpu else String("cpu")
    var glue = Python.import_module(GLUE)
    var builtins = Python.import_module("builtins")
    var py_dims = builtins.list()
    for d in [IN, HID, OUT]:
        _ = py_dims.append(PythonObject(d))

    var g = build(device, String(glue.kernels_path()))
    var text = String(glue.graph_text(g.backend.graph))
    var rt = Runtime(accelerator=gpu)
    var model = g.compile(rt, dir + "/ln_mlp_train_step.mef", device)
    print("[mojo] built and compiled ln_mlp_train_step (with the LayerNorm kernel pair) in",
          g.compile_seconds, "s on", device)

    var data = glue.ln_problem(py_dims, B, 0)
    var inputs = rt.tensor_map()
    var staging = rt.tensor_map()
    var on_device = List[Tensor]()
    for k in range(6):
        lend(inputs, staging, on_device, rt, gpu, k, host_copy(data["init"][param_name(k)]), input_shape(k))
    for k in range(6, 19):
        lend(inputs, staging, on_device, rt, gpu, k, HostBuffer(4 * numbers(input_shape(k))), input_shape(k))
    lend(inputs, staging, on_device, rt, gpu, 19, host_copy(data["x"]), input_shape(19))
    lend(inputs, staging, on_device, rt, gpu, 20, host_copy(data["y"]), input_shape(20))

    var losses = List[Float32]()
    for _ in range(steps):
        var out = model.execute(inputs).tensor("output0")
        losses.append(out.to_host().item[DType.float32]() if gpu else out.item[DType.float32]())
    print("[mojo]", steps, "steps; loss", losses[0], "->", losses[len(losses) - 1])

    var ref_out = glue.ln_reference(py_dims, B, 0, steps, "ln_mlp_train_step", LR, device)
    print("[python] reference compiled in", Float64(py=ref_out["compile_s"]), "s")
    var finals = List[PythonObject]()
    for k in range(19):
        var py_shape = builtins.list()
        for d in input_shape(k):
            _ = py_shape.append(PythonObject(d))
        finals.append(read(inputs, on_device, gpu, k, glue, py_shape))
    var failures = check(text, losses, finals, names(), ref_out)
    var pair = "noeira_layer_norm_fwd" in text and "noeira_layer_norm_bwd" in text
    print("the graph calls both kernels (noeira_layer_norm_fwd, noeira_layer_norm_bwd):", pair)
    if not pair:
        failures += 1
    if failures > 0:
        raise Error("FAIL: the Mojo-built step differs from the Python one")
    print("PASS: the kernel-pair step built in Mojo is the Python prototype's, bit for bit")
