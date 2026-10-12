"""An MLP train step, built in Mojo: the shapes of noeira's RL networks, and
the step's graph.

The step is `noeira_max/autodiff/bench/mlp_step.py`'s, emitted in the same
order: the gate checks the two graphs print the same MLIR.
"""

from std.python import Python

from max_graph_gen import Dim, Graph, TensorType, Value, buffer_load, relu, tanh

from .train import AdamW, Tape, mse


@fieldwise_init
struct Shape(Copyable, Movable):
    var dims: List[Int]
    var act: String
    var batch: Int


def shape_named(name: String) raises -> Shape:
    if name == "small":
        return Shape([8, 16, 16, 4], "relu", 4)
    if name == "ppo":  # mlp_step.py's PPO network
        return Shape([8, 64, 64, 4], "tanh", 64)
    if name == "sac":  # mlp_step.py's SAC critic
        return Shape([23, 256, 256, 1], "relu", 256)
    raise Error("unknown shape " + name)


def step_name(s: Shape) -> String:
    """`mlp_step.py`'s graph name, so both graphs print the same header."""
    var out = String("mlp_")
    for i in range(len(s.dims)):
        if i > 0:
            out += "x"
        out += String(s.dims[i])
    return out + "_" + s.act + "_b" + String(s.batch) + "_train_step"


def f32(var dims: List[Dim], device: String, buffer: Bool = False) -> TensorType:
    return TensorType(DType.float32, dims^, device, is_buffer=buffer)


def param_dims(s: Shape, i: Int) -> List[Int]:
    """Parameter `i` in `w0, b0, w1, b1, ...` order."""
    var layer = i // 2
    if i % 2 == 0:
        return [s.dims[layer], s.dims[layer + 1]]
    return [s.dims[layer + 1]]


def input_shape(s: Shape, k: Int) -> List[Int]:
    """Input `k` of the step: the parameters, the counter, each parameter's m
    and v, then the batch (`x`, `y`)."""
    var n = 2 * (len(s.dims) - 1)
    if k < n:
        return param_dims(s, k)
    if k == n:
        return [1]
    if k <= 3 * n:
        return param_dims(s, (k - n - 1) // 2)
    if k == 3 * n + 1:
        return [s.batch, s.dims[0]]
    return [s.batch, s.dims[len(s.dims) - 1]]


def state_names(s: Shape) -> List[String]:
    """The names of inputs 0 to 3n: the parameters, `step`, then each
    parameter's `m.` and `v.` (`py_grad.reference`'s keys)."""
    var n = 2 * (len(s.dims) - 1)
    var out = List[String]()
    for k in range(n):
        out.append(param_name(k))
    out.append("step")
    for k in range(n):
        out.append("m." + param_name(k))
        out.append("v." + param_name(k))
    return out^


def param_name(i: Int) -> String:
    return ("w" if i % 2 == 0 else "b") + String(i // 2)


def as_dims(shape: List[Int]) -> List[Dim]:
    var out = List[Dim]()
    for d in shape:
        out.append(Dim(d))
    return out^


def numbers(shape: List[Int]) -> Int:
    var count = 1
    for d in shape:
        count *= d
    return count


def build_mlp_step(s: Shape, device: String, lr: Float64) raises -> Graph:
    """The train step. Inputs, as the Python prototype orders them: the
    parameters (w0, b0, w1, b1, ...), the counter, each parameter's m and v,
    then the batch."""
    var depth = len(s.dims) - 1
    var n = 2 * depth
    var types = List[TensorType]()
    for i in range(n):
        types.append(f32(as_dims(param_dims(s, i)), device, buffer=True))
    types.append(f32([Dim(1)], device, buffer=True))
    for i in range(n):
        types.append(f32(as_dims(param_dims(s, i)), device, buffer=True))
        types.append(f32(as_dims(param_dims(s, i)), device, buffer=True))
    types.append(f32([Dim(s.batch), Dim(s.dims[0])], device))
    types.append(f32([Dim(s.batch), Dim(s.dims[depth])], device))

    var g = Graph(step_name(s), types)
    var inputs = g.inputs()
    var params = List[Value]()
    var m = List[Value]()
    var v = List[Value]()
    for i in range(n):
        params.append(inputs[i].copy())
        m.append(inputs[n + 1 + 2 * i].copy())
        v.append(inputs[n + 2 + 2 * i].copy())
    ref step = inputs[n]
    ref x = inputs[3 * n + 1]
    ref y = inputs[3 * n + 2]

    # The parameters' values, loaded in name order (b0, b1, ..., w0, w1, ...),
    # as the Python transform loads a dict of buffers.
    var order = List[Int]()
    for i in range(depth):
        order.append(2 * i + 1)
    for i in range(depth):
        order.append(2 * i)
    var loaded = List[Value]()
    for _ in range(n):
        loaded.append(Value(Python.none(), f32(List[Dim](), device)))
    var primals = List[Value]()
    for k in range(n):
        var value = buffer_load(params[order[k]])
        loaded[order[k]] = value.copy()
        primals.append(value^)

    var tape = Tape()
    var h = x.copy()
    for layer in range(depth):
        h = h @ loaded[2 * layer] + loaded[2 * layer + 1]
        if layer < depth - 1:
            h = relu(h) if s.act == "relu" else tanh(h)
    var loss = mse(h, y)
    var by_order = tape.gradients(primals, loss)

    var grads = List[Value]()
    for _ in range(n):
        grads.append(Value(Python.none(), f32(List[Dim](), device)))
    for k in range(n):
        grads[order[k]] = by_order[k].copy()
    AdamW(lr, 0.9, 0.999, 1e-8).apply(params, m, v, step, grads)
    g.output([loss^])
    return g^
