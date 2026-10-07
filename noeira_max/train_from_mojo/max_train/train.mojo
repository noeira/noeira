"""Training graphs built from Mojo: the gradient tape, a loss, an optimizer.

A Mojo program builds its train step into a MAX graph with `max_graph_gen`:
the model's forward pass, the loss, and the optimizer's update, whose
parameters and moments are buffers the graph stores into. One step is one
execution, run by `maxrt` with no Python.

The gradient is the one step built in Python, at setup: `Tape` records the
graph's ops before the forward pass, and `Tape.gradients` hands the ops
emitted since to the autodiff prototype's transform (`py_grad.py`), which
emits their VJPs into the same graph. That is where the VJP rule registry
lives today; with a registry in the graph layer, this call would be a C call.

Every op here is emitted in the order `noeira_max/autodiff` emits it from
Python (`bench/mlp_step.py`'s `mse`, `optim.AdamW`), so a step built in Mojo
is the same graph as the Python one, op for op.
"""

from std.python import Python, PythonObject

from max_graph_gen import (
    Dim,
    PythonBackend,
    Value,
    buffer_load,
    buffer_store,
    cast,
    constant,
    equal,
    logsoftmax,
    negative,
    pow,
    reduce_mean,
    reduce_sum,
    reshape,
    scalar,
    sqrt,
)
from max_graph_gen.backend import values_from_py

comptime GLUE = "noeira_max.train_from_mojo.py_grad"


struct Tape(Movable):
    """The ops of the current graph before the forward pass. Create it after
    loading the parameters, before the first op that uses them."""

    var graph: PythonObject
    var before: PythonObject

    def __init__(out self) raises:
        self.graph = PythonBackend.current().graph
        self.before = Python.import_module(GLUE).snapshot(self.graph)

    def gradients(self, primals: List[Value], loss: Value) raises -> List[Value]:
        """The gradient of the scalar `loss` with respect to each of
        `primals`, emitted into the graph; one per primal, in order."""
        var builtins = Python.import_module("builtins")
        var handles = builtins.list()
        for i in range(len(primals)):
            _ = handles.append(primals[i].handle)
        return values_from_py(
            Python.import_module(GLUE).backward(self.graph, self.before, handles, loss.handle)
        )


def mse(prediction: Value, target: Value) raises -> Value:
    """The mean squared error, as `[1, 1]`: every axis summed, then scaled."""
    var d = prediction - target
    var total = d * d
    for axis in range(total.type.rank()):
        total = reduce_sum(total, axis)
    var n = 1
    for i in range(prediction.type.rank()):
        n *= prediction.type.shape[i].size
    return total * scalar(1.0 / Float64(n), total)


def cross_entropy(logits: Value, targets: Value) raises -> Value:
    """The mean cross-entropy over every leading position, as `[1, 1]`
    (`models/common.py`'s): `logits [..., classes]`, integer `targets
    [...]`, the one-hot built in the graph."""
    var classes = logits.type.shape[logits.type.rank() - 1].size
    var rows = 1
    for i in range(targets.type.rank()):
        rows *= targets.type.shape[i].size
    var flat = reshape(logits, [Dim(rows), Dim(classes)])
    var labels = reshape(targets, [Dim(rows), Dim(1)])
    var ids = List[Float64]()
    for c in range(classes):
        ids.append(Float64(c))
    var choices = constant(ids, DType.int64, [Dim(classes)], targets.type.device)
    var one_hot = cast(equal(labels, choices), logits.type.dtype)
    var picked = reduce_sum(logsoftmax(flat, 1) * one_hot, 1)
    return negative(reduce_mean(picked, 0))


@fieldwise_init
struct AdamW(Copyable, Movable):
    """`torch.optim.AdamW` with no weight decay and a constant learning rate,
    emitted into the graph. The step counter is a buffer, so the bias
    correction is computed on the device and every call takes the same
    inputs (what capture needs)."""

    var lr: Float64
    var b1: Float64
    var b2: Float64
    var eps: Float64

    def apply(
        self,
        params: List[Value],
        m: List[Value],
        v: List[Value],
        step: Value,
        grads: List[Value],
    ) raises:
        """Updates every parameter buffer and its moments in place, and the
        counter. `params`, `m`, `v` and `grads` are aligned."""
        var it = buffer_load(step)
        var one = scalar(1.0, it)
        var lr = scalar(self.lr, it) * one  # the constant schedule's scale
        var t = it + scalar(1.0, it)
        var b1_t = pow(scalar(self.b1, t), t)
        var bias1 = scalar(1.0, t) - b1_t
        var b2_t = pow(scalar(self.b2, t), t)
        var bias2_sqrt = sqrt(scalar(1.0, t) - b2_t)
        var step_size = lr / bias1

        # Every store after every load: MAX 26.6's compile time explodes when
        # buffer loads and stores alternate.
        var stores = List[Value]()
        var values = List[Value]()
        for i in range(len(params)):
            ref g = grads[i]
            var p = buffer_load(params[i])
            var m0 = buffer_load(m[i])
            var v0 = buffer_load(v[i])
            var dm = g - m0
            var m1 = m0 + dm * scalar(1.0 - self.b1, dm)  # m.lerp_(g, 1 - b1)
            var vb = v0 * scalar(self.b2, v0)
            var gg = g * g
            var v1 = vb + gg * scalar(1.0 - self.b2, gg)  # v.mul_(b2).addcmul_(g, g, 1 - b2)
            var q = sqrt(v1) / bias2_sqrt
            var denom = q + scalar(self.eps, q)
            var p1 = p - step_size * (m1 / denom)
            stores.append(m[i].copy())
            values.append(m1^)
            stores.append(v[i].copy())
            values.append(v1^)
            stores.append(params[i].copy())
            values.append(p1^)
        stores.append(step.copy())
        values.append(t^)
        for i in range(len(stores)):
            buffer_store(stores[i], values[i])
