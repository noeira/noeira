"""The hybrid: noeira nn layers own the weights, a MAX train step built in
Mojo updates them in place.

A SAC critic (23 -> 256 -> 256 -> 1, ReLU, batch 256) is three noeira layers
(`LinearReLU`, `LinearReLU`, `Linear`), initialised by nn (Kaiming). The train
step is `max_train`'s, built in Mojo; its parameter buffers are the layers'
own `Param` memory, lent to MAX by address (host memory on the CPU, nn's
device buffers on CUDA). Adam's moments, the counter and the batch are
buffers this program lends as `train_mlp.mojo` does. So MAX trains the
weights nn holds, and nn's own forward then runs on the trained weights.

Checked:
- the step's graph text, every loss, and the final weights read from nn's
  memory, against the Python prototype's step run from nn's initial weights
  (`py_grad.reference`), bit for bit;
- nn's forward on the trained weights, against a float64 NumPy forward of the
  reference's final weights.

nn caches derived copies of a weight (a zero-padded copy on the GPU; a bf16
cast) and refreshes them when the optimizer bumps the weight's `version`. A
MAX step writes the weights behind nn's back, so the program bumps the
versions itself after training. `--no-bump` skips it, to show the stale copy.

    noeira_max/train_from_mojo/run.sh --nn [--gpu]
    ./train_nn_mlp OUT_DIR [--steps N] [--gpu] [--no-bump]
"""

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from max_train import (
    build_mlp_step, check, host_copy, input_shape, lend, numbers, param_name, read, shape_named,
    state_names, step_name,
)
from maxrt import HostBuffer, Runtime, Tensor

from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor as NNTensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU

comptime GLUE = "noeira_max.train_from_mojo.py_grad"
comptime LR = 3e-4
comptime IN = 23
comptime H = 256
comptime OUT = 1
comptime B = 256


def address[target: StaticString](t: NNTensor) raises -> Int:
    """Where the tensor's values live on `target`. nn owns that memory; it must
    outlive every MAX step that is lent it."""
    comptime if target == "gpu":
        return Int(t.dev.value().unsafe_ptr())
    return Int(t.data.unsafe_ptr())


def to_numpy[target: StaticString](
    mut t: NNTensor, ctx: Optional[DeviceContext], py_shape: PythonObject
) raises -> PythonObject:
    """A NumPy copy of the tensor's values (on the GPU, downloaded first).

    The tensor is this call's argument, so it is alive until Python has
    copied it. `array_at(Int(t.data.unsafe_ptr()), ...)` written inline would
    release `t` at its last use, inside the argument list: the copy then
    reads freed memory, whose first bytes the allocator has overwritten."""
    comptime if target == "gpu":
        t.download(ctx.value())
    return Python.import_module(GLUE).array_at(Int(t.data.unsafe_ptr()), py_shape)


struct Critic[target: StaticString](Movable):
    var l1: LinearReLU[IN, H]
    var l2: LinearReLU[H, H]
    var l3: Linear[H, OUT]

    def __init__(out self, ctx: Optional[DeviceContext]) raises:
        self.l1 = LinearReLU[IN, H].make[Self.target, Kaiming](ctx)
        self.l2 = LinearReLU[H, H].make[Self.target, Kaiming](ctx)
        self.l3 = Linear[H, OUT].make[Self.target, Kaiming](ctx)

    def __init__(out self, *, deinit move: Self):
        self.l1 = move.l1^
        self.l2 = move.l2^
        self.l3 = move.l3^

    def forward(
        mut self, mut x: NNTensor, mut h1: NNTensor, mut h2: NNTensor, mut y: NNTensor,
        ctx: Optional[DeviceContext],
    ) raises:
        self.l1.forward[Self.target, B](TensorRefs[1](x), h1, ctx)
        self.l2.forward[Self.target, B](TensorRefs[1](h1), h2, ctx)
        self.l3.forward[Self.target, B](TensorRefs[1](h2), y, ctx)

    def param_address(self, k: Int) raises -> Int:
        """Parameter `k` in the step's order: w0, b0, w1, b1, w2, b2."""
        if k == 0:
            return address[Self.target](self.l1.weight.val)
        if k == 1:
            return address[Self.target](self.l1.bias.val)
        if k == 2:
            return address[Self.target](self.l2.weight.val)
        if k == 3:
            return address[Self.target](self.l2.bias.val)
        if k == 4:
            return address[Self.target](self.l3.weight.val)
        return address[Self.target](self.l3.bias.val)

    def param_numpy(mut self, k: Int, ctx: Optional[DeviceContext], py_shape: PythonObject) raises -> PythonObject:
        if k == 0:
            return to_numpy[Self.target](self.l1.weight.val, ctx, py_shape)
        if k == 1:
            return to_numpy[Self.target](self.l1.bias.val, ctx, py_shape)
        if k == 2:
            return to_numpy[Self.target](self.l2.weight.val, ctx, py_shape)
        if k == 3:
            return to_numpy[Self.target](self.l2.bias.val, ctx, py_shape)
        if k == 4:
            return to_numpy[Self.target](self.l3.weight.val, ctx, py_shape)
        return to_numpy[Self.target](self.l3.bias.val, ctx, py_shape)

    def bump(mut self):
        """What nn's optimizer does after each step: tells the layers their
        weights changed, so they refresh any cached copy."""
        self.l1.weight.val.version += 1
        self.l1.bias.val.version += 1
        self.l2.weight.val.version += 1
        self.l2.bias.val.version += 1
        self.l3.weight.val.version += 1
        self.l3.bias.val.version += 1


def py_list(values: List[Int]) raises -> PythonObject:
    var out = Python.import_module("builtins").list()
    for v in values:
        _ = out.append(PythonObject(v))
    return out


def nn_forward_error[target: StaticString](
    mut net: Critic[target], mut x: NNTensor, ctx: Optional[DeviceContext],
    params: PythonObject, x_np: PythonObject,
) raises -> Float64:
    """nn's forward on `x`, against a float64 forward from `params`."""
    var h1 = NNTensor.alloc(B * H)
    var h2 = NNTensor.alloc(B * H)
    var y = NNTensor.alloc(B * OUT)
    net.forward(x, h1, h2, y, ctx)
    var glue = Python.import_module(GLUE)
    var got = to_numpy[target](y, ctx, py_list([B, OUT]))
    return Float64(py=glue.forward_error(got, params, x_np, "relu"))


def run[target: StaticString](dir: String, steps: Int, bump: Bool) raises:
    comptime gpu = target == "gpu"
    var ctx = Optional(DeviceContext()) if gpu else Optional[DeviceContext](None)
    var s = shape_named("sac")
    var n = 2 * (len(s.dims) - 1)
    var device = String(target)
    var glue = Python.import_module(GLUE)
    var builtins = Python.import_module("builtins")

    # nn's layers, initialised by nn: their initial weights seed the reference.
    var net = Critic[target](ctx)
    var init = builtins.dict()
    for k in range(n):
        init[PythonObject(param_name(k))] = net.param_numpy(k, ctx, py_list(input_shape(s, k)))
    var data = glue.problem(py_list(s.dims), s.batch, 0)  # the batch (its weights are not used)
    var x = NNTensor.alloc(B * IN)
    var x_values = glue.to_list(data["x"])
    for i in range(B * IN):
        x.data[i] = Float32(Float64(py=x_values[i]))
    comptime if gpu:
        x.upload(ctx.value())
    # One forward before training, as a model in use would have run: it builds
    # nn's cached copies of the weights from the initial values.
    var before = nn_forward_error[target](net, x, ctx, init, data["x"])
    print("[nn] forward on the initial weights: max relative error", before)

    var g = build_mlp_step(s, device, LR)
    var text = String(glue.graph_text(g.backend.graph))
    var rt = Runtime(accelerator=gpu)
    var model = g.compile(rt, dir + "/" + step_name(s) + ".mef", device)
    print("[mojo] built and compiled", step_name(s), "in", g.compile_seconds, "s on", device)

    # The parameters: nn's own memory. The rest: lent as train_mlp.mojo does.
    var inputs = rt.tensor_map()
    var staging = rt.tensor_map()
    var on_device = List[Tensor]()
    for k in range(n):
        inputs.borrow_address("input" + String(k), net.param_address(k), DType.float32, input_shape(s, k), gpu)
    for k in range(n, 3 * n + 1):
        lend(inputs, staging, on_device, rt, gpu, k, HostBuffer(4 * numbers(input_shape(s, k))), input_shape(s, k))
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 1, host_copy(data["x"]), input_shape(s, 3 * n + 1))
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 2, host_copy(data["y"]), input_shape(s, 3 * n + 2))

    var losses = List[Float32]()
    var times = List[Int]()
    for _ in range(steps):
        var t = perf_counter_ns()
        var out = model.execute(inputs).tensor("output0")
        var loss = out.to_host().item[DType.float32]() if gpu else out.item[DType.float32]()
        times.append(Int(perf_counter_ns() - t))
        losses.append(loss)
    sort(times)
    print("[mojo]", steps, "MAX steps on nn's weights; loss", losses[0], "->", losses[len(losses) - 1],
          "; median step", Float64(times[len(times) // 2]) / 1000.0, "us")

    var ref_out = glue.reference(py_list(s.dims), s.batch, 0, steps, step_name(s), s.act, LR, device, init)
    var finals = List[PythonObject]()
    for k in range(n):
        finals.append(net.param_numpy(k, ctx, py_list(input_shape(s, k))))
    for k in range(n, 3 * n + 1):
        finals.append(read(inputs, on_device, gpu, k, glue, py_list(input_shape(s, k)), first_lent=n))
    var failures = check(text, losses, finals, state_names(s), ref_out)

    # nn's forward on the weights MAX trained.
    var tolerance = 5e-3 if gpu else 1e-5  # TF32 on CUDA
    print("[nn] training moved the output by", Float64(py=glue.forward_change(init, ref_out["params"], data["x"], "relu")),
          "(relative): the scale the checks below resolve")
    var stale = nn_forward_error[target](net, x, ctx, ref_out["params"], data["x"])
    print("[nn] forward on the trained weights, versions not bumped: max relative error", stale)
    if bump:
        net.bump()
        var fresh = nn_forward_error[target](net, x, ctx, ref_out["params"], data["x"])
        print("[nn] forward on the trained weights, versions bumped:     max relative error", fresh)
        if fresh > tolerance:
            failures += 1
    _ = x^  # lent to nn's forward until here
    if failures > 0:
        raise Error("FAIL: " + String(failures) + " checks failed")
    print("PASS: MAX trained nn's own weights in place, bit for bit with the Python step;"
          " nn's forward runs on them")


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var steps = 50
    var gpu = False
    var bump = True
    var i = 2
    while i < len(args):
        if args[i] == "--steps":
            steps = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--gpu":
            gpu = True
        elif args[i] == "--no-bump":
            bump = False
        else:
            raise Error("unknown flag " + String(args[i]))
        i += 1
    if gpu:
        run["gpu"](dir, steps, bump)
    else:
        run["cpu"](dir, steps, bump)
