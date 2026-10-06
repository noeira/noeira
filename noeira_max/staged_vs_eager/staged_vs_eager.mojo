"""One MLP definition, two ways to run it: eager on noeira's nn
kernels, or staged as a MAX graph, on the same weight memory.

`MLP[IN, H, OUT, target]` holds noeira layers (`LinearReLU`, `LinearReLU`,
`Linear`: in -> H -> H -> out). `eager[B]` calls their forwards. `staged`
builds the same model with the generated graph builder (`max_graph_gen`,
batch symbolic: one compile per width), compiles it, and runs it through
`maxrt`, lending it the eager path's own weight and bias memory by address:
nothing is copied between the two.

- CPU (default): host memory, lent under the host device.
- `--gpu`: noeira's device buffers, lent under the accelerator. This is the
  CUDA case (`maxrt_tests/probe_cuda_context.mojo`); on Metal, MAX 26.6 reads
  a lent device address as zeros, and the numerics check says so.

Checks the outputs agree at batch 1, 64 and 1024, then times the delivered
latency of both paths across batch sizes and widths: on a GPU, each call
synchronises (Mojo's context before, MAX's device after). One `RESULT
{json}` line per point.

    noeira_max/staged_vs_eager/run.sh [--gpu]
"""

from std.math import sin
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from max_graph_gen import Dim, Graph, TensorType, relu
from maxrt import Model, Runtime, TensorMap

from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU

comptime IN = 64
comptime OUT = 16
comptime MIN_SECONDS = 0.25
comptime MIN_REPS = 3
comptime WARMUP = 3


def f32(var shape: List[Dim], device: String) -> TensorType:
    return TensorType(DType.float32, shape^, device)


def address[target: StaticString](t: Tensor) raises -> Int:
    """A tensor's storage on `target`. The tensor must outlive every use."""
    comptime if target == "gpu":
        return Int(t.dev.value().unsafe_ptr())
    return Int(t.data.unsafe_ptr())


struct MLP[IN: Int, H: Int, OUT: Int, target: StaticString](Movable):
    var l1: LinearReLU[Self.IN, Self.H]
    var l2: LinearReLU[Self.H, Self.H]
    var l3: Linear[Self.H, Self.OUT]

    def __init__(out self, ctx: Optional[DeviceContext]) raises:
        self.l1 = LinearReLU[Self.IN, Self.H].make[Self.target, Kaiming](ctx)
        self.l2 = LinearReLU[Self.H, Self.H].make[Self.target, Kaiming](ctx)
        self.l3 = Linear[Self.H, Self.OUT].make[Self.target, Kaiming](ctx)

    def __init__(out self, *, deinit move: Self):
        self.l1 = move.l1^
        self.l2 = move.l2^
        self.l3 = move.l3^

    def eager[B: Int](
        mut self, mut x: Tensor, mut h1: Tensor, mut h2: Tensor, mut y: Tensor,
        ctx: Optional[DeviceContext],
    ) raises:
        """noeira's kernels: three forwards."""
        self.l1.forward[Self.target, B](TensorRefs[1](x), h1, ctx)
        self.l2.forward[Self.target, B](TensorRefs[1](h1), h2, ctx)
        self.l3.forward[Self.target, B](TensorRefs[1](h2), y, ctx)

    def staged(self, rt: Runtime, dir: String) raises -> Tuple[Model, Float64]:
        """The same model as a MAX graph with a symbolic batch, built in
        Mojo, compiled once; returns it and the compile seconds."""
        var dev = String(Self.target)
        var g = Graph(
            "mlp_" + dev + "_h" + String(Self.H),
            [f32([Dim.symbolic("batch"), Dim(Self.IN)], dev), f32([Self.IN, Self.H], dev),
             f32([Self.H], dev), f32([Self.H, Self.H], dev), f32([Self.H], dev),
             f32([Self.H, Self.OUT], dev), f32([Self.OUT], dev)],
        )
        var p = g.inputs()
        var h = relu(p[0] @ p[1] + p[2])
        h = relu(h @ p[3] + p[4])
        g.output([h @ p[5] + p[6]])
        var model = g.compile(rt, dir + "/mlp_" + dev + "_h" + String(Self.H) + ".mef", dev)
        return (model^, g.compile_seconds)

    def lend(self, mut inputs: TensorMap, x: Tensor, batch: Int) raises:
        """The staged model's inputs: the eager path's own memory."""
        comptime on_device = Self.target == "gpu"
        comptime f = DType.float32
        inputs.borrow_address("input0", address[Self.target](x), f, [batch, Self.IN], on_device)
        inputs.borrow_address("input1", address[Self.target](self.l1.weight.val), f, [Self.IN, Self.H], on_device)
        inputs.borrow_address("input2", address[Self.target](self.l1.bias.val), f, [Self.H], on_device)
        inputs.borrow_address("input3", address[Self.target](self.l2.weight.val), f, [Self.H, Self.H], on_device)
        inputs.borrow_address("input4", address[Self.target](self.l2.bias.val), f, [Self.H], on_device)
        inputs.borrow_address("input5", address[Self.target](self.l3.weight.val), f, [Self.H, Self.OUT], on_device)
        inputs.borrow_address("input6", address[Self.target](self.l3.bias.val), f, [Self.OUT], on_device)


def median_us(mut samples: List[Int]) -> Float64:
    sort(samples)
    return Float64(samples[len(samples) // 2]) / 1000.0


def sync(ctx: Optional[DeviceContext]) raises:
    if ctx:
        ctx.value().synchronize()


def point[H: Int, B: Int, target: StaticString](
    mut mlp: MLP[IN, H, OUT, target], model: Model, rt: Runtime, ctx: Optional[DeviceContext],
    compile_s: Float64, check: Bool, report: Bool = True,
) raises:
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Float32(sin(Float64(i) * 0.37))
    comptime if target == "gpu":
        x.upload(ctx.value())
    var h1 = Tensor.alloc(B * H)
    var h2 = Tensor.alloc(B * H)
    var y = Tensor.alloc(B * OUT)
    mlp.eager[B](x, h1, h2, y, ctx)  # sizes the eager path's buffers
    sync(ctx)
    var inputs = rt.tensor_map()
    mlp.lend(inputs, x, B)

    for _ in range(WARMUP):
        mlp.eager[B](x, h1, h2, y, ctx)
        sync(ctx)
        _ = model.execute(inputs)
        rt.synchronize()
    var eager = List[Int]()
    var total = 0
    while len(eager) < MIN_REPS or Float64(total) < MIN_SECONDS * 1e9:
        var t = perf_counter_ns()
        mlp.eager[B](x, h1, h2, y, ctx)
        sync(ctx)
        var dt = Int(perf_counter_ns() - t)
        eager.append(dt)
        total += dt
    var staged = List[Int]()
    total = 0
    var worst = Float64(0)
    var scale = Float64(0)
    while len(staged) < MIN_REPS or Float64(total) < MIN_SECONDS * 1e9:
        var t = perf_counter_ns()
        sync(ctx)  # the producer (eager's input) is done
        var outputs = model.execute(inputs)
        rt.synchronize()  # the output is ready for a consumer
        var dt = Int(perf_counter_ns() - t)
        staged.append(dt)
        total += dt
        if check and len(staged) == 1:
            # eager's y against the staged output, relative to the largest value.
            comptime if target == "gpu":
                y.download(ctx.value())
            var out = outputs.tensor("output0")
            var host = out.to_host()
            var po = host.data[Float32]()
            for i in range(B * OUT):
                scale = max(scale, abs(Float64(y.data[i])))
            for i in range(B * OUT):
                worst = max(worst, abs(Float64(po[unsafe_offset=i]) - Float64(y.data[i])) / scale)
    var e = median_us(eager)
    var s = median_us(staged)
    _ = x^  # lent by address to `inputs` until here
    if not report:
        return
    print(
        "RESULT {\"target\": \"" + String(target) + "\", \"width\": " + String(H)
        + ", \"batch\": " + String(B)
        + ", \"eager_us\": " + String(e) + ", \"staged_us\": " + String(s)
        + ", \"staged_over_eager\": " + String(s / e)
        + ", \"staged_compile_s\": " + String(compile_s)
        + (", \"max_rel_diff\": " + String(worst) + ", \"output_scale\": " + String(scale) if check else String(""))
        + "}"
    )


def sweep[H: Int, target: StaticString](rt: Runtime, ctx: Optional[DeviceContext], dir: String) raises:
    var mlp = MLP[IN, H, OUT, target](ctx)
    var staged = mlp.staged(rt, dir)
    ref model = staged[0]
    var compile_s = staged[1]
    # A first pass that is not reported: the first calls of a new model and of
    # fresh weights pay one-time costs (page faults, lazy initialisation).
    point[H, 1, target](mlp, model, rt, ctx, compile_s, False, report=False)
    # Numerics are checked at batch 1, 64 and 1024.
    point[H, 1, target](mlp, model, rt, ctx, compile_s, True)
    point[H, 4, target](mlp, model, rt, ctx, compile_s, False)
    point[H, 16, target](mlp, model, rt, ctx, compile_s, False)
    point[H, 64, target](mlp, model, rt, ctx, compile_s, True)
    point[H, 256, target](mlp, model, rt, ctx, compile_s, False)
    point[H, 1024, target](mlp, model, rt, ctx, compile_s, True)
    point[H, 4096, target](mlp, model, rt, ctx, compile_s, False)


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var gpu = len(args) > 2 and args[2] == "--gpu"
    if gpu:
        var rt = Runtime(accelerator=True)
        var ctx = Optional(DeviceContext())
        sweep[256, "gpu"](rt, ctx, dir)
        sweep[1024, "gpu"](rt, ctx, dir)
        sweep[4096, "gpu"](rt, ctx, dir)
    else:
        var rt = Runtime()
        var none = Optional[DeviceContext](None)
        sweep[256, "cpu"](rt, none, dir)
        sweep[1024, "cpu"](rt, none, dir)
        sweep[4096, "cpu"](rt, none, dir)
