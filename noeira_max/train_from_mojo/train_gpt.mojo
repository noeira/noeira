"""The autodiff prototype's GPT train step, built in Mojo, with both of noeira's
kernel pairs: LayerNorm and causal attention (FlashAttention-2 with its
log-sum-exp residuals) as Mojo custom ops.

The model is `models/gpt.py`'s with `layer_norm="kernel"`,
`attention="kernel"` and no dropout: token and position embeddings, pre-norm
blocks (attention, then a GELU MLP), a final LayerNorm, the head tied to the
token embedding, and the cross-entropy loss. AdamW as in `train_mlp.mojo`.
The batch is a fixed one (token ids and targets as graph inputs); the Python
GPT samples it from a corpus in the graph, which this gate leaves out.

Gated like `train_mlp.mojo` against the same step built in Python
(`py_grad.gpt_reference`): graph text, every loss, the final buffers.

    ./train_gpt OUT_DIR [--steps N] [--layers L] [--gpu] [--capture] [--no-reference]
                        [--extra N [--extra-sync]]
"""

from std.python import Python, PythonObject
from std.sys import argv
from std.time import perf_counter_ns

from max_graph_gen import (
    Dim, Graph, TensorType, Value, buffer_load, constant, custom, gather, gelu_tanh, reshape,
    transpose,
)
from max_train import (
    AdamW, Tape, check, cross_entropy, host_copy, lend, median_us, numbers, pipelined_us, read, train,
)
from maxrt import HostBuffer, Runtime, Tensor

comptime GLUE = "noeira_max.train_from_mojo.py_grad"
comptime LR = 3e-4
comptime VOCAB = 65
comptime SEQ = 16
comptime DIM = 64
comptime HEADS = 2
comptime B = 4


def f32(shape: List[Int], device: String, buffer: Bool = False, dtype: DType = DType.float32) -> TensorType:
    var dims = List[Dim]()
    for d in shape:
        dims.append(Dim(d))
    return TensorType(dtype, dims^, device, is_buffer=buffer)


struct Params(Movable):
    """The parameters' values, by name (`gpt.init`'s names)."""

    var names: List[String]
    var values: List[Value]

    def __init__(out self, var names: List[String], var values: List[Value]):
        self.names = names^
        self.values = values^

    def __getitem__(self, name: String) raises -> Value:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.values[i].copy()
        raise Error("no parameter " + name)


def layer_norm(x: Value, gamma: Value, beta: Value, kernels: String) raises -> Value:
    """`models/common.py`'s `layer_norm_kernel`: the rows, epsilon, the custom
    op, the result in `x`'s shape."""
    var dev = x.type.device
    var d = x.type.shape[x.type.rank() - 1].size
    var n = 1
    for i in range(x.type.rank() - 1):
        n *= x.type.shape[i].size
    var rows = reshape(x, [Dim(n), Dim(d)])
    var eps = constant([1e-5], DType.float32, [Dim(1)], dev)
    var outs = custom(
        kernels, "noeira_layer_norm_fwd", [rows^, gamma.copy(), beta.copy(), eps^],
        [f32([n, d], dev), f32([n, 1], dev), f32([n, 1], dev)],
    )
    return reshape(outs[0], x.type.shape)


def attention(qkv: Value, kernels: String) raises -> Value:
    """`models/common.py`'s `attention_kernel`: noeira's fused causal
    attention over `qkv [B, T, 3C]`, heads merged."""
    var dev = qkv.type.device
    var b = qkv.type.shape[0].size
    var t = qkv.type.shape[1].size
    var c = qkv.type.shape[2].size // 3
    var outs = custom(
        kernels, "noeira_attention_fwd", [qkv.copy()],
        [f32([b, t, c], dev), f32([b * HEADS * t], dev)],
        ["B", "H", "S", "HD"], [b, HEADS, t, c // HEADS],
    )
    return outs[0].copy()


def dense(x: Value, w: Value, bias: Value) raises -> Value:
    return x @ w + bias


def forward(p: Params, idx: Value, layers: Int, kernels: String) raises -> Value:
    """`models/gpt.py`'s `forward`: token ids `[B, T]` to logits `[B, T, vocab]`."""
    var x = gather(p["wte"], idx, 0) + p["wpe"]
    for l in range(layers):
        var h = "h" + String(l) + "."
        var a = layer_norm(x, p[h + "ln1.w"], p[h + "ln1.b"], kernels)
        var y = attention(dense(a, p[h + "qkv.w"], p[h + "qkv.b"]), kernels)
        x = x + dense(y, p[h + "proj.w"], p[h + "proj.b"])
        a = layer_norm(x, p[h + "ln2.w"], p[h + "ln2.b"], kernels)
        var f = gelu_tanh(dense(a, p[h + "fc1.w"], p[h + "fc1.b"]))
        x = x + dense(f, p[h + "fc2.w"], p[h + "fc2.b"])
    x = layer_norm(x, p["lnf.w"], p["lnf.b"], kernels)
    return x @ transpose(p["wte"], 0, 1)  # the head, tied to the embedding


def build(
    names: List[String], shapes: List[List[Int]], order: List[Int], layers: Int,
    device: String, kernels: String,
) raises -> Graph:
    """Inputs: the parameters (`gpt.init`'s order), the counter, each one's m
    and v, the token ids and the targets."""
    var n = len(names)
    var types = List[TensorType]()
    for k in range(n):
        types.append(f32(shapes[k], device, buffer=True))
    types.append(f32([1], device, buffer=True))
    for k in range(n):
        types.append(f32(shapes[k], device, buffer=True))
        types.append(f32(shapes[k], device, buffer=True))
    types.append(f32([B, SEQ], device, dtype=DType.int64))
    types.append(f32([B, SEQ], device, dtype=DType.int64))
    var g = Graph("gpt_train_step", types)
    var inputs = g.inputs()
    var buffers = List[Value]()
    var m = List[Value]()
    var v = List[Value]()
    for k in range(n):
        buffers.append(inputs[k].copy())
        m.append(inputs[n + 1 + 2 * k].copy())
        v.append(inputs[n + 2 + 2 * k].copy())

    # Loaded in name order, as the transform loads a dict of buffers.
    var loaded = List[Value]()
    for _ in range(n):
        loaded.append(Value(Python.none(), f32([], device)))
    var primals = List[Value]()
    for k in range(n):
        var value = buffer_load(buffers[order[k]])
        loaded[order[k]] = value.copy()
        primals.append(value^)

    var tape = Tape()
    var p = Params(names.copy(), loaded^)
    var loss = cross_entropy(forward(p, inputs[3 * n + 1], layers, kernels), inputs[3 * n + 2])
    var by_order = tape.gradients(primals, loss)
    var grads = List[Value]()
    for _ in range(n):
        grads.append(Value(Python.none(), f32([], device)))
    for k in range(n):
        grads[order[k]] = by_order[k].copy()
    AdamW(LR, 0.9, 0.999, 1e-8).apply(buffers, m, v, inputs[n], grads)
    g.output([loss^])
    return g^


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var steps = 30
    var layers = 2
    var gpu = False
    var capture = False
    var reference = True
    var extra = 0
    var extra_sync = False
    var i = 2
    while i < len(args):
        if args[i] == "--steps":
            steps = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--layers":
            layers = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--gpu":
            gpu = True
        elif args[i] == "--extra":  # then N more steps and a digest of the buffers
            extra = Int(String(args[i + 1]))
            i += 1
        elif args[i] == "--extra-sync":  # ... each synchronised, not back to back
            extra_sync = True
        elif args[i] == "--no-reference":  # time only: no Python model in the process
            reference = False
        elif args[i] == "--capture":  # CUDA: replay the captured step
            gpu = True
            capture = True
        else:
            raise Error("unknown flag " + String(args[i]))
        i += 1
    var device = String("gpu") if gpu else String("cpu")
    var glue = Python.import_module(GLUE)
    var builtins = Python.import_module("builtins")
    var cfg = builtins.list()
    for d in [VOCAB, SEQ, DIM, HEADS, layers]:
        _ = cfg.append(PythonObject(d))

    var data = glue.gpt_problem(cfg, B, 0)
    var names = List[String]()
    var shapes = List[List[Int]]()
    for k in range(Int(py=builtins.len(data["names"]))):
        names.append(String(data["names"][k]))
        var shape = List[Int]()
        for d in data["shapes"][k]:
            shape.append(Int(py=d))
        shapes.append(shape^)
    var order = List[Int]()
    for k in glue.sorted_order(data["names"]):
        order.append(Int(py=k))
    var n = len(names)

    var t0 = perf_counter_ns()
    var g = build(names, shapes, order, layers, device, String(glue.kernels_path()))
    var build_s = Float64(perf_counter_ns() - t0) / 1e9
    var text = String(glue.graph_text(g.backend.graph))
    var rt = Runtime(accelerator=gpu)
    var model = g.compile(rt, dir + "/gpt_train_step.mef", device)
    print("[mojo] built the", layers, "-layer GPT step (", n, "parameters ) in", build_s,
          "s; compiled in", g.compile_seconds, "s on", device)

    var inputs = rt.tensor_map()
    var staging = rt.tensor_map()
    var on_device = List[Tensor]()
    for k in range(n):
        lend(inputs, staging, on_device, rt, gpu, k, host_copy(data["init"][names[k]]), shapes[k])
    lend(inputs, staging, on_device, rt, gpu, n, HostBuffer(4), [1])
    for k in range(2 * n):
        ref shape = shapes[k // 2]
        lend(inputs, staging, on_device, rt, gpu, n + 1 + k, HostBuffer(4 * numbers(shape)), shape)
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 1, host_copy(data["x"]), [B, SEQ], DType.int64)
    lend(inputs, staging, on_device, rt, gpu, 3 * n + 2, host_copy(data["y"]), [B, SEQ], DType.int64)

    var losses = List[Float32]()
    var times = List[Int]()
    var lent = List[Tensor]()
    var outputs = List[Tensor]()  # rewritten by every replay: alive while replaying
    train(model, inputs, 3 * n + 3, steps, gpu, capture, losses, times, lent, outputs)
    print("[mojo]", steps, "steps", "(captured from step 1)" if capture else "", "; loss", losses[0], "->",
          losses[len(losses) - 1], "; median step", median_us(times, 1), "us, loss copied back each step")

    if extra > 0:
        # Further steps, synchronised (each loss copied back) or back to back,
        # then a digest of every buffer: the two must leave the same state.
        if extra_sync:
            var more = List[Float32]()
            var more_times = List[Int]()
            var more_lent = List[Tensor]()
            var more_outputs = List[Tensor]()
            train(model, inputs, 3 * n + 3, extra, gpu, False, more, more_times, more_lent, more_outputs)
        else:
            _ = pipelined_us(model, inputs, lent, rt, extra, False, keep_outputs=True)
        rt.synchronize()
        var state = builtins.list()
        for k in range(3 * n + 1):
            var shape = shapes[k].copy() if k < n else ([1] if k == n else shapes[(k - n - 1) // 2].copy())
            var py_shape = builtins.list()
            for d in shape:
                _ = py_shape.append(PythonObject(d))
            _ = state.append(read(inputs, on_device, gpu, k, glue, py_shape))
        print("[mojo]", extra, "more steps,", "synchronised" if extra_sync else "back to back",
              "; buffers digest", String(glue.digest(state)))
        _ = outputs^
        _ = on_device^  # MAX reads and writes these through the lent addresses until here
        return
    if not reference:
        if gpu:
            print("[mojo] pipelined, 200 steps, every output map kept until the end:",
                  pipelined_us(model, inputs, lent, rt, 200, False, keep_outputs=True), "us per step")
            print("[mojo] pipelined, 200 steps: executed", pipelined_us(model, inputs, lent, rt, 200, False),
                  "us per step" + (String("; replayed ") + String(pipelined_us(model, inputs, lent, rt, 200, True))
                  + " us per step" if capture else String("")))
        _ = outputs^
        _ = on_device^  # MAX reads and writes these through the lent addresses until here
        return
    var ref_out = glue.gpt_reference(cfg, B, 0, steps, "gpt_train_step", LR, device)
    print("[python] reference compiled in", Float64(py=ref_out["compile_s"]), "s")
    var check_names = List[String]()
    var finals = List[PythonObject]()
    for k in range(3 * n + 1):
        var shape = shapes[k].copy() if k < n else ([1] if k == n else shapes[(k - n - 1) // 2].copy())
        if k < n:
            check_names.append(names[k])
        elif k == n:
            check_names.append("step")
        else:
            check_names.append(("m." if (k - n - 1) % 2 == 0 else "v.") + names[(k - n - 1) // 2])
        var py_shape = builtins.list()
        for d in shape:
            _ = py_shape.append(PythonObject(d))
        finals.append(read(inputs, on_device, gpu, k, glue, py_shape))
    var failures = check(text, losses, finals, check_names, ref_out)
    var kernels = (
        "noeira_layer_norm_fwd" in text and "noeira_layer_norm_bwd" in text
        and "noeira_attention_fwd" in text and "noeira_attention_bwd" in text
    )
    print("the graph calls all four kernels (LayerNorm and attention, forward and backward):", kernels)
    if not kernels:
        failures += 1
    if failures > 0:
        raise Error("FAIL: the Mojo-built GPT step differs from the Python one")
    print("PASS: the GPT step built in Mojo is the Python prototype's, bit for bit")
    if gpu:
        # Further steps, back to back: the step's own cost, with no per-step copy.
        print("[mojo] pipelined, 200 steps: executed", pipelined_us(model, inputs, lent, rt, 200, False),
              "us per step" + (String("; replayed ") + String(pipelined_us(model, inputs, lent, rt, 200, True))
              + " us per step" if capture else String("")))
    _ = outputs^  # replays write into it until here
    _ = on_device^  # MAX reads and writes these through the lent addresses until here
