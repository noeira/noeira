"""A MAX graph built, compiled and run from one Mojo program: the RL actor MLP
(17 -> 256 -> 256 -> 6, batch 64), weights as graph inputs.

Setup builds the graph through `max_graph_gen` (Python's `max.graph`
underneath, through Mojo's interop), compiles it and exports the MEF, and
loads it with `maxrt`. The hot loop then calls only the C API: no Python.

Prints the setup timeline (import, build, compile, load, first call), the
per-call latency, and how many copies of `libmax` the process holds: one
linked into this binary for `maxrt`, one loaded by Python's `max` package,
unless they are the same image.

    noeira_max/graph_mojo/run.sh     # builds and runs it after the tests
"""

from std.ffi import c_char, external_call
from std.math import sin
from std.sys import CompilationTarget, argv
from std.time import perf_counter_ns

from max_graph_gen import Dim, Graph, TensorType, relu
from maxrt import HostBuffer, Runtime

comptime BATCH = 64
comptime CALLS = 1000


def seconds_since(start: Int) -> Float64:
    return Float64(perf_counter_ns() - start) / 1e9


def f32(var shape: List[Dim]) -> TensorType:
    return TensorType(DType.float32, shape^)


def fill(n: Int, seed: Float64, scale: Float64) -> HostBuffer:
    var b = HostBuffer(4 * n)
    var p = b.data[Float32]()
    for i in range(n):
        p[unsafe_offset=i] = Float32(sin(Float64(i) * 0.731 + seed) * scale)
    return b^


def loaded_images(fragment: String) raises -> List[String]:
    """The loaded images (shared libraries) whose path contains `fragment`."""
    var out = List[String]()
    comptime if CompilationTarget.is_macos():
        var n = Int(external_call["_dyld_image_count", UInt32]())
        for i in range(n):
            var name = external_call[
                "_dyld_get_image_name", Pointer[c_char, MutUntrackedOrigin], UInt32
            ](UInt32(i))
            var path = String(unsafe_from_utf8_ptr=name)
            if fragment in path:
                out.append(path)
    else:
        var maps: String
        with open("/proc/self/maps", "r") as f:
            maps = f.read()
        for line in maps.split("\n"):
            var path = String(line.split(" ")[len(line.split(" ")) - 1])
            if fragment in path and path not in out:
                out.append(path)
    return out^


def main() raises:
    var dir = String(argv()[1])
    var t0 = perf_counter_ns()
    var rt = Runtime()
    var before = len(loaded_images("libmax"))

    # Setup: build the graph in Mojo, through Python's max.graph.
    var t = perf_counter_ns()
    var g = Graph(
        "actor_mlp",
        [f32([BATCH, 17]), f32([17, 256]), f32([256]), f32([256, 256]), f32([256]),
         f32([256, 6]), f32([6])],
    )
    var p = g.inputs()
    var h = relu(p[0] @ p[1] + p[2])
    h = relu(h @ p[3] + p[4])
    g.output([h @ p[5] + p[6]])
    var build_s = seconds_since(t)
    var after_python = loaded_images("libmax")

    t = perf_counter_ns()
    var model = g.compile(rt, dir + "/actor_mlp.mef")
    var compile_load_s = seconds_since(t)

    # Inputs: owned by the map, reused by every call.
    var inputs = rt.tensor_map()
    var shapes: List[List[Int]] = [[BATCH, 17], [17, 256], [256], [256, 256], [256], [256, 6], [6]]
    for i in range(len(shapes)):
        var n = 1
        for d in shapes[i]:
            n *= d
        inputs.borrow("input" + String(i), fill(n, Float64(i), 0.1 if i > 0 else 1.0), DType.float32, shapes[i])

    t = perf_counter_ns()
    var first = model.execute(inputs).tensor("output0").data[Float32]()[unsafe_offset=0]
    var first_call_s = seconds_since(t)
    var to_first = seconds_since(t0)

    # The hot loop: C API only.
    for _ in range(100):  # warmup
        _ = model.execute(inputs)
    var samples = List[Int]()
    for _ in range(CALLS):
        var s = perf_counter_ns()
        _ = model.execute(inputs)
        samples.append(Int(perf_counter_ns() - s))
    sort(samples)

    print("libmax images: before Python", before, "| after importing max.graph", len(after_python))
    for i in range(len(after_python)):
        print("   ", after_python[i])
    print("setup: build", build_s, "s, compile + export + load", compile_load_s,
          "s (Python compile", g.compile_seconds, "s), first call", first_call_s, "s")
    print("process start to first inference:", to_first, "s; output[0] =", first)
    print("hot loop (C API, no Python): median", Float64(samples[CALLS // 2]) / 1000.0,
          "us per call over", CALLS, "calls")
