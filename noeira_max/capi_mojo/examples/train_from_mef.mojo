"""Trains with an exported MAX train step from Mojo, on `maxrt`.

The workload of `PROTOTYPE_MAX_AUTODIFF_PLAN.md` M3 (branch
`proto/max-autodiff`, `noeira_max/autodiff/capi/`): a GPT train step
(forward, backward, AdamW) whose parameters, moments, step counter and seed
are buffers it stores into. Its exporter writes `step.mef`, every input's
initial bytes (`step.inputs`) and their layout (`step.manifest`: int64s
`n`, the counter's input index, then per input `dtype rank dims... offset
nbytes`). There, the driver calls the C API directly; here, the binding.

    mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/examples/train_from_mef.mojo \
        -o train_from_mef -Xlinker -L$CONDA_PREFIX/lib -Xlinker -lmax
    ./train_from_mef OUT_DIR STEPS

Prints the same lines as that driver (`loss <step> <value>`, then timings and
the step counter read from this program's memory).
"""

from std.sys import argv
from std.time import perf_counter_ns

from maxrt import HostBuffer, Runtime, from_m_dtype


def read_bytes(path: String) raises -> List[UInt8]:
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(1 << 20)
            if len(chunk) == 0:
                break
            out.extend(chunk^)
    return out^


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var steps = Int(String(args[2]))

    var manifest = read_bytes(dir + "/step.manifest")
    var words = manifest.unsafe_ptr().unsafe_bitcast[Int64]()
    var blob = read_bytes(dir + "/step.inputs")

    var start = perf_counter_ns()
    var rt = Runtime()
    var model = rt.load(dir + "/step.mef")
    var load_ms = Float64(perf_counter_ns() - start) / 1e6

    # Every input: a copy of its initial bytes, owned by the map from now on.
    var inputs = rt.tensor_map()
    var n = Int(words[unsafe_offset=0])
    var counter = Int(words[unsafe_offset=1])
    var at = 2
    for i in range(n):
        var dtype = from_m_dtype(Int32(words[unsafe_offset=at]))
        var rank = Int(words[unsafe_offset=at + 1])
        var shape = List[Int]()
        for d in range(rank):
            shape.append(Int(words[unsafe_offset=at + 2 + d]))
        var offset = Int(words[unsafe_offset=at + 2 + rank])
        var nbytes = Int(words[unsafe_offset=at + 3 + rank])
        at += 4 + rank
        var initial = HostBuffer(copy_from=Int(blob.unsafe_ptr()) + offset, nbytes=nbytes)
        inputs.borrow("input" + String(i), initial^, dtype, shape)

    var times = List[Int]()
    for step in range(steps):
        var t0 = perf_counter_ns()
        var loss = model.execute(inputs).tensor("output0").item[DType.float32]()
        times.append(Int(perf_counter_ns() - t0))
        print("loss", step, loss)

    sort(times)
    print("mojo_median_step_us", Float64(times[len(times) // 2]) / 1000.0)
    print("mef_load_ms", load_ms)
    if counter >= 0:
        var value = inputs.owned_data[Float32]("input" + String(counter))[unsafe_offset=0]
        print("step_counter_in_mojo_memory", value)
