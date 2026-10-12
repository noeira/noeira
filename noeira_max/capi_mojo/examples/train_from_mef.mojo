"""Trains with an exported MAX train step from Mojo, on `maxrt`.

The workload of the autodiff prototype's Mojo driver
(`noeira_max/autodiff/capi/`): a GPT train step
(forward, backward, AdamW) whose parameters, moments, step counter and seed
are buffers it stores into. Its exporter writes `step.mef`, every input's
initial bytes (`step.inputs`) and their layout (`step.manifest`: int64s
`n`, the counter's input index, then per input `dtype rank dims... offset
nbytes`). There, the driver calls the C API directly; here, the binding.

    mojo build -I noeira_max/capi_mojo noeira_max/capi_mojo/examples/train_from_mef.mojo \
        -o train_from_mef -Xlinker -L$CONDA_PREFIX/lib -Xlinker -lmax
    ./train_from_mef OUT_DIR STEPS                # a step exported for the CPU
    ./train_from_mef OUT_DIR STEPS --gpu          # exported with --device gpu
    ./train_from_mef OUT_DIR STEPS --capture      # the same, captured and replayed

On the host, the map owns each input's bytes and the step updates them in
place. On an accelerator, host memory would be copied in on every call and the
step's stores lost, so each input is copied to the device once and its device
copy is lent by address. `--capture` runs step 0, captures the step, and
replays it for the others.

Prints the same lines as that driver (`loss <step> <value>`, then timings and
the step counter, read back from the program's own buffer).
"""

from std.sys import argv
from std.time import perf_counter_ns

from maxrt import HostBuffer, Runtime, Tensor, from_m_dtype

comptime KEY: UInt64 = 1


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
    var capture = False
    var gpu = False
    for i in range(3, len(args)):
        if args[i] == "--capture":
            capture = True
            gpu = True
        elif args[i] == "--gpu":
            gpu = True
        else:
            raise Error("unknown flag " + String(args[i]))

    var manifest = read_bytes(dir + "/step.manifest")
    var words = manifest.unsafe_ptr().unsafe_bitcast[Int64]()
    var blob = read_bytes(dir + "/step.inputs")

    var start = perf_counter_ns()
    var rt = Runtime(accelerator=gpu)
    var model = rt.load(dir + "/step.mef")
    var load_ms = Float64(perf_counter_ns() - start) / 1e6

    # Every input: a copy of its initial bytes, owned by the map (host), or
    # copied to the device and lent by address (accelerator).
    var inputs = rt.tensor_map()
    var staging = rt.tensor_map()
    var on_device = List[Tensor]()
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
        var name = "input" + String(i)
        var initial = HostBuffer(copy_from=Int(blob.unsafe_ptr()) + offset, nbytes=nbytes)
        if gpu:
            staging.borrow(name, initial^, dtype, shape)
            var copy = staging.tensor(name).to_device(rt.device())
            inputs.borrow_address(name, copy.address(), dtype, shape, on_device=True)
            on_device.append(copy^)
        else:
            inputs.borrow(name, initial^, dtype, shape)
    _ = manifest^  # `words` points into it
    _ = blob^  # copied from until here

    var times = List[Int]()
    var replayed = List[Tensor]()
    var lent = List[Tensor]()
    for step in range(steps):
        var t0 = perf_counter_ns()
        var loss: Float32
        if capture and step > 0:
            if step == 1:  # after one plain execution, as from Python
                for i in range(n):
                    lent.append(inputs.tensor("input" + String(i)))
                replayed = model.capture(KEY, lent)
            model.replay(KEY, lent)
            loss = replayed[0].to_host().item[DType.float32]()
        elif gpu:
            loss = model.execute(inputs).tensor("output0").to_host().item[DType.float32]()
        else:
            loss = model.execute(inputs).tensor("output0").item[DType.float32]()
        if step > 1 or not capture:  # the capture step is not timed
            times.append(Int(perf_counter_ns() - t0))
        print("loss", step, loss)

    sort(times)
    print("mojo_median_step_us", Float64(times[len(times) // 2]) / 1000.0)
    print("mef_load_ms", load_ms)
    if capture:
        print("capture_ok", len(replayed), "outputs; steps 1 to", steps - 1, "replayed")
    if counter >= 0:
        var value: Float32
        if gpu:
            value = on_device[counter].to_host().item[DType.float32]()
        else:
            value = inputs.owned_data[Float32]("input" + String(counter))[unsafe_offset=0]
        print("step_counter_in_mojo_memory", value)
