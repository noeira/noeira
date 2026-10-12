"""Trains with a MAX train step from Mojo, through the MAX C API: no Python
in the process.

`export_step.py` writes the compiled step (`step.mef`), every input's initial
bytes (`step.inputs`) and their layout (`step.manifest`). This program owns
those bytes, lends each input to the runtime without copying
(`M_borrowTensorInto`), and calls `M_executeModelSync` once per step. The
step stores its parameters, optimizer state and seed in place, so after the
loop the step counter is read straight from this program's memory.

    mojo build noeira_max/autodiff/capi/train_step.mojo -o train_step \
        -Xlinker -L$CONDA_PREFIX/lib -Xlinker -lmax -Xlinker -rpath -Xlinker $CONDA_PREFIX/lib
    ./train_step OUT_DIR STEPS        # inside `pixi run` (it sets MODULAR_HOME)

Prints one `loss <step> <value>` line per step, then timings, the step
counter, and what `M_captureModelSync` says on this device.
"""

from std.ffi import c_char, external_call
from std.sys import argv
from std.time import perf_counter_ns

comptime M_HOST: Int32 = 0
comptime CHUNK = 1 << 20


def check(status: Int, what: String) raises:
    if external_call["M_isError", Int32](status) != 0:
        var msg = external_call["M_getError", Pointer[c_char, MutUntrackedOrigin]](
            status
        )
        raise Error(what + " failed: " + String(unsafe_from_utf8_ptr=msg))


def read_bytes(path: String) raises -> List[UInt8]:
    """The whole file, read in chunks (one `read` is capped below 2 GiB)."""
    var out = List[UInt8]()
    with open(path, "r") as f:
        while True:
            var chunk = f.read_bytes(CHUNK)
            if len(chunk) == 0:
                break
            out.extend(chunk^)
    return out^


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: train_step OUT_DIR STEPS")
    var dir = String(args[1])
    var steps = Int(String(args[2]))

    # The manifest: n, the step counter's input index (or -1), then per
    # input: dtype rank dims... offset nbytes.
    var manifest = read_bytes(dir + "/step.manifest")
    var words = manifest.unsafe_ptr().unsafe_bitcast[Int64]()
    var n_inputs = Int(words[unsafe_offset=0])
    var counter_index = Int(words[unsafe_offset=1])
    var blob = read_bytes(dir + "/step.inputs")

    var status = external_call["M_newStatus", Int]()
    var runtime_config = external_call["M_newRuntimeConfig", Int]()
    var host = external_call["M_newDevice", Int](M_HOST, Int32(0), status)
    check(status, "M_newDevice")
    external_call["M_runtimeConfigAddDevice", NoneType](runtime_config, host)
    var ctx = external_call["M_newRuntimeContext", Int](runtime_config, status)
    check(status, "M_newRuntimeContext")

    var compile_config = external_call["M_newCompileConfig", Int]()
    var mef = dir + "/step.mef"
    external_call["M_setModelPath", NoneType](
        compile_config, mef.as_c_string_span().ptr()
    )
    var start = perf_counter_ns()
    var compiled = external_call["M_compileModelSync", Int](
        ctx, Pointer(to=compile_config), status
    )
    check(status, "M_compileModelSync")
    var model = external_call["M_initModel", Int](ctx, compiled, Int(0), status)
    check(status, "M_initModel")
    var load_ms = Float64(perf_counter_ns() - start) / 1e6

    # Lend every input to the runtime: the runtime reads, and the step's
    # stores write, this program's `blob`.
    var inputs = external_call["M_newAsyncTensorMap", Int](ctx)
    var names = List[String]()
    var offsets = List[Int]()
    var at = 2
    for i in range(n_inputs):
        var dtype = Int32(words[unsafe_offset=at])
        var rank = Int(words[unsafe_offset=at + 1])
        var shape = List[Int64]()
        for d in range(rank):
            shape.append(words[unsafe_offset=at + 2 + d])
        var offset = Int(words[unsafe_offset=at + 2 + rank])
        at += 4 + rank
        var name = String("input") + String(i)
        var spec = external_call["M_newTensorSpec", Int](
            shape.unsafe_ptr(), Int64(rank), dtype, name.as_c_string_span().ptr(), host
        )
        external_call["M_borrowTensorInto", NoneType](
            inputs, blob.unsafe_ptr().unsafe_offset(offset), spec, status
        )
        check(status, "M_borrowTensorInto " + name)
        names.append(name)
        offsets.append(offset)

    # Train. The loss is the only output; reading it is the host sync.
    var output_name = String("output0")
    var losses = List[Float32]()
    var times = List[Int]()
    for _ in range(steps):
        var t0 = perf_counter_ns()
        var outputs = external_call["M_executeModelSync", Int](ctx, model, inputs, status)
        check(status, "M_executeModelSync")
        var loss = external_call["M_getTensorByNameFrom", Int](
            outputs, output_name.as_c_string_span().ptr(), status
        )
        check(status, "output0")
        var value = external_call["M_getTensorData", Pointer[Float32, MutUntrackedOrigin]](
            loss
        )
        losses.append(value[unsafe_offset=0])
        times.append(Int(perf_counter_ns() - t0))
        external_call["M_freeTensor", NoneType](loss)
        external_call["M_freeAsyncTensorMap", NoneType](outputs)

    for i in range(len(losses)):
        print("loss", i, losses[i])

    # Median step time (insertion sort: a few hundred values).
    for i in range(1, len(times)):
        var v = times[i]
        var j = i - 1
        while j >= 0 and times[j] > v:
            times[j + 1] = times[j]
            j -= 1
        times[j + 1] = v
    if len(times) > 0:
        print("mojo_median_step_us", Float64(times[len(times) // 2]) / 1000.0)
    print("mef_load_ms", load_ms)

    # In place? Read the optimizer's step counter from THIS program's memory:
    # had the runtime copied the borrowed buffers, it would still be 0.
    if counter_index >= 0:
        var counter = blob.unsafe_ptr().unsafe_offset(offsets[counter_index]).unsafe_bitcast[
            Float32
        ]()[unsafe_offset=0]
        print("step_counter_in_mojo_memory", counter)

    # Device-graph capture through the C API, with the same borrowed inputs.
    var tensors = List[Int]()
    for i in range(n_inputs):
        tensors.append(
            external_call["M_getTensorByNameFrom", Int](
                inputs, names[i].as_c_string_span().ptr(), status
            )
        )
    check(status, "input tensors for capture")
    var capture_status = external_call["M_newStatus", Int]()
    var key = UInt64(1)
    var n_outputs = 0
    _ = external_call["M_captureModelSync", Int](
        ctx, model, Pointer(to=key), Int(1), tensors.unsafe_ptr(), n_inputs,
        Pointer(to=n_outputs), capture_status,
    )
    if external_call["M_isError", Int32](capture_status) != 0:
        var msg = external_call["M_getError", Pointer[c_char, MutUntrackedOrigin]](
            capture_status
        )
        print("capture_error", String(unsafe_from_utf8_ptr=msg))
    else:
        print("capture_ok outputs", n_outputs)
    _ = blob^  # the runtime borrowed it until here
