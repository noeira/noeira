"""The gate's plumbing: lending the step's buffers to `maxrt`, reading them
back, and checking a run against the Python prototype's."""

from std.python import Python, PythonObject

from std.time import perf_counter_ns

from maxrt import HostBuffer, Model, Runtime, Tensor, TensorMap


comptime GLUE = "noeira_max.train_from_mojo.py_grad"
comptime CAPTURE_KEY: UInt64 = 1


def host_copy(array: PythonObject) raises -> HostBuffer:
    """A copy of a NumPy array's bytes. The array is this call's argument, so
    it is alive until the copy is done: `HostBuffer(copy_from=address(a),
    nbytes=a.nbytes)` would release `a` at its last use, inside the argument
    list, before the copy runs (its bytes then read as whatever the
    allocator left)."""
    var glue = Python.import_module(GLUE)
    return HostBuffer(copy_from=Int(py=glue.address(array)), nbytes=Int(py=array.nbytes))


def lend(
    mut inputs: TensorMap, mut staging: TensorMap, mut on_device: List[Tensor],
    rt: Runtime, gpu: Bool, index: Int, var buffer: HostBuffer, shape: List[Int],
    dtype: DType = DType.float32,
) raises:
    """Lends `buffer` as input `index`: owned by the map on the host; on an
    accelerator, copied to the device once and lent by address (host memory
    lent under an accelerator is staged on every call, and the step's stores
    into it are lost).

    The map holds only the device copy's address: the caller keeps
    `on_device` alive for as long as the model executes. Freed at its last
    use, the copies went back to MAX's memory pool while the step still
    wrote through their addresses; back to back, the pool handed them out
    again and the step faulted (CUDA_ERROR_ILLEGAL_ADDRESS, reported by
    whichever kernel ran next)."""
    var name = "input" + String(index)
    if gpu:
        staging.borrow(name, buffer^, dtype, shape)
        var copy = staging.tensor(name).to_device(rt.device())
        inputs.borrow_address(name, copy.address(), dtype, shape, on_device=True)
        on_device.append(copy^)
    else:
        inputs.borrow(name, buffer^, dtype, shape)


def read(
    inputs: TensorMap, on_device: List[Tensor], gpu: Bool, index: Int,
    glue: PythonObject, py_shape: PythonObject, first_lent: Int = 0,
) raises -> PythonObject:
    """Input `index` as the step left it, copied into a NumPy array.
    `first_lent`: the first input `lend` lent (`on_device[0]`)."""
    if gpu:
        var host = on_device[index - first_lent].to_host()
        var copied = glue.array_at(Int(host.data[Float32]()), py_shape)
        _ = host^  # alive until the copy is done, not just until `data()`
        return copied
    return glue.array_at(Int(inputs.owned_data[Float32]("input" + String(index))), py_shape)


def check(
    text: String, losses: List[Float32], finals: List[PythonObject], names: List[String],
    reference: PythonObject,
) raises -> Int:
    """Prints how a run compares with a `py_grad` reference run, and returns
    the number of failed checks: the graph text, every loss, and the final
    buffers (`finals`, named by `names`: the parameters, `step`, then each
    parameter's `m.` and `v.`)."""
    var glue = Python.import_module(GLUE)
    var failures = 0
    var same_text = String(reference["text"]) == text
    print("graph text:", "identical" if same_text else "DIFFERS")
    if not same_text:
        failures += 1
        var difflib = Python.import_module("difflib")
        var lines = difflib.unified_diff(
            reference["text"].splitlines(), PythonObject(text).splitlines(), "python", "mojo", n=0
        )
        var shown = 0
        for line in lines:
            if shown < 40:
                print("   ", String(line))
            shown += 1
    var steps = len(losses)
    var loss_diff = 0
    for k in range(steps):
        if Float32(Float64(py=reference["losses"][k])) != losses[k]:
            loss_diff += 1
    print("losses:", steps - loss_diff, "of", steps, "bit-identical")
    if loss_diff > 0:
        failures += 1

    var worst = 0.0
    var differ = 0
    for k in range(len(names)):
        ref key = names[k]
        var want: PythonObject
        if key.startswith("m.") or key.startswith("v.") or key == "step":
            want = reference["state"][key]
        else:
            want = reference["params"][key]
        var result = glue.compare(PythonObject(key), finals[k], want)
        differ += Int(py=result[0])
        worst = max(worst, Float64(py=result[1]))
    print("final parameters, moments, counter:", differ, "elements differ; max relative difference", worst)
    if differ > 0:
        failures += 1
    return failures


def train(
    model: Model, inputs: TensorMap, count: Int, steps: Int, gpu: Bool, capture: Bool,
    mut losses: List[Float32], mut times: List[Int], mut lent: List[Tensor],
    mut outputs: List[Tensor],
) raises:
    """Runs `steps` steps, appending each loss and each step's time (ns; the
    loss is copied back every step, so each step is synchronised).

    With `capture` (CUDA), step 0 executes, then the step is captured as a
    device graph over the `count` inputs (left in `lent`) and every later
    step replays it: the same kernels on the same buffers, with no per-call
    host work in MAX's executor.

    Every replay writes the captured outputs (left in `outputs`), so the
    caller keeps them alive for as long as it replays. Freed when this
    function returned, the next replays wrote freed device memory
    (CUDA_ERROR_ILLEGAL_ADDRESS, reported by a later kernel)."""
    for k in range(steps):
        var t = perf_counter_ns()
        var loss: Float32
        if capture and k > 0:
            if k == 1:
                for i in range(count):
                    lent.append(inputs.tensor("input" + String(i)))
                outputs = model.capture(CAPTURE_KEY, lent)
                t = perf_counter_ns()  # the capture itself is not a step
            model.replay(CAPTURE_KEY, lent)
            loss = outputs[0].to_host().item[DType.float32]()
        else:
            var out = model.execute(inputs).tensor("output0")
            loss = out.to_host().item[DType.float32]() if gpu else out.item[DType.float32]()
        times.append(Int(perf_counter_ns() - t))
        losses.append(loss)


def pipelined_us(
    model: Model, inputs: TensorMap, lent: List[Tensor], rt: Runtime, iters: Int, replay: Bool,
    keep_outputs: Bool = False,
) raises -> Float64:
    """Microseconds per step for `iters` steps back to back, one
    synchronisation at the end: executed, or replayed (after `train` with
    `capture`). These are further training steps."""
    rt.synchronize()
    var keep = List[TensorMap]()  # each call's outputs, alive until the synchronisation
    var t = perf_counter_ns()
    for _ in range(iters):
        if replay:
            model.replay(CAPTURE_KEY, lent)
        elif keep_outputs:
            keep.append(model.execute(inputs))
        else:
            _ = model.execute(inputs)
    rt.synchronize()
    var us = Float64(perf_counter_ns() - t) / Float64(iters) / 1000.0
    _ = keep^
    return us


def median_us(times: List[Int], skip: Int = 0) -> Float64:
    """The median of `times` (ns) after the first `skip`, in microseconds."""
    var rest = List[Int]()
    for i in range(skip, len(times)):
        rest.append(times[i])
    sort(rest)
    return Float64(rest[len(rest) // 2]) / 1000.0
