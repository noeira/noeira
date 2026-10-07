"""The gate's plumbing: lending the step's buffers to `maxrt`, reading them
back, and checking a run against the Python prototype's."""

from std.python import Python, PythonObject

from maxrt import HostBuffer, Runtime, Tensor, TensorMap


comptime GLUE = "noeira_max.train_from_mojo.py_grad"


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
    into it are lost)."""
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
