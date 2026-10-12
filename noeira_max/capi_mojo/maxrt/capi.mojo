"""The MAX C API (`include/max/c/*.h`, MAX 26.6) as typed Mojo functions.

One `def` per C function, grouped and ordered as the headers are. This file
is the only place `maxrt` calls `external_call`.

Conventions:

- An opaque handle (`M_Status*`, `M_RuntimeContext*`, ...) is an `Int`: a C
  pointer and an `Int` are the same register, and Mojo's safe `Pointer` has
  no null, which several functions return on failure.
- A data address crossing the boundary (`void *input`,
  `const void *M_getTensorData`) is an `Int` too; `maxrt.tensor` turns it
  into a typed `Pointer` where Mojo reads it.
- A C string argument is a `String` taken by value (`as_c_string_span`
  appends the NUL, so it needs a mutable copy); a C string result is a
  `String`, copied at once.
- `M_Foo **` out-parameters and arrays are handled inside the wrapper.

Bound: 50 of the 70 exported functions. Not bound: the safetensors reader
(10), `M_newWeightsRegistry` / `M_newWeightsRegistryFromSafetensors` (the
binding passes parameters as inputs, never as weights), the async variants
(`M_compileModel`, `M_waitForCompilation`, `M_waitForModel`), the four debug
setters of `context.h`, and `M_freeTensorNameArray`. Nothing returns a name
array: `types.h` documents `M_getInputNames()` and `M_getOutputNames()`, but
`libmax` does not export them, so a model's input names cannot be queried,
only assumed (`input0`, `input1`, ...).
"""

from std.ffi import c_char, external_call

comptime Handle = Int
"""An opaque C pointer (`M_Status*`, `M_AsyncTensor*`, ...)."""
comptime CString = Pointer[c_char, MutUntrackedOrigin]


def to_string(c: CString) -> String:
    """A copy of a NUL-terminated C string."""
    return String(unsafe_from_utf8_ptr=c)


# ── common.h ────────────────────────────────────────────────────────────


def M_version() -> String:
    return to_string(external_call["M_version", CString]())


def M_newStatus() -> Handle:
    return external_call["M_newStatus", Handle]()


def M_getError(status: Handle) -> String:
    return to_string(external_call["M_getError", CString](status))


def M_isError(status: Handle) -> Bool:
    return external_call["M_isError", Int32](status) != 0


def M_freeStatus(status: Handle):
    external_call["M_freeStatus", NoneType](status)


# ── context.h ───────────────────────────────────────────────────────────


def M_newRuntimeConfig() -> Handle:
    return external_call["M_newRuntimeConfig", Handle]()


def M_freeRuntimeConfig(config: Handle):
    external_call["M_freeRuntimeConfig", NoneType](config)


def M_runtimeConfigAddDevice(config: Handle, device: Handle):
    external_call["M_runtimeConfigAddDevice", NoneType](config, device)


def M_newRuntimeContext(config: Handle, status: Handle) -> Handle:
    return external_call["M_newRuntimeContext", Handle](config, status)


def M_freeRuntimeContext(context: Handle):
    external_call["M_freeRuntimeContext", NoneType](context)


# ── device.h ────────────────────────────────────────────────────────────

comptime M_HOST: Int32 = 0
comptime M_ACCELERATOR: Int32 = 1


def M_newDevice(device_type: Int32, id: Int32, status: Handle) -> Handle:
    return external_call["M_newDevice", Handle](device_type, id, status)


def M_getDeviceType(device: Handle) -> Int32:
    return external_call["M_getDeviceType", Int32](device)


def M_getDeviceId(device: Handle) -> Int32:
    return external_call["M_getDeviceId", Int32](device)


def M_isHostDevice(device: Handle) -> Bool:
    return external_call["M_isHostDevice", Int32](device) != 0


def M_synchronizeDevice(device: Handle, status: Handle):
    external_call["M_synchronizeDevice", NoneType](device, status)


def M_getDeviceLabel(device: Handle) -> String:
    return to_string(external_call["M_getDeviceLabel", CString](device))


def M_freeDevice(device: Handle):
    external_call["M_freeDevice", NoneType](device)


def M_getAcceleratorCount() -> Int:
    return Int(external_call["M_getAcceleratorCount", Int32]())


# ── model.h ─────────────────────────────────────────────────────────────


def M_newCompileConfig() -> Handle:
    return external_call["M_newCompileConfig", Handle]()


def M_setModelPath(config: Handle, var path: String):
    external_call["M_setModelPath", NoneType](config, path.as_c_string_span().ptr())


def M_compileModelSync(context: Handle, var config: Handle, status: Handle) -> Handle:
    """Takes ownership of `config` (the C API frees it and nulls the handle)."""
    return external_call["M_compileModelSync", Handle](
        context, Pointer(to=config), status
    )


def M_initModel(
    context: Handle, compiled: Handle, weights: Handle, status: Handle
) -> Handle:
    return external_call["M_initModel", Handle](context, compiled, weights, status)


def M_executeModelSync(
    context: Handle, model: Handle, inputs: Handle, status: Handle
) -> Handle:
    return external_call["M_executeModelSync", Handle](context, model, inputs, status)


def M_captureModelSync(
    context: Handle,
    model: Handle,
    keys: List[UInt64],
    inputs: List[Handle],
    mut num_outputs: Int,
    status: Handle,
) -> Handle:
    """Returns the address of a `malloc`ed array of `num_outputs` output
    tensors (each freed with `M_freeTensor`, the array with `free`), or 0."""
    return external_call["M_captureModelSync", Handle](
        context, model, keys.unsafe_ptr(), len(keys), inputs.unsafe_ptr(),
        len(inputs), Pointer(to=num_outputs), status,
    )


def M_replayModelSync(
    context: Handle, model: Handle, keys: List[UInt64], inputs: List[Handle], status: Handle
):
    external_call["M_replayModelSync", NoneType](
        context, model, keys.unsafe_ptr(), len(keys), inputs.unsafe_ptr(),
        len(inputs), status,
    )


def M_debugVerifyReplayModelSync(
    context: Handle, model: Handle, keys: List[UInt64], inputs: List[Handle], status: Handle
):
    external_call["M_debugVerifyReplayModelSync", NoneType](
        context, model, keys.unsafe_ptr(), len(keys), inputs.unsafe_ptr(),
        len(inputs), status,
    )


def M_releaseCapturedGraphs(
    context: Handle, model: Handle, keys: List[UInt64], status: Handle
):
    external_call["M_releaseCapturedGraphs", NoneType](
        context, model, keys.unsafe_ptr(), len(keys), status
    )


def M_freeModel(model: Handle):
    external_call["M_freeModel", NoneType](model)


def M_freeCompiledModel(compiled: Handle):
    external_call["M_freeCompiledModel", NoneType](compiled)


def M_freeCompileConfig(config: Handle):
    external_call["M_freeCompileConfig", NoneType](config)


# ── tensor.h ────────────────────────────────────────────────────────────


def M_newTensorSpec(
    shape: List[Int64], dtype: Int32, var name: String, device: Handle
) -> Handle:
    return external_call["M_newTensorSpec", Handle](
        shape.unsafe_ptr(), Int64(len(shape)), dtype,
        name.as_c_string_span().ptr(), device,
    )


def M_isDynamicRanked(spec: Handle) -> Bool:
    return external_call["M_isDynamicRanked", Int32](spec) != 0


def M_getDimAt(spec: Handle, axis: Int) -> Int:
    return Int(external_call["M_getDimAt", Int64](spec, axis))


def M_getRank(spec: Handle) -> Int:
    return Int(external_call["M_getRank", Int64](spec))


def M_getDtype(spec: Handle) -> Int32:
    return external_call["M_getDtype", Int32](spec)


def M_getName(spec: Handle) -> String:
    return to_string(external_call["M_getName", CString](spec))


def M_newAsyncTensorMap(context: Handle) -> Handle:
    return external_call["M_newAsyncTensorMap", Handle](context)


def M_borrowTensorInto(tensors: Handle, address: Int, spec: Handle, status: Handle):
    external_call["M_borrowTensorInto", NoneType](tensors, address, spec, status)


def M_getTensorByNameFrom(tensors: Handle, var name: String, status: Handle) -> Handle:
    return external_call["M_getTensorByNameFrom", Handle](
        tensors, name.as_c_string_span().ptr(), status
    )


def M_getTensorNumElements(tensor: Handle) -> Int:
    return external_call["M_getTensorNumElements", Int](tensor)


def M_getTensorType(tensor: Handle) -> Int32:
    return external_call["M_getTensorType", Int32](tensor)


def M_getTensorData(tensor: Handle) -> Int:
    return external_call["M_getTensorData", Int](tensor)


def M_getTensorSpec(tensor: Handle) -> Handle:
    return external_call["M_getTensorSpec", Handle](tensor)


def M_getDeviceTypeFromSpec(spec: Handle) -> Int32:
    return external_call["M_getDeviceTypeFromSpec", Int32](spec)


def M_getDeviceIdFromSpec(spec: Handle) -> Int32:
    return external_call["M_getDeviceIdFromSpec", Int32](spec)


def M_getTensorDevice(tensor: Handle) -> Handle:
    return external_call["M_getTensorDevice", Handle](tensor)


def M_copyTensorToDevice(tensor: Handle, device: Handle, status: Handle) -> Handle:
    return external_call["M_copyTensorToDevice", Handle](tensor, device, status)


def M_freeTensor(tensor: Handle):
    external_call["M_freeTensor", NoneType](tensor)


def M_freeTensorSpec(spec: Handle):
    external_call["M_freeTensorSpec", NoneType](spec)


def M_freeAsyncTensorMap(tensors: Handle):
    external_call["M_freeAsyncTensorMap", NoneType](tensors)
