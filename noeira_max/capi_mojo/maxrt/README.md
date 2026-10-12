# maxrt — the MAX C API from Mojo

Runs compiled MAX models (MEF files, from Python's `export_mef`) from a Mojo
program, with no Python in the process: load, lend host or device buffers,
execute, capture and replay.

```mojo
from maxrt import HostBuffer, Runtime

var rt = Runtime()                     # Runtime(accelerator=True) for a GPU
var model = rt.load("step.mef")        # loads; compiling happened in Python
var inputs = rt.tensor_map()
inputs.borrow("input0", HostBuffer(4 * 8), DType.float32, [8])
var loss = model.execute(inputs).tensor("output0").item[DType.float32]()
```

| File | What |
|---|---|
| `capi.mojo` | 50 of the C API's 70 functions as typed `def`s, in the headers' order; the only `external_call`s |
| `_core.mojo` | `Status` (C errors raised as Mojo `Error`s) and the shared runtime state |
| `runtime.mojo` | `Runtime`: host or accelerator, `load`, `tensor_map`, `synchronize` |
| `model.mojo` | `Model`: `execute`, `capture`, `replay` |
| `tensor.mojo` | `HostBuffer`, `TensorMap` (`borrow`, `borrow_address`, `owned_data`), `Tensor` (`item`, `data`, `to_device`, `to_host`) |
| `config.mojo` | Without `MODULAR_HOME`, locates `libmax` and writes its own `modular.cfg` |
| `dtypes.mojo` | Mojo `DType` <-> the C API's `M_Dtype` |

**Lifetimes.** Mojo frees a value at its last use, so every pointer `maxrt`
returns carries its owner's origin, and a tensor keeps the map it views
alive. Memory lent by address (`borrow_address`) is the caller's to keep
alive. The runtime context is never freed: in MAX 26.6, freeing it crashes
the process at exit (`_core.RuntimeState`).

**In-place updates.** A buffer the model stores into is updated in the
lender's memory when lent under the host device. Under an accelerator, host
memory is staged on every call and the model's writes are lost; device memory
must be lent instead (`maxrt_tests/test_maxrt.mojo`, `test_accelerator`):

- **CUDA (MAX 26.6):** device memory lent by address is read and updated in
  place, whether MAX allocated it (`Tensor.to_device`) or a Mojo
  `DeviceContext` did (`DeviceBuffer.unsafe_ptr()`): the two share the CUDA
  context and allocator (`maxrt_tests/probe_cuda_context.mojo`). They use
  different streams, so synchronise (`DeviceContext.synchronize()`,
  `Runtime.synchronize()`) at each hand-off. Capture and replay work.
- **Metal (MAX 26.6):** a device tensor's address lent back is read as zeros,
  and capture is refused.

Build with `mojo build -I noeira_max/capi_mojo … -Xlinker -lmax` (`mojo run`
cannot resolve the C API's symbols). Tests: `maxrt_tests/run.sh` (inside pixi,
then outside it with `MODULAR_HOME` unset). A full training workload, on the
host or the GPU, with or without capture: `examples/train_from_mef.mojo`.
Benchmarks against Python and noeira's nn: `../bench/`.
