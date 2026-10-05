"""maxrt: run compiled MAX models (MEF files) from Mojo through the MAX C API,
with no Python in the process.

    from maxrt import Runtime, HostBuffer

    var rt = Runtime()                        # or Runtime(accelerator=True)
    var model = rt.load("graph.mef")          # from Python's `export_mef`
    var inputs = rt.tensor_map()
    inputs.borrow("input0", HostBuffer(4 * 8), DType.float32, [8])
    var outputs = model.execute(inputs)
    var y = outputs.tensor("output0").item[DType.float32]()

Every C object is freed by its Mojo owner, except the runtime context, which
lives until the process exits: freeing it makes MAX 26.6 crash in the Mojo
runtime's exit-time teardown (`_core.RuntimeState`). Models, maps and tensors
share the context, and a tensor keeps the map it views alive. Errors from the
C API raise, with the C API's message.

Build with `mojo build -I noeira_max/capi_mojo ... -Xlinker -lmax`;
`mojo run` cannot resolve the C API's symbols.
"""

from .dtypes import from_m_dtype, m_dtype
from .model import Model
from .runtime import Runtime, accelerator_count, version
from .tensor import HostBuffer, Tensor, TensorMap
