"""A compiled and initialised model: execute, capture, replay."""

from std.ffi import external_call

from . import capi
from ._core import Shared, Status
from .tensor import Tensor, TensorMap, typed


struct Model(Movable):
    """An `M_AsyncModel` and the `M_AsyncCompiledModel` it came from."""

    var _shared: Shared
    var _compiled: capi.Handle
    var _model: capi.Handle

    def __init__(out self, shared: Shared, compiled: capi.Handle, model: capi.Handle):
        self._shared = shared
        self._compiled = compiled
        self._model = model

    def __init__(out self, *, deinit move: Self):
        self._shared = move._shared^
        self._compiled = move._compiled
        self._model = move._model

    def execute(self, inputs: TensorMap) raises -> TensorMap:
        """One synchronous execution. Buffers lent to `inputs` that the model
        stores into are updated in place (see `maxrt.tensor`)."""
        var status = Status()
        var outputs = capi.M_executeModelSync(
            self._shared[].context, self._model, inputs.handle(), status.handle
        )
        status.check("M_executeModelSync")
        return TensorMap(self._shared, outputs)

    def capture(self, key: UInt64, inputs: List[Tensor]) raises -> List[Tensor]:
        """Records one execution as a device graph (CUDA / HIP only). The
        returned outputs are rewritten by every `replay`."""
        var status = Status()
        var keys: List[UInt64] = [key]
        var handles = _handles(inputs)
        var n = 0
        var array = capi.M_captureModelSync(
            self._shared[].context, self._model, keys, handles, n, status.handle
        )
        status.check("M_captureModelSync")
        var outputs = List[Tensor]()
        var slots = typed[Int](array)
        for i in range(n):
            outputs.append(Tensor(self._shared, None, slots[unsafe_offset=i]))
        external_call["free", NoneType](array)
        return outputs^

    def replay(self, key: UInt64, inputs: List[Tensor]) raises:
        """Replays the graph captured under `key`, on the same buffers."""
        var status = Status()
        var keys: List[UInt64] = [key]
        capi.M_replayModelSync(
            self._shared[].context, self._model, keys, _handles(inputs), status.handle
        )
        status.check("M_replayModelSync")

    def __deinit__(deinit self):
        capi.M_freeModel(self._model)
        capi.M_freeCompiledModel(self._compiled)


def _handles(tensors: List[Tensor]) -> List[capi.Handle]:
    var out = List[capi.Handle]()
    for i in range(len(tensors)):  # `for t in` needs Copyable; a Tensor owns a handle
        out.append(tensors[i].handle())
    return out^
