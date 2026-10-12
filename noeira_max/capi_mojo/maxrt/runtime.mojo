"""The entry point: a runtime on the host or an accelerator, which loads
models and makes tensor maps."""

from std.memory import ArcPointer

from . import capi
from ._core import RuntimeState, Shared, Status
from .config import ensure_modular_home
from .model import Model
from .tensor import TensorMap


def accelerator_count() -> Int:
    return capi.M_getAcceleratorCount()


def version() -> String:
    return capi.M_version()


struct Runtime(Copyable, Movable):
    """The MAX runtime context and its devices. Copies share them."""

    var _shared: Shared

    def __init__(out self, accelerator: Bool = False, accelerator_id: Int = 0) raises:
        """A runtime on the host, plus one accelerator if `accelerator`. Sets
        `MODULAR_HOME` first if the environment did not (`maxrt.config`)."""
        ensure_modular_home()
        var status = Status()
        var config = capi.M_newRuntimeConfig()
        var host = capi.M_newDevice(capi.M_HOST, 0, status.handle)
        status.check("M_newDevice(host)")
        capi.M_runtimeConfigAddDevice(config, host)
        var device = host
        if accelerator:
            device = capi.M_newDevice(
                capi.M_ACCELERATOR, Int32(accelerator_id), status.handle
            )
            status.check("M_newDevice(accelerator " + String(accelerator_id) + ")")
            capi.M_runtimeConfigAddDevice(config, device)
        var context = capi.M_newRuntimeContext(config, status.handle)
        status.check("M_newRuntimeContext")
        self._shared = ArcPointer(RuntimeState(config, host, device, context))

    def __init__(out self, *, copy: Self):
        self._shared = copy._shared

    def __init__(out self, *, deinit move: Self):
        self._shared = move._shared^

    def load(self, path: String) raises -> Model:
        """Loads a MEF (from `export_mef`): no compilation, only the load."""
        var status = Status()
        var config = capi.M_newCompileConfig()
        capi.M_setModelPath(config, path)
        var compiled = capi.M_compileModelSync(self._shared[].context, config, status.handle)
        status.check("loading " + path)
        var model = capi.M_initModel(self._shared[].context, compiled, 0, status.handle)
        status.check("M_initModel " + path)
        return Model(self._shared, compiled, model)

    def tensor_map(self) -> TensorMap:
        return TensorMap(self._shared, capi.M_newAsyncTensorMap(self._shared[].context))

    def device(self) -> capi.Handle:
        """The execution device: the accelerator, or the host."""
        return self._shared[].device

    def host(self) -> capi.Handle:
        return self._shared[].host

    def device_label(self) -> String:
        return capi.M_getDeviceLabel(self._shared[].device)

    def synchronize(self) raises:
        var status = Status()
        capi.M_synchronizeDevice(self._shared[].device, status.handle)
        status.check("M_synchronizeDevice")
