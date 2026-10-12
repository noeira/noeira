"""What every `maxrt` object shares: the runtime's C objects, and statuses.

Mojo destroys a value at its last use, not at the end of its scope. A model,
a tensor map or a tensor therefore holds an `ArcPointer` to the runtime's
C objects, so the context outlives every object made from it, whatever order
the program stops using them in. (In MAX 26.6 the context itself is kept until
the process exits: see `RuntimeState`.)
"""

from std.memory import ArcPointer

from . import capi


struct Status(Movable):
    """An `M_Status`, freed at its last use."""

    var handle: capi.Handle

    def __init__(out self):
        self.handle = capi.M_newStatus()

    def __init__(out self, *, deinit move: Self):
        self.handle = move.handle

    def check(self, what: String) raises:
        """Raises `what: <the C API's message>` if the call failed."""
        if capi.M_isError(self.handle):
            raise Error(what + ": " + capi.M_getError(self.handle))

    def __deinit__(deinit self):
        capi.M_freeStatus(self.handle)


struct RuntimeState(Movable):
    """The runtime config, its devices and the context.

    Deliberately NOT freed. In MAX 26.6, a Mojo program that calls
    `M_freeRuntimeContext` after running a multi-threaded CPU model crashes at
    exit: SIGSEGV in the Mojo runtime's own teardown
    (`KGEN_CompilerRT_DestroyGlobals` -> AsyncRT). Measured 2/2 runs on the
    2-layer GPT train step, against 2/2 clean exits with the context leaked
    (`examples/train_from_mef.mojo`). The program and MAX share
    `libKGENCompilerRTShared`, and the two teardowns collide. A runtime is made
    once per process, so it lives until exit; models, maps, tensors, specs and
    statuses are still freed by their owners.
    """

    var config: capi.Handle
    var host: capi.Handle
    var device: capi.Handle
    """The execution device: `host`, or an accelerator."""
    var context: capi.Handle

    def __init__(
        out self,
        config: capi.Handle,
        host: capi.Handle,
        device: capi.Handle,
        context: capi.Handle,
    ):
        self.config = config
        self.host = host
        self.device = device
        self.context = context

    def __init__(out self, *, deinit move: Self):
        self.config = move.config
        self.host = move.host
        self.device = move.device
        self.context = move.context

    def __deinit__(deinit self):
        pass  # see the struct's docstring: freeing the context crashes at exit


comptime Shared = ArcPointer[RuntimeState]
