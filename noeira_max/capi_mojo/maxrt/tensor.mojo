"""Host buffers, tensor maps and tensors.

A `TensorMap` lends memory to the runtime (`M_borrowTensorInto`): either a
`HostBuffer` it then owns, or a caller-owned address (`borrow_address`), for
device memory.

**Lifetimes.** Mojo destroys a value at its last use, not at the end of its
scope, and a C handle carries no lifetime. So:

- every pointer `maxrt` hands out is typed with its owner's origin
  (`Pointer[T, origin_of(self)]`), and the compiler keeps the owner alive
  while the pointer is used;
- a `Tensor` taken from a map keeps that map alive: the C API lends a map's
  tensors (`M_getTensorByNameFrom`), it does not copy them;
- memory lent by address (`borrow_address`) is the caller's to keep alive
  until the last `execute` that uses it.

With untracked pointers instead, a map freed at its last use takes its
outputs and buffers with it, and the reads return stale values, silently.

**In-place writes.** Lent under the HOST device, the model writes the
lender's memory. Lent under an accelerator, the memory must already be on
that device to be borrowed in place; a host address is copied to the device
on every call, and the model's writes never come back (`tensor.h`,
`M_borrowTensorInto`).
"""

from std.ffi import external_call
from std.memory import ArcPointer

from . import capi
from ._core import Shared, Status
from .dtypes import m_dtype

comptime ALIGNMENT = 64


def typed[T: AnyType](address: Int) -> Pointer[T, MutUntrackedOrigin]:
    """The C address `address` as a typed, untracked pointer.

    Mojo 1.1's `Pointer` has no constructor from an address, so this asks
    libc: `memmove(p, p, 0)` returns `p`, now typed. Public accessors rebind
    the result to their owner's origin.
    """
    return external_call[
        "memmove", Pointer[T, MutUntrackedOrigin], Int, Int, Int
    ](address, address, 0)


def _aligned_zeros(nbytes: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    var rounded = max(ALIGNMENT, (nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT)
    var ptr = external_call[
        "aligned_alloc", Pointer[UInt8, MutUntrackedOrigin], Int, Int
    ](ALIGNMENT, rounded)
    _ = external_call["memset", Int](Int(ptr), 0, rounded)
    return ptr


struct HostBuffer(Movable):
    """Host memory, aligned to 64 bytes and zero-initialised."""

    var _ptr: Pointer[UInt8, MutUntrackedOrigin]
    var nbytes: Int

    def __init__(out self, nbytes: Int):
        self._ptr = _aligned_zeros(nbytes)
        self.nbytes = nbytes

    def __init__(out self, *, copy_from: Int, nbytes: Int):
        """A copy of the `nbytes` bytes at address `copy_from`."""
        self._ptr = _aligned_zeros(nbytes)
        self.nbytes = nbytes
        _ = external_call["memcpy", Int](Int(self._ptr), copy_from, nbytes)

    def __init__(out self, *, deinit move: Self):
        self._ptr = move._ptr
        self.nbytes = move.nbytes

    def address(self) -> Int:
        """The raw address: whoever lends it keeps this buffer alive."""
        return Int(self._ptr)

    def data[T: AnyType](ref self) -> Pointer[T, origin_of(self)]:
        return rebind[Pointer[T, origin_of(self)]](self._ptr.unsafe_bitcast[T]())

    def __deinit__(deinit self):
        # One C signature per symbol in a binary: `free` always takes an Int.
        external_call["free", NoneType](Int(self._ptr))


struct _MapState(Movable):
    """An `M_AsyncTensorMap` and the host buffers lent to it, freed together
    (the map first)."""

    var map: capi.Handle
    var owned: List[HostBuffer]
    var names: List[String]

    def __init__(out self, map: capi.Handle):
        self.map = map
        self.owned = List[HostBuffer]()
        self.names = List[String]()

    def __init__(out self, *, deinit move: Self):
        self.map = move.map
        self.owned = move.owned^
        self.names = move.names^

    def __deinit__(deinit self):
        capi.M_freeAsyncTensorMap(self.map)


struct Tensor(Movable):
    """An `M_AsyncTensor`: a model output, a lent input, or a device copy."""

    var _shared: Shared
    var _source: Optional[ArcPointer[_MapState]]
    """The map this tensor is lent from, kept alive with it."""
    var _t: capi.Handle

    def __init__(
        out self,
        shared: Shared,
        source: Optional[ArcPointer[_MapState]],
        handle: capi.Handle,
    ):
        self._shared = shared
        self._source = source
        self._t = handle

    def __init__(out self, *, deinit move: Self):
        self._shared = move._shared^
        self._source = move._source^
        self._t = move._t

    def handle(self) -> capi.Handle:
        return self._t

    def num_elements(self) -> Int:
        return capi.M_getTensorNumElements(self._t)

    def dtype_code(self) -> Int32:
        return capi.M_getTensorType(self._t)

    def address(self) -> Int:
        """Where the data is (host or device memory, per `on_host`). Whoever
        lends this address keeps this tensor alive."""
        return capi.M_getTensorData(self._t)

    def on_host(self) -> Bool:
        var device = capi.M_getTensorDevice(self._t)  # ours to free (tensor.h)
        var host = capi.M_isHostDevice(device)
        capi.M_freeDevice(device)
        return host

    def data[T: AnyType](ref self) raises -> Pointer[T, origin_of(self)]:
        if not self.on_host():
            raise Error("maxrt: tensor is on a device; copy it with to_host() first")
        return rebind[Pointer[T, origin_of(self)]](typed[T](self.address()))

    def item[dtype: DType](self) raises -> Scalar[dtype]:
        """The first element (a loss, for example), copied out."""
        return self.data[Scalar[dtype]]()[unsafe_offset=0]

    def to_device(self, device: capi.Handle) raises -> Tensor:
        """A copy on `device`, which owns its memory, complete on return.

        `M_copyTensorToDevice` returns before the copy is done: on the host
        (MAX 26.6, M1), a read right after it saw stale values in up to 20% of
        the elements, and none after `M_synchronizeDevice`. So both devices
        are synchronised here.
        """
        var status = Status()
        var copy = capi.M_copyTensorToDevice(self._t, device, status.handle)
        status.check("M_copyTensorToDevice")
        var source = capi.M_getTensorDevice(self._t)
        var synced = Status()
        capi.M_synchronizeDevice(source, synced.handle)
        capi.M_freeDevice(source)
        synced.check("M_synchronizeDevice (the copy's source)")
        var done = Status()
        capi.M_synchronizeDevice(device, done.handle)
        done.check("M_synchronizeDevice (the copy's destination)")
        return Tensor(self._shared, None, copy)

    def to_host(self) raises -> Tensor:
        return self.to_device(self._shared[].host)

    def __deinit__(deinit self):
        capi.M_freeTensor(self._t)


struct TensorMap(Movable):
    """An `M_AsyncTensorMap`: a model's inputs, or its outputs."""

    var _shared: Shared
    var _state: ArcPointer[_MapState]

    def __init__(out self, shared: Shared, handle: capi.Handle):
        self._shared = shared
        self._state = ArcPointer(_MapState(handle))

    def __init__(out self, *, deinit move: Self):
        self._shared = move._shared^
        self._state = move._state^

    def handle(self) -> capi.Handle:
        return self._state[].map

    def borrow(
        mut self, name: String, var buffer: HostBuffer, dtype: DType, shape: List[Int]
    ) raises:
        """Lends `buffer` as the input `name`. The map owns it from now on;
        `owned_data(name)` reads what the model wrote into it."""
        self._lend(name, buffer.address(), dtype, shape, self._shared[].host)
        self._state[].owned.append(buffer^)
        self._state[].names.append(name)

    def borrow_address(
        mut self, name: String, address: Int, dtype: DType, shape: List[Int],
        on_device: Bool,
    ) raises:
        """Lends memory the caller owns and keeps alive (device memory, say,
        from `Tensor.to_device`). `on_device` selects the spec's device."""
        var device = self._shared[].device if on_device else self._shared[].host
        self._lend(name, address, dtype, shape, device)

    def _lend(
        mut self, name: String, address: Int, dtype: DType, shape: List[Int],
        device: capi.Handle,
    ) raises:
        var dims = List[Int64]()
        for d in shape:
            dims.append(Int64(d))
        var spec = capi.M_newTensorSpec(dims, m_dtype(dtype), name, device)
        var status = Status()
        capi.M_borrowTensorInto(self._state[].map, address, spec, status.handle)
        capi.M_freeTensorSpec(spec)  # the map keeps its own copy
        status.check("M_borrowTensorInto " + name)

    def owned_data[T: AnyType](ref self, name: String) raises -> Pointer[T, origin_of(self)]:
        """The host buffer this map owns for input `name`, as `T`s: what the
        model last wrote into it."""
        var names = Pointer(to=self._state[].names)
        for i in range(len(names[])):
            if names[][i] == name:
                var raw = self._state[].owned[i].data[T]()
                return rebind[Pointer[T, origin_of(self)]](raw)
        raise Error("maxrt: no owned buffer named " + name)

    def tensor(self, name: String) raises -> Tensor:
        """The tensor `name` (an output, or a lent input). It keeps this map,
        whose memory it views, alive."""
        var status = Status()
        var t = capi.M_getTensorByNameFrom(self._state[].map, name, status.handle)
        status.check("tensor " + name)
        return Tensor(self._shared, Optional(self._state), t)
