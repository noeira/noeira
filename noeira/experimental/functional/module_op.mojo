"""Comptime nn `Module`s as functional-tape ops — the bridge.

`tape.module[B](Linear[64, 256].make[…]())` hands a module to the tape and
returns an `FModule` handle; `m(x)` records ONE node whose forward and
backward are the module's own `forward` / `vjp`. Every primitive the
comptime nn has (LayerNorm, attention, Conv2D, …) is then usable eagerly
with no second implementation to drift from the first.

How a tape that knows no module types calls one: the module lives on the
heap (owned by the tape — Mojo destroys a value at its LAST USE, so a
borrowed module would be gone by the backward), and the tape keeps an
opaque pointer to it plus THIN function pointers to thunks instantiated for
its type — the `ParamVisitorRef` idiom of `nn/core/param.mojo`.

Limits, by construction:
* a module's batch `B` is a compile-time parameter of its `forward`, so it
  is fixed when the module is handed over; a call with another batch raises;
* a module keeps the caches of its LAST call, which its `vjp` reads, so it
  may be called ONCE per step (the tape raises on a second call before
  `reset`); weight sharing goes through native ops;
* `ARITY == 1` and fp32 activations, for this spike.
"""

from std.memory import alloc
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitorRef


comptime ModState = Pointer[NoneType, MutUntrackedOrigin]
comptime ModFwdFn = def(
    ModState, mut Tensor, mut Tensor, Optional[DeviceContext]
) raises thin
comptime ModVjpFn = def(
    ModState, mut Tensor, mut Tensor, mut Tensor, Optional[DeviceContext]
) raises thin
comptime ModZeroFn = def(ModState, Optional[DeviceContext]) raises thin
comptime ModWalkFn = def(
    ModState, mut ParamVisitorRef, Optional[DeviceContext], String
) raises thin
comptime ModDropFn = def(ModState) thin


def _mod_fwd[M: Module, B: Int, target: StaticString](
    s: ModState, mut x: Tensor, mut out: Tensor, ctx: Optional[DeviceContext]
) raises:
    ref m = s.unsafe_bitcast[M]()[]
    m.forward[target, B](
        rebind[TensorRefs[M.ARITY, origin_of(x), M.ACT_DT]](TensorRefs[1](x)),
        rebind[TensorImpl[M.ACT_DT]](out),
        ctx,
    )


def _mod_vjp[M: Module, B: Int, target: StaticString](
    s: ModState,
    mut x: Tensor,
    mut go: Tensor,
    mut gi: Tensor,
    ctx: Optional[DeviceContext],
) raises:
    ref m = s.unsafe_bitcast[M]()[]
    m.vjp[target, B](
        rebind[TensorRefs[M.ARITY, origin_of(x), M.ACT_DT]](TensorRefs[1](x)),
        rebind[TensorImpl[M.ACT_DT]](go),
        rebind[TensorRefs[M.ARITY, origin_of(gi), M.ACT_DT]](TensorRefs[1](gi)),
        ctx,
    )


def _mod_zero[M: Module, target: StaticString](
    s: ModState, ctx: Optional[DeviceContext]
) raises:
    s.unsafe_bitcast[M]()[].zero_grad[target](ctx)


def _mod_walk[M: Module, target: StaticString](
    s: ModState, mut v: ParamVisitorRef, ctx: Optional[DeviceContext],
    prefix: String,
) raises:
    s.unsafe_bitcast[M]()[].for_each_param[target](v, ctx, prefix)


def _mod_walk_state[M: Module, target: StaticString](
    s: ModState, mut v: ParamVisitorRef, ctx: Optional[DeviceContext],
    prefix: String,
) raises:
    s.unsafe_bitcast[M]()[].for_each_state[target](v, ctx, prefix)


def _mod_drop[M: Module](s: ModState):
    var p = s.unsafe_bitcast[M]()
    _ = p.unsafe_take_pointee()
    p.unsafe_free()


struct ModEntry(Copyable):
    """One module the tape owns: where it is, how to call it, its shape."""

    var state: ModState
    var fwd: ModFwdFn
    var vjp: ModVjpFn
    var zero: ModZeroFn
    var walk: ModWalkFn
    var walk_state: ModWalkFn
    var drop: ModDropFn
    var batch: Int
    var in_dim: Int
    var out_dim: Int
    var used: Bool
    """Called since the last `reset`: its caches hold that call."""
    var name: String
    """The prefix of its parameters' names in the tape's walk."""

    def __init__(
        out self,
        state: ModState,
        fwd: ModFwdFn,
        vjp: ModVjpFn,
        zero: ModZeroFn,
        walk: ModWalkFn,
        walk_state: ModWalkFn,
        drop: ModDropFn,
        batch: Int,
        in_dim: Int,
        out_dim: Int,
    ):
        self.state = state
        self.fwd = fwd
        self.vjp = vjp
        self.zero = zero
        self.walk = walk
        self.walk_state = walk_state
        self.drop = drop
        self.batch = batch
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.used = False
        self.name = String("")


def make_entry[M: Module, B: Int](var m: M, gpu: Bool) -> ModEntry:
    """Move `m` to the heap and build its entry, with the thunks of the
    tape's device (a module made for the GPU has device parameters)."""
    comptime assert M.ARITY == 1, "tape.module: ARITY 1 modules only (spike)"
    comptime assert M.ACT_DT == DT, "tape.module: fp32 activations only"
    comptime in_dim = M.IN_DIMS[0]
    var p = alloc[M](1)
    p.unsafe_write(m^)
    var st = p.unsafe_bitcast[NoneType]().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    if gpu:
        return ModEntry(
            st, _mod_fwd[M, B, "gpu"], _mod_vjp[M, B, "gpu"],
            _mod_zero[M, "gpu"], _mod_walk[M, "gpu"],
            _mod_walk_state[M, "gpu"], _mod_drop[M],
            B, in_dim, M.OUT_DIM,
        )
    return ModEntry(
        st, _mod_fwd[M, B, "cpu"], _mod_vjp[M, B, "cpu"],
        _mod_zero[M, "cpu"], _mod_walk[M, "cpu"],
        _mod_walk_state[M, "cpu"], _mod_drop[M],
        B, in_dim, M.OUT_DIM,
    )
