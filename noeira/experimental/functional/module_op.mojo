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
* a module's batch `B` is a compile-time parameter of its `forward`, so the
  batches it serves are fixed when it is handed over (up to three:
  `tape.module[64, 1](m)` instantiates both); another batch raises;
* a module keeps the caches of its LAST call, which its `vjp` reads, so it
  may be called ONCE per step (the tape raises on a second call before
  `reset`); weight sharing goes through native ops;
* ARITY 1 to 3, fp32 activations.
"""

from std.memory import alloc
from std.memory.alloc import Layout as AllocLayout
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitorRef


comptime ModState = Pointer[NoneType, MutUntrackedOrigin]
comptime TP = Pointer[Tensor, MutUntrackedOrigin]
"""A node's value or gradient storage. Every slot of one call carries the
SAME (untracked) origin: a module takes its inputs as one `TensorRefs`
pack, whose entries must share an origin."""
comptime ModFwdFn = def(
    ModState, TP, TP, TP, TP, Optional[DeviceContext]
) raises thin
"""(state, x0, x1, x2, y, ctx) — inputs past the module's ARITY unused."""
comptime ModVjpFn = def(
    ModState, TP, TP, TP, TP, TP, TP, TP, Optional[DeviceContext]
) raises thin
"""(state, x0, x1, x2, grad_out, g0, g1, g2, ctx): the module WRITES g_i."""
comptime ModZeroFn = def(ModState, Optional[DeviceContext]) raises thin
comptime ModWalkFn = def(
    ModState, mut ParamVisitorRef, Optional[DeviceContext], String
) raises thin
comptime ModDropFn = def(ModState) thin


def _mod_fwd[M: Module, B: Int, target: StaticString](
    s: ModState, x0: TP, x1: TP, x2: TP, y: TP, ctx: Optional[DeviceContext]
) raises:
    ref m = s.unsafe_bitcast[M]()[]
    comptime RT = TensorRefs[M.ARITY, MutUntrackedOrigin, M.ACT_DT]
    comptime if M.ARITY == 1:
        m.forward[target, B](rebind[RT](TensorRefs[1](x0[])), rebind[TensorImpl[M.ACT_DT]](y[]), ctx)
    elif M.ARITY == 2:
        m.forward[target, B](rebind[RT](TensorRefs[2](x0[], x1[])), rebind[TensorImpl[M.ACT_DT]](y[]), ctx)
    else:
        m.forward[target, B](rebind[RT](TensorRefs[3](x0[], x1[], x2[])), rebind[TensorImpl[M.ACT_DT]](y[]), ctx)


def _mod_vjp[M: Module, B: Int, target: StaticString](
    s: ModState, x0: TP, x1: TP, x2: TP, go: TP, g0: TP, g1: TP, g2: TP,
    ctx: Optional[DeviceContext],
) raises:
    ref m = s.unsafe_bitcast[M]()[]
    comptime RT = TensorRefs[M.ARITY, MutUntrackedOrigin, M.ACT_DT]
    comptime if M.ARITY == 1:
        m.vjp[target, B](
            rebind[RT](TensorRefs[1](x0[])), rebind[TensorImpl[M.ACT_DT]](go[]),
            rebind[RT](TensorRefs[1](g0[])), ctx,
        )
    elif M.ARITY == 2:
        m.vjp[target, B](
            rebind[RT](TensorRefs[2](x0[], x1[])), rebind[TensorImpl[M.ACT_DT]](go[]),
            rebind[RT](TensorRefs[2](g0[], g1[])), ctx,
        )
    else:
        m.vjp[target, B](
            rebind[RT](TensorRefs[3](x0[], x1[], x2[])), rebind[TensorImpl[M.ACT_DT]](go[]),
            rebind[RT](TensorRefs[3](g0[], g1[], g2[])), ctx,
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
    """One module the tape owns: where it is, how to call it per batch, its
    input widths."""

    var state: ModState
    var batches: List[Int]
    var fwds: List[ModFwdFn]
    var vjps: List[ModVjpFn]
    """`fwds[i]` / `vjps[i]` are the thunks instantiated for `batches[i]`."""
    var zero: ModZeroFn
    var walk: ModWalkFn
    var walk_state: ModWalkFn
    var drop: ModDropFn
    var arity: Int
    var in_dims: List[Int]
    var out_dim: Int
    var used: Bool
    """Called since the last `reset`: its caches hold that call."""
    var name: String
    """The prefix of its parameters' names in the tape's walk."""

    def __init__(
        out self, state: ModState, zero: ModZeroFn, walk: ModWalkFn,
        walk_state: ModWalkFn, drop: ModDropFn, arity: Int,
        var in_dims: List[Int], out_dim: Int,
    ):
        self.state = state
        self.batches = List[Int]()
        self.fwds = List[ModFwdFn]()
        self.vjps = List[ModVjpFn]()
        self.zero = zero
        self.walk = walk
        self.walk_state = walk_state
        self.drop = drop
        self.arity = arity
        self.in_dims = in_dims^
        self.out_dim = out_dim
        self.used = False
        self.name = String("")

    def batch_index(self, rows: Int) -> Int:
        for i in range(len(self.batches)):
            if self.batches[i] == rows:
                return i
        return -1


def _add_batch[M: Module, B: Int](mut e: ModEntry, gpu: Bool):
    e.batches.append(B)
    if gpu:
        e.fwds.append(_mod_fwd[M, B, "gpu"])
        e.vjps.append(_mod_vjp[M, B, "gpu"])
    else:
        e.fwds.append(_mod_fwd[M, B, "cpu"])
        e.vjps.append(_mod_vjp[M, B, "cpu"])


def make_entry[M: Module, B: Int, B2: Int, B3: Int](var m: M, gpu: Bool) -> ModEntry:
    """Move `m` to the heap and build its entry: thunks for the tape's
    device (a module made for the GPU has device parameters) and for each
    batch `B`, `B2`, `B3` that is > 0."""
    comptime assert M.ARITY >= 1 and M.ARITY <= 3, "tape.module: ARITY 1 to 3"
    comptime assert M.ACT_DT == DT, "tape.module: fp32 activations only"
    var in_dims = List[Int]()
    comptime for i in range(M.ARITY):
        comptime d = M.IN_DIMS[i]
        in_dims.append(d)
    var p = alloc(AllocLayout[M](count=1)).unsafe_leak()
    p.unsafe_write(m^)
    var st = p.unsafe_bitcast[NoneType]().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    var e: ModEntry
    if gpu:
        e = ModEntry(
            st, _mod_zero[M, "gpu"], _mod_walk[M, "gpu"],
            _mod_walk_state[M, "gpu"], _mod_drop[M], M.ARITY, in_dims^, M.OUT_DIM,
        )
    else:
        e = ModEntry(
            st, _mod_zero[M, "cpu"], _mod_walk[M, "cpu"],
            _mod_walk_state[M, "cpu"], _mod_drop[M], M.ARITY, in_dims^, M.OUT_DIM,
        )
    _add_batch[M, B](e, gpu)
    comptime if B2 > 0:
        _add_batch[M, B2](e, gpu)
    comptime if B3 > 0:
        _add_batch[M, B3](e, gpu)
    return e^
