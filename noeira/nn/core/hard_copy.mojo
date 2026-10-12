"""HardCopy — verbatim Module→Module weight+state copy (tau=1 promotion).

The storage analog of legacy `map_params.hard_copy_params`: copies every Param
value AND every State buffer (e.g. BatchNorm running stats) from `src` into
`dst`, leaving optimizer moments untouched (target / arena nets don't optimize).
Used to initialize target nets and to promote arena winners.

The State copy is ESSENTIAL for BatchNorm nets: a hard copy that moves
weights/γ/β but not running_mean/var produces a net whose EVAL-mode forward runs
trained weights under stale (init mean-0 / var-1) normalization constants —
activations explode and the policy head emits non-finite outputs (the AlphaZero
post-promotion collapse). Stateless nets (MLPs) have an empty state walk.

Pointer-free: reuses the checkpoint walk order (`for_each_param` then
`for_each_state`) with two single-tree visitors — `_CollectVisitor` reads src
values into owned host `List`s, `_InjectVisitor` writes them into dst in the same
order, validating name + size per section (topology-drift catch). The round-trip
keeps native `Scalar[DT]` values (no text), so the copy is BIT-IDENTICAL (unlike
a save/load checkpoint round-trip, which goes through float text).
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from .tensor import Tensor
from .param import ParamVisitor, ParamVisitorRT, walk_params, ParamVisitorRef
from .module import Module


struct _CollectVisitor(ParamVisitor, ParamVisitorRT):
    """Reads each visited Param/State's values into an owned host `List`, in
    walk order. GPU params download first. Moments/grad are ignored."""

    var names: List[String]
    var vals: List[List[Scalar[DT]]]

    def __init__(out self):
        self.names = List[String]()
        self.vals = List[List[Scalar[DT]]]()

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        var buf = List[Scalar[DT]](capacity=n)
        for i in range(n):
            buf.append(param.data[i])
        self.names.append(name)
        self.vals.append(buf^)

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
struct _InjectVisitor(ParamVisitor, ParamVisitorRT):
    """Writes collected values into each visited Param/State in the same walk
    order, validating name + size. GPU params upload after."""

    var names: List[String]
    var vals: List[List[Scalar[DT]]]
    var cur: Int

    def __init__(
        out self, var names: List[String], var vals: List[List[Scalar[DT]]]
    ):
        self.names = names^
        self.vals = vals^
        self.cur = 0

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.cur >= len(self.vals):
            raise Error("hard_copy: dst has more params/states than src")
        if self.names[self.cur] != name:
            raise Error(
                "hard_copy: name mismatch — src '"
                + self.names[self.cur]
                + "' vs dst '"
                + name
                + "' (topology drift)"
            )
        ref buf = self.vals[self.cur]
        if len(buf) != n:
            raise Error(
                "hard_copy: size mismatch for '"
                + name
                + "' — src "
                + String(len(buf))
                + ", dst "
                + String(n)
            )
        for i in range(n):
            param.data[i] = buf[i]
        # A WRITE: advance `version`, or the version-gated derived weight caches
        # of the destination (`Linear` / `LinearAct` / `Conv2D` `w_pad`, the
        # bf16 `w_bf` of those and `Embedding`) keep serving the PRE-COPY
        # weight once it has run a forward — a promoted / target net silently
        # evaluated with its old weights. `checkpoint.mojo` load does the same.
        param.version += 1
        comptime if target == "gpu":
            param.upload(ctx.value())
        self.cur += 1

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
def hard_copy[
    target: StaticString, M: Module
](mut src: M, mut dst: M, ctx: Optional[DeviceContext] = None) raises:
    """Copy `src` → `dst` verbatim (params + states, bit-identical). Optimizer
    moments are NOT copied. Stateless models do a params-only copy (empty state
    walk = no-op)."""
    var c = _CollectVisitor()
    walk_params[target](src, c, ctx)
    var _sref1 = ParamVisitorRef.of[type_of(c), target](c)
    src.for_each_state[target](_sref1, ctx)
    # Copy the small collected lists (arena promotion is infrequent); moving
    # individual fields out of `c` would partially destroy it.
    var inj = _InjectVisitor(c.names.copy(), c.vals.copy())
    walk_params[target](dst, inj, ctx)
    var _sref2 = ParamVisitorRef.of[type_of(inj), target](inj)
    dst.for_each_state[target](_sref2, ctx)
    if inj.cur != len(inj.vals):
        raise Error("hard_copy: src has more params/states than dst")
