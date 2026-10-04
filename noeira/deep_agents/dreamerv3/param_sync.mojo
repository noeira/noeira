"""Name-keyed parameter sync between two ComputeGraphs (storage-native).

The DreamerV3 trainer optimizes the RSSM core/prior params inside the
`WMCoreGraph`, but the imagination rollout runs them through a separate
`WMImagineGraph` (no obs/token path). Both graphs name the shared nodes
identically (`x0`/`x1`/`x2`/`dhin`/`h`/`gru` core + `pr0`/`pr1`/`prior`),
so their `for_each_param` walks emit matching dotted names for the shared
sub-modules.

`collect_graph_params` snapshots the trained source graph's params into a
`name → values` Dict (downloads on GPU). `apply_graph_params` then copies,
by NAME, every value that exists in the snapshot into the destination graph
(uploads on GPU) — the imagine graph becomes a read-only mirror of the
trained core each AC step. The src/dst graphs are DIFFERENT types sharing
some node names; the dict skip-on-miss handles the non-shared params.

Two functions (not one) because Mojo forbids two variadic `*DECLS` packs in
one signature — collect over `src`, apply over `dst`, threaded by the Dict.
"""

from std.collections import Dict
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT, walk_params
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import GraphDecl


# Snapshot visitor: copy each param's values into the dict by name (downloads
# on GPU first so `param.data` is current).
struct _SnapshotVisitor(Movable, ParamVisitor, ParamVisitorRT):
    var d: Dict[String, List[Scalar[DT]]]

    def __init__(out self):
        self.d = Dict[String, List[Scalar[DT]]]()

    def take(deinit self) -> Dict[String, List[Scalar[DT]]]:
        return self.d^

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
        var vals = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
        for i in range(n):
            vals[i] = param.data[i]
        self.d[name] = vals^


# Named-import visitor: for each dst param, if its name is in the snapshot,
# copy the values into the slab (uploads on GPU). Names not present are left
# at their current value (the dst-only params with no source match).

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
struct _NamedImportVisitor(ParamVisitor, ParamVisitorRT):
    var d: Dict[String, List[Scalar[DT]]]
    var missing: Int

    def __init__(out self, var d: Dict[String, List[Scalar[DT]]]):
        self.d = d^
        self.missing = 0

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
        if name not in self.d:
            self.missing += 1
            return
        ref vals = self.d[name]
        param.ensure(n)
        var nn = len(vals) if len(vals) < n else n
        for i in range(nn):
            param.data[i] = vals[i]
        param.n = n
        # A written weight must advance `version`: `Linear`'s GPU forward reads a
        # version-gated padded copy (`w_pad`) and would keep the old weight.
        param.version += 1
        comptime if target == "gpu":
            param.upload_resident(ctx.value())

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
def collect_graph_params[
    target: StaticString, *DECLS: GraphDecl
](
    mut src: ComputeGraph[*DECLS],
    ctx: Optional[DeviceContext] = None,
) raises -> Dict[String, List[Scalar[DT]]]:
    """Snapshot every param of `src` into a `name → values` Dict (downloads
    on GPU). Target-agnostic result (host values either way)."""
    var v = _SnapshotVisitor()
    walk_params[target](src, v, ctx)
    return v^.take()


def apply_graph_params[
    target: StaticString, *DECLS: GraphDecl
](
    mut dst: ComputeGraph[*DECLS],
    imm snap: Dict[String, List[Scalar[DT]]],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy every shared-name param value from the snapshot into `dst` (skips
    names with no match; uploads on GPU)."""
    var v = _NamedImportVisitor(snap.copy())
    walk_params[target](dst, v, ctx)


# ── Device-direct variant (Stage 3 P5; GPU capture path) ──────────────────
#
# The collect/apply above round-trip params through host (download → upload),
# which is ILLEGAL inside a CUDA-graph capture (no H2D/D2H mid-capture). For
# the captured discrete WM+AC step the sync must stay on-device: snapshot each
# source param's DEVICE pointer by name (no download), then `copy_from_device`
# it straight into the matching dst param's device buffer (no upload). Result
# is bit-identical to the host path (same values, device→device) so the eager
# train_step is unchanged.


struct _DevSnapshotVisitor(Movable, ParamVisitor, ParamVisitorRT):
    """name → source param DEVICE pointer (GPU-only; no host download)."""

    var d: Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]]

    def __init__(out self):
        self.d = Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]]()

    def take(deinit self) -> Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]]:
        return self.d^

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
            self.d[name] = rebind[
                Pointer[Scalar[DT], MutUntrackedOrigin]
            ](param.dev.value().unsafe_ptr())

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
struct _DevImportVisitor(ParamVisitor, ParamVisitorRT):
    """For each dst param whose name is in the snapshot, device→device copy the
    source buffer in (no host upload). Names with no match are left as-is."""

    var d: Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]]
    var ctx: DeviceContext

    def __init__(
        out self,
        var d: Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]],
        ctx: DeviceContext,
    ):
        self.d = d^
        self.ctx = ctx

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
            if name not in self.d:
                return
            param.copy_from_device(self.ctx, self.d[name].as_unsafe_any_origin(), n)
            param.version += 1  # see the host-dict apply above

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
def collect_graph_params_device[
    target: StaticString, *DECLS: GraphDecl
](
    mut src: ComputeGraph[*DECLS],
    ctx: Optional[DeviceContext] = None,
) raises -> Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]]:
    """Snapshot each `src` param's DEVICE pointer by name (GPU; no download)."""
    var v = _DevSnapshotVisitor()
    walk_params[target](src, v, ctx)
    return v^.take()


def apply_graph_params_device[
    target: StaticString, *DECLS: GraphDecl
](
    mut dst: ComputeGraph[*DECLS],
    var snap: Dict[String, Pointer[Scalar[DT], MutUntrackedOrigin]],
    ctx: DeviceContext,
) raises:
    """Device→device copy each shared-name source buffer into `dst` (GPU; no
    upload). Capture-safe core→imagine mirror."""
    var v = _DevImportVisitor(snap^, ctx)
    walk_params[target](dst, v, Optional[DeviceContext](ctx))
