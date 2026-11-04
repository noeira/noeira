"""Parameter snapshots across targets.

`sm_snapshot[target](model, ctx)` reads every param (or grad) of a model on
either target into host lists, in walk order; `sm_restore[target](model,
vals, ctx)` writes them back (uploading on "gpu", bumping `version` so the
layers' derived weight caches rebuild). With them a GPU model starts from a
CPU model's exact weights, and the GPU-vs-CPU gates compare like with like.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.param import ParamVisitor, ParamWalkable
from noeira.nn.core.tensor import Tensor


struct _Snap(ParamVisitor):
    var grads: Bool
    var out: List[List[Scalar[DT]]]

    def __init__(out self, grads: Bool):
        self.grads = grads
        self.out = List[List[Scalar[DT]]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            if self.grads:
                grad.download(ctx.value())
            else:
                param.download(ctx.value())
        var l = List[Scalar[DT]](length=N, fill=0)
        for i in range(N):
            l[i] = grad.data[i] if self.grads else param.data[i]
        self.out.append(l^)


struct _Restore(ParamVisitor):
    var vals: List[List[Scalar[DT]]]
    var k: Int

    def __init__(out self, var vals: List[List[Scalar[DT]]]):
        self.vals = vals^
        self.k = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if self.k >= len(self.vals) or len(self.vals[self.k]) != N:
            raise Error("sm_restore: param " + name + " does not match the snapshot")
        param.ensure(N)
        for i in range(N):
            param.data[i] = self.vals[self.k][i]
        comptime if target == "gpu":
            param.upload_resident(ctx.value())
        param.version += 1
        self.k += 1


def sm_snapshot[target: StaticString, M: ParamWalkable](
    mut model: M, ctx: Optional[DeviceContext], grads: Bool = False
) raises -> List[List[Scalar[DT]]]:
    """Every param's values (or grads), host-side, in walk order."""
    var v = _Snap(grads)
    model.for_each_param[target](v, ctx)
    return v.out.copy()


def sm_restore[target: StaticString, M: ParamWalkable](
    mut model: M, var vals: List[List[Scalar[DT]]], ctx: Optional[DeviceContext]
) raises:
    """Write a snapshot back into a model of the same architecture."""
    var v = _Restore(vals^)
    model.for_each_param[target](v, ctx)
    if v.k != len(v.vals):
        raise Error("sm_restore: the snapshot has " + String(len(v.vals)) + " params, the model " + String(v.k))
