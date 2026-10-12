"""Target-net syncs reach every parameter, state and weight cache.

`Module.polyak_from` defaults to a NO-OP, and `polyak_tensor` / `hard_copy`
write weights in place. Two ways for a target net to stay frozen, silently:

  1. a module owning params (or state) with no `polyak_from` override — the
     target keeps its init values (LayerNorm, then RMSNorm in DreamerV3's slow
     value, BatchNorm / ConvRMSNorm / Embedding / GaussianHead: audit
     2026-10-07);
  2. a version-gated derived weight cache (`w_pad`, `w_bf`) that the write
     does not invalidate — the target keeps serving its pre-sync weight once
     it has run a forward (Conv2D under Polyak froze Rainbow's target on the
     padded MAX path; `hard_copy` never bumped `version` at all).

Gates:
  - per module: fill the source's params AND state with random values, then
    `tgt.polyak_from(src, tau=1)` must make every param and state of the
    target equal the source's (module-agnostic: catches any field an
    override skips);
  - per module with a weight cache, on the GPU: run the target's forward
    FIRST (warms the cache), then `hard_copy(src -> tgt)`; the next target
    forward must equal the source's.

    pixi run -e apple mojo run -I . tests/nn/test_polyak_and_copy_sync.mojo
"""

from std.math import abs
from std.random import seed, random_float64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, LAYOUT_NCHW
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT, walk_params, ParamVisitorRef
from noeira.nn.core.hard_copy import hard_copy, _CollectVisitor
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.conv_rms_norm import ConvRMSNorm
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D
from noeira.nn.primitives.batch_norm_2d import BatchNorm2D
from noeira.nn.primitives.embedding import Embedding
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.conv2d import Conv2D
from noeira.deep_agents.primitives.gaussian_head import GaussianHead


struct _Fill(ParamVisitor, ParamVisitorRT):
    """Overwrites every visited param / state with random values (a write:
    bumps `version`, uploads on the GPU)."""

    def __init__(out self):
        pass

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(n):
            param.data[i] = Scalar[DT](random_float64(0.5, 1.5))
        param.version += 1
        comptime if target == "gpu":
            param.upload(ctx.value())

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def _dump[target: StaticString, M: Module](mut m: M, ctx: Optional[DeviceContext]) raises -> _CollectVisitor:
    var c = _CollectVisitor()
    walk_params[target](m, c, ctx)
    var r = ParamVisitorRef.of[type_of(c), target](c)
    m.for_each_state[target](r, ctx)
    return c^


def check_polyak[target: StaticString, M: Module](name: String, ctx: Optional[DeviceContext]) raises -> Bool:
    seed(1)
    var src = M.make[target, Kaiming](ctx)
    seed(2)
    var tgt = M.make[target, Kaiming](ctx)
    var f = _Fill()
    walk_params[target](src, f, ctx)
    var fr = ParamVisitorRef.of[type_of(f), target](f)
    src.for_each_state[target](fr, ctx)
    tgt.polyak_from[target](src, Scalar[DT](1.0), ctx)
    var a = _dump[target](src, ctx)
    var b = _dump[target](tgt, ctx)
    var worst = 0.0
    var n_fields = len(a.vals)
    var stale = String("")
    for i in range(n_fields):
        var w = 0.0
        for j in range(len(a.vals[i])):
            w = max(w, abs(Float64(a.vals[i][j]) - Float64(b.vals[i][j])))
        if w > 1e-6:
            stale += " " + a.names[i]
        worst = max(worst, w)
    var ok = worst <= 1e-6 and n_fields > 0
    print(
        "  polyak ", name, " [", target, "]: ", n_fields, " params+states, max |tgt - src| ",
        worst, "" if ok else " ✗ not synced:" + stale, sep="",
    )
    return ok


def check_copy_linear[IN: Int, OUT: Int, B: Int](ctx: DeviceContext) raises -> Bool:
    """`hard_copy` into a `Linear` whose weight cache is warm (it ran a
    forward): the next forward must use the copied weight."""
    comptime L = Linear[IN, OUT]
    var octx = Optional[DeviceContext](ctx)
    seed(3)
    var src = L.make["gpu", Kaiming](octx)
    seed(4)
    var tgt = L.make["gpu", Kaiming](octx)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
    x.upload(ctx)
    var ys = Tensor()
    var yt = Tensor()
    src.forward["gpu", B](TensorRefs[1](x), ys, octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)  # warms tgt's caches
    hard_copy["gpu"](src, tgt, octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)
    ctx.synchronize()
    ys.download(ctx)
    yt.download(ctx)
    return _report("Linear[" + String(IN) + "->" + String(OUT) + "]", ys, yt, B * OUT)


def check_copy_conv[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, B: Int
](ctx: DeviceContext) raises -> Bool:
    """Same as `check_copy_linear`, for a `Conv2D`."""
    comptime Cv = Conv2D[IC, OC, K, S, P, H, W]
    var octx = Optional[DeviceContext](ctx)
    seed(3)
    var src = Cv.make["gpu", Kaiming](octx)
    seed(4)
    var tgt = Cv.make["gpu", Kaiming](octx)
    var x = Tensor.alloc(B * Cv.IN_FLAT)
    for i in range(B * Cv.IN_FLAT):
        x.data[i] = Scalar[DT](random_float64(-1, 1))
    x.upload(ctx)
    var ys = Tensor()
    var yt = Tensor()
    src.forward["gpu", B](TensorRefs[1](x), ys, octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)  # warms tgt's caches
    hard_copy["gpu"](src, tgt, octx)
    tgt.forward["gpu", B](TensorRefs[1](x), yt, octx)
    ctx.synchronize()
    ys.download(ctx)
    yt.download(ctx)
    return _report("Conv2D[" + String(IC) + "->" + String(OC) + " k" + String(K) + "]", ys, yt, B * Cv.OUT_FLAT)


def _report(name: String, ys: Tensor, yt: Tensor, n: Int) -> Bool:
    var w = 0.0
    for i in range(n):
        w = max(w, abs(Float64(ys.data[i]) - Float64(yt.data[i])))
    var ok = w <= 1e-5
    print("  hard_copy ", name, ": max |tgt - src| after copy ", w, "" if ok else " ✗ (stale weight cache)", sep="")
    return ok


def main() raises:
    print("Target-net syncs: polyak_from(tau=1) and hard_copy reach every param, state and cache")
    var ok = True
    var none = Optional[DeviceContext](None)
    ok = check_polyak["cpu", RMSNorm[32]]("RMSNorm", none) and ok
    ok = check_polyak["cpu", ConvRMSNorm[8, 16]]("ConvRMSNorm", none) and ok
    ok = check_polyak["cpu", BatchNorm1D[16]]("BatchNorm1D", none) and ok
    ok = check_polyak["cpu", BatchNorm2D[8, 4, 4]]("BatchNorm2D", none) and ok
    ok = check_polyak["cpu", Embedding[20, 8]]("Embedding", none) and ok
    ok = check_polyak["cpu", GaussianHead[16, 4]]("GaussianHead", none) and ok
    ok = check_polyak["cpu", LayerNorm[16]]("LayerNorm", none) and ok
    ok = check_polyak["cpu", Linear[12, 8]]("Linear", none) and ok
    ok = check_polyak["cpu", Conv2D[3, 4, 3, 1, 1, 6, 6]]("Conv2D", none) and ok
    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)
    ok = check_polyak["gpu", RMSNorm[32]]("RMSNorm", octx) and ok
    ok = check_polyak["gpu", BatchNorm2D[8, 4, 4]]("BatchNorm2D", octx) and ok
    # Weight caches: a padded `Linear` (MAX's padded path on Apple) and a conv
    # that keeps MAX's padded path on NVIDIA too (`Conv2D.KEEP_MAX`).
    ok = check_copy_linear[100, 64, 96](ctx) and ok
    ok = check_copy_conv[1, 16, 5, 2, 0, 28, 28, 8](ctx) and ok   # MNIST conv1: KEEP_MAX
    assert_true(ok, "a target-net sync missed a param, a state or a weight cache")
    print("PASS")
