"""Adam through `ParamVisitorRef` (the type-erased walk, runtime size, SIMD
update) against an INDEPENDENT scalar Adam written here — not against the
optimizer's own generic path, which now forwards to the same code and would
make the check blind. CPU. Also asserts the walk visits every param once,
and that a BARE walk (`model.for_each_param(opt)`, what graph trainers do
instead of `opt.step`) advances every weight's `version` — the gate on the
derived caches (`Linear`'s padded `w_pad`, the bf16 `w_bf`)."""
from std.math import abs, sqrt
from max.gpu.host import DeviceContext
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam

comptime NET = Sequential[Linear[37, 53], Linear[53, 29], Linear[29, 7]]
comptime LR = Scalar[DT](1e-2)
comptime B1 = Scalar[DT](0.9)
comptime B2 = Scalar[DT](0.999)
comptime EPS = Scalar[DT](1e-8)
comptime STEPS = 7


struct _SetGrad(ParamVisitor):
    var seed: Int
    def __init__(out self):
        self.seed = 0
    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for k in range(N):
            grad.data[k] = Scalar[DT](0.01) * Scalar[DT]((k + self.seed) % 17 - 8)
        self.seed += 3


struct _Versions(ParamVisitor):
    var v: List[Int]
    def __init__(out self):
        self.v = List[Int]()
    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        self.v.append(param.version)


struct _Snapshot(ParamVisitor):
    """Flattened (param, grad, apply_decay) of the whole model, walk order."""
    var vals: List[Scalar[DT]]
    var grads: List[Scalar[DT]]
    var decay: List[Bool]
    var n_params: Int
    def __init__(out self):
        self.vals = List[Scalar[DT]]()
        self.grads = List[Scalar[DT]]()
        self.decay = List[Bool]()
        self.n_params = 0
    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for k in range(N):
            self.vals.append(param.data[k])
            self.grads.append(grad.data[k])
            self.decay.append(apply_decay)
        self.n_params += 1


def main() raises:
    var net = NET.make["cpu", Deterministic](None)
    var sg = _SetGrad()
    net.for_each_param["cpu"](sg, None)
    var s0 = _Snapshot()
    net.for_each_param["cpu"](s0, None)
    if s0.n_params != 6 or len(s0.vals) != 37 * 53 + 53 + 53 * 29 + 29 + 29 * 7 + 7:
        raise Error("walk did not visit every param exactly once")
    # the reference: scalar Adam over the flattened snapshot, same defaults
    var want = s0.vals.copy()
    var m = List[Scalar[DT]](length=len(want), fill=Scalar[DT](0))
    var v = List[Scalar[DT]](length=len(want), fill=Scalar[DT](0))
    var opt = Adam(lr=LR)
    var worst = Scalar[DT](0)
    var n = 0
    for s in range(STEPS):
        var b1t = Scalar[DT](1)
        var b2t = Scalar[DT](1)
        for _ in range(s + 1):
            b1t *= B1
            b2t *= B2
        var bc1 = Scalar[DT](1) - b1t
        var bc2 = Scalar[DT](1) - b2t
        for k in range(len(want)):
            var g = s0.grads[k]
            var p = want[k]
            if s0.decay[k]:
                p -= LR * opt.wd * p
            m[k] = B1 * m[k] + (Scalar[DT](1) - B1) * g
            v[k] = B2 * v[k] + (Scalar[DT](1) - B2) * g * g
            want[k] = p - LR * (m[k] / bc1) / (sqrt(v[k] / bc2) + EPS)
        opt.step["cpu"](net, None)  # the erased path
        var snap = _Snapshot()
        net.for_each_param["cpu"](snap, None)
        for k in range(len(want)):
            var d = abs(snap.vals[k] - want[k])
            if d > worst:
                worst = d
            n += 1
    print("compared", n, "param values over", STEPS, "steps; worst |erased - reference| =", worst)
    if worst > Scalar[DT](2e-6):
        raise Error("FAIL: Adam through ParamVisitorRef disagrees with the scalar reference")
    print("PASS: Adam via ParamVisitorRef matches the independent reference")

    # A BARE walk must bump `version` too. `Linear` re-pads its K-aligned
    # weight copy only when the version moves, so a walk that updates the
    # weight without bumping leaves the GPU forward on the pre-update weight
    # forever, with no error. Shipped twice: fixed at this leaf in c5a527c7e,
    # lost again when e09997780 moved the bump into `step` only — and found
    # by the LeWM GPU checkpoint round-trip, not by any optimizer gate.
    var before = _Versions()
    net.for_each_param["cpu"](before, None)
    opt.begin_step()
    net.for_each_param["cpu"](opt, None)
    var after = _Versions()
    net.for_each_param["cpu"](after, None)
    for k in range(len(before.v)):
        if after.v[k] <= before.v[k]:
            raise Error(
                "FAIL: a bare Adam walk left param " + String(k)
                + "'s version at " + String(after.v[k])
                + " — version-gated weight caches would serve stale weights"
            )
    print("PASS: a bare Adam walk advances every param's version")
