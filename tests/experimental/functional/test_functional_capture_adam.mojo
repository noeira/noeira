"""A captured TRAINING step on the functional tape: forward, backward AND
nn's Adam, replayed from one CUDA graph — gates.

The trick is nn's own: `adam.adopt["gpu"](tape, ctx)` (arena mode). The
tape is `ParamWalkable`, so the arena packs its native parameters and its
modules' alike; the arena step is one fused kernel, and its bias correction
`β^t` lives on the device and advances on EVERY replay. An un-adopted Adam
bakes the host's `1 - β^t` of the capture step into the graph instead.

Arms, each N steps over the same sequence of input batches:
  A  eager, un-adopted Adam (per-parameter, host bias correction)
  B  eager, adopted Adam (arena)
  C  adopted Adam, one eager step, the next CAPTURED, the rest REPLAYED
  D  CONTROL: un-adopted Adam captured the same way — must NOT match B

Gates: B ~ A (two Adam implementations), C == B bit for bit, D != B
(the gate can see a frozen bias correction), and the loss falls. On a GPU
without capture (Metal) only A vs B runs and the capture rows say so.

Run: pixi run -e default mojo run -I . tests/experimental/functional/test_functional_capture_adam.mojo
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor, FModule
import noeira.experimental.functional.F as F


comptime B = 64
comptime D = 12
comptime H = 48
comptime N_STEPS = 20
comptime N_BATCH = 5
comptime TOL: Float64 = 1e-5


def _rand(n: Int, scale: Float64) -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for _ in range(n):
        out.append(Scalar[DT]((random_float64() * 2.0 - 1.0) * scale))
    return out^


struct Run:
    """One arm's record: the loss of every step and the final parameters."""

    var losses: List[Scalar[DT]]
    var params: List[Scalar[DT]]
    var captured: Bool

    def __init__(out self):
        self.losses = List[Scalar[DT]]()
        self.params = List[Scalar[DT]]()
        self.captured = False


def _arm(ctx: DeviceContext, adopt: Bool, capture: Bool) raises -> Run:
    seed(17)
    var w2v = _rand(H * H, 0.2)
    var b2v = _rand(H, 0.1)
    var w3v = _rand(H, 0.3)
    var b3v = _rand(1, 0.1)
    var l1v = _rand(D * H, 0.4)
    var batches = List[List[Scalar[DT]]]()
    for _ in range(N_BATCH):
        batches.append(_rand(B * D, 1.0))
    var yv = List[Scalar[DT]]()
    for i in range(B):
        yv.append(Scalar[DT](0.5) * batches[0][i * D] - Scalar[DT](0.25))

    var src = Tensor.alloc(B * D)
    var ys = Tensor.alloc(B)
    for k in range(B):
        ys.data[k] = yv[k]
    src.upload(ctx)
    ys.upload(ctx)

    var tape = Tape(ctx)
    var w2 = tape.param(H, H, w2v)
    var b2 = tape.param(1, H, b2v)
    var w3 = tape.param(H, 1, w3v)
    var b3 = tape.param(1, 1, b3v)
    var l1 = tape.module[B](Linear[D, H].make["gpu", Deterministic](ctx))
    for k in range(D * H):
        l1.get().weight.val.data[k] = l1v[k]
    l1.get().weight.val.upload(ctx)
    var opt = Adam(lr=3e-3)
    if adopt:
        opt.adopt["gpu"](tape, ctx)

    var run = Run()

    def _step(mut opt: Adam) raises {mut tape, mut src, mut ys, mut l1, w2, b2, w3, b3, ctx} -> FTensor:
        tape.reset()
        tape.zero_grad()
        var x = tape.input_device(B, D, src.dev.value())
        var h = l1(x).relu()
        h = F.linear(h, w2, b2, "tanh")
        var loss = F.mse(F.linear(h, w3, b3), tape.input_device(B, 1, ys.dev.value()))
        loss.backward()
        opt.step["gpu"](tape, ctx)
        return loss

    def _feed(i: Int) raises {mut src, batches, ctx}:
        var b = batches[i % N_BATCH].copy()
        for k in range(B * D):
            src.data[k] = b[k]
        src.upload_resident(ctx)

    var loss: FTensor
    _feed(0)
    loss = _step(opt)  # eager: allocations, cuBLASLt plans
    run.losses.append(loss.item())
    if capture:
        _feed(1)
        tape.capture_begin()
        loss = _step(opt)
        tape.capture_end()
        run.captured = tape.captured()
        if not run.captured:
            return run^
        run.losses.append(loss.item())
        for i in range(2, N_STEPS):
            _feed(i)
            tape.replay()
            run.losses.append(loss.item())
    else:
        for i in range(1, N_STEPS):
            _feed(i)
            loss = _step(opt)
            run.losses.append(loss.item())
    run.params = w2.values()
    run.params.extend(b3.values())
    l1.get().weight.val.download(ctx)
    for k in range(D * H):
        run.params.append(l1.get().weight.val.data[k])
    return run^


def _cmp(name: String, got: Run, want: Run, exact: Bool) raises -> Bool:
    var worst: Float64 = 0
    var differ = 0
    var n = 0
    for k in range(len(want.losses)):
        n += 1
        if got.losses[k] != want.losses[k]:
            differ += 1
        worst = max(worst, abs(Float64(got.losses[k]) - Float64(want.losses[k])) / max(1.0, abs(Float64(want.losses[k]))))
    for k in range(len(want.params)):
        n += 1
        if got.params[k] != want.params[k]:
            differ += 1
        worst = max(worst, abs(Float64(got.params[k]) - Float64(want.params[k])) / max(1.0, abs(Float64(want.params[k]))))
    var ok = (differ == 0) if exact else (worst <= TOL)
    print("    ", name, ": ", n, " compared (", N_STEPS, " losses + final params), ",
          differ, " not bit-identical, worst rel ", worst, "  ",
          "OK" if ok else "FAIL", sep="")
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var a = _arm(ctx, adopt=False, capture=False)
    var b = _arm(ctx, adopt=True, capture=False)
    print("  loss over", N_STEPS, "Adam steps (B):", b.losses[0], "->", b.losses[N_STEPS - 1])
    var ok = b.losses[N_STEPS - 1] < b.losses[0] * 0.8
    ok = _cmp("B adopted (arena) vs A per-param Adam, eager", b, a, False) and ok
    var c = _arm(ctx, adopt=True, capture=True)
    if not c.captured:
        print("    capture unavailable on this device (NVIDIA only) — C and D not tested")
    else:
        ok = _cmp("C adopted, CAPTURED + replayed vs B eager", c, b, True) and ok
        var d = _arm(ctx, adopt=False, capture=True)
        var same = True
        for k in range(len(b.params)):
            if d.params[k] != b.params[k]:
                same = False
        var caught = not same
        print("    control D (un-adopted Adam captured: frozen bias correction) differs from B:",
              caught, " final loss", d.losses[N_STEPS - 1], "vs", b.losses[N_STEPS - 1])
        ok = caught and ok
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("functional capture+Adam gates failed")
