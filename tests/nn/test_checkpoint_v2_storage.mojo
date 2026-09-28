"""Checkpoint: named sections + optimizer-moment (m/v) round-trip + drift.

1. Train a model a few Adam steps (populates per-Param m/v), save (v3 binary),
   load into a fresh model, and confirm params AND moments restore EXACTLY (a
   capture visitor concatenates param+m+v values; model1 vs model2 must match)
   — proving the optimizer-state save enables exact resume. CPU + GPU.
2. Topology drift: loading the checkpoint into a differently-sized model RAISES
   (name/size validation), which the positional v1 format could not catch.
3. Legacy v2 compat: a checkpoint written with the old TEXT writer still loads
   through the dispatching `load_params` (pre-v3 files remain readable).

Run: pixi run -e apple mojo run -I . tests/nn/test_checkpoint_v2_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, load_params, LegacyV2CheckpointWriter,
)
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]
comptime BAD = Sequential[Linear[D, 8], Linear[8, O]]


struct _MVCapture(ParamVisitor):
    """Concatenates each Param's value, then m, then v (when populated)."""
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(N):
            self.vals.append(param.data[i])
        if m.n >= N and v.n >= N:
            comptime if target == "gpu":
                m.download(ctx.value())
                v.download(ctx.value())
            for i in range(N):
                self.vals.append(m.data[i])
            for i in range(N):
                self.vals.append(v.data[i])


def _train3[target: StaticString](
    mut net: NET, ctx: Optional[DeviceContext]
) raises:
    var opt = Adam(lr=1e-3)
    for step in range(3):
        var x = Tensor.alloc(B * D)
        var go = Tensor.alloc(B * O)
        for i in range(B * D):
            x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
        for i in range(B * O):
            go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
        var out = Tensor.alloc(B * O)
        var gi = Tensor.alloc(B * D)
        comptime if target == "gpu":
            x.upload(ctx.value()); go.upload(ctx.value())
        net.forward[target, B](TensorRefs[1](x), out, ctx)
        net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        opt.step[target](net, ctx)


def _check[target: StaticString](
    ctx: Optional[DeviceContext], path: String
) raises -> Bool:
    var a = NET.make[target, Deterministic](ctx)
    _train3[target](a, ctx)
    save_params[target](a, path, ctx)

    var b = NET.make[target, Deterministic](ctx)
    load_params[target](b, path, ctx)

    var ca = _MVCapture(); a.for_each_param[target](ca, ctx)
    var cb = _MVCapture(); b.for_each_param[target](cb, ctx)
    if len(ca.vals) != len(cb.vals) or len(ca.vals) == 0:
        return False
    # m/v must actually be present (else we'd only be testing param round-trip).
    var n_params_vals = (D * H + H) + (H * O + O)
    if len(ca.vals) <= n_params_vals:
        return False  # moments were not captured → save failed
    for i in range(len(ca.vals)):
        if abs(ca.vals[i] - cb.vals[i]) > Scalar[DT](1e-6):
            return False
    return True


def main() raises:
    print("Checkpoint v3 binary (+ legacy v2 text compat)")
    var oc = _check["cpu"](None, String("/tmp/ckpt_v2_cpu.txt"))
    print("  CPU round-trip+moments:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c), String("/tmp/ckpt_v2_gpu.txt"))
    print("  GPU round-trip+moments:", "OK" if og else "FAIL")

    # Topology drift: load the CPU checkpoint into a wrong-sized model → raises.
    var drift_caught = False
    try:
        var bad = BAD.make["cpu", Deterministic](None)
        load_params["cpu"](bad, String("/tmp/ckpt_v2_cpu.txt"), None)
    except:
        drift_caught = True
    print("  topology-drift raises:", "OK" if drift_caught else "FAIL")

    # Legacy v2 text compat: write with the OLD text writer, load through the
    # dispatching load_params, values must match.
    var a2 = NET.make["cpu", Deterministic](None)
    _train3["cpu"](a2, None)
    var w = LegacyV2CheckpointWriter(save_moments=True)
    w.mode = 0
    a2.for_each_param["cpu"](w, None)
    w.mode = 1
    a2.for_each_state["cpu"](w, None)
    with open("/tmp/ckpt_legacy_v2.txt", "w") as f:
        f.write(w.content)
    var b2 = NET.make["cpu", Deterministic](None)
    load_params["cpu"](b2, String("/tmp/ckpt_legacy_v2.txt"), None)
    var ca2 = _MVCapture(); a2.for_each_param["cpu"](ca2, None)
    var cb2 = _MVCapture(); b2.for_each_param["cpu"](cb2, None)
    var legacy_ok = len(ca2.vals) == len(cb2.vals) and len(ca2.vals) > 0
    if legacy_ok:
        for i in range(len(ca2.vals)):
            # text round-trip is lossy at ~1e-6 (String(float) precision)
            if abs(ca2.vals[i] - cb2.vals[i]) > Scalar[DT](1e-5):
                legacy_ok = False
                break
    print("  legacy v2 text load:", "OK" if legacy_ok else "FAIL")

    assert_true(oc and og and drift_caught and legacy_ok, "checkpoint")
    print("CHECKPOINT V3 + LEGACY V2 OK")
