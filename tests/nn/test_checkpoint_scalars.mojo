"""Checkpoint v3 `K` scalar sections + Adam / ScalarAdam step-state resume.

1. Scalars round-trip BIT-EXACTLY through `save_params` / `load_params` and
   `save_params_multi` / `load_params_multi` — including a counter past 2^24
   (where float32 stops counting) and a value whose decimal text would not
   round-trip.
2. A v3 file written without scalars loads with an empty set.
3. A foreign header after the last tensor raises (drift is still caught).
4. Adam resume: train A, checkpoint params + moments + step state, restore
   into B, then take ONE more identical step on both — the params must be
   identical. The control restores the same file WITHOUT the step state and
   must DIFFER, or the gate would pass on an unchanged binary. CPU, and GPU
   with the arena adopted both before and after the restore.
5. ScalarAdam: `put_state` / `take_state` then one identical step on each.

Run: pixi run -e apple mojo run -I . tests/nn/test_checkpoint_scalars.mojo
"""

from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import (
    save_params, load_params, save_params_multi, load_params_multi,
    CheckpointScalars,
)
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.scalar_adam import ScalarAdam


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]
comptime SMALL = Sequential[Linear[D, O]]


struct _Capture(ParamVisitor):
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


def _step[target: StaticString](
    mut net: NET, mut opt: Adam, step: Int, ctx: Optional[DeviceContext]
) raises:
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
    net.zero_grad[target](ctx)
    net.forward[target, B](TensorRefs[1](x), out, ctx)
    net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    opt.step[target](net, ctx)


def _params[target: StaticString](
    mut net: NET, ctx: Optional[DeviceContext]
) raises -> List[Scalar[DT]]:
    var c = _Capture()
    net.for_each_param[target](c, ctx)
    return c.vals.copy()


def _same(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Bool:
    if len(a) != len(b) or len(a) == 0:
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def test_scalars_round_trip() raises:
    var net = NET.make["cpu", Deterministic](None)
    var sc = CheckpointScalars()
    sc.set_int("counter", (1 << 40) + 3)
    sc.set("eps.epsilon", 0.1)
    sc.set("alpha.value", -1.2345678901234567)
    sc.set("eps.epsilon", 0.05)  # overwrite, not a second section
    save_params["cpu"](net, "/tmp/ckpt_scalars.bin", None, scalars=sc)
    var back = load_params["cpu"](net, "/tmp/ckpt_scalars.bin", None)
    assert_equal(len(back), 3)
    assert_equal(back.get_int("counter", 0), (1 << 40) + 3)
    assert_true(back.get("eps.epsilon", 0.0) == 0.05)
    assert_true(back.get("alpha.value", 0.0) == -1.2345678901234567)
    assert_true(back.get("absent", 7.0) == 7.0)

    var a = NET.make["cpu", Deterministic](None)
    var s = SMALL.make["cpu", Deterministic](None)
    var sc2 = CheckpointScalars()
    sc2.set_int("steps", 123456789)
    save_params_multi["cpu"]("/tmp/ckpt_scalars_multi.bin", None, False, a, s, scalars=sc2)
    var back2 = load_params_multi["cpu"]("/tmp/ckpt_scalars_multi.bin", None, a, s)
    assert_equal(back2.get_int("steps", 0), 123456789)


def test_no_scalars_is_empty() raises:
    var net = NET.make["cpu", Deterministic](None)
    save_params["cpu"](net, "/tmp/ckpt_scalars_none.bin", None)
    var back = load_params["cpu"](net, "/tmp/ckpt_scalars_none.bin", None)
    assert_equal(len(back), 0)


def test_bad_name_raises() raises:
    var sc = CheckpointScalars()
    var raised = False
    try:
        sc.set("has space", 1.0)
    except:
        raised = True
    assert_true(raised, "a name with a space must be refused")


def test_extra_tensor_is_drift() raises:
    # Two models' sections, read back as ONE model: the second model's `P`
    # header sits where scalars are expected and must raise.
    var a = NET.make["cpu", Deterministic](None)
    var s = SMALL.make["cpu", Deterministic](None)
    save_params_multi["cpu"]("/tmp/ckpt_scalars_drift.bin", None, False, a, s)
    var raised = False
    try:
        _ = load_params["cpu"](a, "/tmp/ckpt_scalars_drift.bin", None)
    except:
        raised = True
    assert_true(raised, "unread tensor sections must raise")


def _adam_resume[target: StaticString](
    ctx: Optional[DeviceContext], adopt_before_take: Bool, path: String
) raises:
    var a = NET.make[target, Deterministic](ctx)
    var opt_a = Adam(lr=1e-2)
    opt_a.adopt[target](a, ctx)
    for s in range(7):
        _step[target](a, opt_a, s, ctx)
    var sc = CheckpointScalars()
    opt_a.put_step_state(sc, "opt")
    save_params[target](a, path, ctx, save_moments=True, scalars=sc)

    # Resumed WITH step state.
    var b = NET.make[target, Deterministic](ctx)
    var opt_b = Adam(lr=1e-2)
    if adopt_before_take:
        opt_b.adopt[target](b, ctx)
    var back = load_params[target](b, path, ctx)
    opt_b.take_step_state(back, "opt")
    if not adopt_before_take:
        opt_b.adopt[target](b, ctx)
    assert_equal(opt_b.t, 7)

    # Control: same file, step state NOT restored.
    var c = NET.make[target, Deterministic](ctx)
    var opt_c = Adam(lr=1e-2)
    opt_c.adopt[target](c, ctx)
    _ = load_params[target](c, path, ctx)

    _step[target](a, opt_a, 7, ctx)
    _step[target](b, opt_b, 7, ctx)
    _step[target](c, opt_c, 7, ctx)
    var pa = _params[target](a, ctx)
    var pb = _params[target](b, ctx)
    var pc = _params[target](c, ctx)
    var db = Scalar[DT](0); var dc = Scalar[DT](0)
    for i in range(len(pa)):
        db = max(db, abs(pa[i] - pb[i])); dc = max(dc, abs(pa[i] - pc[i]))
    print("  [", target, "] max|a-b| =", db, " max|a-c| =", dc, " t:", opt_a.t, opt_b.t, opt_c.t)
    assert_true(_same(pa, _params[target](b, ctx)), "resume with step state")
    assert_true(
        not _same(pa, _params[target](c, ctx)),
        "control resumed WITHOUT step state must differ — else vacuous",
    )


def test_adam_resume_cpu() raises:
    _adam_resume["cpu"](None, False, "/tmp/ckpt_adam_resume_cpu.bin")


def test_adam_resume_gpu() raises:
    var c = DeviceContext()
    _adam_resume["gpu"](Optional(c), True, "/tmp/ckpt_adam_resume_gpu_a.bin")
    _adam_resume["gpu"](Optional(c), False, "/tmp/ckpt_adam_resume_gpu_b.bin")


def test_scalar_adam_state() raises:
    var a = ScalarAdam.new(0.3, 3e-2)
    for i in range(9):
        a.step(Scalar[DT](i % 4) - 1.5)
    var sc = CheckpointScalars()
    a.put_state(sc, "alpha")
    var b = ScalarAdam.new(0.0, 3e-2)
    b.take_state(sc, "alpha")
    a.step(0.7)
    b.step(0.7)
    assert_true(a.value == b.value and a.m == b.m and a.v == b.v)
    assert_equal(b.t, 10)


def main() raises:
    test_scalars_round_trip()
    test_no_scalars_is_empty()
    test_bad_name_raises()
    test_extra_tensor_is_drift()
    test_adam_resume_cpu()
    test_adam_resume_gpu()
    test_scalar_adam_state()
    print("CHECKPOINT SCALARS OK")
