"""SuperMimic with an object: the joint world-model loss on the device against the CPU, same weights, same inputs.

    pixi run -e nvidia mojo run -I . tests/deep_agents/supermimic/test_supermimic_obj_gpu.mojo

The CPU path is gated against a torch transcription (`test_supermimic_obj`);
this gate shows the device runs the SAME computation: the object's per-lane
phases are one body for both targets (`object_lanes.mojo`), so what it checks
is that they lower and run on the device, and that the two networks and the
reductions agree there. `-D SM_MJLAB=1`: on mjlab's G1.

  [1] the joint k-step loss with BOTH networks perturbed off their zero
      start: every loss term, every gradient of the world model and of the
      object network
  [2] the same with the object network as `make` leaves it (last layer
      zero): the device's robot terms are the device's own
      `sm_wm_window_loss`

Vectors compare by cosine >= 0.99999 and max err <= 1e-4 of the largest, as
`test_supermimic_gpu`.
"""

from std.math import sqrt
from std.random import random_float64, seed
from std.sys import get_defined_int

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.object_model import SMObjectModel, SMObjWindowBatch, sm_obj_window_loss
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1Robot, SMDims
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWorldModel, sm_wm_window_loss
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import TensorImpl


comptime TEST_MJLAB = get_defined_int["SM_MJLAB", 0]() != 0
comptime R = G1MjlabRobot if TEST_MJLAB else G1Robot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime H = 64
comptime TH = 32
comptime HO = 48
comptime B = 48
comptime K = 4


struct Tally:
    var ok: Int
    var total: Int

    def __init__(out self):
        self.ok = 0
        self.total = 0

    def check(mut self, cond: Bool, msg: String):
        self.total += 1
        if cond:
            self.ok += 1
            print("  ok:", msg)
        else:
            print("  FAIL:", msg)


def _u(lo: Float64, hi: Float64) -> Scalar[DT]:
    return Scalar[DT](random_float64(lo, hi))


def _unit4(mut t: TensorImpl[DT], off: Int):
    var q0 = random_float64(-1, 1)
    var q1 = random_float64(-0.6, 0.6)
    var q2 = random_float64(-0.6, 0.6)
    var q3 = random_float64(-1, 1)
    var nq = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    t.data[off + 0] = Scalar[DT](q0 / nq)
    t.data[off + 1] = Scalar[DT](q1 / nq)
    t.data[off + 2] = Scalar[DT](q2 / nq)
    t.data[off + 3] = Scalar[DT](q3 / nq)


def _rand_state(mut t: TensorImpl[DT], off: Int):
    """A plausible robot state: a unit quaternion, joints and velocities in range."""
    t.data[off + 0] = _u(-1, 1)
    t.data[off + 1] = _u(-1, 1)
    t.data[off + 2] = _u(0.5, 0.9)
    _unit4(t, off + 3)
    for j in range(n):
        t.data[off + 7 + j] = _u(-0.8, 0.8)
        t.data[off + D.NQ + 6 + j] = _u(-3, 3)
    for k in range(6):
        t.data[off + D.NQ + k] = _u(-1, 1)


def _rand_object(mut t: TensorImpl[DT], off: Int):
    """An object within a metre of the origin, any attitude, moving."""
    t.data[off + 0] = _u(-1, 1)
    t.data[off + 1] = _u(-1, 1)
    t.data[off + 2] = _u(0.2, 1.2)
    _unit4(t, off + 3)
    for k in range(3):
        t.data[off + 7 + k] = _u(-1, 1)
        t.data[off + 10 + k] = _u(-3, 3)


def _compare(name: String, a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]], mut tl: Tally):
    """Flattened cosine and max abs err relative to the largest |a|."""
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    var big = 0.0
    var err = 0.0
    var cnt = 0
    for i in range(len(a)):
        for k in range(len(a[i])):
            var x = Float64(a[i][k])
            var y = Float64(b[i][k])
            dot += x * y
            na += x * x
            nb += y * y
            big = max(big, abs(x))
            err = max(err, abs(x - y))
            cnt += 1
    var cos = dot / sqrt(max(na * nb, 1e-300))
    tl.check(cos >= 0.99999 and err <= 1e-4 * big and big > 1e-6,
        name + ": " + String(cnt) + " values, cosine " + String(cos) + ", max err " + String(err / max(big, 1e-300)) + " of the largest (" + String(big) + ")")


def _close(name: String, g: Float64, c: Float64, mut tl: Tally):
    var rel = abs(g - c) / max(abs(c), 1e-12)
    tl.check(rel < 1e-5, name + " " + String(g) + " (cpu " + String(c) + "), rel " + String(rel))


def _robot_batch() -> SMWindowBatch[R]:
    var w = SMWindowBatch[R](K, B)
    for t in range(K + 1):
        for b in range(B):
            _rand_state(w.states, (t * B + b) * D.S)
    for i in range(K * B * n):
        w.a_exec.data[i] = _u(-1, 1)
    for b in range(B):
        w.xi.data[b * D.X] = _u(0.3, 1.2)
        for k in range(1, D.X):
            w.xi.data[b * D.X + k] = _u(-0.02, 0.02)
    for i in range(K * B):
        w.done.data[i] = Scalar[DT](1) if random_float64() < 0.2 else Scalar[DT](0)
    for i in range(K * B):
        var o = i * 7
        w.ref_root.data[o + 2] = _u(0.6, 0.8)
        w.ref_root.data[o + 3] = 1
    return w^


def _object_batch() -> SMObjWindowBatch:
    var wo = SMObjWindowBatch(K, B)
    for t in range(K + 1):
        for b in range(B):
            _rand_object(wo.states, (t * B + b) * SM_OBJ)
    for b in range(B):
        wo.ctx.data[b * SM_OBJ_CTX] = _u(0.1, 2.0)
        wo.ctx.data[b * SM_OBJ_CTX + 1] = _u(0.5, 1.3)
    return wo^


def test_joint_loss(ctx: DeviceContext, mut tl: Tally) raises:
    print("[1] the joint k-step loss and gradients, both networks off their zero start")
    seed(11)
    var wc = SMWorldModel[R, H, TH].make[Kaiming]()
    var oc = SMObjectModel[R, HO].make[Kaiming]()
    # both last layers start at zero: perturb them so the chains move
    var snap = sm_snapshot["cpu"](wc, None)
    for i in range(len(snap)):
        for k in range(len(snap[i])):
            snap[i][k] += Scalar[DT](random_float64(-0.02, 0.02))
    sm_restore["cpu"](wc, snap.copy(), None)
    var osnap = sm_snapshot["cpu"](oc, None)
    for i in range(len(osnap)):
        for k in range(len(osnap[i])):
            osnap[i][k] += Scalar[DT](random_float64(-0.02, 0.02))
    sm_restore["cpu"](oc, osnap.copy(), None)
    var wg = SMWorldModel[R, H, TH].make[Kaiming, "gpu"](ctx)
    sm_restore["gpu"](wg, snap.copy(), ctx)
    var og = SMObjectModel[R, HO].make[Kaiming, "gpu"](ctx)
    sm_restore["gpu"](og, osnap.copy(), ctx)

    var w = _robot_batch()
    var wo = _object_batch()
    wc.zero_grad["cpu"](None)
    oc.zero_grad["cpu"](None)
    var lc = sm_obj_window_loss[R, H, TH, HO, B](wc, oc, w, wo, True, w_obj=0.7)
    w.to_device(ctx)
    wo.to_device(ctx)
    wg.zero_grad["gpu"](ctx)
    og.zero_grad["gpu"](ctx)
    var lg = sm_obj_window_loss[R, H, TH, HO, B, "gpu"](wg, og, w, wo, True, w_obj=0.7, ctx=ctx)
    _close("loss", lg.loss, lc.loss, tl)
    _close("robot loss", lg.state_loss, lc.state_loss, tl)
    _close("object loss", lg.obj_loss, lc.obj_loss, tl)
    _close("BCE", lg.term_bce, lc.term_bce, tl)
    _close("object k-step error", lg.obj_nse_kstep, lc.obj_nse_kstep, tl)
    _compare("world-model gradients", sm_snapshot["cpu"](wc, None, grads=True), sm_snapshot["gpu"](wg, ctx, grads=True), tl)
    _compare("object-network gradients", sm_snapshot["cpu"](oc, None, grads=True), sm_snapshot["gpu"](og, ctx, grads=True), tl)


def test_zero_start(ctx: DeviceContext, mut tl: Tally) raises:
    print("[2] the object network as made: the device's robot terms are sm_wm_window_loss's")
    seed(12)
    var wc = SMWorldModel[R, H, TH].make[Kaiming]()
    var snap = sm_snapshot["cpu"](wc, None)
    for i in range(len(snap)):
        for k in range(len(snap[i])):
            snap[i][k] += Scalar[DT](random_float64(-0.02, 0.02))
    var wg = SMWorldModel[R, H, TH].make[Kaiming, "gpu"](ctx)
    sm_restore["gpu"](wg, snap.copy(), ctx)
    var og = SMObjectModel[R, HO].make[Kaiming, "gpu"](ctx)
    var w = _robot_batch()
    var wo = _object_batch()
    w.to_device(ctx)
    wo.to_device(ctx)
    wg.zero_grad["gpu"](ctx)
    var plain = sm_wm_window_loss[R, H, TH, B, "gpu"](wg, w, True, ctx=ctx)
    var gp = sm_snapshot["gpu"](wg, ctx, grads=True)
    wg.zero_grad["gpu"](ctx)
    og.zero_grad["gpu"](ctx)
    var joint = sm_obj_window_loss[R, H, TH, HO, B, "gpu"](wg, og, w, wo, True, ctx=ctx)
    _close("robot loss", joint.state_loss, plain.state_loss, tl)
    _close("BCE", joint.term_bce, plain.term_bce, tl)
    _compare("world-model gradients", gp, sm_snapshot["gpu"](wg, ctx, grads=True), tl)
    tl.check(joint.obj_loss > 0.0, "the object loss is the constant-velocity chain's: " + String(joint.obj_loss))


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    test_joint_loss(ctx, tl)
    test_zero_start(ctx, tl)
    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_supermimic_obj_gpu: " + String(tl.total - tl.ok) + " failed")
