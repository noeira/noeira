"""SuperMimic's GPU path against its CPU path, same weights, same inputs.

The CPU path is gated against the reproduction's torch code
(test_supermimic_wm / _imag); this gate shows the device runs the SAME
computation: the per-lane phases are one body (`lanes.mojo`), so what it
checks is that they lower and run on the device, and that the nn modules,
optimiser and reductions agree there.

  [1] world-model k-step loss: loss terms, every gradient
  [2] the imagined rollout (H = 8, obs noise on): J, every output, the
      policy gradient
  [3] one policy update (actor step + critic fit, same minibatch draws):
      metrics, values / targets, the policy's and critic's weight deltas
  [4] one world-model update (clip + Adam): loss, the weight deltas
  [5] the action step on host tensors through the device
  [6] the checkpoint across targets: a GPU trainer's file loaded into a
      fresh CPU trainer (and a CPU file into a fresh GPU trainer) restores
      every weight bit-identically and the loop's counters

Vectors compare by cosine >= 0.99999 and max err <= 1e-4 of the largest.

Run: pixi run -e apple mojo run -I . tests/deep_agents/supermimic/test_supermimic_gpu.mojo
"""

from std.math import sqrt
from std.random import random_float64, seed

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.blocks import sm_random_perm
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.imagination import SMImagBatch, sm_imagine
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from std.sys import get_defined_int

from noeira.deep_agents.supermimic.robot import G1Robot, G1Sim, SMDims, TrackingSim, G1MjlabRobot
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWorldModel, sm_wm_window_loss
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import TensorImpl


# `-D SM_MJLAB=1`: the same gate on mjlab's G1.
comptime TEST_MJLAB = get_defined_int["SM_MJLAB", 0]() != 0
comptime R = G1MjlabRobot if TEST_MJLAB else G1Robot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NB = 16          # policy batch
comptime HZ = 8           # horizon
comptime TC = SMConfig["cpu", R, 32, 16, 24, 20, 16, 12, 28, 22, 18, 14, 16, 12, 8, 32, NB, HZ]
comptime TG = SMConfig["gpu", R, 32, 16, 24, 20, 16, 12, 28, 22, 18, 14, 16, 12, 8, 32, NB, HZ]


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


def _rand_state(mut t: TensorImpl[DT], off: Int):
    """A plausible state: a unit quaternion, joints and velocities in range."""
    var q0 = random_float64(-1, 1)
    var q1 = random_float64(-0.3, 0.3)
    var q2 = random_float64(-0.3, 0.3)
    var q3 = random_float64(-1, 1)
    var nq = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    t.data[off + 0] = _u(-1, 1)
    t.data[off + 1] = _u(-1, 1)
    t.data[off + 2] = _u(0.5, 0.9)
    t.data[off + 3] = Scalar[DT](q0 / nq)
    t.data[off + 4] = Scalar[DT](q1 / nq)
    t.data[off + 5] = Scalar[DT](q2 / nq)
    t.data[off + 6] = Scalar[DT](q3 / nq)
    for j in range(n):
        t.data[off + 7 + j] = _u(-0.8, 0.8)
        t.data[off + D.NQ + 6 + j] = _u(-3, 3)
    for k in range(6):
        t.data[off + D.NQ + k] = _u(-1, 1)


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
    tl.check(cos >= 0.99999 and err <= 1e-4 * big, name + ": " + String(cnt) + " values, cosine " + String(cos) + ", max err " + String(err / max(big, 1e-300)) + " of the largest")


def test_wm_loss(ctx: DeviceContext, mut tl: Tally) raises:
    print("[1] world-model k-step loss and gradients")
    comptime H = 64
    comptime TH = 32
    comptime B = 48
    comptime K = 4
    seed(7)
    var wc = SMWorldModel[R, H, TH].make[Kaiming]()
    # the dynamics' last layer starts at zero: perturb it so the chain moves
    var snap = sm_snapshot["cpu"](wc, None)
    for i in range(len(snap)):
        for k in range(len(snap[i])):
            snap[i][k] += Scalar[DT](random_float64(-0.02, 0.02))
    sm_restore["cpu"](wc, snap.copy(), None)
    var wg = SMWorldModel[R, H, TH].make[Kaiming, "gpu"](ctx)
    sm_restore["gpu"](wg, snap.copy(), ctx)

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
    var lc = sm_wm_window_loss[R, H, TH, B](wc, w, True)
    w.to_device(ctx)
    wg.zero_grad["gpu"](ctx)
    var lg = sm_wm_window_loss[R, H, TH, B, "gpu"](wg, w, True, ctx=ctx)
    var rel = abs(lg.loss - lc.loss) / abs(lc.loss)
    tl.check(rel < 1e-5, "loss " + String(lg.loss) + " (cpu " + String(lc.loss) + "), rel " + String(rel))
    tl.check(
        abs(lg.term_bce - lc.term_bce) < 1e-5 * abs(lc.term_bce) + 1e-7 and abs(lg.nse_kstep - lc.nse_kstep) < 1e-5 * lc.nse_kstep,
        "BCE " + String(lg.term_bce) + " (cpu " + String(lc.term_bce) + "), k-step nse " + String(lg.nse_kstep) + " (cpu " + String(lc.nse_kstep) + ")",
    )
    _compare("gradients", sm_snapshot["cpu"](wc, None, grads=True), sm_snapshot["gpu"](wg, ctx, grads=True), tl)


def _rand_quat(mut t: TensorImpl[DT], off: Int):
    var q = List[Float64](length=4, fill=0.0)
    var nq = 0.0
    for k in range(4):
        q[k] = random_float64(-1, 1)
        nq += q[k] * q[k]
    for k in range(4):
        t.data[off + k] = Scalar[DT](q[k] / sqrt(nq))


def _imag_batch() -> SMImagBatch[R]:
    var ib = SMImagBatch[R](NB, HZ)
    for b in range(NB):
        _rand_state(ib.state, b * D.S)
        ib.xi.data[b * D.X] = _u(0.3, 1.2)
        for k in range(1, D.X):
            ib.xi.data[b * D.X + k] = _u(-0.02, 0.02)
    for i in range(NB * n):
        ib.prev_action.data[i] = _u(-0.5, 0.5)
    for i in range(D.HIST * NB * D.OBS):
        ib.hist0.data[i] = _u(-0.5, 0.5)
    for i in range((HZ + 1) * NB * D.GOAL):
        ib.goals.data[i] = _u(-1, 1)
    for r in range((HZ + 1) * NB):
        var o = r * D.REF
        for k in range(R.N_TRACKED + 1):
            var bo = o + k * 13
            ib.refs.data[bo + 0] = _u(-0.3, 0.3)
            ib.refs.data[bo + 1] = _u(-0.3, 0.3)
            ib.refs.data[bo + 2] = _u(0.2, 1.2)
            _rand_quat(ib.refs, bo + 3)
            if k < R.N_TRACKED:
                for c in range(6):
                    ib.refs.data[bo + 7 + c] = _u(-1, 1)
    return ib^


def _compare_steps(name: String, a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]], mut tl: Tally):
    """Optimiser steps: Adam moves a well-conditioned element by ~lr and an
    element whose gradient is near eps by an amount that is ill-conditioned
    in g (a 1e-7 change in g can change it by its own size). So: the elements
    that moved (|Δ| >= half the largest move) must agree to 1e-3 of the
    largest move; the rest are reported and bounded by the largest move."""
    var big = 0.0
    for i in range(len(a)):
        for k in range(len(a[i])):
            big = max(big, abs(Float64(a[i][k])))
    var em = 0.0
    var er = 0.0
    var nm = 0
    var nr = 0
    for i in range(len(a)):
        for k in range(len(a[i])):
            var x = Float64(a[i][k])
            var e = abs(x - Float64(b[i][k]))
            if abs(x) >= 0.5 * big:
                em = max(em, e)
                nm += 1
            else:
                er = max(er, e)
                nr += 1
    tl.check(em <= 1e-3 * big, name + ": " + String(nm) + " elements that moved agree to " + String(em / big) + " of the largest move (" + String(big) + ")")
    tl.check(er <= big, name + ": " + String(nr) + " that barely moved within " + String(er / big) + " of the largest move")


def _delta(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> List[List[Scalar[DT]]]:
    var d = List[List[Scalar[DT]]]()
    for i in range(len(a)):
        var l = List[Scalar[DT]](length=len(a[i]), fill=0)
        for k in range(len(a[i])):
            l[k] = b[i][k] - a[i][k]
        d.append(l^)
    return d^


def _tv(t: TensorImpl[DT]) -> List[Float64]:
    var l = List[Float64](length=t.n, fill=0.0)
    for i in range(t.n):
        l[i] = Float64(t.data[i])
    return l^


def _close(name: String, a: Float64, b: Float64, rel: Float64, mut tl: Tally):
    var e = abs(a - b) / max(abs(a), 1e-12)
    tl.check(e <= rel, name + " " + String(b) + " (cpu " + String(a) + "), rel " + String(e))


def _vec(name: String, a: List[Float64], b: List[Float64], mut tl: Tally):
    var la = List[List[Scalar[DT]]]()
    var lb = List[List[Scalar[DT]]]()
    var x = List[Scalar[DT]](length=len(a), fill=0)
    var y = List[Scalar[DT]](length=len(b), fill=0)
    for i in range(len(a)):
        x[i] = Scalar[DT](a[i])
        y[i] = Scalar[DT](b[i])
    la.append(x^)
    lb.append(y^)
    _compare(name, la, lb, tl)


def _cpu_trainer() raises -> SuperMimicTrainer[TC]:
    """A CPU trainer, the dynamics' zero last layer perturbed so the rollout
    moves."""
    seed(11)
    var tc = SuperMimicTrainer[TC].for_robot(R.SIM.create())
    var w = sm_snapshot["cpu"](tc.wm, None)
    for i in range(len(w)):
        for k in range(len(w[i])):
            w[i][k] += Scalar[DT](random_float64(-0.02, 0.02))
    sm_restore["cpu"](tc.wm, w^, None)
    return tc^


def _gpu_twin(mut tc: SuperMimicTrainer[TC], ctx: DeviceContext) raises -> SuperMimicTrainer[TG]:
    """A GPU trainer holding the CPU one's weights."""
    var tg = SuperMimicTrainer[TG].for_robot(R.SIM.create(), ctx)
    sm_restore["gpu"](tg.wm, sm_snapshot["cpu"](tc.wm, None), ctx)
    sm_restore["gpu"](tg.pol, sm_snapshot["cpu"](tc.pol, None), ctx)
    sm_restore["gpu"](tg.critic, sm_snapshot["cpu"](tc.critic, None), ctx)
    return tg^


def test_imagine(ctx: DeviceContext, mut tl: Tally) raises:
    print("[2] the imagined rollout and its policy gradient (H = 8, noise on)")
    var tc = _cpu_trainer()
    var tg = _gpu_twin(tc, ctx)
    seed(12)
    var ib = _imag_batch()
    tc.pol.zero_grad["cpu"](None)
    var oc = sm_imagine[TC, NB](tc.wm, tc.pol, tc.critic, tc.tables, ib, 0.99, True, UInt32(1234))
    ib.to_device(ctx)
    tg.pol.zero_grad["gpu"](ctx)
    var og = sm_imagine[TG, NB, "gpu"](tg.wm, tg.pol, tg.critic, tg.tables, ib, 0.99, True, UInt32(1234), ctx)
    og.to_host(ctx)
    _close("J", oc.j, og.j, 1e-5, tl)
    _close("undiscounted return", oc.return_undisc, og.return_undisc, 1e-5, tl)
    _vec("states", _tv(oc.states), _tv(og.states), tl)
    _vec("actions", _tv(oc.actions), _tv(og.actions), tl)
    _vec("rewards", _tv(oc.rewards), _tv(og.rewards), tl)
    _vec("dhat", _tv(oc.dhat), _tv(og.dhat), tl)
    _vec("survival", _tv(oc.survival), _tv(og.survival), tl)
    _vec("e_root", _tv(oc.e_root), _tv(og.e_root), tl)
    _vec("value_h", _tv(oc.value_h), _tv(og.value_h), tl)
    _compare("policy gradient", sm_snapshot["cpu"](tc.pol, None, grads=True), sm_snapshot["gpu"](tg.pol, ctx, grads=True), tl)


def test_policy_update(ctx: DeviceContext, mut tl: Tally) raises:
    print("[3] one policy update: actor step + critic fit")
    var tc = _cpu_trainer()
    var tg = _gpu_twin(tc, ctx)
    seed(13)
    var ib = _imag_batch()
    var perms = List[Int]()
    for _ in range(5):
        var q = sm_random_perm(HZ * NB)
        for i in range(len(q)):
            perms.append(q[i])
    var pc0 = sm_snapshot["cpu"](tc.pol, None)
    var cc0 = sm_snapshot["cpu"](tc.critic, None)
    var vc = List[Float64]()
    var rc = List[Float64]()
    var vg = List[Float64]()
    var rg = List[Float64]()
    var mc = tc.policy_update[NB](ib, perms, vc, rc)
    var mg = tg.policy_update[NB](ib, perms, vg, rg)
    _close("objective", mc.imag_objective, mg.imag_objective, 1e-5, tl)
    _close("policy grad norm", mc.policy_grad_norm, mg.policy_grad_norm, 1e-4, tl)
    _close("last critic loss", mc.critic_loss, mg.critic_loss, 1e-3, tl)
    _vec("values", vc, vg, tl)
    _vec("TD(lambda) targets", rc, rg, tl)
    _compare_steps("policy step", _delta(pc0, sm_snapshot["cpu"](tc.pol, None)), _delta(pc0, sm_snapshot["gpu"](tg.pol, ctx)), tl)
    _compare_steps("critic 20 steps", _delta(cc0, sm_snapshot["cpu"](tc.critic, None)), _delta(cc0, sm_snapshot["gpu"](tg.critic, ctx)), tl)


def test_wm_update(ctx: DeviceContext, mut tl: Tally) raises:
    print("[4] one world-model update (clip + Adam)")
    var tc = _cpu_trainer()
    var tg = _gpu_twin(tc, ctx)
    seed(14)
    comptime B = 32
    comptime K = 4
    var w = SMWindowBatch[R](K, B)
    for t in range(K + 1):
        for b in range(B):
            _rand_state(w.states, (t * B + b) * D.S)
    for i in range(K * B * n):
        w.a_exec.data[i] = _u(-1, 1)
    for b in range(B):
        w.xi.data[b * D.X] = _u(0.3, 1.2)
    for i in range(K * B):
        w.done.data[i] = Scalar[DT](1) if random_float64() < 0.2 else Scalar[DT](0)
        w.ref_root.data[i * 7 + 2] = _u(0.6, 0.8)
        w.ref_root.data[i * 7 + 3] = 1
    var w0 = sm_snapshot["cpu"](tc.wm, None)
    var lc = tc.wm_update[B](w)
    var lg = tg.wm_update[B](w)
    _close("loss", lc.loss, lg.loss, 1e-5, tl)
    _compare_steps("weight deltas", _delta(w0, sm_snapshot["cpu"](tc.wm, None)), _delta(w0, sm_snapshot["gpu"](tg.wm, ctx)), tl)


def test_act(ctx: DeviceContext, mut tl: Tally) raises:
    print("[5] the action step")
    var tc = _cpu_trainer()
    var tg = _gpu_twin(tc, ctx)
    seed(15)
    comptime L = 8
    var hist = TensorImpl[DT].alloc(L * D.HIST * D.OBS)
    var goals = TensorImpl[DT].alloc(L * D.GOAL)
    for i in range(hist.n):
        hist.data[i] = _u(-1, 1)
    for i in range(goals.n):
        goals.data[i] = _u(-1, 1)
    var ac = tc.act[L](hist, goals)
    var ag = tg.act[L](hist, goals)
    var a = List[Float64]()
    var b = List[Float64]()
    for i in range(ac.n):
        a.append(Float64(ac.data[i]))
        b.append(Float64(ag.data[i]))
    _vec("actions", a, b, tl)


def _same(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Int:
    """Mismatching elements (bitwise equality on float32)."""
    var bad = 0
    if len(a) != len(b):
        return -1
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return -1
        for k in range(len(a[i])):
            bad += 1 if a[i][k] != b[i][k] else 0
    return bad


def test_checkpoint(ctx: DeviceContext, mut tl: Tally) raises:
    print("[6] the checkpoint across targets")
    var tc = _cpu_trainer()
    var tg = _gpu_twin(tc, ctx)
    # move the GPU weights away from the CPU's: one world-model update
    seed(16)
    comptime B = 32
    comptime K = 4
    var w = SMWindowBatch[R](K, B)
    for t in range(K + 1):
        for b in range(B):
            _rand_state(w.states, (t * B + b) * D.S)
    for i in range(K * B):
        w.ref_root.data[i * 7 + 3] = 1
    _ = tg.wm_update[B](w)
    var path = String("/tmp/sm_ckpt_gpu_test.ckpt")
    tg.save_state(path, 123, 45)
    var fresh = SuperMimicTrainer[TC].for_robot(R.SIM.create())
    var ctr = fresh.load_state(path)
    var bad = _same(sm_snapshot["gpu"](tg.wm, ctx), sm_snapshot["cpu"](fresh.wm, None))
    bad += _same(sm_snapshot["gpu"](tg.pol, ctx), sm_snapshot["cpu"](fresh.pol, None))
    bad += _same(sm_snapshot["gpu"](tg.critic, ctx), sm_snapshot["cpu"](fresh.critic, None))
    tl.check(bad == 0 and ctr[0] == 123 and ctr[1] == 45, "GPU file -> CPU trainer: " + String(bad) + " mismatching weights, counters " + String(ctr[0]) + " / " + String(ctr[1]))
    tc.save_state(path, 7, 3)
    var fg = SuperMimicTrainer[TG].for_robot(R.SIM.create(), ctx)
    var cg = fg.load_state(path)
    var bad2 = _same(sm_snapshot["cpu"](tc.wm, None), sm_snapshot["gpu"](fg.wm, ctx))
    bad2 += _same(sm_snapshot["cpu"](tc.pol, None), sm_snapshot["gpu"](fg.pol, ctx))
    bad2 += _same(sm_snapshot["cpu"](tc.critic, None), sm_snapshot["gpu"](fg.critic, ctx))
    tl.check(bad2 == 0 and cg[0] == 7 and cg[1] == 3, "CPU file -> GPU trainer: " + String(bad2) + " mismatching weights, counters " + String(cg[0]) + " / " + String(cg[1]))


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    test_wm_loss(ctx, tl)
    test_imagine(ctx, tl)
    test_policy_update(ctx, tl)
    test_wm_update(ctx, tl)
    test_act(ctx, tl)
    test_checkpoint(ctx, tl)
    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_supermimic_gpu: " + String(tl.total - tl.ok) + " failed")
