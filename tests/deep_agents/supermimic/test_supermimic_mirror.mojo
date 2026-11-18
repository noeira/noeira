"""The left-right mirror of world-model windows (`lanes.sm_mirror_windows`,
`SuperMimicTrainer.set_mirror`), on mjlab's G1.

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_mirror.mojo

Known answers and the simulator itself as the reference: no dump.

WHAT EACH SECTION CAN CATCH
===========================
[1] the joint table built from the robot's names: left and right hips are
    partners, pitch joints keep their sign, roll and yaw joints change it,
    the waist's three joints are their own partners; a robot that declares
    no names is refused.
[2] the mirror applied twice is the identity, to the bit, on a random window
    batch; lanes beyond the mirrored count are untouched.
[3] THE PHYSICS: one control step from a mirrored state under the mirrored
    action lands on the mirror of the step from the original, in the air
    and standing on the floor. This is what makes a mirrored window valid
    training data, and it would not hold with a wrong sign or partner
    anywhere in the table (a deliberately wrong table is shown to miss).
[4] in the trainer: with a share of 0.5 on a batch of 8, a world-model update
    leaves the first 4 windows mirrored and the last 4 as sampled.
"""

from std.math import abs
from std.random import random_float64, seed

from noeira.deep_agents.supermimic.agent import sm_agent
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.lanes import sm_mirror_windows, tp
from noeira.deep_agents.supermimic.prompt_eval import sm_stand_library
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1Robot, SMDims
from noeira.deep_agents.supermimic.world_model import SMWindowBatch
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime n = R.NDOF
comptime NQ = R.NQ
comptime NV = R.NV
comptime C = SMConfig["cpu", R, 32, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 8, 8, 8, 8]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _mirror_one(mut tab: TensorImpl[DT], s: List[Float64], a: List[Float64]) raises -> Tuple[List[Float64], List[Float64]]:
    """The mirror of one state and one action, through the code under test (a
    1-lane, 1-step window)."""
    var w = SMWindowBatch[R](1, 1)
    for i in range(D.S):
        w.states.data[i] = Scalar[DT](s[i])
    for j in range(n):
        w.a_exec.data[j] = Scalar[DT](a[j])
    w.ref_root.data[3] = 1
    sm_mirror_windows["cpu", R](None, 1, 1, 1, tp["cpu"](w.states), tp["cpu"](w.a_exec), tp["cpu"](w.xi), tp["cpu"](w.ref_root), tp["cpu"](tab))
    var so = List[Float64](length=D.S, fill=0.0)
    var ao = List[Float64](length=n, fill=0.0)
    for i in range(D.S):
        so[i] = Float64(w.states.data[i])
    for j in range(n):
        ao[j] = Float64(w.a_exec.data[j])
    return (so^, ao^)


def _step(mut sim: R.SIM, s: List[Float64], a: List[Float64]) -> List[Float64]:
    """One control step of the CPU simulator from state `s` (qpos, then qvel)."""
    var qpos = List[Float64](length=NQ, fill=0.0)
    var qvel = List[Float64](length=NV, fill=0.0)
    for i in range(NQ):
        qpos[i] = s[i]
    for i in range(NV):
        qvel[i] = s[NQ + i]
    sim.set_state(qpos, qvel)
    var bias = List[Float64](length=n, fill=0.0)
    sim.control_step(a, bias)
    var o = List[Float64](length=D.S, fill=0.0)
    for i in range(NQ):
        o[i] = sim.qpos(i)
    for i in range(NV):
        o[NQ + i] = sim.qvel(i)
    return o^


def _state(root_z: Float64, tilt: Float64) -> List[Float64]:
    """An asymmetric state: the default pose with every joint moved, the root
    at `root_z` turned by `tilt` rad about a skew axis, with velocities."""
    var s = List[Float64](length=D.S, fill=0.0)
    s[0] = 0.1
    s[1] = -0.2
    s[2] = root_z
    var h = 0.5 * tilt
    s[3] = 1.0 - 0.5 * h * h
    s[4] = 0.6 * h
    s[5] = 0.3 * h
    s[6] = -0.74 * h
    for j in range(n):
        s[7 + j] = R.default_joint_pos(j) + random_float64(-0.15, 0.15)
        s[NQ + 6 + j] = random_float64(-0.5, 0.5)
    for k in range(6):
        s[NQ + k] = random_float64(-0.3, 0.3)
    return s^


def _mismatch(a: List[Float64], b: List[Float64]) -> Tuple[Float64, Float64]:
    """Largest difference over the positions (qpos) and over the velocities."""
    var mp = 0.0
    var mv = 0.0
    for i in range(NQ):
        # q and −q are one rotation: compare the root quaternion up to its sign
        var d = abs(a[i] - b[i])
        if i >= 3 and i < 7:
            var dot = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]
            d = abs(a[i] - b[i]) if dot >= 0.0 else abs(a[i] + b[i])
        mp = max(mp, d)
    for i in range(NV):
        mv = max(mv, abs(a[NQ + i] - b[NQ + i]))
    return (mp, mv)


def main() raises:
    var t = Tally()
    seed(41)
    var sim0 = R.SIM.create()
    var agent = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0, utd=1.0, warmup=14)

    print("\n[1] the joint table")
    agent.trainer.set_mirror(0.5)
    var tb = List[Float64](length=2 * n, fill=0.0)
    for i in range(2 * n):
        tb[i] = Float64(agent.trainer.mirror_tab.data[i])
    t.truth(tb[0] == 6.0 and tb[6] == 0.0 and tb[n + 0] == 1.0, "left_hip_pitch (0) and right_hip_pitch (6) are partners, sign +")
    t.truth(tb[1] == 7.0 and tb[n + 1] == -1.0 and tb[2] == 8.0 and tb[n + 2] == -1.0, "hip roll and hip yaw change sign")
    t.truth(tb[3] == 9.0 and tb[n + 3] == 1.0, "the knees keep theirs")
    t.truth(
        tb[12] == 12.0 and tb[n + 12] == -1.0 and tb[13] == 13.0 and tb[n + 13] == -1.0 and tb[14] == 14.0 and tb[n + 14] == 1.0,
        "the waist's yaw, roll and pitch are their own partners with signs -, -, +",
    )
    t.truth(tb[15] == 22.0 and tb[22] == 15.0 and tb[21] == 28.0 and tb[n + 21] == -1.0, "the arms pair by 7 (left shoulder pitch 15 with right 22, wrist yaw 21 with 28, sign -)")
    t.truth(G1Robot.joint_name(0).byte_length() == 0 and R.joint_name(0) == "left_hip_pitch_joint", "a robot without declared joint names reads \"\" (the mirror refuses it)")

    print("\n[2] twice is the identity")
    comptime K = 3
    comptime B = 6
    var w = SMWindowBatch[R](K, B)
    for i in range(w.states.n):
        w.states.data[i] = Scalar[DT](random_float64(-1.0, 1.0))
    for i in range(w.a_exec.n):
        w.a_exec.data[i] = Scalar[DT](random_float64(-1.0, 1.0))
    for i in range(w.xi.n):
        w.xi.data[i] = Scalar[DT](random_float64(-1.0, 1.0))
    for i in range(w.ref_root.n):
        w.ref_root.data[i] = Scalar[DT](random_float64(-1.0, 1.0))
    var s0 = w.states.data.copy()
    var a0 = w.a_exec.data.copy()
    var x0 = w.xi.data.copy()
    var r0 = w.ref_root.data.copy()
    sm_mirror_windows["cpu", R](None, 4, K, B, tp["cpu"](w.states), tp["cpu"](w.a_exec), tp["cpu"](w.xi), tp["cpu"](w.ref_root), tp["cpu"](agent.trainer.mirror_tab))
    var changed = 0
    var beyond = 0
    for tt in range(K + 1):
        for l in range(B):
            for i in range(D.S):
                var same = w.states.data[(tt * B + l) * D.S + i] == s0[(tt * B + l) * D.S + i]
                if l < 4 and not same:
                    changed += 1
                if l >= 4 and not same:
                    beyond += 1
    for l in range(4, B):
        for i in range(D.X):
            beyond += 0 if w.xi.data[l * D.X + i] == x0[l * D.X + i] else 1
    t.truth(changed > 4 * (K + 1) * 30 and beyond == 0, "one pass changes the first 4 lanes (" + String(changed) + " state entries) and none of the other 2")
    sm_mirror_windows["cpu", R](None, 4, K, B, tp["cpu"](w.states), tp["cpu"](w.a_exec), tp["cpu"](w.xi), tp["cpu"](w.ref_root), tp["cpu"](agent.trainer.mirror_tab))
    var back = 0
    for i in range(w.states.n):
        back += 0 if w.states.data[i] == s0[i] else 1
    for i in range(w.a_exec.n):
        back += 0 if w.a_exec.data[i] == a0[i] else 1
    for i in range(w.xi.n):
        back += 0 if w.xi.data[i] == x0[i] else 1
    for i in range(w.ref_root.n):
        back += 0 if w.ref_root.data[i] == r0[i] else 1
    t.truth(back == 0, "a second pass restores states, actions, sim context and reference roots to the bit")

    print("\n[3] the physics: step(mirror(s), mirror(a)) = mirror(step(s, a))")
    var sim = R.SIM.create()
    var wrong = TensorImpl[DT].alloc(2 * n)
    for i in range(2 * n):
        wrong.data[i] = agent.trainer.mirror_tab.data[i]
    wrong.data[n + 1] = 1                     # a deliberately wrong table: hip roll keeps its sign
    wrong.data[n + 7] = 1
    for scene in range(2):
        var root_z = 1.6 if scene == 0 else sim0.qpos(2)
        var label = String("in the air") if scene == 0 else String("standing on the floor")
        var worst_p = 0.0
        var worst_v = 0.0
        var worst_wrong = 0.0
        var move = 0.0
        for _ in range(4):
            var s = _state(root_z, 0.3 if scene == 0 else 0.05)
            var a = List[Float64](length=n, fill=0.0)
            for j in range(n):
                a[j] = random_float64(-0.5, 0.5)
            var s1 = _step(sim, s, a)
            var m = _mirror_one(agent.trainer.mirror_tab, s, a)
            var s1m = _step(sim, m[0], m[1])
            var none_a = List[Float64](length=n, fill=0.0)
            var want = _mirror_one(agent.trainer.mirror_tab, s1, none_a)
            var mm = _mismatch(s1m, want[0])
            worst_p = max(worst_p, mm[0])
            worst_v = max(worst_v, mm[1])
            for i in range(NV):
                move = max(move, abs(s1[NQ + i] - s[NQ + i]))
            var mw = _mirror_one(wrong, s, a)
            var s1w = _step(sim, mw[0], mw[1])
            var wantw = _mirror_one(wrong, s1, none_a)
            worst_wrong = max(worst_wrong, _mismatch(s1w, wantw[0])[1])
        print("   ", label, ": positions differ by", worst_p, "| velocities by", worst_v, "(the step changes velocities by up to", move, ") | with a wrong hip-roll sign:", worst_wrong)
        t.truth(worst_v <= 0.02 * move and worst_p <= 1e-3, label + ": the mirrored step is the mirror of the step, to " + String(worst_v / move) + " of the step's velocity change")
        t.truth(worst_wrong > 10.0 * max(worst_v, 1e-9) and worst_wrong > 0.05 * move, label + ": a wrong sign in the table misses by " + String(worst_wrong / move) + " of it")

    print("\n[4] in the trainer")
    agent.train(16, 0)
    var wb = agent.window_batch()
    var before = wb.states.data.copy()
    var l = agent.trainer.wm_update[8](wb)
    var mirrored = 0
    var kept = 0
    comptime KW = C.DEF_WM_K
    for lane in range(8):
        var diff = 0
        for tt in range(KW + 1):
            for i in range(D.S):
                diff += 0 if wb.states.data[(tt * 8 + lane) * D.S + i] == before[(tt * 8 + lane) * D.S + i] else 1
        if lane < 4 and diff > 0:
            mirrored += 1
        if lane >= 4 and diff == 0:
            kept += 1
    t.truth(mirrored == 4 and kept == 4, "a share of 0.5 mirrors the first 4 of 8 windows and keeps the other 4 (" + String(mirrored) + ", " + String(kept) + ")")
    # lane 0's y positions flipped at every step
    var yflip = True
    for tt in range(KW + 1):
        yflip = yflip and wb.states.data[(tt * 8) * D.S + 1] == -before[(tt * 8) * D.S + 1]
    t.truth(yflip and l.loss == l.loss, "the mirrored windows have their root's y negated, and the update's loss is finite (" + String(l.loss) + ")")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_mirror: " + String(t.fails) + " failed")
