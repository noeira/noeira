"""SuperMimic's fast-training options (`fast.mojo`, `trainer.mojo`): the lr
ramp, the reference skip, the averaged policy and its checkpoint.

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_fast.mojo

Answers known by construction, no reference implementation. CPU, tiny widths.

WHAT EACH SECTION CAN CATCH
===========================
[1] the lr ramp (`SMFastState`): the multiplier holds until the trigger, the
    trigger fires on the training return or at the deadline and only at a
    test (every 50 updates), then the multiplier is linear to 1.
[2] the reference skip in `act`: it adds exactly `(q_ref − q0) / scale`, so
    the skip alone commands the target pose `q0 + scale·a = q_ref`; a robot
    that declares no action scale reads 0.
[3] the reference skip in imagination: the policy gradient with the skip on
    equals a central finite difference of −J/H on three decoder weights (the
    reverse pass is unchanged because the skip is data); the gradient
    WITHOUT the skip does not match that difference (the check has power).
[4] the averaged policy: the first update copies the live policy, the second
    is `d·avg + (1 − d)·live`, and it is not the live policy.
[5] the checkpoint: the file holds the AVERAGE in the policy's place (a
    plain reader deploys it), `.live` restores the live policy on a resume,
    and the reference skip follows the file.
[6] the loop: the world model takes `wm_batches` steps per update, the
    policy's lr is the base × the multiplier while the others keep the base,
    and the deadline starts the ramp at update 50.
[7] analytic termination: the logit is 0 on a threshold and −1 at 95 % of
    it, its gradient goes to the larger ratio; in a rollout started 0.2 m
    above the reference (80 % of the height threshold) the policy gradient
    equals a finite difference, and the learned head's gradient does not.
[8] the value ceiling: J with the ceiling is J without it plus
    γᴴ·mean(c_H·(min(V, cap) − V)), with some lanes capped and some not; the
    policy gradient equals a finite difference; the critic's values in an
    update stay under the ceiling when it is on and exceed it when it is off.
[9] the drift weight's guard: a checkpoint written under another weight is
    refused.
"""

from std.math import abs

from noeira.deep_agents.supermimic.agent import SuperMimicAgent, sm_agent
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.fast import SMFastOpts, SMFastState
from noeira.deep_agents.supermimic.imagination import SMImagBatch, sm_imagine
from noeira.deep_agents.supermimic.lanes import sm_cap, sm_cap_back, sm_term_analytic, sm_term_analytic_back, tp
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from noeira.deep_agents.supermimic.prompt_eval import sm_stand_library
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1Robot, SMDims
from noeira.deep_agents.supermimic.state_spec import SM_WM_DRIFT_MILLI
from noeira.nn.core.checkpoint import CheckpointScalars, save_params_multi
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime R = G1MjlabRobot
comptime D = SMDims[R]
comptime LANES = 8
comptime H = 8
comptime C = SMConfig["cpu", R, 32, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, LANES, 8, 8, H]


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


def _near(a: Float64, b: Float64, rel: Float64 = 1e-12) -> Bool:
    return abs(a - b) <= rel * max(abs(b), 1e-30)


def _max_diff(a: List[List[Scalar[DT]]], b: List[List[Scalar[DT]]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        for j in range(len(a[i])):
            m = max(m, abs(Float64(a[i][j]) - Float64(b[i][j])))
    return m


def _max_diff_t(mut a: TensorImpl[DT], mut b: TensorImpl[DT], n: Int) -> Float64:
    var m = 0.0
    for i in range(n):
        m = max(m, abs(Float64(a.data[i]) - Float64(b.data[i])))
    return m


def _hist() -> TensorImpl[DT]:
    """A fixed observation history for every lane."""
    var hist = TensorImpl[DT].alloc(LANES * D.HIST * D.OBS)
    for i in range(LANES * D.HIST * D.OBS):
        hist.data[i] = Scalar[DT](0.01 * Float64((i * 7) % 23) - 0.1)
    return hist^


def _goals() -> TensorImpl[DT]:
    """A fixed goal window for every lane."""
    var goals = TensorImpl[DT].alloc(LANES * D.GOAL)
    for i in range(LANES * D.GOAL):
        goals.data[i] = Scalar[DT](0.02 * Float64((i * 5) % 19) - 0.15)
    return goals^


comptime FD_EPS = 0.02


def _bias_grads(mut agent: SuperMimicAgent[C], mut ib: SMImagBatch[R], analytic: Bool, v_max: Float64) raises -> List[Float64]:
    """d(−J/H) / d(decoder's last bias) at joints 0, 7, 20, by the reverse scan."""
    agent.trainer.pol.zero_grad["cpu"](None)
    _ = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, True, 0, None, None, analytic, v_max)
    var g = sm_snapshot["cpu"](agent.trainer.pol, None, True)
    var last = len(g) - 1
    return [Float64(g[last][0]), Float64(g[last][7]), Float64(g[last][20])]


def _bias_fd(mut agent: SuperMimicAgent[C], mut ib: SMImagBatch[R], analytic: Bool, v_max: Float64) raises -> List[Float64]:
    """The same three derivatives by central finite differences of −J/H."""
    var w0 = sm_snapshot["cpu"](agent.trainer.pol, None)
    var last = len(w0) - 1
    var out = List[Float64]()
    for e in range(3):
        var k = [0, 7, 20][e]
        var wp = w0.copy()
        wp[last][k] += Scalar[DT](FD_EPS)
        sm_restore["cpu"](agent.trainer.pol, wp^, None)
        var op = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, False, 0, None, None, analytic, v_max)
        var wm_ = w0.copy()
        wm_[last][k] -= Scalar[DT](FD_EPS)
        sm_restore["cpu"](agent.trainer.pol, wm_^, None)
        var om = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, False, 0, None, None, analytic, v_max)
        out.append(-(op.j - om.j) / (2.0 * FD_EPS * Float64(H)))
    sm_restore["cpu"](agent.trainer.pol, w0.copy(), None)
    return out^


def _worst(a: List[Float64], b: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def _largest(a: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i]))
    return m


def main() raises:
    var t = Tally()

    print("\n[1] the lr ramp")
    var st = SMFastState(SMFastOpts(0.0, 0, 4.0, 0.75, 300, 1400, -1.0, False, False, 0.0))
    t.truth(st.pi_mult(0) == 4.0 and st.pi_mult(900) == 4.0, "x 4 until the trigger")
    st.note_episode(0.9)
    t.truth(not st.tick(49) and st.decay_at < 0, "no test between two multiples of 50, whatever the return")
    st.note_episode(0.5)
    t.truth(not st.tick(50) and _near(st.last_return, 0.7), "a test below 0.75 (mean 0.7) does not start it")
    st.note_episode(0.8)
    t.truth(st.tick(100) and st.decay_at == 100, "a test at 0.8 starts it, at update 100")
    t.truth(_near(st.pi_mult(100), 4.0) and _near(st.pi_mult(250), 2.5) and _near(st.pi_mult(400), 1.0) and _near(st.pi_mult(5000), 1.0), "linear 4 -> 1 over 300 updates, then 1")
    var dl = SMFastState(SMFastOpts(0.0, 0, 4.0, 0.75, 300, 1400, -1.0, False, False, 0.0))
    t.truth(not dl.tick(1350) and dl.tick(1400) and dl.decay_at == 1400, "without the return, the deadline starts it at update 1400")
    var off = SMFastState(SMFastOpts.off())
    t.truth(off.pi_mult(7) == 1.0 and not off.tick(50) and off.wm_batches(10) == 10, "off: x 1, no ramp, the config's world-model batches")
    t.truth(SMFastState(SMFastOpts.blink()).wm_batches(10) == 5, "the preset: 5 world-model batches")

    # one agent for [2]–[5]: 8 lanes on a 2-clip stand library, one update per step after 14
    var sim0 = R.SIM.create()
    var agent = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0, utd=1.0, warmup=14)

    print("\n[2] the reference skip in act")
    var hist = _hist()
    var goals = _goals()
    var a0 = agent.trainer.act[LANES](hist, goals)
    agent.trainer.set_ref_skip(True)
    var a1 = agent.trainer.act[LANES](hist, goals)
    var worst = 0.0
    var worst_pose = 0.0
    for b in range(LANES):
        for j in range(R.NDOF):
            var skip = Float64(a1.data[b * R.NDOF + j]) - Float64(a0.data[b * R.NDOF + j])
            var want = Float64(goals.data[b * D.GOAL + j]) / R.action_scale(j)
            worst = max(worst, abs(skip - want) / max(abs(want), 1e-3))
            # the PD target the skip alone commands, against the goal's q_ref − q0
            worst_pose = max(worst_pose, abs(R.action_scale(j) * skip - Float64(goals.data[b * D.GOAL + j])))
    t.truth(worst < 1e-4, "it adds (q_ref - q0) / scale to every action (worst relative " + String(worst) + ")")
    t.truth(worst_pose < 1e-5, "so the skip alone commands the target pose (worst " + String(worst_pose) + " rad)")
    t.truth(G1Robot.action_scale(0) == 0.0 and R.action_scale(0) > 0.0, "a robot without a declared action scale reads 0 (the skip refuses it)")
    agent.trainer.set_ref_skip(False)

    print("\n[3] the reference skip in imagination")
    agent.train(20, 0)                       # steps 14..20 update: the world model answers to actions
    t.truth(agent.updates == 7, "7 updates after the warm-up (" + String(agent.updates) + ")")
    agent.trainer.set_ref_skip(True)
    var ib = agent.start_batch()
    var skip_p = agent.trainer._skip()
    agent.trainer.pol.zero_grad["cpu"](None)
    _ = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, True, 0, None, skip_p)
    var g_skip = sm_snapshot["cpu"](agent.trainer.pol, None, True)
    agent.trainer.pol.zero_grad["cpu"](None)
    _ = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, True, 0, None, None)
    var g_plain = sm_snapshot["cpu"](agent.trainer.pol, None, True)
    var w0 = sm_snapshot["cpu"](agent.trainer.pol, None)
    var last = len(w0) - 1                   # the decoder's last bias: one entry per joint
    comptime EPS = 0.02
    var fd_err = 0.0
    var fd_scale = 0.0
    var plain_err = 0.0
    for e in range(3):
        var k = [0, 7, 20][e]
        var wp = w0.copy()
        wp[last][k] += Scalar[DT](EPS)
        sm_restore["cpu"](agent.trainer.pol, wp^, None)
        var op = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, False, 0, None, skip_p)
        var jp = op.j
        var wm_ = w0.copy()
        wm_[last][k] -= Scalar[DT](EPS)
        sm_restore["cpu"](agent.trainer.pol, wm_^, None)
        var om = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib, 0.99, False, 0, None, skip_p)
        var jm = om.j
        var fd = -(jp - jm) / (2.0 * EPS * Float64(H))
        print("    decoder bias", k, ": finite difference", fd, "| with the skip", Float64(g_skip[last][k]), "| without", Float64(g_plain[last][k]))
        fd_err = max(fd_err, abs(fd - Float64(g_skip[last][k])))
        plain_err = max(plain_err, abs(fd - Float64(g_plain[last][k])))
        fd_scale = max(fd_scale, abs(fd))
    sm_restore["cpu"](agent.trainer.pol, w0.copy(), None)
    t.truth(fd_err <= 0.03 * fd_scale, "the gradient with the skip is the finite difference of -J/H (worst " + String(fd_err) + " of " + String(fd_scale) + ")")
    t.truth(plain_err > 0.2 * fd_scale, "the gradient without the skip is not (worst " + String(plain_err) + "): the check has power")
    agent.trainer.set_ref_skip(False)

    print("\n[4] the averaged policy")
    comptime DECAY = 0.9
    agent.trainer.set_ema(DECAY)
    agent.train(21, 0)                       # one more update: the average starts as a copy
    var live1 = sm_snapshot["cpu"](agent.trainer.pol, None)
    var avg1 = sm_snapshot["cpu"](agent.trainer.pol_avg, None)
    t.truth(agent.trainer.ema_n == 1 and _max_diff(live1, avg1) == 0.0, "the first update copies the live policy")
    agent.train(22, 0)
    var live2 = sm_snapshot["cpu"](agent.trainer.pol, None)
    var avg2 = sm_snapshot["cpu"](agent.trainer.pol_avg, None)
    var mix_err = 0.0
    var moved = 0.0
    for i in range(len(avg2)):
        for j in range(len(avg2[i])):
            var want = DECAY * Float64(avg1[i][j]) + (1.0 - DECAY) * Float64(live2[i][j])
            mix_err = max(mix_err, abs(Float64(avg2[i][j]) - want))
            moved = max(moved, abs(Float64(live2[i][j]) - Float64(live1[i][j])))
    t.truth(mix_err <= 1e-6 * max(moved, 1e-6) + 1e-7, "the second is 0.9 x average + 0.1 x live (worst " + String(mix_err) + ", the policy moved " + String(moved) + ")")
    t.truth(_max_diff(avg2, live2) > 0.5 * moved, "and it is not the live policy")
    var a_live = agent.trainer.act[LANES](hist, goals)
    var a_dep = agent.trainer.act[LANES](hist, goals, deployed=True)
    t.truth(_max_diff_t(a_live, a_dep, LANES * R.NDOF) > 0.0, "act(deployed) answers with the average, act with the live policy")

    print("\n[5] the checkpoint")
    agent.trainer.set_ref_skip(True)
    var a_live_s = agent.trainer.act[LANES](hist, goals)
    var a_dep_s = agent.trainer.act[LANES](hist, goals, deployed=True)
    var path = String("/tmp/sm_fast_test.ckpt")
    agent.trainer.save_state(path, 22, agent.updates)
    var reader = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0)
    var ctr = reader.trainer.load_state(path)
    t.truth(ctr[0] == 22 and ctr[1] == agent.updates, "the counters come back")
    t.truth(reader.trainer.ref_skip, "the reference skip follows the file")
    var a_read = reader.trainer.act[LANES](hist, goals)
    t.truth(_max_diff_t(a_read, a_dep_s, LANES * R.NDOF) == 0.0, "a plain reader deploys the AVERAGE (its act = the trainer's act(deployed))")
    var resumed = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, False, 0)
    resumed.set_fast(SMFastOpts(DECAY, 0, 1.0, 0.0, 300, 0, -1.0, False, False, 0.0))
    _ = resumed.trainer.load_state(path)
    var r_live = resumed.trainer.act[LANES](hist, goals)
    var r_dep = resumed.trainer.act[LANES](hist, goals, deployed=True)
    t.truth(_max_diff_t(r_live, a_live_s, LANES * R.NDOF) == 0.0, "a resume with the average on restores the live policy from .live")
    t.truth(_max_diff_t(r_dep, a_dep_s, LANES * R.NDOF) == 0.0, "and the average from the file")

    agent.trainer.set_ref_skip(False)

    print("\n[7] analytic termination")
    var tin = TensorImpl[DT].alloc(8)
    var lg = TensorImpl[DT].alloc(2)
    tin.data[2] = 1.0                        # lane 0: the height ratio on its threshold
    tin.data[3] = 0.3
    tin.data[6] = 0.5                        # lane 1: the tilt ratio at 95 % of its threshold
    tin.data[7] = 0.95
    sm_term_analytic["cpu"](None, 2, tp["cpu"](tin), tp["cpu"](lg))
    t.truth(abs(Float64(lg.data[0])) < 1e-6 and abs(Float64(lg.data[1]) + 1.0) < 1e-5, "logit 0 on a threshold, -1 at 95 % of it (" + String(Float64(lg.data[0])) + ", " + String(Float64(lg.data[1])) + ")")
    var glg = TensorImpl[DT].alloc(2)
    var gtin = TensorImpl[DT].alloc(8)
    glg.data[0] = 2.0
    glg.data[1] = 3.0
    for i in range(8):
        gtin.data[i] = 99.0
    sm_term_analytic_back["cpu"](None, 2, tp["cpu"](tin), tp["cpu"](glg), tp["cpu"](gtin))
    var want_g: List[Float64] = [0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 60.0]
    var gt_ok = True
    for i in range(8):
        gt_ok = gt_ok and abs(Float64(gtin.data[i]) - want_g[i]) < 1e-4
    t.truth(gt_ok, "its gradient is glg / 0.05 on the larger ratio and 0 elsewhere")
    var ib7 = agent.start_batch()
    for b in range(8):
        ib7.state.data[b * D.S + 2] += Scalar[DT](0.2)      # 0.2 m above the reference: 80 % of the height threshold
    var probe = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib7, 0.99, False, 0, None, None, True, 0.0)
    var d_lo = 1.0
    var d_hi = 0.0
    for i in range(8, (H + 1) * 8):
        d_lo = min(d_lo, Float64(probe.dhat.data[i]))
        d_hi = max(d_hi, Float64(probe.dhat.data[i]))
    t.truth(d_lo > 1e-4 and d_hi < 0.9, "the rollout's termination probabilities are in the sigmoid's live range (" + String(d_lo) + " .. " + String(d_hi) + ")")
    var g7 = _bias_grads(agent, ib7, True, 0.0)
    var g7_head = _bias_grads(agent, ib7, False, 0.0)
    var fd7 = _bias_fd(agent, ib7, True, 0.0)
    print("    finite differences", fd7[0], fd7[1], fd7[2], "| analytic", g7[0], g7[1], g7[2], "| learned head", g7_head[0], g7_head[1], g7_head[2])
    t.truth(_worst(g7, fd7) <= 0.03 * _largest(fd7), "the gradient with the analytic termination is the finite difference (worst " + String(_worst(g7, fd7)) + " of " + String(_largest(fd7)) + ")")
    t.truth(_worst(g7_head, fd7) > 0.2 * _largest(fd7), "the learned head's gradient is not (worst " + String(_worst(g7_head, fd7)) + ")")

    print("\n[8] the value ceiling")
    var cv = TensorImpl[DT].alloc(3)
    var cg = TensorImpl[DT].alloc(3)
    cv.data[0] = 0.5
    cv.data[1] = 2.0
    cv.data[2] = -1.0
    cg.data[0] = 7.0
    cg.data[1] = 8.0
    cg.data[2] = 9.0
    sm_cap["cpu"](None, 3, tp["cpu"](cv), Scalar[DT](1.0))
    sm_cap_back["cpu"](None, 3, tp["cpu"](cv), Scalar[DT](1.0), tp["cpu"](cg))
    t.truth(cv.data[0] == 0.5 and cv.data[1] == 1.0 and cv.data[2] == -1.0, "min(V, cap): only the value above the cap moves")
    t.truth(cg.data[0] == 7.0 and cg.data[1] == 0.0 and cg.data[2] == 9.0, "and only that value loses its gradient")
    var ib8 = agent.start_batch()
    var plain = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib8, 0.99, False)
    var vmean = 0.0
    for b in range(8):
        vmean += Float64(plain.value_h.data[b]) / 8.0
    var cap = vmean if vmean > 1e-3 else 1e-3          # between the lanes' values: some capped, some not
    var gH = 1.0
    for _ in range(H):
        gH *= 0.99
    var want_j = plain.j
    var n_capped = 0
    for b in range(8):
        var v = Float64(plain.value_h.data[b])
        if v > cap:
            n_capped += 1
            want_j += gH * Float64(plain.survival.data[H * 8 + b]) * (cap - v) / 8.0
    var capped = sm_imagine[C, 8, "cpu"](agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, agent.trainer.tables, ib8, 0.99, False, 0, None, None, False, cap)
    t.truth(n_capped > 0 and n_capped < 8, String(n_capped) + " of 8 lanes above the cap " + String(cap))
    t.truth(abs(capped.j - want_j) <= 1e-5 * max(abs(want_j), 1e-3) and abs(capped.j - plain.j) > 1e-6, "J with the cap = J + g^H mean(c_H (cap - V)) over the capped lanes (" + String(capped.j) + " against " + String(plain.j) + " without)")
    var g8 = _bias_grads(agent, ib8, False, cap)
    var fd8 = _bias_fd(agent, ib8, False, cap)
    print("    finite differences", fd8[0], fd8[1], fd8[2], "| with the cap", g8[0], g8[1], g8[2])
    t.truth(_worst(g8, fd8) <= 0.03 * _largest(fd8), "the gradient with the cap is the finite difference (worst " + String(_worst(g8, fd8)) + " of " + String(_largest(fd8)) + ")")
    var vals = List[Float64]()
    var tgts = List[Float64]()
    var none = List[Int]()
    agent.trainer.v_max = 0.0
    _ = agent.trainer.policy_update[8](ib8, none, vals, tgts)
    var v_hi = 0.0
    for i in range(len(vals)):
        v_hi = max(v_hi, vals[i])
    var cap2 = 0.5 * v_hi
    agent.trainer.v_max = cap2
    _ = agent.trainer.policy_update[8](ib8, none, vals, tgts)
    var v_hi2 = -1e9
    var t_hi2 = -1e9
    for i in range(len(vals)):
        v_hi2 = max(v_hi2, vals[i])
    for i in range(len(tgts)):
        t_hi2 = max(t_hi2, tgts[i])
    t.truth(v_hi > 0.0 and v_hi2 <= cap2 * (1.0 + 1e-6), "in an update, the critic's values reach " + String(v_hi) + " without the cap and stay under " + String(cap2) + " with it (" + String(v_hi2) + ")")
    t.truth(t_hi2 <= cap2 + 0.1 * Float64(H), "and its targets stay under the cap plus the horizon's rewards (" + String(t_hi2) + ")")
    agent.trainer.v_max = 0.0

    print("\n[9] the drift weight's guard")
    var other = 625 if SM_WM_DRIFT_MILLI != 625 else 1000
    var sc9 = CheckpointScalars()
    sc9.set_int("iteration", 1)
    sc9.set_int("updates", 1)
    sc9.set_int("wm_drift_milli", other)
    var path9 = String("/tmp/sm_fast_test_drift.ckpt")
    save_params_multi["cpu"](path9, None, False, agent.trainer.wm, agent.trainer.pol, agent.trainer.critic, scalars=sc9)
    var refused = False
    try:
        _ = reader.trainer.load_state(path9)
    except e:
        refused = True
        print("   ", e)
    t.truth(refused, "a checkpoint written under the weight " + String(other) + "/1000 is refused by this build (" + String(SM_WM_DRIFT_MILLI) + "/1000)")

    print("\n[6] the loop")
    var looped = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, True, 0, utd=1.0, warmup=14)
    t.truth(looped.env.action_noise == 0.1, "with noise, collection's action noise starts at 0.1")
    looped.set_fast(SMFastOpts(0.995, 3, 4.0, 0.0, 4, 50, 0.0, True, True, 10.0))
    looped.train(65, 0)                      # 52 updates: the deadline's test is at update 50
    var u = looped.updates
    t.truth(u == 52, "52 updates (" + String(u) + ")")
    t.truth(looped.trainer.wm_blk.opt.t == 3 * u, "the world model took 3 steps per update (" + String(looped.trainer.wm_blk.opt.t) + ")")
    t.truth(looped.fast.decay_at == 50, "the deadline starts the ramp at update 50 (" + String(looped.fast.decay_at) + ")")
    # the last update (number 52) was scheduled with 51 done: one ramp step of four
    var want_mult = 4.0 + (1.0 - 4.0) * 0.25
    t.truth(_near(looped.fast.pi_mult(u - 1), want_mult), "one step of four down the ramp: x " + String(want_mult))
    t.truth(_near(Float64(looped.trainer.actor_blk.opt.lr), looped.lr * want_mult, 1e-6), "the policy's lr is the base x the multiplier")
    t.truth(
        _near(Float64(looped.trainer.wm_blk.opt.lr), looped.lr, 1e-6) and _near(Float64(looped.trainer.critic_blk.opt.lr), looped.lr, 1e-6),
        "the world model's and the critic's keep the base",
    )
    t.truth(looped.trainer.ema_n == u and looped.trainer.ref_skip and looped.env.action_noise == 0.0, "the average, the skip and the action noise reached the trainer and the env")
    t.truth(looped.trainer.analytic_term and looped.trainer.v_max == 10.0, "and the analytic termination and the value ceiling too")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_fast: " + String(t.fails) + " failed")
