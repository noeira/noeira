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
"""

from std.math import abs

from noeira.deep_agents.supermimic.agent import sm_agent
from noeira.deep_agents.supermimic.config import SMConfig
from noeira.deep_agents.supermimic.fast import SMFastOpts, SMFastState
from noeira.deep_agents.supermimic.imagination import sm_imagine
from noeira.deep_agents.supermimic.params import sm_restore, sm_snapshot
from noeira.deep_agents.supermimic.prompt_eval import sm_stand_library
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1Robot, SMDims
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


def main() raises:
    var t = Tally()

    print("\n[1] the lr ramp")
    var st = SMFastState(SMFastOpts(0.0, 0, 4.0, 0.75, 300, 1400, -1.0, False))
    t.truth(st.pi_mult(0) == 4.0 and st.pi_mult(900) == 4.0, "x 4 until the trigger")
    st.note_episode(0.9)
    t.truth(not st.tick(49) and st.decay_at < 0, "no test between two multiples of 50, whatever the return")
    st.note_episode(0.5)
    t.truth(not st.tick(50) and _near(st.last_return, 0.7), "a test below 0.75 (mean 0.7) does not start it")
    st.note_episode(0.8)
    t.truth(st.tick(100) and st.decay_at == 100, "a test at 0.8 starts it, at update 100")
    t.truth(_near(st.pi_mult(100), 4.0) and _near(st.pi_mult(250), 2.5) and _near(st.pi_mult(400), 1.0) and _near(st.pi_mult(5000), 1.0), "linear 4 -> 1 over 300 updates, then 1")
    var dl = SMFastState(SMFastOpts(0.0, 0, 4.0, 0.75, 300, 1400, -1.0, False))
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
    resumed.set_fast(SMFastOpts(DECAY, 0, 1.0, 0.0, 300, 0, -1.0, False))
    _ = resumed.trainer.load_state(path)
    var r_live = resumed.trainer.act[LANES](hist, goals)
    var r_dep = resumed.trainer.act[LANES](hist, goals, deployed=True)
    t.truth(_max_diff_t(r_live, a_live_s, LANES * R.NDOF) == 0.0, "a resume with the average on restores the live policy from .live")
    t.truth(_max_diff_t(r_dep, a_dep_s, LANES * R.NDOF) == 0.0, "and the average from the file")

    print("\n[6] the loop")
    var looped = sm_agent[C](sm_stand_library[R](2, 200, 0.05, sim0.qpos(2)), 2000, True, 0, utd=1.0, warmup=14)
    t.truth(looped.env.action_noise == 0.1, "with noise, collection's action noise starts at 0.1")
    looped.set_fast(SMFastOpts(0.995, 3, 4.0, 0.0, 4, 50, 0.0, True))
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

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_fast: " + String(t.fails) + " failed")
