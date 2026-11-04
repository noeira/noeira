"""SuperMimic M2c: the tracking environment on our CPU G1 (`deep_agents/supermimic/env`).

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_env.mojo

No reference implementation runs here (the reproduction's env is mjlab on
MuJoCo Warp); every section checks a behaviour whose answer is known BY
CONSTRUCTION. The reference library is built in the test from the robot's
pose after 2 s of zero action from its `stand` start (through its own
tracking sim). On both G1s that pose is LYING (open-loop PD on the default
pose does not balance the robot). It is still a pose the PD holds, so a zero
policy tracks it. `-D SM_MJLAB=1` runs every section on mjlab's G1.

[1] holding the pose: the settled reference, zero action, no noise, no DR —
    every step's reward within 2 % of its maximum (0.1), no termination.
[2] resets: without noise the reset state IS the reference row; with noise,
    every component of the difference inside its configured range.
[3] the encoder bias: the observation's joint block is q + bias − q0 (no
    obs noise), and the PD target carries −bias (a biased lane settles ~ −b
    from an unbiased twin); the foot geoms (8 spheres; 14 capsules on mjlab's G1) carry priority 1 and ξ's
    friction.
[4] termination: a reference 0.3 m above the robot terminates the first
    step: a transition row and a terminal row labelled 1, then a reset.
[5] the clip's end: a 60-frame clip times out after its last frame (not
    terminated), the sampler's log flags it, and the lane restarts.
[6] bookkeeping: a lane reset to frame f shows frame f + 1 after the step;
    consecutive replay rows chain (row t's next state is row t+1's state),
    rewards and frames match the env's.
"""

from std.sys import argv, get_defined_int

from noeira.core.cont_action import ContAction
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv, SM_EPISODE_STEPS
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import G1MjlabRobot, G1Robot, SMDims

# `-D SM_MJLAB=1`: the same gate on mjlab's G1.
comptime TEST_MJLAB = get_defined_int["SM_MJLAB", 0]() != 0
comptime TRobot = G1MjlabRobot if TEST_MJLAB else G1Robot

comptime SM_HIST = SMDims[TRobot].HIST
comptime SM_OBS = SMDims[TRobot].OBS
comptime SMG1Env = SMTrackingEnv[TRobot]
comptime SMReplayBuffer = SMReplayOf[TRobot]
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.nn.constants import DT
from noeira.physics3d.gpu.constants import GEOM_IDX_FRICTION, GEOM_IDX_PRIORITY, MODEL_GEOM_SIZE


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


def _standing_row() raises -> List[Scalar[DT]]:
    """The robot under test settled for 2 s under zero action (no bias) from
    its `stand` start, through its own tracking sim."""
    var sim = TRobot.SIM.create()
    var a = List[Float64](length=G1_N_DOF, fill=0.0)
    for _ in range(100):
        sim.control_step(a, a)
    var row = List[Scalar[DT]](length=71, fill=0)
    for k in range(36):
        row[k] = Scalar[DT](sim.qpos(k))
    # zero velocities: a still reference
    return row^


def _library(stand: List[Scalar[DT]], lens: List[Int], dz: Float64) raises -> SMMotionLibrary[TRobot]:
    var total = 0
    for c in range(len(lens)):
        total += lens[c]
    var rows = List[Scalar[DT]](length=total * 71, fill=0)
    for f in range(total):
        for k in range(71):
            rows[f * 71 + k] = stand[k]
        rows[f * 71 + 2] += Scalar[DT](dz)
    var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
    for j in range(G1_N_DOF):
        q0[j] = TRobot.default_joint_pos(j)
    return SMMotionLibrary[TRobot](rows^, lens, q0)


def main() raises:
    var t = Tally()
    var stand = _standing_row()
    print("  standing pelvis height", stand[2])
    var zero = List[Scalar[DT]](length=2 * G1_N_DOF, fill=0)

    print("\n[1] holding the standing pose (no noise, no DR, zero action)")
    var lens: List[Int] = [600]
    var env = SMG1Env(2, _library(stand, lens, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=1)
    var rb = SMReplayBuffer(2, 200)
    var rmin = 1.0
    var nterm = 0
    for _ in range(50):
        var info = env.step(zero, rb)
        nterm += info.n_terminated + info.n_truncated
        for i in range(2):
            rmin = min(rmin, env.last_reward[i])
    t.truth(rmin >= 0.098 and nterm == 0, "50 steps: min reward " + String(rmin) + " (max 0.1), " + String(nterm) + " episode ends")

    print("\n[2] resets")
    var env2 = SMG1Env(2, _library(stand, lens, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=2)
    var e0 = 0.0
    for i in range(2):
        var f = env2.lanes[i].frame
        for k in range(36):
            e0 = max(e0, abs(Float64(env2.lanes[i].s[k]) - Float64(env2.lib.rows[f * 71 + k])))
    t.truth(e0 <= 1e-6, "no noise: the reset state is the reference row (max err " + String(e0) + ")")
    var env3 = SMG1Env(8, _library(stand, lens, 0.0), dr=True, obs_noise=True, reset_noise=True, rng_seed=3)
    var bad = 0
    for i in range(8):
        ref ln = env3.lanes[i]
        var f = ln.frame
        if abs(Float64(ln.s[0] - env3.lib.rows[f * 71])) > 0.05 or abs(Float64(ln.s[2] - env3.lib.rows[f * 71 + 2])) > 0.01:
            bad += 1
        for j in range(G1_N_DOF):
            if abs(Float64(ln.s[7 + j] - env3.lib.rows[f * 71 + 7 + j])) > 0.1 + 1e-6:
                bad += 1
        if abs(Float64(ln.s[36]) - Float64(env3.lib.rows[f * 71 + 36])) > 0.5 or abs(Float64(ln.s[38]) - Float64(env3.lib.rows[f * 71 + 38])) > 0.2:
            bad += 1
        if Float64(ln.xi[0]) < 0.3 or Float64(ln.xi[0]) > 1.2:
            bad += 1
        for j in range(G1_N_DOF):
            if abs(Float64(ln.xi[1 + j])) > 0.01:
                bad += 1
        if abs(Float64(ln.xi[30])) > 0.025 or abs(Float64(ln.xi[31])) > 0.05 or abs(Float64(ln.xi[32])) > 0.05:
            bad += 1
    t.truth(bad == 0, "with noise and DR, 8 lanes: pose, joints, velocities and xi inside their ranges")

    print("\n[3] the encoder bias and the feet")
    var e_obs = 0.0
    var env4 = SMG1Env(2, _library(stand, lens, 0.0), dr=True, obs_noise=False, reset_noise=False, rng_seed=4)
    for i in range(2):
        ref ln = env4.lanes[i]
        var last = (SM_HIST - 1) * SM_OBS
        for j in range(G1_N_DOF):
            var want = Float64(ln.s[7 + j]) + Float64(ln.xi[1 + j]) - TRobot.default_joint_pos(j)
            e_obs = max(e_obs, abs(Float64(ln.hist[last + j]) - want))
    t.truth(e_obs <= 1e-6, "observed joints = q + bias - q0 (max err " + String(e_obs) + ")")
    var bad_f = 0
    for i in range(2):
        for g in env4.tables.feet:
            if env4.lanes[i].sim.geom_field(g, GEOM_IDX_PRIORITY) != 1.0:
                bad_f += 1
            if abs(env4.lanes[i].sim.geom_field(g, GEOM_IDX_FRICTION) - Float64(env4.lanes[i].xi[0])) > 1e-6:
                bad_f += 1
    # BFM-Zero's G1: 4 contact spheres per foot; mjlab's: 7 capsules per foot
    var want_feet = 14 if TEST_MJLAB else 8
    t.truth(bad_f == 0 and len(env4.tables.feet) == want_feet, String(len(env4.tables.feet)) + " foot geoms found by body name: priority 1, friction = xi[0]")

    # the bias in the PD target: the same env with bias b settles ~ −b away
    var ea = SMG1Env(1, _library(stand, lens, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=9)
    var eb = SMG1Env(1, _library(stand, lens, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=9)
    for j in range(G1_N_DOF):
        eb.lanes[0].xi[1 + j] = Scalar[DT](0.01 if j % 2 == 0 else -0.01)
    var rba = SMReplayBuffer(1, 100)
    var rbb = SMReplayBuffer(1, 100)
    var z1 = List[Scalar[DT]](length=G1_N_DOF, fill=0)
    for _ in range(25):
        _ = ea.step(z1, rba)
        _ = eb.step(z1, rbb)
    var num = 0.0
    var den = 0.0
    for j in range(G1_N_DOF):
        var dq = Float64(eb.lanes[0].s[7 + j] - ea.lanes[0].s[7 + j])
        var b = Float64(eb.lanes[0].xi[1 + j])
        num += dq * (-b)
        den += b * b
    t.truth(num / den > 0.5 and num / den < 1.5, "the PD target carries -bias: settled joints move by " + String(num / den) + " x (-bias) (1 = all of it)")

    print("\n[4] termination")
    var env5 = SMG1Env(2, _library(stand, lens, 0.3), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=5)
    # the reset puts the robot AT the elevated reference; drop it back to the floor height
    for i in range(2):
        var qpos = List[Float64](length=36, fill=0.0)
        var qvel = List[Float64](length=35, fill=0.0)
        for k in range(36):
            qpos[k] = Float64(stand[k])
        env5.lanes[i].sim.set_state(qpos, qvel)
        env5.lanes[i].read_state()
        env5._remember(i)
    var rb5 = SMReplayBuffer(2, 50)
    var inf5 = env5.step(zero, rb5)
    var labels_ok = rb5.valid[0] and rb5.valid[1] and not rb5.terminal[0] and rb5.terminal[1] and Float64(rb5.done[0]) == 1.0 and Float64(rb5.done[1]) == 1.0
    t.truth(inf5.n_terminated == 2 and labels_ok and rb5.ep_counter[0] == 1, "both lanes terminate at step 1; transition + terminal rows labelled 1; episode counted")

    print("\n[5] the clip's end")
    var short: List[Int] = [60]
    var env6 = SMG1Env(1, _library(stand, short, 0.0), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, rng_seed=6)
    var f_start = env6.lanes[0].frame
    var rb6 = SMReplayBuffer(1, 200)
    var steps = 0
    var ended_by = String("")
    for _ in range(80):
        var info = env6.step(zero, rb6)
        steps += 1
        if info.n_terminated > 0:
            ended_by = "terminated"
            break
        if info.n_truncated > 0:
            ended_by = "time-out"
            break
    # step k targets frame f_start + k − 1; the frame passes the last (59) at
    # the end of step 60 − f_start, which sets the flag; the next step ends it
    var want_steps = 61 - f_start
    t.truth(ended_by == "time-out" and steps == want_steps, "a 60-frame clip started at frame " + String(f_start) + " ends by " + ended_by + " after " + String(steps) + " steps (want " + String(want_steps) + ")")

    print("\n[6] frame and replay bookkeeping")
    var env7 = SMG1Env(2, _library(stand, lens, 0.0), dr=True, obs_noise=True, reset_noise=True, rng_seed=7)
    var f0 = env7.lanes[0].frame
    var rb7 = SMReplayBuffer(2, 100)
    var acts = List[Scalar[DT]](length=2 * G1_N_DOF, fill=0)
    for j in range(G1_N_DOF):
        acts[j] = Scalar[DT](0.05)
    var rewards = List[Float64]()
    var after = List[List[Scalar[DT]]]()
    for _ in range(5):
        _ = env7.step(acts, rb7)
        rewards.append(env7.last_reward[0])
        after.append(env7.lanes[0].s.copy())
    t.truth(env7.lanes[0].frame == f0 + 5, "frame advanced by one per step (" + String(f0) + " -> " + String(env7.lanes[0].frame) + ")")
    var chain = 0.0
    for r in range(4):
        for k in range(71):   # row r+1 holds the state reached by row r's transition
            chain = max(chain, abs(Float64(rb7.s[(r + 1) * 71 + k]) - Float64(after[r][k])))
    t.truth(chain == 0.0, "4 consecutive rows chain: row t+1's state = the state after step t")
    var e_r = 0.0
    var e_f = 0
    for r in range(5):
        e_r = max(e_r, abs(Float64(rb7.r[r]) - rewards[r]))
        if rb7.frame[r] != f0 + r:
            e_f += 1
    t.truth(e_r <= 1e-6 and e_f == 0 and rb7.ep_start[0] and not rb7.ep_start[1], "5 rows: rewards = the env's, frames f0..f0+4, ep_start on the first only")
    # the 5th row's state is the state the env holds before step 6 (prev_s)
    var e_last = 0.0
    for k in range(71):
        e_last = max(e_last, abs(Float64(env7.lanes[0].prev_s[k]) - Float64(env7.lanes[0].s[k])))
    var a_noise = 0.0
    for j in range(G1_N_DOF):
        a_noise = max(a_noise, abs(Float64(rb7.a_exec[j]) - Float64(rb7.a[j])))
    t.truth(e_last == 0.0 and a_noise > 0.0 and rb7.a[0] == Scalar[DT](0.05), "the next row starts from the current state; executed = clean + noise (max |noise| " + String(a_noise) + ")")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_env: " + String(t.fails) + " failed")
