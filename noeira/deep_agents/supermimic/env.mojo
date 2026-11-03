# +--------------------------------------------------------------------------+ #
# | SuperMimic: the tracking environment, any `TrackingRobot`, CPU lanes
# +--------------------------------------------------------------------------+ #
"""The reproduction's mjlab environment (`envs/tracking_env_cfg.py`, the
`MultiClipMotionCommand`, `NoisyJointPositionAction`, `SimContext` and the
`ReplayRecorder`) for robot R, one CPU simulator (`R.SIM`) per lane.

ONE STEP, in mjlab's `ManagerBasedRlEnv.step` order:
    1. a_exec = a + N(0, 0.1²); the lane runs the PD target
       q0 + scale · a_exec − bias (`TrackingSim.control_step`)
    2. ep_len += 1; terminated = |h_ref − h| > 0.25 or |tilt| > 0.8, against
       the step's target frame f_t; time-out at 500 steps or at the clip's end
    3. r = the 8 BeyondMimic terms of the new state (the heads, on the SIM
       state) against f_t; logged with f_t's bin for the reset sampler
    4. ending lanes: the transition row and a terminal row into the replay;
       their episodes folded into the sampler; reset (ξ re-drawn, a frame
       from the sampler, the reference state there + noise)
    5. every lane's frame += 1 (a lane just reset starts at its frame + 1, as
       mjlab's command update runs for every env after the resets, and once
       more inside `reset()`); the clip's end is flagged when passed
    6. observation (3n + 6, noisy) into the 10-step history (a reset lane's
       first observation fills all 10); the next rows are remembered

ROBOT-SPECIFIC CHOICES (made by the robot's conformers, see `robot.mojo`)
* The action scale is mjlab's (`q0 + 0.25·effort/kp·a`): `TrackingSim` maps
  it onto its env's own action convention (the G1's env takes a/5).
* The encoder bias enters the PD target (`target − bias`, as mjlab does) and
  the observation (`q + bias`), not the physics.
* FOOT FRICTION: the env gives the robot's foot geoms (`tables.feet`)
  priority 1, so their friction is used as is (mjlab's G1 does the same). On
  our G1 the feet have priority 0 and friction 1, which MuJoCo combines with
  the floor's by max — randomising them would do nothing otherwise.
* Rewards and terminations read the state exactly; mjlab's read derived
  kinematics one substep stale (its documented trade-off).

THE GPU PATH replaces the lanes by one batched simulator and steps 2–6 by
kernels over the same per-lane functions (`sm_heads_lane`,
`sm_observation_lane`, …); the replay becomes the episodic ring's device
sibling.
"""

from std.math import cos, log, sin, sqrt
from std.random import random_float64, seed

from noeira.deep_agents.data import EpisodicReplay
from noeira.deep_agents.supermimic.heads import (
    SM_N_TERMS, SMHeadLayout, qapply_inv, sm_heads_lane, sm_observation_lane,
)
from noeira.deep_agents.supermimic.kinematics import Q4, qmul
from noeira.deep_agents.supermimic.motion_library import (
    SMEpisodeLog, SMMotionLibrary, SMResetSampler,
)
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.robot import SMDims, TrackingRobot, TrackingSim
from noeira.deep_agents.supermimic.tables import SMTables, sm_soft_limits, sm_tables
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import BODY_IDX_IPOS_X, GEOM_IDX_FRICTION, GEOM_IDX_PRIORITY


comptime SM_EPISODE_STEPS: Int = 500        # 10 s at 50 Hz
comptime SM_MAX_LOGGED: Int = 502           # min(ceil(10 / 0.02) + 2, 1024)
comptime SM_ACTION_NOISE: Float64 = 0.1
comptime SM_HEIGHT_TERM: Float64 = 0.25
comptime SM_TILT_TERM: Float64 = 0.8

comptime SMReplayOf[R: TrackingRobot] = EpisodicReplay[SMDims[R].S, R.NDOF, SMDims[R].X]
"""The replay ring for robot R: state S, action n, context ξ (X)."""


@always_inline
def _u(lo: Float64, hi: Float64) -> Float64:
    return random_float64(lo, hi)


@always_inline
def _randn() -> Float64:
    var u1 = max(random_float64(0.0, 1.0), 1e-300)
    var u2 = random_float64(0.0, 1.0)
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


@always_inline
def quat_from_euler_xyz(roll: Float64, pitch: Float64, yaw: Float64) -> Q4[DType.float64]:
    """mjlab's `quat_from_euler_xyz` (WXYZ)."""
    var cy = cos(yaw * 0.5)
    var sy = sin(yaw * 0.5)
    var cr = cos(roll * 0.5)
    var sr = sin(roll * 0.5)
    var cp = cos(pitch * 0.5)
    var sp = sin(pitch * 0.5)
    return Q4[DType.float64](
        cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp,
    )


struct SMLane[R: TrackingRobot](Movable):
    var sim: Self.R.SIM
    var frame: Int
    var clip_end: Bool
    var ep_len: Int
    var log: SMEpisodeLog
    var xi: List[Scalar[DT]]           # X: friction | bias n | CoM offset 3
    var com_nominal: List[Float64]
    var hist: List[Scalar[DT]]         # 10 × OBS, oldest first
    var hist_fresh: Bool
    var a_last: List[Scalar[DT]]       # the clean action of the last step (0 after a reset)
    var s: List[Scalar[DT]]            # the current state (S)
    var prev_s: List[Scalar[DT]]
    var prev_xi: List[Scalar[DT]]
    var prev_frame: Int
    var ep_start: Bool

    def __init__(out self) raises:
        comptime D = SMDims[Self.R]
        self.sim = Self.R.SIM.create()
        self.frame = 0
        self.clip_end = False
        self.ep_len = 0
        self.log = SMEpisodeLog()
        self.xi = List[Scalar[DT]](length=D.X, fill=0)
        self.com_nominal = List[Float64](length=3, fill=0.0)
        self.hist = List[Scalar[DT]](length=D.HIST * D.OBS, fill=0)
        self.hist_fresh = True
        self.a_last = List[Scalar[DT]](length=Self.R.NDOF, fill=0)
        self.s = List[Scalar[DT]](length=D.S, fill=0)
        self.prev_s = List[Scalar[DT]](length=D.S, fill=0)
        self.prev_xi = List[Scalar[DT]](length=D.X, fill=0)
        self.prev_frame = 0
        self.ep_start = True

    def __init__(out self, *, deinit move: Self):
        self.sim = move.sim^
        self.frame = move.frame
        self.clip_end = move.clip_end
        self.ep_len = move.ep_len
        self.log = move.log^
        self.xi = move.xi^
        self.com_nominal = move.com_nominal^
        self.hist = move.hist^
        self.hist_fresh = move.hist_fresh
        self.a_last = move.a_last^
        self.s = move.s^
        self.prev_s = move.prev_s^
        self.prev_xi = move.prev_xi^
        self.prev_frame = move.prev_frame
        self.ep_start = move.ep_start

    def read_state(mut self):
        for k in range(Self.R.NQ):
            self.s[k] = Scalar[DT](self.sim.qpos(k))
        for k in range(Self.R.NV):
            self.s[Self.R.NQ + k] = Scalar[DT](self.sim.qvel(k))


@fieldwise_init
struct SMStepInfo(Copyable, Movable):
    var reward_mean: Float64
    var n_terminated: Int
    var n_truncated: Int


struct SMTrackingEnv[R: TrackingRobot](Movable):
    var lanes: List[SMLane[Self.R]]
    var lib: SMMotionLibrary[Self.R]
    var sampler: SMResetSampler
    var tables: SMTables[DT]
    var dr: Bool
    var obs_noise: Bool
    var reset_noise: Bool
    var action_noise: Float64
    var last_reward: List[Float64]
    var last_terminated: List[Bool]

    def __init__(
        out self, n: Int, var lib: SMMotionLibrary[Self.R], dr: Bool = True, obs_noise: Bool = True,
        reset_noise: Bool = True, action_noise: Float64 = SM_ACTION_NOISE, rng_seed: Int = 0,
    ) raises:
        seed(rng_seed)
        self.lanes = List[SMLane[Self.R]]()
        for _ in range(n):
            self.lanes.append(SMLane[Self.R]())
        self.sampler = SMResetSampler(lib.num_bins)
        var q0 = List[Float64](length=Self.R.NDOF, fill=0.0)
        for j in range(Self.R.NDOF):
            q0[j] = Float64(lib.q0[j])
        self.lib = lib^
        self.tables = sm_tables[DT, Self.R](self.lanes[0].sim, q0)
        self.dr = dr
        self.obs_noise = obs_noise
        self.reset_noise = reset_noise
        self.action_noise = action_noise
        self.last_reward = List[Float64](length=n, fill=0.0)
        self.last_terminated = List[Bool](length=n, fill=False)
        for i in range(n):
            ref ln = self.lanes[i]
            var cb = self.tables.com_body
            for c in range(3):
                ln.com_nominal[c] = ln.sim.body_field(cb, BODY_IDX_IPOS_X + c)
            for g in self.tables.feet:
                ln.sim.set_geom_field(g, GEOM_IDX_PRIORITY, 1.0)
            if dr:   # startup event: the CoM offset (once per env)
                var rng: List[Float64] = [0.025, 0.05, 0.05]
                for c in range(3):
                    var d = _u(-rng[c], rng[c])
                    ln.sim.set_body_field(cb, BODY_IDX_IPOS_X + c, ln.com_nominal[c] + d)
                    ln.xi[1 + Self.R.NDOF + c] = Scalar[DT](d)
            else:
                self._set_friction(i, 1.0)      # the evaluation protocol: friction fixed at 1
        var all = List[Int]()
        for i in range(n):
            all.append(i)
        self._reset(all)
        # mjlab's `reset()` runs the command update once too (`compute(dt=0)`):
        # the first target is the sampled frame + 1, as after an auto-reset
        for i in range(n):
            self._advance_frame(i)
        for i in range(n):
            self._observe(i)
            self._remember(i)
            self.lanes[i].ep_start = True

    def __init__(out self, *, deinit move: Self):
        self.lanes = move.lanes^
        self.lib = move.lib^
        self.sampler = move.sampler^
        self.tables = move.tables^
        self.dr = move.dr
        self.obs_noise = move.obs_noise
        self.reset_noise = move.reset_noise
        self.action_noise = move.action_noise
        self.last_reward = move.last_reward^
        self.last_terminated = move.last_terminated^

    def num_lanes(self) -> Int:
        return len(self.lanes)

    def _set_friction(mut self, i: Int, mu: Float64):
        ref ln = self.lanes[i]
        for g in self.tables.feet:
            ln.sim.set_geom_field(g, GEOM_IDX_FRICTION, mu)
        ln.xi[0] = Scalar[DT](mu)

    def _reset(mut self, ids: List[Int]) raises:
        """Reset events, a frame per lane, the reference state + noise."""
        if len(ids) == 0:
            return
        var p = self.sampler.probabilities()
        var cdf = List[Float64](length=len(p), fill=0.0)
        var acc = 0.0
        for b in range(len(p)):
            acc += p[b]
            cdf[b] = acc
        for i in ids:
            if self.dr:
                self._set_friction(i, _u(0.3, 1.2))
                for j in range(Self.R.NDOF):
                    self.lanes[i].xi[1 + j] = Scalar[DT](_u(-0.01, 0.01))
            var x = random_float64(0.0, acc)
            var b = 0
            while b < len(cdf) - 1 and cdf[b] < x:
                b += 1
            var f = SMResetSampler.frame_from[Self.R](self.lib, b, random_float64(0.0, 1.0))
            self._write_reference(i, f)
            ref ln = self.lanes[i]
            ln.frame = f
            ln.clip_end = False
            ln.ep_len = 0
            ln.log = SMEpisodeLog()
            ln.hist_fresh = True
            for j in range(Self.R.NDOF):
                ln.a_last[j] = 0

    def _write_reference(mut self, i: Int, f: Int) raises:
        comptime NQ = Self.R.NQ
        comptime n = Self.R.NDOF
        var r = f * SMDims[Self.R].S
        ref rows = self.lib.rows
        var pos = List[Float64](length=3, fill=0.0)
        var lin = List[Float64](length=3, fill=0.0)
        var ang = List[Float64](length=3, fill=0.0)
        for c in range(3):
            pos[c] = Float64(rows[r + c])
            lin[c] = Float64(rows[r + NQ + c])
            ang[c] = Float64(rows[r + NQ + 3 + c])
        var q = Q4[DType.float64](Float64(rows[r + 3]), Float64(rows[r + 4]), Float64(rows[r + 5]), Float64(rows[r + 6]))
        var qj = List[Float64](length=n, fill=0.0)
        for j in range(n):
            qj[j] = Float64(rows[r + 7 + j])
        if self.reset_noise:
            pos[0] += _u(-0.05, 0.05)
            pos[1] += _u(-0.05, 0.05)
            pos[2] += _u(-0.01, 0.01)
            var e = quat_from_euler_xyz(_u(-0.1, 0.1), _u(-0.1, 0.1), _u(-0.2, 0.2))
            q = qmul(e, q)
            lin[0] += _u(-0.5, 0.5)
            lin[1] += _u(-0.5, 0.5)
            lin[2] += _u(-0.2, 0.2)
            ang[0] += _u(-0.52, 0.52)
            ang[1] += _u(-0.52, 0.52)
            ang[2] += _u(-0.78, 0.78)
            for j in range(n):
                qj[j] += _u(-0.1, 0.1)
        ref ln = self.lanes[i]
        var qpos = List[Float64](length=NQ, fill=0.0)
        var qvel = List[Float64](length=Self.R.NV, fill=0.0)
        for c in range(3):
            qpos[c] = pos[c]
            qvel[c] = lin[c]
        for k in range(4):
            qpos[3 + k] = q[k]
        var wb = qapply_inv(q, Q4[DType.float64](ang[0], ang[1], ang[2], 0))   # the free joint's ω is in the body frame
        for c in range(3):
            qvel[3 + c] = wb[c]
        for j in range(n):
            var lim = sm_soft_limits[Self.R](ln.sim, j)
            qpos[7 + j] = min(max(qj[j], lim[0]), lim[1])
            qvel[6 + j] = Float64(rows[r + NQ + 6 + j])
        ln.sim.set_state(qpos, qvel)
        ln.read_state()

    def _observe(mut self, i: Int):
        """The noisy observation of the current state into the history."""
        comptime D = SMDims[Self.R]
        ref ln = self.lanes[i]
        var o = List[Scalar[DT]](length=D.OBS, fill=0)
        sm_observation_lane[DT, Self.R](
            self.tables.head_ptr(), ln.s.unsafe_ptr().as_unsafe_any_origin(), ln.a_last.unsafe_ptr().as_unsafe_any_origin(),
            ln.xi.unsafe_ptr().as_unsafe_any_origin(), o.unsafe_ptr().as_unsafe_any_origin(),
        )
        if self.obs_noise:
            for k in range(D.OBS):
                var h = sm_obs_noise_half[Self.R.NDOF](k)
                if h > 0:
                    o[k] += Scalar[DT](_u(-h, h))
        if ln.hist_fresh:
            for hh in range(D.HIST):
                for k in range(D.OBS):
                    ln.hist[hh * D.OBS + k] = o[k]
            ln.hist_fresh = False
        else:
            for hh in range(D.HIST - 1):
                for k in range(D.OBS):
                    ln.hist[hh * D.OBS + k] = ln.hist[(hh + 1) * D.OBS + k]
            for k in range(D.OBS):
                ln.hist[(D.HIST - 1) * D.OBS + k] = o[k]

    def _advance_frame(mut self, i: Int):
        """`_update_command`: frame + 1, the clip's end flagged once passed
        (judged against the CURRENT frame's clip)."""
        ref ln = self.lanes[i]
        var last = self.lib.last_frame_of(ln.frame)
        ln.frame += 1
        if ln.frame > last:
            ln.clip_end = True
            ln.frame = last

    def _remember(mut self, i: Int):
        ref ln = self.lanes[i]
        for k in range(SMDims[Self.R].S):
            ln.prev_s[k] = ln.s[k]
        for k in range(SMDims[Self.R].X):
            ln.prev_xi[k] = ln.xi[k]
        ln.prev_frame = ln.frame

    def policy_inputs(mut self, mut hist: TensorImpl[DT], mut goals: TensorImpl[DT]):
        """Every lane's history [L · 10 · OBS] and goal window [L · GOAL]
        (`ActStep`'s inputs)."""
        comptime D = SMDims[Self.R]
        for i in range(len(self.lanes)):
            for k in range(D.HIST * D.OBS):
                hist.data[i * D.HIST * D.OBS + k] = self.lanes[i].hist[k]
            self.lib.goal_window(self.lanes[i].frame, goals.data.unsafe_ptr().unsafe_offset(i * D.GOAL).as_unsafe_any_origin())

    def step(mut self, actions: List[Scalar[DT]], mut rb: SMReplayOf[Self.R]) raises -> SMStepInfo:
        """One control step for every lane; `actions` is [L·n] (clean)."""
        comptime D = SMDims[Self.R]
        comptime n = Self.R.NDOF
        var nl = len(self.lanes)
        var a_exec = List[Scalar[DT]](length=nl * n, fill=0)
        var resets = List[Int]()
        var ending = List[SMEpisodeLog]()
        var rsum = 0.0
        var n_term = 0
        var n_trunc = 0
        var terms = List[Scalar[DT]](length=SM_N_TERMS, fill=0)
        var feat = List[Scalar[DT]](length=2, fill=0)
        var obs_tmp = List[Scalar[DT]](length=D.OBS, fill=0)
        var scratch = List[Scalar[DT]](length=SMHeadLayout[Self.R].SCRATCH, fill=0)
        var rf = List[Scalar[DT]](length=D.REF, fill=0)
        var a_cmd = List[Float64](length=n, fill=0.0)
        var bias = List[Float64](length=n, fill=0.0)
        for i in range(nl):
            ref ln = self.lanes[i]
            # 1. the executed action through the PD (bias folded into the target)
            for j in range(n):
                var ae = Float64(actions[i * n + j])
                if self.action_noise > 0:
                    ae += self.action_noise * _randn()
                a_exec[i * n + j] = Scalar[DT](ae)
                a_cmd[j] = ae
                bias[j] = Float64(ln.xi[1 + j])
            ln.sim.control_step(a_cmd, bias)
            ln.read_state()
            ln.ep_len += 1
            # 2.–3. termination and reward against f_t
            self.lib.reference_row(ln.frame, self.tables, rf.unsafe_ptr().as_unsafe_any_origin())
            var a_now = List[Scalar[DT]](length=n, fill=0)
            for j in range(n):
                a_now[j] = actions[i * n + j]
            var r = sm_heads_lane[DT, Self.R](
                self.tables.tree_ptr(), self.tables.head_ptr(), ln.s.unsafe_ptr().as_unsafe_any_origin(),
                a_now.unsafe_ptr().as_unsafe_any_origin(), ln.a_last.unsafe_ptr().as_unsafe_any_origin(),
                ln.xi.unsafe_ptr().as_unsafe_any_origin(), rf.unsafe_ptr().as_unsafe_any_origin(),
                terms.unsafe_ptr().as_unsafe_any_origin(), feat.unsafe_ptr().as_unsafe_any_origin(),
                obs_tmp.unsafe_ptr().as_unsafe_any_origin(), scratch.unsafe_ptr().as_unsafe_any_origin(),
            )
            var terminated = abs(Float64(feat[0])) > SM_HEIGHT_TERM or abs(Float64(feat[1])) > SM_TILT_TERM
            var time_out = ln.ep_len >= SM_EPISODE_STEPS or ln.clip_end
            self.last_reward[i] = Float64(r)
            self.last_terminated[i] = terminated
            rsum += Float64(r)
            # the sampler's per-step log, with f_t's bin
            if len(ln.log.rewards) < SM_MAX_LOGGED:
                ln.log.rewards.append(Float64(r))
                ln.log.bins.append(self.lib.bin_of_frame(ln.frame))
            else:
                ln.log.rewards[SM_MAX_LOGGED - 1] = Float64(r)
                ln.log.bins[SM_MAX_LOGGED - 1] = self.lib.bin_of_frame(ln.frame)
            for j in range(n):
                ln.a_last[j] = a_now[j]
            if terminated or time_out:
                # 4. record_pre_reset: the last transition and the terminal row
                var ae_l = List[Scalar[DT]](length=n, fill=0)
                for j in range(n):
                    ae_l[j] = a_exec[i * n + j]
                var lbl = 1.0 if terminated else 0.0
                rb.add(i, ln.prev_s, a_now, ae_l, ln.prev_xi, Float64(r), lbl, ln.prev_frame, False, ln.ep_start)
                var zero = List[Scalar[DT]](length=n, fill=0)
                rb.add(i, ln.s, zero, zero, ln.prev_xi, 0.0, lbl, ln.frame, True, False)
                rb.end_episode(i)
                ln.log.terminated = terminated
                ln.log.clip_end = ln.clip_end
                ending.append(ln.log.copy())
                resets.append(i)
                if terminated:
                    n_term += 1
                else:
                    n_trunc += 1
        self.sampler.finalize(ending)
        self._reset(resets)
        # 5. every lane's command update
        for i in range(nl):
            self._advance_frame(i)
        # 6. observations, then the recorder's post-step / post-reset rows
        var is_reset = List[Bool](length=nl, fill=False)
        for i in resets:
            is_reset[i] = True
        for i in range(nl):
            self._observe(i)
            ref ln = self.lanes[i]
            if not is_reset[i]:
                var a_now = List[Scalar[DT]](length=n, fill=0)
                var ae_l = List[Scalar[DT]](length=n, fill=0)
                for j in range(n):
                    a_now[j] = actions[i * n + j]
                    ae_l[j] = a_exec[i * n + j]
                rb.add(i, ln.prev_s, a_now, ae_l, ln.prev_xi, self.last_reward[i], 0.0, ln.prev_frame, False, ln.ep_start)
                ln.ep_start = False
            else:
                ln.ep_start = True
            self._remember(i)
        return SMStepInfo(rsum / Float64(nl), n_term, n_trunc)
