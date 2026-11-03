# +--------------------------------------------------------------------------+ #
# | SuperMimic M2c: the tracking environment on our CPU G1
# +--------------------------------------------------------------------------+ #
"""The reproduction's mjlab environment (`envs/tracking_env_cfg.py`, the
`MultiClipMotionCommand`, `NoisyJointPositionAction`, `SimContext` and the
`ReplayRecorder`) on noeira's BFM-Zero G1, one CPU simulator per lane.

ONE STEP, in mjlab's `ManagerBasedRlEnv.step` order:
    1. a_exec = a + N(0, 0.1²); the simulator gets the PD target
       q0 + 0.25·effort/kp · a_exec − bias (4 substeps of 1/200 s, PD at each)
    2. ep_len += 1; terminated = |h_ref − h| > 0.25 or |tilt| > 0.8, against
       the step's target frame f_t; time-out at 500 steps or at the clip's end
    3. r = the 8 BeyondMimic terms of the new state (the heads, on the SIM
       state) against f_t; logged with f_t's bin for the reset sampler
    4. ending lanes: the transition row and a terminal row into the replay;
       their episodes folded into the sampler; reset (ξ re-drawn, a frame
       from the sampler, the reference state there + noise)
    5. every lane's frame += 1 (a lane just reset starts at its frame + 1, as
       mjlab's command update runs for every env after the resets); the
       clip's end is flagged when passed
    6. observation (93, noisy) into the 10-step history (a reset lane's first
       observation fills all 10); the next rows are remembered

WHAT IS OURS (plan D-1), NOT THE REPRODUCTION'S
* The body, its PD gains and default pose (BFM-Zero's). The ACTION SCALE is
  mjlab's: target = q0 + 0.25·effort/kp·a. Our env multiplies by 5 and clips
  at ±5 first, so it is handed a/5 (its clip binds at |a| > 5 only).
* The encoder bias enters the PD target as in mjlab (`target − bias`) and the
  observation (`q + bias`), not the physics.
* FOOT FRICTION: mjlab's G1 gives its foot geoms priority 1, so their friction
  is used as is. Our feet have priority 0 and friction 1, which MuJoCo
  combines with the floor's by max — randomising them would do nothing. This
  env sets the 8 foot spheres' priority to 1 and their friction to ξ's.
* Rewards and terminations read the state exactly; mjlab's read derived
  kinematics one substep stale (its documented trade-off).
"""

from std.math import cos, log, sin, sqrt
from std.random import random_float64, seed

from noeira.core.cont_action import ContAction
from noeira.deep_agents.supermimic.g1_fk import G1FKTree, Q4, g1_fk_tree, qmul
from noeira.deep_agents.supermimic.heads import (
    G1HeadsTable, SM_N_TERMS, SM_REF_DIM, SM_SCRATCH, g1_heads_lane, g1_heads_table,
    g1_soft_limits, qapply_inv, sm_observation_lane, sm_term_features,
)
from noeira.deep_agents.supermimic.motion_library import (
    SMEpisodeLog, SMMotionLibrary, SMResetSampler,
)
from noeira.deep_agents.supermimic.policy import SM_HIST, SM_OBS, sm_obs_noise_half
from noeira.deep_agents.supermimic.replay import SMReplayBuffer
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos, g1_effort, g1_kp
from noeira.envs.robots.unitree_g1_xml import (
    LEFT_ANKLE_ROLL_BODY_IDX, RIGHT_ANKLE_ROLL_BODY_IDX, TORSO_BODY_IDX,
)
from noeira.nn.constants import DT
from noeira.physics3d.constants import GEOM_SPHERE
from noeira.physics3d.gpu.constants import (
    BODY_IDX_IPOS_X, BODY_IDX_PARENT, GEOM_IDX_BODY, GEOM_IDX_CONAFFINITY, GEOM_IDX_CONTYPE,
    GEOM_IDX_FRICTION, GEOM_IDX_PRIORITY, GEOM_IDX_TYPE, MODEL_BODY_SIZE, MODEL_GEOM_SIZE,
)


comptime SM_EPISODE_STEPS: Int = 500        # 10 s at 50 Hz
comptime SM_MAX_LOGGED: Int = 502           # min(ceil(10 / 0.02) + 2, 1024)
comptime SM_ACTION_NOISE: Float64 = 0.1
comptime SM_HEIGHT_TERM: Float64 = 0.25
comptime SM_TILT_TERM: Float64 = 0.8


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


struct SMLane(Movable):
    var sim: UnitreeG1[DType.float64]
    var frame: Int
    var clip_end: Bool
    var ep_len: Int
    var log: SMEpisodeLog
    var xi: List[Scalar[DT]]           # 33: friction | bias 29 | torso CoM offset 3
    var com_nominal: List[Float64]
    var hist: List[Scalar[DT]]         # 10 × 93, oldest first
    var hist_fresh: Bool
    var a_last: List[Scalar[DT]]       # the clean action of the last step (0 after a reset)
    var s: List[Scalar[DT]]            # the current state (71)
    var prev_s: List[Scalar[DT]]
    var prev_xi: List[Scalar[DT]]
    var prev_frame: Int
    var ep_start: Bool

    def __init__(out self) raises:
        self.sim = UnitreeG1[DType.float64]()
        _ = self.sim.reset()
        self.frame = 0
        self.clip_end = False
        self.ep_len = 0
        self.log = SMEpisodeLog()
        self.xi = List[Scalar[DT]](length=33, fill=0)
        self.com_nominal = List[Float64](length=3, fill=0.0)
        self.hist = List[Scalar[DT]](length=SM_HIST * SM_OBS, fill=0)
        self.hist_fresh = True
        self.a_last = List[Scalar[DT]](length=G1_N_DOF, fill=0)
        self.s = List[Scalar[DT]](length=71, fill=0)
        self.prev_s = List[Scalar[DT]](length=71, fill=0)
        self.prev_xi = List[Scalar[DT]](length=33, fill=0)
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
        for k in range(36):
            self.s[k] = Scalar[DT](self.sim.d.qpos.data[k])
        for k in range(35):
            self.s[36 + k] = Scalar[DT](self.sim.d.qvel.data[k])


@fieldwise_init
struct SMStepInfo(Copyable, Movable):
    var reward_mean: Float64
    var n_terminated: Int
    var n_truncated: Int


struct SMG1Env(Movable):
    var lanes: List[SMLane]
    var lib: SMMotionLibrary
    var sampler: SMResetSampler
    var tree: G1FKTree[DT]
    var table: G1HeadsTable[DT]
    var foot_geoms: List[Int]
    var dr: Bool
    var obs_noise: Bool
    var reset_noise: Bool
    var action_noise: Float64
    var last_reward: List[Float64]
    var last_terminated: List[Bool]

    def __init__(
        out self, n: Int, var lib: SMMotionLibrary, dr: Bool = True, obs_noise: Bool = True,
        reset_noise: Bool = True, action_noise: Float64 = SM_ACTION_NOISE, rng_seed: Int = 0,
    ) raises:
        seed(rng_seed)
        self.lanes = List[SMLane]()
        for _ in range(n):
            self.lanes.append(SMLane())
        self.sampler = SMResetSampler(lib.num_bins)
        self.lib = lib^
        var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
        for j in range(G1_N_DOF):
            q0[j] = g1_default_pos(j)
        self.tree = g1_fk_tree[DT](self.lanes[0].sim)
        self.table = g1_heads_table[DT](self.lanes[0].sim, q0)
        self.foot_geoms = List[Int]()
        self.dr = dr
        self.obs_noise = obs_noise
        self.reset_noise = reset_noise
        self.action_noise = action_noise
        self.last_reward = List[Float64](length=n, fill=0.0)
        self.last_terminated = List[Bool](length=n, fill=False)
        # the 8 foot spheres (as in bones_seed_filter): on an ankle-roll link or a frame hanging off one
        comptime E = UnitreeG1[DType.float64]
        ref sim0 = self.lanes[0].sim
        for g in range(E.NGEOM):
            var o = g * MODEL_GEOM_SIZE
            if Int(sim0.mf.geoms.data[o + GEOM_IDX_TYPE]) != GEOM_SPHERE:
                continue
            if Int(sim0.mf.geoms.data[o + GEOM_IDX_CONTYPE]) == 0 and Int(sim0.mf.geoms.data[o + GEOM_IDX_CONAFFINITY]) == 0:
                continue
            var b = Int(sim0.mf.geoms.data[o + GEOM_IDX_BODY])
            var side = b
            if side != LEFT_ANKLE_ROLL_BODY_IDX and side != RIGHT_ANKLE_ROLL_BODY_IDX:
                side = Int(sim0.mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            if side == LEFT_ANKLE_ROLL_BODY_IDX or side == RIGHT_ANKLE_ROLL_BODY_IDX:
                self.foot_geoms.append(g)
        if len(self.foot_geoms) != 8:
            raise Error("SMG1Env: expected 8 foot spheres, found " + String(len(self.foot_geoms)))
        for i in range(n):
            ref ln = self.lanes[i]
            var ob = TORSO_BODY_IDX * MODEL_BODY_SIZE + BODY_IDX_IPOS_X
            for c in range(3):
                ln.com_nominal[c] = Float64(ln.sim.mf.bodies.data[ob + c])
            for g in self.foot_geoms:
                ln.sim.mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_PRIORITY] = 1.0
            # startup event: the torso CoM offset (once per env)
            if dr:
                var rng: List[Float64] = [0.025, 0.05, 0.05]
                for c in range(3):
                    var d = _u(-rng[c], rng[c])
                    ln.sim.mf.bodies.data[ob + c] = ln.com_nominal[c] + d
                    ln.xi[30 + c] = Scalar[DT](d)
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
        self.tree = move.tree^
        self.table = move.table^
        self.foot_geoms = move.foot_geoms^
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
        for g in self.foot_geoms:
            ln.sim.mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_FRICTION] = mu
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
                for j in range(G1_N_DOF):
                    self.lanes[i].xi[1 + j] = Scalar[DT](_u(-0.01, 0.01))
            var x = random_float64(0.0, acc)
            var b = 0
            while b < len(cdf) - 1 and cdf[b] < x:
                b += 1
            var f = SMResetSampler.frame_from(self.lib, b, random_float64(0.0, 1.0))
            self._write_reference(i, f)
            ref ln = self.lanes[i]
            ln.frame = f
            ln.clip_end = False
            ln.ep_len = 0
            ln.log = SMEpisodeLog()
            ln.hist_fresh = True
            for j in range(G1_N_DOF):
                ln.a_last[j] = 0

    def _write_reference(mut self, i: Int, f: Int) raises:
        var r = f * 71
        ref rows = self.lib.rows
        var pos = List[Float64](length=3, fill=0.0)
        var lin = List[Float64](length=3, fill=0.0)
        var ang = List[Float64](length=3, fill=0.0)
        for c in range(3):
            pos[c] = Float64(rows[r + c])
            lin[c] = Float64(rows[r + 36 + c])
            ang[c] = Float64(rows[r + 39 + c])
        var q = Q4[DType.float64](Float64(rows[r + 3]), Float64(rows[r + 4]), Float64(rows[r + 5]), Float64(rows[r + 6]))
        var qj = List[Float64](length=G1_N_DOF, fill=0.0)
        for j in range(G1_N_DOF):
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
            for j in range(G1_N_DOF):
                qj[j] += _u(-0.1, 0.1)
        ref ln = self.lanes[i]
        var qpos = List[Float64](length=36, fill=0.0)
        var qvel = List[Float64](length=35, fill=0.0)
        for c in range(3):
            qpos[c] = pos[c]
            qvel[c] = lin[c]
        for k in range(4):
            qpos[3 + k] = q[k]
        var wb = qapply_inv(q, Q4[DType.float64](ang[0], ang[1], ang[2], 0))   # the free joint's ω is in the body frame
        for c in range(3):
            qvel[3 + c] = wb[c]
        for j in range(G1_N_DOF):
            var lim = g1_soft_limits(ln.sim, j)
            qpos[7 + j] = min(max(qj[j], lim[0]), lim[1])
            qvel[6 + j] = Float64(rows[r + 42 + j])
        ln.sim.set_state(qpos, qvel)
        ln.read_state()

    def _observe(mut self, i: Int):
        """The noisy 93-d observation of the current state into the history."""
        ref ln = self.lanes[i]
        var o = List[Scalar[DT]](length=SM_OBS, fill=0)
        sm_observation_lane[DT](
            self.table.ptr(), ln.s.unsafe_ptr().as_unsafe_any_origin(), ln.a_last.unsafe_ptr().as_unsafe_any_origin(),
            ln.xi.unsafe_ptr().as_unsafe_any_origin(), o.unsafe_ptr().as_unsafe_any_origin(),
        )
        if self.obs_noise:
            for k in range(SM_OBS):
                var h = sm_obs_noise_half(k)
                if h > 0:
                    o[k] += Scalar[DT](_u(-h, h))
        if ln.hist_fresh:
            for hh in range(SM_HIST):
                for k in range(SM_OBS):
                    ln.hist[hh * SM_OBS + k] = o[k]
            ln.hist_fresh = False
        else:
            for hh in range(SM_HIST - 1):
                for k in range(SM_OBS):
                    ln.hist[hh * SM_OBS + k] = ln.hist[(hh + 1) * SM_OBS + k]
            for k in range(SM_OBS):
                ln.hist[(SM_HIST - 1) * SM_OBS + k] = o[k]

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
        for k in range(71):
            ln.prev_s[k] = ln.s[k]
        for k in range(33):
            ln.prev_xi[k] = ln.xi[k]
        ln.prev_frame = ln.frame

    def goal(mut self, i: Int, mut dst: List[Scalar[DT]]):
        """The 680-d goal window of lane i's current target frame."""
        self.lib.goal_window(self.lanes[i].frame, dst.unsafe_ptr().as_unsafe_any_origin())

    def step(mut self, actions: List[Scalar[DT]], mut rb: SMReplayBuffer) raises -> SMStepInfo:
        """One control step for every lane; `actions` is [n·29] (clean)."""
        var n = len(self.lanes)
        var a_exec = List[Scalar[DT]](length=n * G1_N_DOF, fill=0)
        var resets = List[Int]()
        var ending = List[SMEpisodeLog]()
        var rsum = 0.0
        var n_term = 0
        var n_trunc = 0
        var terms = List[Scalar[DT]](length=SM_N_TERMS, fill=0)
        var feat = List[Scalar[DT]](length=2, fill=0)
        var obs_tmp = List[Scalar[DT]](length=SM_OBS, fill=0)
        var scratch = List[Scalar[DT]](length=SM_SCRATCH, fill=0)
        var rf = List[Scalar[DT]](length=SM_REF_DIM, fill=0)
        for i in range(n):
            ref ln = self.lanes[i]
            # 1. the executed action through the PD (bias folded into the target)
            var act = ContAction[G1_N_DOF]()
            for j in range(G1_N_DOF):
                var ae = Float64(actions[i * G1_N_DOF + j])
                if self.action_noise > 0:
                    ae += self.action_noise * _randn()
                a_exec[i * G1_N_DOF + j] = Scalar[DT](ae)
                act.data[j] = ae / 5.0 - Float64(ln.xi[1 + j]) * g1_kp(j) / (1.25 * g1_effort(j))
            _ = ln.sim.step(act)
            ln.read_state()
            ln.ep_len += 1
            # 2.–3. termination and reward against f_t
            self.lib.reference_row(ln.frame, self.tree, self.table, rf.unsafe_ptr().as_unsafe_any_origin())
            var a_now = List[Scalar[DT]](length=G1_N_DOF, fill=0)
            for j in range(G1_N_DOF):
                a_now[j] = actions[i * G1_N_DOF + j]
            var r = g1_heads_lane[DT](
                self.tree.ptr(), self.table.ptr(), ln.s.unsafe_ptr().as_unsafe_any_origin(),
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
            for j in range(G1_N_DOF):
                ln.a_last[j] = a_now[j]
            if terminated or time_out:
                # 4. record_pre_reset: the last transition and the terminal row
                var ae_l = List[Scalar[DT]](length=G1_N_DOF, fill=0)
                for j in range(G1_N_DOF):
                    ae_l[j] = a_exec[i * G1_N_DOF + j]
                var lbl = 1.0 if terminated else 0.0
                rb.add(i, ln.prev_s, a_now, ae_l, ln.prev_xi, Float64(r), lbl, ln.prev_frame, False, ln.ep_start)
                var zero = List[Scalar[DT]](length=G1_N_DOF, fill=0)
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
        for i in range(n):
            self._advance_frame(i)
        # 6. observations, then the recorder's post-step / post-reset rows
        var is_reset = List[Bool](length=n, fill=False)
        for i in resets:
            is_reset[i] = True
        for i in range(n):
            self._observe(i)
            ref ln = self.lanes[i]
            if not is_reset[i]:
                var a_now = List[Scalar[DT]](length=G1_N_DOF, fill=0)
                var ae_l = List[Scalar[DT]](length=G1_N_DOF, fill=0)
                for j in range(G1_N_DOF):
                    a_now[j] = actions[i * G1_N_DOF + j]
                    ae_l[j] = a_exec[i * G1_N_DOF + j]
                rb.add(i, ln.prev_s, a_now, ae_l, ln.prev_xi, self.last_reward[i], 0.0, ln.prev_frame, False, ln.ep_start)
                ln.ep_start = False
            else:
                ln.ep_start = True
            self._remember(i)
        return SMStepInfo(rsum / Float64(n), n_term, n_trunc)
