# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the device tracking env that carries it
# +--------------------------------------------------------------------------+ #
"""`env_gpu.SMTrackingEnvGpu` over a simulator that also holds ONE free object
after the robot (the carry-box family, `G1MjlabBoxBatchedSim`), and the
object's side of the replay. `docs/OBJECT_INTERACTION_PLAN.md` O1.2 / O1.3.

The robot's env is used UNCHANGED: its reward, termination, observation and
reset sampler never see the object (the tracker is blind to it). This file
adds what the object needs around it:

    the reference   `sm_object_track`: [frames · 13] = pose, linear velocity
                    (world) and angular velocity (its own frame) per library
                    frame, from the object's pose track
    at a reset      the object goes to its reference state at the lane's
                    reset frame (`_k_obj_inject`), and with `friction_dr` the
                    lane draws the object's friction, U(0.5, 1.3), as the
                    object body's contact-friction override
    at every step   the object's state is read from the simulator AFTER the
                    control step and BEFORE the resets (`_k_obj_read`) and
                    written to the object's ring

THE OBJECT'S RING (`SMObjReplayGpu`) holds, for row (e, t) of the robot's ring
(`EpisodicReplayGpu`), the object's state at that row's robot state and the
context (mass, friction) in force: 15 numbers. It has NO pointer of its own:
`add_step` runs after the robot ring's and reads ITS pointer, so the two cannot
drift apart, and a flat row index sampled from the robot's ring (`sample_windows`)
addresses the object's ring too (`windows_at`).

⚠ ONE RULE OF THE ROBOT'S RING IS RESTATED HERE: a step writes one transition
row per env, plus a terminal row when the episode ends (`_k_add_step`).
`_k_obj_add` derives the rows from the pointer after that write and the same
`META_ENDING`. `test_supermimic_obj_env_gpu` holds the two together.

WHERE THE OBJECT IS: the simulator's last body, a free one, so its qpos is at
the robot's `NQ`, its qvel at the robot's `NV` and its body index the robot's
`NBODY + 1`. The struct asserts the simulator's widths say exactly that.
"""

from std.math import atan2, ceildiv, sqrt

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.data.episodic_replay_gpu import META_ENDING
from noeira.deep_agents.supermimic.env import SM_ACTION_NOISE, SM_EPISODE_STEPS, SMStepInfo
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf, SMTrackingEnvGpu
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_hash_u01, talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.robot import TrackingBatchedSim, TrackingRobot
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.constraints.lane_dr import LANE_DR_FRICTION, LANE_DR_STRIDE
from noeira.physics3d.gpu.constants import BODY_IDX_MASS, GEOM_IDX_FRICTION


comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 64

comptime SM_OBJ_ROW: Int = SM_OBJ + SM_OBJ_CTX
"""A row of the object's ring: its state (13), then its context (2)."""

comptime SM_OBJ_MU_MID: Float64 = 0.9     # the object's friction per lane: U(mid − half, mid + half)
comptime SM_OBJ_MU_HALF: Float64 = 0.4
comptime _RNG_OBJ_MU: UInt32 = 7          # a hash stream the robot's env does not use (`env_gpu.RNG_*`: 1..6)


# ── the reference track ─────────────────────────────────────────────────────
def sm_object_track(poses: List[Float64], clip_len: List[Int], fps: Float64 = 50.0) raises -> List[Scalar[DT]]:
    """The object's reference state per library frame, [frames · 13], from its
    pose per frame (`poses`: [frames · 7] = position, quaternion WXYZ, the
    clips one after the other as in the library).

    Velocities are the finite difference to the NEXT frame of the same clip
    (a clip's last frame takes the one before it): linear in the world, and
    angular in the object's own frame, i.e. the rotation vector of
    `conj(q_f) ⊗ q_{f+1}` times fps. That is MuJoCo's free-joint qvel, so a
    row is the simulator's (qpos 7, qvel 6) of the object as it stands."""
    var total = 0
    for c in range(len(clip_len)):
        total += clip_len[c]
    if len(poses) != total * 7:
        raise Error("sm_object_track: " + String(len(poses)) + " pose values for " + String(total) + " frames (7 each)")
    var out = List[Scalar[DT]](length=total * SM_OBJ, fill=0)
    var f0 = 0
    for c in range(len(clip_len)):
        var n = clip_len[c]
        for f in range(n):
            var a = (f0 + f) * 7
            var qn = sqrt(poses[a + 3] * poses[a + 3] + poses[a + 4] * poses[a + 4] + poses[a + 5] * poses[a + 5] + poses[a + 6] * poses[a + 6])
            if qn < 1e-9:
                raise Error("sm_object_track: a zero quaternion at frame " + String(f) + " of clip " + String(c))
            var o = (f0 + f) * SM_OBJ
            for k in range(3):
                out[o + k] = Scalar[DT](poses[a + k])
            for k in range(4):
                out[o + 3 + k] = Scalar[DT](poses[a + 3 + k] / qn)
            if n < 2:
                continue
            var i0 = f if f + 1 < n else f - 1
            var pa = (f0 + i0) * 7
            var pb = pa + 7
            for k in range(3):
                out[o + 7 + k] = Scalar[DT]((poses[pb + k] - poses[pa + k]) * fps)
            var aw = poses[pa + 3]
            var ax = poses[pa + 4]
            var ay = poses[pa + 5]
            var az = poses[pa + 6]
            var bw = poses[pb + 3]
            var bx = poses[pb + 4]
            var by = poses[pb + 5]
            var bz = poses[pb + 6]
            # conj(a) ⊗ b
            var dw = aw * bw + ax * bx + ay * by + az * bz
            var dx = aw * bx - ax * bw - (ay * bz - az * by)
            var dy = aw * by - ay * bw - (az * bx - ax * bz)
            var dz = aw * bz - az * bw - (ax * by - ay * bx)
            if dw < 0.0:          # the short way round
                dw = -dw
                dx = -dx
                dy = -dy
                dz = -dz
            var s = sqrt(dx * dx + dy * dy + dz * dz)
            if s > 1e-12:
                var g = 2.0 * atan2(s, dw) / s * fps
                out[o + 10] = Scalar[DT](dx * g)
                out[o + 11] = Scalar[DT](dy * g)
                out[o + 12] = Scalar[DT](dz * g)
        f0 += n
    return out^


# ── kernels ─────────────────────────────────────────────────────────────────
def _k_obj_read[OQ: Int, OV: Int, QS: Int, VS: Int](nl: Int64, qpos: FP[DT], qvel: FP[DT], o: FP[DT]):
    """o[l] = lane l's object state, from the simulator."""
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    comptime for k in range(7):
        o[unsafe_offset=l * SM_OBJ + k] = qpos[unsafe_offset=l * QS + OQ + k]
    comptime for k in range(6):
        o[unsafe_offset=l * SM_OBJ + 7 + k] = qvel[unsafe_offset=l * VS + OV + k]


def _k_obj_inject[OQ: Int, OV: Int, OB: Int, QS: Int, VS: Int, NB: Int](
    nl: Int64, seed: UInt32, dr: Int64, mass: Scalar[DT], mu0: Scalar[DT],
    orows: FP[DT], reset_f: FPi, qpos: FP[DT], qvel: FP[DT], lane_dr: FP[DT], octx: FP[DT],
):
    """The lanes the robot's env just reset (`reset_f[l]` ≥ 0): the object at
    its reference state of that frame, its friction drawn, its context."""
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var f = Int(reset_f[unsafe_offset=l])
    if f < 0:
        return
    comptime for k in range(7):
        qpos[unsafe_offset=l * QS + OQ + k] = orows[unsafe_offset=f * SM_OBJ + k]
    comptime for k in range(6):
        qvel[unsafe_offset=l * VS + OV + k] = orows[unsafe_offset=f * SM_OBJ + 7 + k]
    var mu = mu0
    if dr != 0:
        var u = Scalar[DT](2) * Scalar[DT](sm_hash_u01(seed, UInt32(l))) - 1
        mu = Scalar[DT](SM_OBJ_MU_MID) + Scalar[DT](SM_OBJ_MU_HALF) * u
        lane_dr[unsafe_offset=(l * NB + OB) * LANE_DR_STRIDE + LANE_DR_FRICTION] = mu
    octx[unsafe_offset=l * SM_OBJ_CTX] = mass
    octx[unsafe_offset=l * SM_OBJ_CTX + 1] = mu


def _k_obj_add(E: Int64, T: Int64, ptr: FPi, meta: FPi, o0: FP[DT], o1: FP[DT], c0: FP[DT], ring: FP[DT]):
    """Env e's rows of the step the robot's ring just took: `ptr` is that
    ring's pointer AFTER its write. The transition row gets (o0, c0); an
    ending env's terminal row, the next one, gets (o1, c0)."""
    var e = Int(global_idx.x)
    if e >= Int(E):
        return
    var t_ = Int(T)
    var ending = meta[unsafe_offset=e * 4 + META_ENDING] != 0
    var rows = 2 if ending else 1
    var t0 = (Int(ptr[unsafe_offset=e]) - rows + 2 * t_) % t_
    var r = ring.unsafe_offset((e * t_ + t0) * SM_OBJ_ROW)
    for k in range(SM_OBJ):
        r[unsafe_offset=k] = o0[unsafe_offset=e * SM_OBJ + k]
    for k in range(SM_OBJ_CTX):
        r[unsafe_offset=SM_OBJ + k] = c0[unsafe_offset=e * SM_OBJ_CTX + k]
    if ending:
        var r1 = ring.unsafe_offset((e * t_ + (t0 + 1) % t_) * SM_OBJ_ROW)
        for k in range(SM_OBJ):
            r1[unsafe_offset=k] = o1[unsafe_offset=e * SM_OBJ + k]
        for k in range(SM_OBJ_CTX):
            r1[unsafe_offset=SM_OBJ + k] = c0[unsafe_offset=e * SM_OBJ_CTX + k]


def _k_obj_windows(n: Int64, k: Int64, T: Int64, idx: FPi, ring: FP[DT], o_s: FP[DT], o_c: FP[DT]):
    """`EpisodicReplayGpu._k_windows` for the object's ring, one thread per
    window: states t-major [(k+1)·n·13], the first row's context [n·2]."""
    var c = Int(global_idx.x)
    var nn = Int(n)
    if c >= nn:
        return
    var t_ = Int(T)
    var row0 = Int(idx[unsafe_offset=c])
    if row0 < 0:
        return
    var e = row0 // t_
    var t = row0 % t_
    for q in range(SM_OBJ_CTX):
        o_c[unsafe_offset=c * SM_OBJ_CTX + q] = ring[unsafe_offset=row0 * SM_OBJ_ROW + SM_OBJ + q]
    for j in range(Int(k) + 1):
        var i = e * t_ + (t + j) % t_
        for q in range(SM_OBJ):
            o_s[unsafe_offset=(j * nn + c) * SM_OBJ + q] = ring[unsafe_offset=i * SM_OBJ_ROW + q]


# ── the object's ring ───────────────────────────────────────────────────────
struct SMObjReplayGpu(Movable):
    """The object's column of a robot ring of the same `[env, time]` shape."""
    var E: Int
    var T: Int
    var ring: DeviceBuffer[DT]        # [E · T · SM_OBJ_ROW]

    def __init__(out self, ctx: DeviceContext, num_envs: Int, capacity: Int) raises:
        if capacity < 2:
            raise Error("SMObjReplayGpu: a capacity of at least 2 rows per env")
        self.E = num_envs
        self.T = capacity
        self.ring = ctx.enqueue_create_buffer[DT](num_envs * capacity * SM_OBJ_ROW)
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.E = move.E
        self.T = move.T
        self.ring = move.ring^

    def add_step(
        mut self, ctx: DeviceContext, rb_ptr: FPi, rb_envs: Int, rb_capacity: Int, meta: FPi,
        o0: FP[DT], o1: FP[DT], c0: FP[DT],
    ) raises:
        """The object's rows of the step the robot's ring JUST took (call it
        right after `rb.add_step`, with the same `meta`). `rb_ptr` is that
        ring's `ptr`, `rb_envs` / `rb_capacity` its shape; o0 [E·13] the
        object at the transition's state, o1 [E·13] at the final state (read
        for ending envs only), c0 [E·2] the context in force during the step."""
        if rb_envs != self.E or rb_capacity != self.T:
            raise Error(
                "SMObjReplayGpu.add_step: a robot ring of " + String(rb_envs) + " x " + String(rb_capacity) + ", this one is "
                + String(self.E) + " x " + String(self.T)
            )
        ctx.enqueue_function[_k_obj_add](
            Int64(self.E), Int64(self.T), rb_ptr, meta, o0, o1, c0, mptr(self.ring.unsafe_ptr()),
            grid_dim=ceildiv(self.E, _TPB), block_dim=_TPB,
        )

    def windows_at(mut self, ctx: DeviceContext, idx: FPi, n: Int, k: Int, o_states: FP[DT], o_ctx: FP[DT]) raises:
        """The object's side of `EpisodicReplayGpu.windows_at` for the same
        flat rows `idx`: o_states [(k+1)·n·13] t-major, o_ctx [n·2]."""
        ctx.enqueue_function[_k_obj_windows](
            Int64(n), Int64(k), Int64(self.T), idx, mptr(self.ring.unsafe_ptr()), o_states, o_ctx,
            grid_dim=ceildiv(n, _TPB), block_dim=_TPB,
        )


# ── the env ─────────────────────────────────────────────────────────────────
struct SMObjTrackingEnvGpu[R: TrackingRobot, BS: TrackingBatchedSim](Movable):
    """The robot's device env plus the object: see the module docstring.
    `env` is the robot's (its `hist`, `policy_inputs`, `sampler`, `logs` are
    used as they are); `o` / `octx` are the object's state and context per
    lane as of the last `step`, before that step's resets."""
    comptime N = Self.BS.N
    comptime OQ = Self.R.NQ
    comptime OV = Self.R.NV
    comptime OB = Self.R.NBODY + 1
    var env: SMTrackingEnvGpu[Self.R, Self.BS]
    var orows: DeviceBuffer[DT]          # the object's reference, [frames · 13]
    var o: TensorImpl[DT]                # [N · 13] after the last control step, before its resets
    var prev_o: TensorImpl[DT]           # [N · 13] at every lane's current state (the next row's)
    var octx: TensorImpl[DT]             # [N · 2] mass, friction in force
    var prev_octx: TensorImpl[DT]        # … during the step being logged
    var mass: Float64
    var mu0: Float64                     # the model's friction of the object's geom
    var friction_dr: Bool
    var seed_base: UInt32

    def __init__(
        out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.R], track: List[Scalar[DT]], var sim: Self.BS,
        dr: Bool = True, obs_noise: Bool = True, reset_noise: Bool = True,
        action_noise: Float64 = SM_ACTION_NOISE, friction_period: Int = SM_EPISODE_STEPS, rng_seed: Int = 0,
        max_episode_steps: Int = SM_EPISODE_STEPS, friction_dr: Bool = True,
    ) raises:
        """`track` is `sm_object_track`'s rows for `lib`'s frames; the other
        arguments are the robot env's. `friction_dr` needs a simulator with
        `LANE_DR` (without it the model's friction stays)."""
        comptime assert Self.BS.SIM_NQ == Self.R.NQ + 7, "the simulator must hold one free object after the robot (qpos)"
        comptime assert Self.BS.SIM_NV == Self.R.NV + 6, "the simulator must hold one free object after the robot (qvel)"
        comptime assert Self.BS.SIM_NBODY == Self.R.NBODY + 2, "the simulator must hold one body after the robot's (world included)"
        comptime NL = Self.BS.N
        if len(track) != lib.num_frames * SM_OBJ:
            raise Error(
                "SMObjTrackingEnvGpu: an object track of " + String(len(track) // SM_OBJ) + " frames for a library of "
                + String(lib.num_frames)
            )
        self.orows = ctx.enqueue_create_buffer[DT](len(track))
        var ht = ctx.enqueue_create_host_buffer[DT](len(track))
        ctx.synchronize()
        for i in range(len(track)):
            ht[i] = track[i]
        ctx.enqueue_copy(self.orows, ht)
        ctx.synchronize()
        # the object's mass and its geom's friction, from a CPU lane of the same model (its last body and geom)
        var cpu = Self.R.SIM.create()
        self.mass = cpu.body_field(Self.OB, BODY_IDX_MASS)
        self.mu0 = cpu.geom_field(cpu.num_geoms() - 1, GEOM_IDX_FRICTION)
        self.friction_dr = friction_dr
        self.seed_base = UInt32(rng_seed)
        self.o = talloc["gpu"](NL * SM_OBJ, ctx)
        self.prev_o = talloc["gpu"](NL * SM_OBJ, ctx)
        self.octx = talloc["gpu"](NL * SM_OBJ_CTX, ctx)
        self.prev_octx = talloc["gpu"](NL * SM_OBJ_CTX, ctx)
        self.env = SMTrackingEnvGpu[Self.R, Self.BS](
            ctx, lib^, sim^, dr=dr, obs_noise=obs_noise, reset_noise=reset_noise, action_noise=action_noise,
            friction_period=friction_period, rng_seed=rng_seed, max_episode_steps=max_episode_steps,
        )
        # the robot's env reset every lane: the object follows
        self._inject(ctx)
        self._remember(ctx)

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.orows = move.orows^
        self.o = move.o^
        self.prev_o = move.prev_o^
        self.octx = move.octx^
        self.prev_octx = move.prev_octx^
        self.mass = move.mass
        self.mu0 = move.mu0
        self.friction_dr = move.friction_dr
        self.seed_base = move.seed_base

    def num_lanes(self) -> Int:
        return Self.BS.N

    def _read(mut self, ctx: DeviceContext, dst: FP[DT]) raises:
        ctx.enqueue_function[_k_obj_read[Self.OQ, Self.OV, Self.BS.SIM_NQ, Self.BS.SIM_NV]](
            Int64(Self.BS.N), self.env.sim.qpos_ptr(), self.env.sim.qvel_ptr(), dst,
            grid_dim=ceildiv(Self.BS.N, _TPB), block_dim=_TPB,
        )

    def _inject(mut self, ctx: DeviceContext) raises:
        """The object of every lane the robot's env reset LAST (`reset_f`),
        then the kinematics again (the env refreshed before the object moved)."""
        var seed = UInt32(self.env.step_count) * UInt32(16) + _RNG_OBJ_MU + self.seed_base * UInt32(0x9E3779B1)
        var dr = Self.BS.LANE_DR and self.friction_dr
        ctx.enqueue_function[_k_obj_inject[Self.OQ, Self.OV, Self.OB, Self.BS.SIM_NQ, Self.BS.SIM_NV, Self.BS.SIM_NBODY]](
            Int64(Self.BS.N), seed, Int64(1 if dr else 0), Scalar[DT](self.mass), Scalar[DT](self.mu0),
            mptr(self.orows.unsafe_ptr()), mptr(self.env.reset_f.unsafe_ptr()),
            self.env.sim.qpos_ptr(), self.env.sim.qvel_ptr(), self.env.sim.lane_dr_ptr(), tp["gpu"](self.octx),
            grid_dim=ceildiv(Self.BS.N, _TPB), block_dim=_TPB,
        )
        self.env.sim.refresh(ctx)

    def _remember(mut self, ctx: DeviceContext) raises:
        """The object and its context as the next row's (`prev_*`)."""
        self._read(ctx, tp["gpu"](self.prev_o))
        ctx.enqueue_copy(self.prev_octx.dev.value(), self.octx.dev.value())

    def restart(mut self, ctx: DeviceContext, frames: List[Int]) raises:
        """`SMTrackingEnvGpu.restart`, the object at its reference of each
        lane's frame."""
        self.env.restart(ctx, frames)
        self._inject(ctx)
        self._remember(ctx)

    def step(
        mut self, ctx: DeviceContext, mut actions: TensorImpl[DT], mut rb: SMReplayGpuOf[Self.R], mut orb: SMObjReplayGpu,
    ) raises -> SMStepInfo:
        """One control step for every lane, both rings written."""
        self.env.step_sim(ctx, actions, rb)
        self._read(ctx, tp["gpu"](self.o))
        orb.add_step(
            ctx, mptr(rb.ptr.unsafe_ptr()), rb.E, rb.T, mptr(self.env.meta.unsafe_ptr()),
            tp["gpu"](self.prev_o), tp["gpu"](self.o), tp["gpu"](self.prev_octx),
        )
        var si = self.env.step_finish(ctx)
        if si.n_terminated + si.n_truncated > 0:
            self._inject(ctx)
        self._remember(ctx)
        return si^
