# +--------------------------------------------------------------------------+ #
# | SuperMimic: the tracking environment on a batched device simulator
# +--------------------------------------------------------------------------+ #
"""`env.mojo`'s tracking environment for N lanes of a `TrackingBatchedSim`,
every per-lane quantity in device memory, one kernel per phase of the step
(same order as the CPU env, whose docstring describes the method):

    1. a_exec = a + N(0, 0.1²)              (_k_act)  → BS.control_step
    2.–4. state, reward and termination against f_t, the replay rows of the
       step and the ending lanes   (_k_post) → EpisodicReplayGpu.add_step
       → the host: per-lane rewards, frames and flags (one small download)
       for the RETURN-BASED RESET SAMPLER, which stays on the host (its bins,
       EMA and draws are `motion_library.SMResetSampler`, one copy of the
       rule); it returns a reset frame per ending lane
    4b. BS.reset_lanes(ending) → _k_inject (the shared `sm_reset_state_lane`,
        ξ's bias re-drawn) → BS.refresh
    5.–6. every lane's frame + 1, the noisy observation into the history,
        the next rows remembered  (_k_obs)

The per-lane math is the CPU env's: `sm_heads_lane` (reward, termination
features), `sm_observation_lane`, `sm_reset_state_lane`, the library's
`sm_reference_row_lane` / `sm_goal_window_lane`. Noise comes from a counter
hash of (seed, lane, entry) instead of the host RNG.

MODEL RANDOMISATION, PER LANE when the simulator has `LANE_DR` (the G1's
does): the foot friction is drawn per lane at every reset (U(0.3, 1.2)) and
written as that lane's contact-friction override on the foot BODIES, the
CoM offset per lane once at start-up (mjlab's startup event), both through
`physics3d/constraints/lane_dr.mojo` — the CPU env's draws, lane for lane.
A simulator without it shares one model: the CoM offset is drawn once for
the batch and the friction every `friction_period` steps (0 = once). Either
way ξ records the values in force. The encoder bias is per lane (it lives in
the action and the observation, not the model).

The env never touches the host's lists in the step except through the one
download / upload pair the sampler needs (N rewards, 2N ints; N frames).
"""

from std.math import ceildiv, cos, log, sqrt
from std.random import random_float64, random_ui64

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.data.episodic_replay_gpu import (
    EpisodicReplayGpu, META_ENDING, META_EP_START, META_FRAME0, META_FRAME1,
)
from noeira.deep_agents.supermimic.env import (
    SM_ACTION_NOISE, SM_EPISODE_STEPS, SM_HEIGHT_TERM, SM_MAX_LOGGED, SM_RESET_UNIFORMS, SM_TILT_TERM,
    SMStepInfo, sm_reset_state_lane,
)
from noeira.deep_agents.supermimic.heads import SM_N_TERMS, SMHeadLayout, sm_heads_lane, sm_observation_lane
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_hash_u01, talloc, tp
from noeira.deep_agents.supermimic.motion_library import (
    SM_REF_SCRATCH, SMEpisodeLog, SMMotionLibrary, SMResetSampler, sm_goal_window_lane, sm_reference_row_lane,
)
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_prompt_fell, sm_prompt_reward_lane
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim, TrackingRobot
from noeira.deep_agents.supermimic.tables import SMTables, sm_tables
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.constraints.lane_dr import LANE_DR_FRICTION, LANE_DR_IPOS_X, LANE_DR_STRIDE
from noeira.physics3d.gpu.constants import BODY_IDX_IPOS_X, GEOM_IDX_BODY, GEOM_IDX_FRICTION, GEOM_IDX_PRIORITY


comptime SMReplayGpuOf[R: TrackingRobot] = EpisodicReplayGpu[SMDims[R].S, R.NDOF, SMDims[R].X]
"""The device replay ring for robot R."""

comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 64

# per-lane int32 flags
comptime LF_CLIP_END: Int32 = 1
comptime LF_HIST_FRESH: Int32 = 2
comptime LF_EP_START: Int32 = 4
# per-lane step status, to the host
comptime ST_TERMINATED: Int32 = 1
comptime ST_ENDING: Int32 = 2
comptime ST_CLIP_END: Int32 = 4

# hash streams (the seed's low bits): one per use, so draws never collide
comptime RNG_ACT: UInt32 = 1
comptime RNG_RESET: UInt32 = 2
comptime RNG_BIAS: UInt32 = 3
comptime RNG_OBS: UInt32 = 4
comptime RNG_FRICTION: UInt32 = 5
comptime RNG_COM: UInt32 = 6


struct SMEnvGpuScratch[R: TrackingRobot]:
    """One lane's scratch: the heads' FK block, terms, features, an
    observation, the reference row and its FK scratch, the reset uniforms."""
    comptime HEADS = 0
    comptime TERMS = SMHeadLayout[Self.R].SCRATCH
    comptime FEAT = Self.TERMS + SM_N_TERMS
    comptime OBS = Self.FEAT + 2
    comptime REF = Self.OBS + SMDims[Self.R].OBS
    comptime REF_SCRATCH = Self.REF + SMDims[Self.R].REF
    comptime U = Self.REF_SCRATCH + SM_REF_SCRATCH[Self.R]
    comptime SIZE = Self.U + SM_RESET_UNIFORMS[Self.R]


@always_inline
def _seed(step: Int, stream: UInt32, base: UInt32) -> UInt32:
    return UInt32(step) * UInt32(16) + stream + base * UInt32(0x9E3779B1)


@always_inline
def _uni(seed: UInt32, l: Int, k: Int) -> Scalar[DT]:
    """A uniform in [−1, 1) for lane l, entry k."""
    return Scalar[DT](2) * Scalar[DT](sm_hash_u01(seed, UInt32(l * 256 + k))) - 1


@always_inline
def _randn(seed: UInt32, l: Int, k: Int) -> Scalar[DT]:
    var u1 = max(Scalar[DT](sm_hash_u01(seed, UInt32(l * 512 + 2 * k))), Scalar[DT](1e-7))
    var u2 = Scalar[DT](sm_hash_u01(seed, UInt32(l * 512 + 2 * k + 1)))
    return sqrt(Scalar[DT](-2) * log(u1)) * cos(Scalar[DT](6.283185307179586) * u2)


# ── kernels ─────────────────────────────────────────────────────────────────
def _k_act[n: Int](nl: Int64, seed: UInt32, std: Scalar[DT], a: FP[DT], ae: FP[DT]):
    var i = Int(global_idx.x)
    if i >= Int(nl) * n:
        return
    var l = i // n
    var j = i % n
    ae[unsafe_offset=i] = a[unsafe_offset=i] + (std * _randn(seed, l, j) if std > 0 else Scalar[DT](0))


def _k_prompt_r[R: TrackingRobot, QS: Int, VS: Int](
    nl: Int64, from_lane: Int64, tree: FP[DT], table: FP[DT], qpos: FP[DT], qvel: FP[DT],
    a: FP[DT], a_last: FP[DT], spec: FP[DT], s: FP[DT], scratch: FP[DT], out_r: FP[DT],
):
    """A prompt lane's task reward, before `_k_post` overwrites `a_last`; its
    FK and atoms land in the heads' scratch, which `_k_post` rewrites."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMEnvGpuScratch[R]
    var l = Int(global_idx.x)
    if l >= Int(nl) or l < Int(from_lane):
        return
    var sl = s.unsafe_offset(l * D.S)
    for k in range(R.NQ):
        sl[unsafe_offset=k] = qpos[unsafe_offset=l * QS + k]
    for k in range(R.NV):
        sl[unsafe_offset=R.NQ + k] = qvel[unsafe_offset=l * VS + k]
    out_r[unsafe_offset=l] = sm_prompt_reward_lane[DT, R](
        tree, table, sl, a.unsafe_offset(l * n), a_last.unsafe_offset(l * n), spec.unsafe_offset(l * SMSpec.SIZE),
        scratch.unsafe_offset(l * SC.SIZE),
    )


def _k_post[R: TrackingRobot, QS: Int, VS: Int](
    nl: Int64, max_steps: Int64, prompt_from: Int64, tree: FP[DT], table: FP[DT], rows: FP[DT], qpos: FP[DT], qvel: FP[DT],
    a: FP[DT], s: FP[DT], a_last: FP[DT], xi: FP[DT], scratch: FP[DT],
    frame: FPi, prev_frame: FPi, ep_len: FPi, flags: FPi,
    out_r: FP[DT], out_i: FPi, rd: FP[DT], meta: FPi, mask: FP[DT], out_terms: FP[DT],
):
    """Steps 2–4 for lane l: the state after the control step, the reward and
    termination against f_t, the replay row metadata, the ending flag."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMEnvGpuScratch[R]
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var sl = s.unsafe_offset(l * D.S)
    for k in range(R.NQ):
        sl[unsafe_offset=k] = qpos[unsafe_offset=l * QS + k]
    for k in range(R.NV):
        sl[unsafe_offset=R.NQ + k] = qvel[unsafe_offset=l * VS + k]
    var el = ep_len[unsafe_offset=l] + 1
    ep_len[unsafe_offset=l] = el
    var f = Int(frame[unsafe_offset=l])
    var sc = scratch.unsafe_offset(l * SC.SIZE)
    var rf = sc.unsafe_offset(SC.REF)
    sm_reference_row_lane[R](rows, f, tree, table, sc.unsafe_offset(SC.REF_SCRATCH), rf)
    var al = a.unsafe_offset(l * n)
    var r = sm_heads_lane[DT, R](
        tree, table, sl, al, a_last.unsafe_offset(l * n), xi.unsafe_offset(l * D.X), rf,
        sc.unsafe_offset(SC.TERMS), sc.unsafe_offset(SC.FEAT), sc.unsafe_offset(SC.OBS), sc,
    )
    var fl = flags[unsafe_offset=l]
    var terminated = abs(sc[unsafe_offset=SC.FEAT]) > Scalar[DT](SM_HEIGHT_TERM) or abs(sc[unsafe_offset=SC.FEAT + 1]) > Scalar[DT](SM_TILT_TERM)
    if l >= Int(prompt_from):
        terminated = sm_prompt_fell[DT](sl)
    var clip_end = (fl & LF_CLIP_END) != 0
    var ending = terminated or (max_steps > 0 and Int(el) >= Int(max_steps)) or clip_end
    for k in range(SM_N_TERMS):
        out_terms[unsafe_offset=l * SM_N_TERMS + k] = sc[unsafe_offset=SC.TERMS + k]
    for j in range(n):
        a_last[unsafe_offset=l * n + j] = al[unsafe_offset=j]
    out_r[unsafe_offset=l] = r
    out_i[unsafe_offset=2 * l] = Int32(f)
    out_i[unsafe_offset=2 * l + 1] = (ST_TERMINATED if terminated else Int32(0)) | (ST_ENDING if ending else Int32(0)) | (ST_CLIP_END if clip_end else Int32(0))
    rd[unsafe_offset=2 * l] = r
    rd[unsafe_offset=2 * l + 1] = Scalar[DT](1) if terminated else Scalar[DT](0)
    var m = meta.unsafe_offset(4 * l)
    m[unsafe_offset=META_FRAME0] = prev_frame[unsafe_offset=l]
    m[unsafe_offset=META_FRAME1] = Int32(f)
    m[unsafe_offset=META_EP_START] = 1 if (fl & LF_EP_START) != 0 else 0
    m[unsafe_offset=META_ENDING] = 1 if ending else 0
    flags[unsafe_offset=l] = (fl & ~LF_EP_START) | (LF_EP_START if ending else Int32(0))
    mask[unsafe_offset=l] = Scalar[DT](1) if ending else Scalar[DT](0)


def _k_inject[R: TrackingRobot, QS: Int, VS: Int, NB: Int](
    nl: Int64, seed_reset: UInt32, seed_bias: UInt32, seed_mu: UInt32, noise: Int64, dr: Int64, lane_mu: Int64,
    rows: FP[DT], table: FP[DT], reset_f: FPi, qpos: FP[DT], qvel: FP[DT], scratch: FP[DT],
    xi: FP[DT], a_last: FP[DT], frame: FPi, ep_len: FPi, flags: FPi,
    lane_dr: FP[DT], feet: FPi, n_feet: Int64,
):
    """Step 4b for the lanes the host gave a frame: the reset state (shared
    law) into the simulator, ξ's bias re-drawn, the episode bookkeeping."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime L = SMHeadLayout[R]
    comptime SC = SMEnvGpuScratch[R]
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var f = Int(reset_f[unsafe_offset=l])
    if f < 0:
        return
    var u = scratch.unsafe_offset(l * SC.SIZE + SC.U)
    for k in range(SM_RESET_UNIFORMS[R]):
        u[unsafe_offset=k] = _uni(seed_reset, l, k)
    sm_reset_state_lane[DT, R](
        rows, f, u, noise != 0, table.unsafe_offset(L.LO), table.unsafe_offset(L.HI),
        qpos.unsafe_offset(l * QS), qvel.unsafe_offset(l * VS),
    )
    if dr != 0:
        for j in range(n):
            xi[unsafe_offset=l * D.X + 1 + j] = Scalar[DT](0.01) * _uni(seed_bias, l, j)
        if lane_mu != 0:
            # this lane's foot friction, U(0.3, 1.2): the override on every foot body
            var mu = Scalar[DT](0.75) + Scalar[DT](0.45) * _uni(seed_mu, l, 0)
            for k in range(Int(n_feet)):
                var b = Int(feet[unsafe_offset=k])
                lane_dr[unsafe_offset=(l * NB + b) * LANE_DR_STRIDE + LANE_DR_FRICTION] = mu
            xi[unsafe_offset=l * D.X] = mu
    for j in range(n):
        a_last[unsafe_offset=l * n + j] = 0
    frame[unsafe_offset=l] = Int32(f)
    ep_len[unsafe_offset=l] = 0
    flags[unsafe_offset=l] = LF_HIST_FRESH | LF_EP_START


def _k_obs[R: TrackingRobot, QS: Int, VS: Int](
    nl: Int64, seed: UInt32, noise: Int64, table: FP[DT], last: FPi, qpos: FP[DT], qvel: FP[DT],
    s: FP[DT], prev_s: FP[DT], xi: FP[DT], prev_xi: FP[DT], a_last: FP[DT], hist: FP[DT], scratch: FP[DT],
    frame: FPi, prev_frame: FPi, flags: FPi,
):
    """Steps 5–6 for lane l: frame + 1 (the clip's end flagged once passed),
    the state, the noisy observation into the history (a fresh lane's first
    observation fills it), the next rows remembered."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime SC = SMEnvGpuScratch[R]
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var f = Int(frame[unsafe_offset=l])
    var lf = Int(last[unsafe_offset=f])
    var fl = flags[unsafe_offset=l]
    f += 1
    if f > lf:
        fl |= LF_CLIP_END
        f = lf
    frame[unsafe_offset=l] = Int32(f)
    var sl = s.unsafe_offset(l * D.S)
    for k in range(R.NQ):
        sl[unsafe_offset=k] = qpos[unsafe_offset=l * QS + k]
    for k in range(R.NV):
        sl[unsafe_offset=R.NQ + k] = qvel[unsafe_offset=l * VS + k]
    var o = scratch.unsafe_offset(l * SC.SIZE + SC.OBS)
    sm_observation_lane[DT, R](table, sl, a_last.unsafe_offset(l * n), xi.unsafe_offset(l * D.X), o)
    if noise != 0:
        for k in range(D.OBS):
            var hw = sm_obs_noise_half[n, DT](k)
            if hw > 0:
                o[unsafe_offset=k] += hw * _uni(seed, l, k)
    var h = hist.unsafe_offset(l * D.HIST * D.OBS)
    if (fl & LF_HIST_FRESH) != 0:
        for hh in range(D.HIST):
            for k in range(D.OBS):
                h[unsafe_offset=hh * D.OBS + k] = o[unsafe_offset=k]
        fl &= ~LF_HIST_FRESH
    else:
        for hh in range(D.HIST - 1):
            for k in range(D.OBS):
                h[unsafe_offset=hh * D.OBS + k] = h[unsafe_offset=(hh + 1) * D.OBS + k]
        for k in range(D.OBS):
            h[unsafe_offset=(D.HIST - 1) * D.OBS + k] = o[unsafe_offset=k]
    flags[unsafe_offset=l] = fl
    for k in range(D.S):
        prev_s[unsafe_offset=l * D.S + k] = sl[unsafe_offset=k]
    for k in range(D.X):
        prev_xi[unsafe_offset=l * D.X + k] = xi[unsafe_offset=l * D.X + k]
    prev_frame[unsafe_offset=l] = Int32(f)


def _k_goals[R: TrackingRobot](nl: Int64, rows: FP[DT], table: FP[DT], last: FPi, frame: FPi, goals: FP[DT]):
    """Lane l's goal window at its current frame."""
    comptime D = SMDims[R]
    comptime L = SMHeadLayout[R]
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var f = Int(frame[unsafe_offset=l])
    sm_goal_window_lane[R](rows, f, Int(last[unsafe_offset=f]), table.unsafe_offset(L.Q0), goals.unsafe_offset(l * D.GOAL))


def _k_lane_com[R: TrackingRobot, NB: Int](nl: Int64, seed: UInt32, body: Int64, lane_dr: FP[DT], xi: FP[DT]):
    """Lane l's CoM offset of `body` (mjlab's startup event: ±(0.025, 0.05,
    0.05) m), as its lane override and in ξ's last three entries."""
    comptime D = SMDims[R]
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var o = (l * NB + Int(body)) * LANE_DR_STRIDE + LANE_DR_IPOS_X
    comptime for c in range(3):
        var h = Scalar[DT](0.025) if c == 0 else Scalar[DT](0.05)
        var d = h * _uni(seed, l, c)
        lane_dr[unsafe_offset=o + c] = d
        xi[unsafe_offset=l * D.X + 1 + R.NDOF + c] = d


def _k_set_xi(nl: Int64, x: Int64, k: Int64, v: Scalar[DT], xi: FP[DT]):
    """ξ[l, k] = v for every lane (a shared model field's value)."""
    var l = Int(global_idx.x)
    if l < Int(nl):
        xi[unsafe_offset=l * Int(x) + Int(k)] = v


# ── the env ─────────────────────────────────────────────────────────────────
struct SMTrackingEnvGpu[R: TrackingRobot, BS: TrackingBatchedSim](Movable):
    comptime N = Self.BS.N
    var sim: Self.BS
    var lib: SMMotionLibrary[Self.R]
    var sampler: SMResetSampler
    var tables: SMTables[DT]
    var dr: Bool
    var obs_noise: Bool
    var reset_noise: Bool
    var action_noise: Float64
    var friction_period: Int
    var seed_base: UInt32
    var step_count: Int
    var logs: List[SMEpisodeLog]
    var last_reward: List[Float64]
    # per-lane device state
    var s: TensorImpl[DT]
    var prev_s: TensorImpl[DT]
    var xi: TensorImpl[DT]
    var prev_xi: TensorImpl[DT]
    var hist: TensorImpl[DT]
    var a_last: TensorImpl[DT]
    var a_exec: TensorImpl[DT]
    var scratch: TensorImpl[DT]
    var rd: TensorImpl[DT]
    var mask: TensorImpl[DT]
    var out_r: TensorImpl[DT]
    var frame: DeviceBuffer[DType.int32]
    var prev_frame: DeviceBuffer[DType.int32]
    var ep_len: DeviceBuffer[DType.int32]
    var flags: DeviceBuffer[DType.int32]
    var meta: DeviceBuffer[DType.int32]
    var out_i: DeviceBuffer[DType.int32]
    var reset_f: DeviceBuffer[DType.int32]
    var h_out_i: List[Int32]                 # the last step's (frame, ST_* status) per lane, host
    var feet: DeviceBuffer[DType.int32]      # the foot geoms' bodies
    var n_feet: Int
    var max_episode_steps: Int               # 0: no cap (the evaluation)
    var next_frame: List[Int]                # ≥ 0: the lane's next reset frame instead of a draw
    var out_terms: TensorImpl[DT]            # [N · 8] the last step's weighted reward terms
    var prompt_from: Int                     # lanes from here on run a prompt interface (`env.mojo`)
    var prompt_r: TensorImpl[DT]             # [N] their task reward (`step` given `specs`)

    def __init__(
        out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.R], var sim: Self.BS,
        dr: Bool = True, obs_noise: Bool = True, reset_noise: Bool = True,
        action_noise: Float64 = SM_ACTION_NOISE, friction_period: Int = SM_EPISODE_STEPS, rng_seed: Int = 0,
        max_episode_steps: Int = SM_EPISODE_STEPS,
    ) raises:
        """`tables` must be built from a CPU lane of the same robot (the model
        records); `sim` is the batched simulator, already created."""
        comptime D = SMDims[Self.R]
        comptime n = Self.R.NDOF
        comptime NL = Self.BS.N
        self.sim = sim^
        var q0 = List[Float64](length=n, fill=0.0)
        for j in range(n):
            q0[j] = Float64(lib.q0[j])
        var cpu_lane = Self.R.SIM.create()
        self.tables = sm_tables[DT, Self.R](cpu_lane, q0)
        self.tables.to_device(ctx)
        lib.to_device(ctx)
        self.sampler = SMResetSampler(lib.num_bins)
        self.lib = lib^
        self.dr = dr
        self.obs_noise = obs_noise
        self.reset_noise = reset_noise
        self.action_noise = action_noise
        self.friction_period = friction_period
        self.seed_base = UInt32(rng_seed)
        self.step_count = 0
        self.logs = List[SMEpisodeLog]()
        for _ in range(NL):
            self.logs.append(SMEpisodeLog())
        self.last_reward = List[Float64](length=NL, fill=0.0)
        self.s = talloc["gpu"](NL * D.S, ctx)
        self.prev_s = talloc["gpu"](NL * D.S, ctx)
        self.xi = talloc["gpu"](NL * D.X, ctx)
        self.prev_xi = talloc["gpu"](NL * D.X, ctx)
        self.hist = talloc["gpu"](NL * D.HIST * D.OBS, ctx)
        self.a_last = talloc["gpu"](NL * n, ctx)
        self.a_exec = talloc["gpu"](NL * n, ctx)
        self.scratch = talloc["gpu"](NL * SMEnvGpuScratch[Self.R].SIZE, ctx)
        self.rd = talloc["gpu"](NL * 2, ctx)
        self.mask = talloc["gpu"](NL, ctx)
        self.out_r = talloc["gpu"](NL, ctx)
        self.frame = ctx.enqueue_create_buffer[DType.int32](NL)
        self.prev_frame = ctx.enqueue_create_buffer[DType.int32](NL)
        self.ep_len = ctx.enqueue_create_buffer[DType.int32](NL)
        self.flags = ctx.enqueue_create_buffer[DType.int32](NL)
        self.meta = ctx.enqueue_create_buffer[DType.int32](NL * 4)
        self.out_i = ctx.enqueue_create_buffer[DType.int32](NL * 2)
        self.reset_f = ctx.enqueue_create_buffer[DType.int32](NL)
        self.h_out_i = List[Int32](length=NL * 2, fill=0)
        self.max_episode_steps = max_episode_steps
        self.next_frame = List[Int](length=NL, fill=-1)
        self.out_terms = talloc["gpu"](NL * SM_N_TERMS, ctx)
        self.prompt_from = NL
        self.prompt_r = talloc["gpu"](NL, ctx)
        # the foot geoms' bodies (the per-lane friction overrides are by body)
        var fb = List[Int]()
        for g in self.tables.feet:
            var b = Int(cpu_lane.geom_field(g, GEOM_IDX_BODY))
            var seen = False
            for k in range(len(fb)):
                seen = seen or fb[k] == b
            if not seen:
                fb.append(b)
        self.n_feet = len(fb)
        self.feet = ctx.enqueue_create_buffer[DType.int32](max(len(fb), 1))
        var hfb = ctx.enqueue_create_host_buffer[DType.int32](max(len(fb), 1))
        ctx.synchronize()
        for k in range(len(fb)):
            hfb[k] = Int32(fb[k])
        ctx.enqueue_copy(self.feet, hfb)
        # the model: feet get priority 1 (their friction is used as is)
        for g in self.tables.feet:
            self.sim.set_geom_field_all(ctx, g, GEOM_IDX_PRIORITY, 1.0)
        var cb = self.tables.com_body
        comptime if Self.BS.LANE_DR:
            # per lane: the CoM offset now, the friction at every reset
            if dr:
                ctx.enqueue_function[_k_lane_com[Self.R, Self.BS.SIM_NBODY]](
                    Int64(NL), _seed(0, RNG_COM, self.seed_base), Int64(cb), self.sim.lane_dr_ptr(), tp["gpu"](self.xi),
                    grid_dim=ceildiv(NL, _TPB), block_dim=_TPB,
                )
            else:
                self._set_xi_all(ctx, 0, 1.0)     # the model's foot friction, 1
        else:
            # one model: the CoM offset once, the friction per period
            if dr:
                var rng: List[Float64] = [0.025, 0.05, 0.05]
                for c in range(3):
                    var d = random_float64(-rng[c], rng[c])
                    self.sim.set_body_field_all(ctx, cb, BODY_IDX_IPOS_X + c, self.sim.body_field(cb, BODY_IDX_IPOS_X + c) + d)
                    self._set_xi_all(ctx, 1 + n + c, d)
            self._draw_friction(ctx)
        # every lane reset, then mjlab's `reset()` command update (frame + 1)
        var all = List[Int]()
        for l in range(NL):
            all.append(l)
        self._reset(ctx, all)
        self._observe(ctx)

    def __init__(out self, *, deinit move: Self):
        self.sim = move.sim^
        self.lib = move.lib^
        self.sampler = move.sampler^
        self.tables = move.tables^
        self.dr = move.dr
        self.obs_noise = move.obs_noise
        self.reset_noise = move.reset_noise
        self.action_noise = move.action_noise
        self.friction_period = move.friction_period
        self.seed_base = move.seed_base
        self.step_count = move.step_count
        self.logs = move.logs^
        self.last_reward = move.last_reward^
        self.s = move.s^
        self.prev_s = move.prev_s^
        self.xi = move.xi^
        self.prev_xi = move.prev_xi^
        self.hist = move.hist^
        self.a_last = move.a_last^
        self.a_exec = move.a_exec^
        self.scratch = move.scratch^
        self.rd = move.rd^
        self.mask = move.mask^
        self.out_r = move.out_r^
        self.frame = move.frame^
        self.prev_frame = move.prev_frame^
        self.ep_len = move.ep_len^
        self.flags = move.flags^
        self.meta = move.meta^
        self.out_i = move.out_i^
        self.reset_f = move.reset_f^
        self.h_out_i = move.h_out_i^
        self.feet = move.feet^
        self.n_feet = move.n_feet
        self.max_episode_steps = move.max_episode_steps
        self.next_frame = move.next_frame^
        self.out_terms = move.out_terms^
        self.prompt_from = move.prompt_from
        self.prompt_r = move.prompt_r^

    def num_lanes(self) -> Int:
        return Self.BS.N

    def restart(mut self, ctx: DeviceContext, frames: List[Int]) raises:
        """Every lane reset to `frames[l]` (they also become each lane's next
        reset frame), then the construction's command update and observation
        — the evaluation's start. Set `next_frame` again to queue clips."""
        var all = List[Int]()
        for l in range(Self.BS.N):
            self.next_frame[l] = frames[l]
            all.append(l)
        self._reset(ctx, all)
        self._observe(ctx)

    def _set_xi_all(mut self, ctx: DeviceContext, k: Int, v: Float64) raises:
        ctx.enqueue_function[_k_set_xi](
            Int64(Self.BS.N), Int64(SMDims[Self.R].X), Int64(k), Scalar[DT](v), tp["gpu"](self.xi),
            grid_dim=ceildiv(Self.BS.N, _TPB), block_dim=_TPB,
        )

    def _draw_friction(mut self, ctx: DeviceContext) raises:
        """The shared foot friction: U(0.3, 1.2) with DR, 1 without."""
        var mu = random_float64(0.3, 1.2) if self.dr else 1.0
        for g in self.tables.feet:
            self.sim.set_geom_field_all(ctx, g, GEOM_IDX_FRICTION, mu)
        self._set_xi_all(ctx, 0, mu)

    def _reset(mut self, ctx: DeviceContext, lanes: List[Int]) raises:
        """Reset `lanes`: a frame each from the sampler (host), the masked
        simulator reset, the reset states (device), the kinematics."""
        comptime NL = Self.BS.N
        var hf = ctx.enqueue_create_host_buffer[DType.int32](NL)
        var hm = ctx.enqueue_create_host_buffer[DT](NL)
        ctx.synchronize()
        for l in range(NL):
            hf[l] = -1
            hm[l] = 0
        var p = self.sampler.probabilities()
        var cdf = List[Float64](length=len(p), fill=0.0)
        var acc = 0.0
        for b in range(len(p)):
            acc += p[b]
            cdf[b] = acc
        for l in lanes:
            var x = random_float64(0.0, acc)
            var b = SMResetSampler.bin_from_cdf(cdf, x)
            self.sampler.note_draw(b)
            var f = SMResetSampler.frame_from[Self.R](self.lib, b, random_float64(0.0, 1.0))
            if self.next_frame[l] >= 0:      # the evaluation's clip queue
                f = self.next_frame[l]
            hf[l] = Int32(f)
            hm[l] = 1
            self.logs[l] = SMEpisodeLog()
        ctx.enqueue_copy(self.reset_f, hf)
        ctx.enqueue_copy(self.mask.dev.value(), hm)
        var st = self.step_count
        self.sim.reset_lanes(ctx, tp["gpu"](self.mask), UInt64(_seed(st, RNG_RESET, self.seed_base)))
        ctx.enqueue_function[_k_inject[Self.R, Self.BS.SIM_NQ, Self.BS.SIM_NV, Self.BS.SIM_NBODY]](
            Int64(NL), _seed(st, RNG_RESET, self.seed_base), _seed(st, RNG_BIAS, self.seed_base),
            _seed(st, RNG_FRICTION, self.seed_base),
            Int64(1 if self.reset_noise else 0), Int64(1 if self.dr else 0), Int64(1 if Self.BS.LANE_DR else 0),
            mptr(self.lib.rows_dev.value().unsafe_ptr()), self.tables.head_p["gpu"](), mptr(self.reset_f.unsafe_ptr()),
            self.sim.qpos_ptr(), self.sim.qvel_ptr(), tp["gpu"](self.scratch), tp["gpu"](self.xi), tp["gpu"](self.a_last),
            mptr(self.frame.unsafe_ptr()), mptr(self.ep_len.unsafe_ptr()), mptr(self.flags.unsafe_ptr()),
            self.sim.lane_dr_ptr(), mptr(self.feet.unsafe_ptr()), Int64(self.n_feet),
            grid_dim=ceildiv(NL, _TPB), block_dim=_TPB,
        )
        self.sim.refresh(ctx)
        ctx.synchronize()      # the host staging buffers go out of scope

    def _observe(mut self, ctx: DeviceContext) raises:
        comptime NL = Self.BS.N
        ctx.enqueue_function[_k_obs[Self.R, Self.BS.SIM_NQ, Self.BS.SIM_NV]](
            Int64(NL), _seed(self.step_count, RNG_OBS, self.seed_base), Int64(1 if self.obs_noise else 0),
            self.tables.head_p["gpu"](), mptr(self.lib.last_dev.value().unsafe_ptr()), self.sim.qpos_ptr(), self.sim.qvel_ptr(),
            tp["gpu"](self.s), tp["gpu"](self.prev_s), tp["gpu"](self.xi), tp["gpu"](self.prev_xi), tp["gpu"](self.a_last),
            tp["gpu"](self.hist), tp["gpu"](self.scratch),
            mptr(self.frame.unsafe_ptr()), mptr(self.prev_frame.unsafe_ptr()), mptr(self.flags.unsafe_ptr()),
            grid_dim=ceildiv(NL, _TPB), block_dim=_TPB,
        )

    def policy_inputs(mut self, ctx: DeviceContext, mut goals: TensorImpl[DT]) raises:
        """Every lane's goal window [N · GOAL] into `goals` (device); the
        history is `self.hist` [N · 10 · OBS] (`ActStep`'s layout)."""
        ctx.enqueue_function[_k_goals[Self.R]](
            Int64(Self.BS.N), mptr(self.lib.rows_dev.value().unsafe_ptr()), self.tables.head_p["gpu"](),
            mptr(self.lib.last_dev.value().unsafe_ptr()), mptr(self.frame.unsafe_ptr()), tp["gpu"](goals),
            grid_dim=ceildiv(Self.BS.N, _TPB), block_dim=_TPB,
        )

    def step(
        mut self, ctx: DeviceContext, mut actions: TensorImpl[DT], mut rb: SMReplayGpuOf[Self.R], specs: Optional[FP[DT]] = None,
    ) raises -> SMStepInfo:
        """One control step for every lane; `actions` [N·n] (clean, device).
        Prompt lanes as in `env.mojo`: `specs` [N·SPEC] on the device."""
        self.step_sim(ctx, actions, rb, specs)
        return self.step_finish(ctx)

    def step_sim(
        mut self, ctx: DeviceContext, mut actions: TensorImpl[DT], mut rb: SMReplayGpuOf[Self.R], specs: Optional[FP[DT]] = None,
    ) raises:
        """The first half of `step` (1.–4.): the control step, the reward and
        the replay rows. The simulator then holds every lane's state AFTER the
        step and BEFORE any reset, `meta` the rows `rb` just took and `mask`
        the ending lanes: what a caller that logs more of the simulator than
        the robot (`object_env_gpu.mojo`) reads before `step_finish`."""
        comptime NL = Self.BS.N
        comptime n = Self.R.NDOF
        comptime D = SMDims[Self.R]
        var st = self.step_count
        # 1. the executed action through the PD
        ctx.enqueue_function[_k_act[n]](
            Int64(NL), _seed(st, RNG_ACT, self.seed_base), Scalar[DT](self.action_noise), tp["gpu"](actions), tp["gpu"](self.a_exec),
            grid_dim=ceildiv(NL * n, _TPB), block_dim=_TPB,
        )
        self.sim.control_step(ctx, tp["gpu"](self.a_exec), tp["gpu"](self.xi).unsafe_offset(1), D.X, UInt64(_seed(st, RNG_ACT, self.seed_base)))
        if specs:
            ctx.enqueue_function[_k_prompt_r[Self.R, Self.BS.SIM_NQ, Self.BS.SIM_NV]](
                Int64(NL), Int64(self.prompt_from), self.tables.tree_p["gpu"](), self.tables.head_p["gpu"](),
                self.sim.qpos_ptr(), self.sim.qvel_ptr(), tp["gpu"](actions), tp["gpu"](self.a_last), specs.value(),
                tp["gpu"](self.s), tp["gpu"](self.scratch), tp["gpu"](self.prompt_r),
                grid_dim=ceildiv(NL, _TPB), block_dim=_TPB,
            )
        # 2.–4. reward, termination, the replay rows
        ctx.enqueue_function[_k_post[Self.R, Self.BS.SIM_NQ, Self.BS.SIM_NV]](
            Int64(NL), Int64(self.max_episode_steps), Int64(self.prompt_from), self.tables.tree_p["gpu"](), self.tables.head_p["gpu"](), mptr(self.lib.rows_dev.value().unsafe_ptr()),
            self.sim.qpos_ptr(), self.sim.qvel_ptr(), tp["gpu"](actions), tp["gpu"](self.s), tp["gpu"](self.a_last),
            tp["gpu"](self.xi), tp["gpu"](self.scratch),
            mptr(self.frame.unsafe_ptr()), mptr(self.prev_frame.unsafe_ptr()), mptr(self.ep_len.unsafe_ptr()), mptr(self.flags.unsafe_ptr()),
            tp["gpu"](self.out_r), mptr(self.out_i.unsafe_ptr()), tp["gpu"](self.rd), mptr(self.meta.unsafe_ptr()), tp["gpu"](self.mask),
            tp["gpu"](self.out_terms),
            grid_dim=ceildiv(NL, _TPB), block_dim=_TPB,
        )
        rb.add_step(
            ctx, tp["gpu"](self.prev_s), tp["gpu"](actions), tp["gpu"](self.a_exec), tp["gpu"](self.prev_xi),
            tp["gpu"](self.rd), tp["gpu"](self.s), mptr(self.meta.unsafe_ptr()),
        )

    def step_finish(mut self, ctx: DeviceContext) raises -> SMStepInfo:
        """The second half of `step`: the host's share (the sampler's log, the
        ending lanes), their resets (4b.), and 5.–6. for every lane. After a
        step that reset any lane, `reset_f` holds that step's frame per reset
        lane and −1 elsewhere."""
        comptime NL = Self.BS.N
        # the sampler's per-step log and the ending episodes (host)
        var hi = ctx.enqueue_create_host_buffer[DType.int32](NL * 2)
        self.out_r.download_enqueue(ctx)
        ctx.enqueue_copy(hi, self.out_i)
        ctx.synchronize()
        self.out_r.download_finalize()
        var ending = List[SMEpisodeLog]()
        var lanes = List[Int]()
        var rsum = 0.0
        var n_term = 0
        var n_trunc = 0
        for l in range(NL):
            var r = Float64(self.out_r.data[l])
            var f = Int(hi[2 * l])
            var stt = hi[2 * l + 1]
            self.h_out_i[2 * l] = hi[2 * l]           # the last step's (frame, status), for callers
            self.h_out_i[2 * l + 1] = stt
            self.last_reward[l] = r
            rsum += r
            var tracker_lane = l < self.prompt_from
            if tracker_lane:
                self.logs[l].log(r, self.lib.bin_of_frame(f), SM_MAX_LOGGED)
            if (stt & ST_ENDING) != 0:
                var term = (stt & ST_TERMINATED) != 0
                self.logs[l].terminated = term
                self.logs[l].clip_end = (stt & ST_CLIP_END) != 0
                if tracker_lane:
                    ending.append(self.logs[l].copy())
                lanes.append(l)
                if term:
                    n_term += 1
                else:
                    n_trunc += 1
        self.sampler.finalize(ending)
        if len(lanes) > 0:
            self._reset(ctx, lanes)
        self.step_count += 1
        comptime if not Self.BS.LANE_DR:
            if self.dr and self.friction_period > 0 and self.step_count % self.friction_period == 0:
                self._draw_friction(ctx)
        # 5.–6. frames, observations, the next rows
        self._observe(ctx)
        return SMStepInfo(rsum / Float64(NL), n_term, n_trunc)
