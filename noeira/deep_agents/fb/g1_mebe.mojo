"""The G1 pieces of FB-MEBE — `noeira-docs/FB_MEBE_G1_PLAN.md` §2 (G1
decisions), §3 (the base-command suite) and §4 rung G.

Everything here reads the G1's 527-D observation (`state 64 | priv 463`,
`UnitreeG1Batched`) or the env's own device fields, and nothing reads LAFAN:
the arm this serves is data-free.

    ϑ_g   B's input, 10-D: projected gravity (58:61), pelvis height (64),
          heading-frame pelvis linear velocity (341:344) and angular
          velocity (434:437)
    φ_g   the density's space, 4-D: (vx, vy, ωz, h) — gravity is left out
          (falls dominate its extremes); the exploration filter keeps
          upright, in-box states (the release's `g_z <= -0.93`, |vx| <= 1.8,
          |vy| <= 1.3)
    fall  a lane whose pelvis is below `h_min` or tilted past `g_max` for
          `delay` consecutive control steps is reset, and the transition is
          flagged as a BOUNDARY (truncation), never a terminal — the plan's
          answer to `_a_rare_terminal_is_a_cliff_the_critic_chases`
    r_env the contact/limit half of the regularity reward, which only the
          simulator sees: undesired contacts, feet orientation, soft joint
          limits, feet slide (RoboParty's weights, `unitree_g1_walk_rp_config`)
          — fed to `FBMebeOnlineAgent` as the env reward with `w_env = 1`;
          joint acceleration and action rate come from the ring as on the
          walker
    suite the 22 base commands of §3, closed-form on ϑ: one shared reward
          function, called by the device kernel that scores rollouts AND by
          the host code that relabels the ring pool into z, so the two cannot
          drift (`_a_rule_written_inline_twice_drifts`)
"""

from std.math import exp, sqrt, sin
from layout import Layout, LayoutTensor
from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.nn.combinators.parallel import Parallel
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.slice import Slice
from noeira.physics3d.gpu.constants import CONTACT_SIZE, METADATA_SIZE
from noeira.envs.robots.unitree_g1_xml import ROOT_QPOS_SIZE
from noeira.envs.robots.unitree_g1_config import _projected_gravity
from noeira.envs.robots.unitree_g1_pd import (
    G1_N_DOF,
    g1_default_pos,
    g1_pos_lower,
    g1_pos_upper,
)
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    g1r_contacts,
    G1R_L_ANKLE_ROLL,
    G1R_R_ANKLE_ROLL,
)

from .bfm_towers import BFMBNet


# ── observation columns (the 527-D G1 row) ───────────────────────────────

comptime G1M_COL_GX: Int = 58
comptime G1M_COL_GY: Int = 59
comptime G1M_COL_GZ: Int = 60
comptime G1M_COL_H: Int = 64
comptime G1M_COL_VX: Int = 341
comptime G1M_COL_VY: Int = 342
comptime G1M_COL_WZ: Int = 436
comptime G1M_COL_QD: Int = 29      # joint velocities, 29 from here
comptime G1M_THETA: Int = 10

comptime G1BNetProj[OBS: Int, D: Int, HB: Int] = Sequential[
    Parallel[
        Slice[OBS, 58, 61],     # projected gravity
        Slice[OBS, 64, 65],     # pelvis height
        Slice[OBS, 341, 344],   # pelvis linear velocity, heading frame
        Slice[OBS, 434, 437],   # pelvis angular velocity, heading frame
    ],
    BFMBNet[G1M_THETA, D, HB],
]
"""B on ϑ_g. Under `normalize_obs` the slices read the NORMALISED row, so
ϑ is standardised for free (§10.4's walker B read raw velocities)."""


# ── fall → truncation ────────────────────────────────────────────────────

def g1_fall_kernel[OBS_W: Int, LANES: Int](
    obs: Pointer[Scalar[DT], MutAnyOrigin],
    env_done: Pointer[Scalar[DT], MutAnyOrigin],
    fall_cnt: Pointer[Scalar[DT], MutAnyOrigin],
    done_rec: Pointer[Scalar[DT], MutAnyOrigin],
    n_falls: Pointer[Scalar[DT], MutAnyOrigin],
    h_min: Scalar[DT],
    g_max: Scalar[DT],
    delay: Int32,
):
    """One thread per lane, AFTER `step_batch`, BEFORE `record_batch_gpu`.
    `done_rec = env time limit OR fallen for `delay` steps`; it is written
    back into the env's `_done` so `selective_reset_batch` resets exactly the
    lanes the ring was told about. `n_falls` counts fall resets per lane
    (a diagnostic the driver reads and zeroes)."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    var h = obs[unsafe_offset=l * OBS_W + G1M_COL_H]
    var gz = obs[unsafe_offset=l * OBS_W + G1M_COL_GZ]
    var down = h < h_min or gz > g_max or h != h or gz != gz
    var c = fall_cnt[unsafe_offset=l] + Scalar[DT](1.0) if down else Scalar[DT](0.0)
    var fell = c >= Scalar[DT](Float64(Int(delay)))
    var d = env_done[unsafe_offset=l] > Scalar[DT](0.5) or fell
    if fell:
        n_falls[unsafe_offset=l] = n_falls[unsafe_offset=l] + Scalar[DT](1.0)
    fall_cnt[unsafe_offset=l] = Scalar[DT](0.0) if d else c
    var dv = Scalar[DT](1.0) if d else Scalar[DT](0.0)
    done_rec[unsafe_offset=l] = dv
    env_done[unsafe_offset=l] = dv


@always_inline
def _hash_u01(a: UInt64, b: UInt64) -> Float64:
    """SplitMix64 of (a, b) → [0, 1)."""
    var z = a * UInt64(0x9E3779B97F4A7C15) + b + UInt64(0x632BE59BD9B4E019)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> UInt64(31))
    return Float64(z >> UInt64(11)) * (1.0 / 9007199254740992.0)


def g1_reset_noise_kernel[NQ: Int, LANES: Int](
    qpos: Pointer[Scalar[DT], MutAnyOrigin],
    reset_mask: Pointer[Scalar[DT], MutAnyOrigin],
    amp: Scalar[DT],
    seed: UInt64,
):
    """One thread per lane: a lane that `selective_reset_batch` just reset
    (stand keyframe, every joint at its default) gets each joint moved by
    U(−amp, amp) inside its range. The plan's "stand keyframe plus joint
    noise"; the root is left at the keyframe. Run FK + velocities + obs
    after it."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    if reset_mask[unsafe_offset=l] < Scalar[DT](0.5):
        return
    comptime for i in range(G1_N_DOF):
        var u = _hash_u01(seed + UInt64(l), UInt64(i + 1))
        var q = g1_default_pos(i) + (2.0 * u - 1.0) * Float64(amp)
        comptime lo = g1_pos_lower(i)
        comptime hi = g1_pos_upper(i)
        if q < lo:
            q = lo
        if q > hi:
            q = hi
        qpos[unsafe_offset=l * NQ + ROOT_QPOS_SIZE + i] = Scalar[DT](q)


# ── the regularity reward's simulator half ───────────────────────────────

comptime G1M_SOFT_LIMIT: Float64 = 0.9


@fieldwise_init
struct G1RegWeights(Copyable, ImplicitlyCopyable, Movable):
    """Penalty weights (positive numbers, subtracted) per CONTROL step.
    Defaults are RoboParty's relative weights (`g1r_weight`): undesired
    contacts 1, feet orientation 0.1, soft joint limits 1, feet slide 0.3."""

    var undesired: Float64
    var feet_orient: Float64
    var limits: Float64
    var slide: Float64

    @staticmethod
    def default() -> Self:
        return Self(1.0, 0.1, 1.0, 0.3)


def g1_reg_env_kernel[
    N: Int, NQ: Int, NBODY: Int, MC: Int,
](
    qpos: LayoutTensor[DT, Layout.row_major(N, NQ), MutAnyOrigin],
    xquat: LayoutTensor[DT, Layout.row_major(N, NBODY * 4), MutAnyOrigin],
    xvel: LayoutTensor[DT, Layout.row_major(N, NBODY * 3), MutAnyOrigin],
    contacts: LayoutTensor[DT, Layout.row_major(N, MC * CONTACT_SIZE), MutAnyOrigin],
    meta: LayoutTensor[DT, Layout.row_major(N, METADATA_SIZE), MutAnyOrigin],
    r_env: Pointer[Scalar[DT], MutAnyOrigin],
    w_und: Scalar[DT],
    w_fo: Scalar[DT],
    w_lim: Scalar[DT],
    w_sl: Scalar[DT],
):
    """One thread per lane, after `step_batch` (the env syncs FK after the
    step, so `xquat` / `xvel` are the post-step bodies). See `G1RegWeights`."""
    var l = Int(global_idx.x)
    if l >= N:
        return
    var fc = g1r_contacts[DT, N, MC](contacts, meta, l)
    # feet orientation: gravity in each ankle-roll frame, xy² (xquat is XYZW)
    var fo = 0.0
    for f in range(2):
        var b4 = (G1R_L_ANKLE_ROLL if f == 0 else G1R_R_ANKLE_ROLL) * 4
        var gf = _projected_gravity(
            Float64(rebind[Scalar[DT]](xquat[l, b4 + 3])),
            Float64(rebind[Scalar[DT]](xquat[l, b4])),
            Float64(rebind[Scalar[DT]](xquat[l, b4 + 1])),
            Float64(rebind[Scalar[DT]](xquat[l, b4 + 2])),
        )
        fo += gf[0] * gf[0] + gf[1] * gf[1]
    # soft joint limits: distance outside 0.9 of each range around its middle
    var lim = 0.0
    comptime for i in range(G1_N_DOF):
        var q = Float64(rebind[Scalar[DT]](qpos[l, ROOT_QPOS_SIZE + i]))
        comptime c = 0.5 * (g1_pos_lower(i) + g1_pos_upper(i))
        comptime r = 0.5 * (g1_pos_upper(i) - g1_pos_lower(i)) * G1M_SOFT_LIMIT
        if q < c - r:
            lim += (c - r) - q
        elif q > c + r:
            lim += q - (c + r)
    # feet slide: the foot body's world xy speed while in contact
    var sl = 0.0
    var la = G1R_L_ANKLE_ROLL * 3
    var ra = G1R_R_ANKLE_ROLL * 3
    if fc.left:
        sl += sqrt(
            Float64(rebind[Scalar[DT]](xvel[l, la])) ** 2
            + Float64(rebind[Scalar[DT]](xvel[l, la + 1])) ** 2
        )
    if fc.right:
        sl += sqrt(
            Float64(rebind[Scalar[DT]](xvel[l, ra])) ** 2
            + Float64(rebind[Scalar[DT]](xvel[l, ra + 1])) ** 2
        )
    r_env[unsafe_offset=l] = -(
        w_und * Scalar[DT](Float64(fc.undesired))
        + w_fo * Scalar[DT](fo)
        + w_lim * Scalar[DT](lim)
        + w_sl * Scalar[DT](sl)
    )


def g1_angvel_price_kernel[OBS_W: Int, LANES: Int](
    obs: Pointer[Scalar[DT], MutAnyOrigin],
    r_env: Pointer[Scalar[DT], MutAnyOrigin],
    w_yaw: Scalar[DT],
    w_xy: Scalar[DT],
):
    """One thread per lane, AFTER `g1_reg_env_kernel` (it ADDS to `r_env`):
    `−w_yaw·ωz² − w_xy·(ωx² + ωy²)` of the post-step pelvis angular velocity
    (heading frame, obs 434:437). Plan §12.6: MEBE's density on φ ∋ ωz
    bought yaw coverage with spin, and IsaacLab's set prices ang_vel_xy
    only, so nothing in Q_reg priced the spin."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    var o = l * OBS_W + G1M_COL_WZ
    var wx = obs[unsafe_offset=o - 2]
    var wy = obs[unsafe_offset=o - 1]
    var wz = obs[unsafe_offset=o]
    r_env[unsafe_offset=l] = r_env[unsafe_offset=l] - (
        w_yaw * wz * wz + w_xy * (wx * wx + wy * wy)
    )


# ── the base-command suite (§3) ──────────────────────────────────────────

comptime G1M_SIG_V: Float64 = 0.3
comptime G1M_SIG_W: Float64 = 0.2
comptime G1M_SIG_G: Float64 = 0.1
comptime G1M_SIG_H: Float64 = 0.05
comptime G1M_CMD_P: Int = 8   # floats per task on the device
comptime G1M_STAND_H: Float64 = 0.78


@fieldwise_init
struct G1Cmd(Copyable, Movable):
    """One base command. `use_vw`: the velocity kernels r_v·r_ω are in the
    product; `use_h`: r_h is. r_g (gravity xy against `(gx, gy)`) always is.
    """

    var name: String
    var family: Int     # 0 velocity, 1 posture, 2 composite
    var vx: Float64
    var vy: Float64
    var wz: Float64
    var gx: Float64
    var gy: Float64
    var h: Float64
    var use_vw: Bool
    var use_h: Bool

    def params(self) -> List[Float64]:
        var p = List[Float64](length=G1M_CMD_P, fill=0.0)
        p[0] = self.vx
        p[1] = self.vy
        p[2] = self.wz
        p[3] = self.gx
        p[4] = self.gy
        p[5] = self.h
        p[6] = 1.0 if self.use_vw else 0.0
        p[7] = 1.0 if self.use_h else 0.0
        return p^


@always_inline
def g1_cmd_reward(
    p0: Float64, p1: Float64, p2: Float64, p3: Float64, p4: Float64,
    p5: Float64, p6: Float64, p7: Float64,
    vx: Float64, vy: Float64, wz: Float64, gx: Float64, gy: Float64,
    h: Float64,
) -> Float64:
    """THE reward of every suite task, for the device rollout and the host
    relabel alike. Gaussian kernels `exp(−e²/σ²)`; ϑ is RAW (unnormalised)."""
    var eg = (gx - p3) ** 2 + (gy - p4) ** 2
    var r = exp(-eg / (G1M_SIG_G * G1M_SIG_G))
    if p6 > 0.5:
        var ev = (vx - p0) ** 2 + (vy - p1) ** 2
        var ew = (wz - p2) ** 2
        r *= exp(-ev / (G1M_SIG_V * G1M_SIG_V)) * exp(-ew / (G1M_SIG_W * G1M_SIG_W))
    if p7 > 0.5:
        var eh = (h - p5) ** 2
        r *= exp(-eh / (G1M_SIG_H * G1M_SIG_H))
    return r


def g1_cmd_reward_obs(ref c: G1Cmd, ref rows: List[Scalar[DT]], off: Int) -> Float64:
    """`g1_cmd_reward` of the RAW 527-D row starting at `rows[off]`."""
    var p = c.params()
    return g1_cmd_reward(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        Float64(rows[off + G1M_COL_VX]), Float64(rows[off + G1M_COL_VY]),
        Float64(rows[off + G1M_COL_WZ]), Float64(rows[off + G1M_COL_GX]),
        Float64(rows[off + G1M_COL_GY]), Float64(rows[off + G1M_COL_H]),
    )


def g1_cmd_suite() -> List[G1Cmd]:
    """§3's 22 tasks. Pitch/roll targets are the projected gravity of a
    10° tilt; their sign is a convention, which is why pitch comes in a ±
    pair."""
    var s10 = sin(10.0 * 3.141592653589793 / 180.0)
    var h0 = G1M_STAND_H
    var out = List[G1Cmd]()
    # velocity (13): r_v · r_ω · r_g(upright)
    var vxs: List[Float64] = [-0.5, 0.0, 0.5, 1.0, 1.5]
    var vys: List[Float64] = [-0.3, 0.3]
    var wzs: List[Float64] = [-1.0, -0.5, 0.5, 1.0]
    var hs: List[Float64] = [0.55, 0.65, 0.78]
    for vx in vxs:
        out.append(G1Cmd("vx" + String(vx), 0, vx, 0.0, 0.0, 0.0, 0.0, h0, True, False))
    for vy in vys:
        out.append(G1Cmd("vy" + String(vy), 0, 0.0, vy, 0.0, 0.0, 0.0, h0, True, False))
    for vy in vys:
        out.append(G1Cmd("vx0.5_vy" + String(vy), 0, 0.5, vy, 0.0, 0.0, 0.0, h0, True, False))
    for wz in wzs:
        out.append(G1Cmd("wz" + String(wz), 0, 0.0, 0.0, wz, 0.0, 0.0, h0, True, False))
    # posture (6): r_g · r_h for heights; r_g alone for a tilt
    for h in hs:
        out.append(G1Cmd("h" + String(h), 1, 0.0, 0.0, 0.0, 0.0, 0.0, h, False, True))
    out.append(G1Cmd("pitch+10", 1, 0.0, 0.0, 0.0, s10, 0.0, h0, False, False))
    out.append(G1Cmd("pitch-10", 1, 0.0, 0.0, 0.0, -s10, 0.0, h0, False, False))
    out.append(G1Cmd("roll10", 1, 0.0, 0.0, 0.0, 0.0, s10, h0, False, False))
    # composite (3): the product of every kernel the task names
    out.append(G1Cmd("vx0.5_h0.65", 2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.65, True, True))
    out.append(G1Cmd("wz0.5_h0.78", 2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.78, True, True))
    out.append(G1Cmd("vx0.5_pitch10", 2, 0.5, 0.0, 0.0, s10, 0.0, h0, True, False))
    return out^


def g1_cmd_step_kernel[OBS_W: Int, LANES: Int](
    obs: Pointer[Scalar[DT], MutAnyOrigin],
    lane_task: Pointer[Scalar[DType.int32], MutAnyOrigin],
    params: Pointer[Scalar[DT], MutAnyOrigin],
    ret: Pointer[Scalar[DT], MutAnyOrigin],
    fell: Pointer[Scalar[DT], MutAnyOrigin],
    act: Pointer[Scalar[DT], MutAnyOrigin],
    prev_act: Pointer[Scalar[DT], MutAnyOrigin],
    ar: Pointer[Scalar[DT], MutAnyOrigin],
    h_fall: Scalar[DT],
    act_dim: Int32,
):
    """One thread per eval lane, after each eval step: the task reward into
    `ret`, a sticky fall flag (pelvis below `h_fall`), and the summed action
    rate ‖a − a_prev‖² (a naturalness readout). Lanes with task < 0 idle."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    var k = Int(lane_task[unsafe_offset=l])
    if k < 0:
        return
    var o = l * OBS_W
    var b = k * G1M_CMD_P
    var r = g1_cmd_reward(
        Float64(params[unsafe_offset=b]), Float64(params[unsafe_offset=b + 1]),
        Float64(params[unsafe_offset=b + 2]), Float64(params[unsafe_offset=b + 3]),
        Float64(params[unsafe_offset=b + 4]), Float64(params[unsafe_offset=b + 5]),
        Float64(params[unsafe_offset=b + 6]), Float64(params[unsafe_offset=b + 7]),
        Float64(obs[unsafe_offset=o + G1M_COL_VX]), Float64(obs[unsafe_offset=o + G1M_COL_VY]),
        Float64(obs[unsafe_offset=o + G1M_COL_WZ]), Float64(obs[unsafe_offset=o + G1M_COL_GX]),
        Float64(obs[unsafe_offset=o + G1M_COL_GY]), Float64(obs[unsafe_offset=o + G1M_COL_H]),
    )
    ret[unsafe_offset=l] = ret[unsafe_offset=l] + Scalar[DT](r)
    if obs[unsafe_offset=o + G1M_COL_H] < h_fall:
        fell[unsafe_offset=l] = Scalar[DT](1.0)
    var na = Int(act_dim)
    var s = Scalar[DT](0.0)
    for j in range(na):
        var d = act[unsafe_offset=l * na + j] - prev_act[unsafe_offset=l * na + j]
        s += d * d
        prev_act[unsafe_offset=l * na + j] = act[unsafe_offset=l * na + j]
    ar[unsafe_offset=l] = ar[unsafe_offset=l] + s
