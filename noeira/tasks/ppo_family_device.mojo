"""The family PPO rollout on the GPU (`-D TASK_PPO_DEVICE_ROLLOUT`).

`ppo_family_driver.run_ppo`'s per-step host loop — every option of it — as
kernels over the env's device buffers, so a control step has no host round
trip. One kernel per host stage, each the device twin of the host code it is
named after (read that code for the WHY; this file only mirrors it):

  `_exec_noise_k`     `--exec-noise`: Gaussian noise on the EXECUTED action
  `_smooth_k`         `--smooth-penalty`: the arm words' squared change
  `_targets_k`        `--action delta|target` (`delta_target` / `target_step`)
  `_hist_push_k`      `TASK_PPO_ACT_HIST` (`_hist_push`)
  `_to_env_k`         one tick of `_targets_to_env` (`ServoLag.apply`)
  `_tick_k`           the per-tick lane walk: divergence guard, summed reward,
                      `--dive-penalty`, success, the terminal row
  `_post_k`           the policy step's transition (`--repeat` terminal rows,
                      the smoothing penalty, the done write-back)
  `_augment_k`        `_augment` (history, `TASK_PPO_TARGET_OBS` lead)
  `_rms_*_k`          `RunningMeanStd.update` (diverged lanes masked) and
                      `normalize_into`
  `_ret_k` / `_rew_k` CleanRL's `NormalizeReward` and the episode records
  `_reset_k`          the lanes that ended: `ServoLag.reset_lane`,
                      `_target_reset`, `_hist_clear`, the smoothing's
                      `has_prev`

⚠ NOT BIT-IDENTICAL TO THE HOST LOOP, by construction: the host draws its
noise and its servo dynamics from the host RNG and keeps `ServoLag` and the
running statistics in Float64; here the draws are Philox (`seed`, lane,
an iteration-keyed offset) and everything is DT (Float32, which Metal also
runs). `tests/tasks/test_ppo_family_device_kernels.mojo` holds each kernel
against its host function on identical inputs instead.

Host work per control step: none but enqueues. Episode records (done,
success, raw return) come back through a deferred ring (`_EpisodeRing`, one
sync per `sync_every` steps); the diverged / smoothing / dive counters are
per-lane device sums read at log cadence (`read_window_stats`).
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.math import exp as fexp, log as flog, sqrt as fsqrt, cos as fcos
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import DT, TPB
from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.training.batched_env import BatchedEnv
from noeira.deep_agents.training.driver_onpolicy import OnPolicyBatchedCore
from noeira.physics3d.gpu.constants import METADATA_SIZE, META_IDX_GOAL_HELD
from noeira.tasks.delta_action import (
    DELTA_ACT, LAG_MAX_DELAY, ServoLag, ACT_HIST, TARGET_OBS,
)


comptime ACT = DELTA_ACT
comptime _V[n: Int] = LayoutTensor[DT, Layout.row_major(n), MutAnyOrigin]
comptime _Ptr = Pointer[Scalar[DT], MutAnyOrigin]

# ServoLag's configuration, packed for the device (`_lag_cfg`).
comptime _L_ON = 0
comptime _L_TAU_LO = 1
comptime _L_TAU_HI = 2
comptime _L_D_LO = 3
comptime _L_D_HI = 4
comptime _L_DT = 5
comptime _L_VMAX_LO = 6
comptime _L_VMAX_HI = 7
comptime _L_ELBOW = 8
comptime _L_PER_JOINT = 9
comptime _L_HAS_OFF = 10
comptime _L_DT_LO = 11
comptime _L_DT_HI = 12
comptime _L_JT_LO = 13
comptime _L_JT_HI = _L_JT_LO + ACT
comptime _L_JD_LO = _L_JT_HI + ACT
comptime _L_JD_HI = _L_JD_LO + ACT
comptime _L_JV_LO = _L_JD_HI + ACT
comptime _L_JV_HI = _L_JV_LO + ACT
comptime _L_OFF_LO = _L_JV_HI + ACT
comptime _L_OFF_HI = _L_OFF_LO + ACT
comptime _L_SIZE = _L_OFF_HI + ACT

# Per-lane counters summed on the device (`_stats`, [N, _S_SIZE]).
comptime _S_DIVERGED = 0
comptime _S_SPEN_SUM = 1
comptime _S_SPEN_N = 2
comptime _S_DIVE = 3
comptime _S_TICKS = 4
comptime _S_SIZE = 5

# The episode record a lane emits at each control step (`_ep`, [N, 3]).
comptime _EP_DONE = 0
comptime _EP_SUCC = 1
comptime _EP_RET = 2


@always_inline
def _clip1(v: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) if v > Scalar[DT](1.0) else (
        Scalar[DT](-1.0) if v < Scalar[DT](-1.0) else v
    )


@always_inline
def _uniform(
    seed: UInt64, lane: Int, offset: UInt64, slot: Int
) -> Scalar[DT]:
    """One uniform per (lane, offset, slot): Philox's own counters, as
    `tasks/sampler._uniform01` uses them."""
    var rng = PhiloxRandom(
        seed=seed,
        subsequence=(UInt64(lane) << 16) | UInt64(slot),
        offset=offset,
    )
    return Scalar[DT](rng.step_uniform()[0])


# ═══════════════════════════════════════════════════════════════════════════
# the policy step: the executed action
# ═══════════════════════════════════════════════════════════════════════════


def _exec_noise_k[N: Int](
    act: _V[N * ACT], sigma: Scalar[DT], seed: UInt64, offset: UInt64,
):
    """`--exec-noise`: a <- clip(a + sigma * N(0,1), -1, 1) per word."""
    var i = Int(global_idx.x)
    if i >= N * ACT:
        return
    var u1 = _uniform(seed, i, offset, 0)
    var u2 = _uniform(seed, i, offset, 1)
    if u1 < Scalar[DT](1e-12):
        u1 = Scalar[DT](1e-12)
    var g = fsqrt(Scalar[DT](-2.0) * flog(u1)) * fcos(
        Scalar[DT](6.283185307179586) * u2
    )
    act[i] = _clip1(rebind[Scalar[DT]](act[i]) + sigma * g)


def _smooth_k[N: Int](
    act: _V[N * ACT],
    a_prev: _V[N * ACT],
    has_prev: _V[N],
    spen: _V[N],
    stats: _V[N * _S_SIZE],
    smooth_w: Scalar[DT],
):
    """`--smooth-penalty`: the mean squared change of the five arm words (as
    executed, clipped) from the lane's last step; none on an episode's first."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var d2: Scalar[DT] = 0.0
    var hp = rebind[Scalar[DT]](has_prev[e]) > Scalar[DT](0.5)
    for j in range(ACT - 1):
        var v = _clip1(rebind[Scalar[DT]](act[e * ACT + j]))
        if hp:
            var dd = v - rebind[Scalar[DT]](a_prev[e * ACT + j])
            d2 += dd * dd
        a_prev[e * ACT + j] = v
    var p = smooth_w * d2 / Scalar[DT](ACT - 1)
    spen[e] = p
    has_prev[e] = Scalar[DT](1.0)
    stats[e * _S_SIZE + _S_SPEN_SUM] = (
        rebind[Scalar[DT]](stats[e * _S_SIZE + _S_SPEN_SUM]) + p
    )
    stats[e * _S_SIZE + _S_SPEN_N] = (
        rebind[Scalar[DT]](stats[e * _S_SIZE + _S_SPEN_N]) + Scalar[DT](1.0)
    )


@always_inline
def _delta_target(
    q: Scalar[DT], a: Scalar[DT], j: Int, lo: Scalar[DT], hi: Scalar[DT],
    d_arm: Scalar[DT], d_grip: Scalar[DT],
) -> Scalar[DT]:
    """`delta_action.delta_target`."""
    var sc = d_grip if j == ACT - 1 else d_arm
    var t = q + a * sc
    if t < lo:
        return lo
    if t > hi:
        return hi
    return t


def _targets_k[N: Int](
    act: _V[N * ACT],
    arm_q: _V[N * ACT],
    tg: _V[N * ACT],
    tprev: _V[N * ACT],
    a_lo: _V[ACT],
    a_hi: _V[ACT],
    mode: Int32,
    d_arm: Scalar[DT],
    d_grip: Scalar[DT],
    lead: Scalar[DT],
):
    """`_delta_targets` (mode 1) / `_target_targets` (mode 2, the target
    becomes the lane's previous target) — the held targets of this step."""
    var i = Int(global_idx.x)
    if i >= N * ACT:
        return
    var j = i % ACT
    var lo = rebind[Scalar[DT]](a_lo[j])
    var hi = rebind[Scalar[DT]](a_hi[j])
    var q = rebind[Scalar[DT]](arm_q[i])
    var a = rebind[Scalar[DT]](act[i])
    if mode == 1:
        tg[i] = _delta_target(q, a, j, lo, hi, d_arm, d_grip)
    else:
        # `target_step`
        var t = _delta_target(
            rebind[Scalar[DT]](tprev[i]), a, j, lo, hi, d_arm, d_grip
        )
        if lead > Scalar[DT](0.0):
            if t > q + lead:
                t = q + lead
            elif t < q - lead:
                t = q - lead
            if t < lo:
                t = lo
            elif t > hi:
                t = hi
        tg[i] = t
        tprev[i] = t


def _hist_push_k[N: Int, W: Int](hist: _V[N * W], act: _V[N * ACT]):
    """`_hist_push`: shift the lane's history by one, the executed action
    (clipped) in slot 0."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    for k in range(W - 1, ACT - 1, -1):
        hist[e * W + k] = hist[e * W + k - ACT]
    for j in range(ACT):
        hist[e * W + j] = _clip1(rebind[Scalar[DT]](act[e * ACT + j]))


# ═══════════════════════════════════════════════════════════════════════════
# the ticks
# ═══════════════════════════════════════════════════════════════════════════


def _to_env_k[N: Int](
    tg: _V[N * ACT],
    act: _V[N * ACT],
    env_act: _V[N * ACT],
    a_lo: _V[ACT],
    a_hi: _V[ACT],
    stepped: Int32,
    tick: Int32,
    cfg: _V[_L_SIZE],
    alpha: _V[N * ACT],
    delay: _V[N * ACT],
    y: _V[N * ACT],
    ring: _V[N * LAG_MAX_DELAY * ACT],
    vcap: _V[N * ACT],
    off: _V[N * ACT],
):
    """One tick of `_targets_to_env` (stepped modes): the held target through
    `ServoLag.apply`, normalised onto the env's absolute action. Absolute
    mode: the policy's action as is."""
    var i = Int(global_idx.x)
    if i >= N * ACT:
        return
    if stepped == 0:
        env_act[i] = act[i]
        return
    var e = i // ACT
    var j = i % ACT
    var u = rebind[Scalar[DT]](tg[i])
    var tgt = u
    if rebind[Scalar[DT]](cfg[_L_ON]) > Scalar[DT](0.5):
        var w = Int(tick) % LAG_MAX_DELAY
        ring[(e * LAG_MAX_DELAY + w) * ACT + j] = u
        var d = Int(rebind[Scalar[DT]](delay[i]))
        var r = (Int(tick) - d + LAG_MAX_DELAY * 8) % LAG_MAX_DELAY
        var ud = rebind[Scalar[DT]](ring[(e * LAG_MAX_DELAY + r) * ACT + j])
        var yi = rebind[Scalar[DT]](y[i])
        var dy = rebind[Scalar[DT]](alpha[i]) * (ud - yi)
        var cap = rebind[Scalar[DT]](vcap[i])
        if cap > Scalar[DT](0.0):
            dy = cap if dy > cap else (-cap if dy < -cap else dy)
        yi = yi + dy
        var em = rebind[Scalar[DT]](cfg[_L_ELBOW])
        if em > Scalar[DT](0.0) and j == 2 and yi > em:
            yi = em
        y[i] = yi
        tgt = yi + rebind[Scalar[DT]](off[i])
    var lo = rebind[Scalar[DT]](a_lo[j])
    var hi = rebind[Scalar[DT]](a_hi[j])
    var mid = Scalar[DT](0.5) * (lo + hi)
    var half = Scalar[DT](0.5) * (hi - lo)
    env_act[i] = (tgt - mid) / half


def _tick_k[N: Int, E_OBS: Int](
    raw: _V[N * E_OBS],
    rew: _V[N],
    done: _V[N],
    meta: _V[N * METADATA_SIZE],
    qa: _V[ACT],
    dmac: _V[N],
    diverged: _V[N],
    rsum: _V[N],
    succ: _V[N],
    term: _V[N * E_OBS],
    stats: _V[N * _S_SIZE],
    dive_w: Scalar[DT],
    obs_bound: Scalar[DT],
    rew_bound: Scalar[DT],
):
    """The tick's lane walk of `run_ppo`: a lane that already ended skips; a
    non-finite or out-of-bound word or reward DIVERGES it (it ends, no
    reward); else the reward is summed, the dive penalty paid, the goal bit
    read, and a done ends it. A lane that ended at this tick keeps its row
    as the terminal row."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    if rebind[Scalar[DT]](dmac[e]) > Scalar[DT](0.5):
        return
    var bad = False
    var rv = rebind[Scalar[DT]](rew[e])
    if not (rv == rv) or abs(rv) > rew_bound:
        bad = True
    for k in range(E_OBS):
        var v = rebind[Scalar[DT]](raw[e * E_OBS + k])
        if not (v == v) or abs(v) > obs_bound:
            bad = True
            break
    var ended = False
    if bad:
        diverged[e] = Scalar[DT](1.0)
        stats[e * _S_SIZE + _S_DIVERGED] = (
            rebind[Scalar[DT]](stats[e * _S_SIZE + _S_DIVERGED])
            + Scalar[DT](1.0)
        )
        ended = True
    else:
        var rs = rebind[Scalar[DT]](rsum[e]) + rv
        if dive_w > Scalar[DT](0.0):
            stats[e * _S_SIZE + _S_TICKS] = (
                rebind[Scalar[DT]](stats[e * _S_SIZE + _S_TICKS])
                + Scalar[DT](1.0)
            )
            var q1 = rebind[Scalar[DT]](
                raw[e * E_OBS + Int(rebind[Scalar[DT]](qa[1]))]
            )
            var q2 = rebind[Scalar[DT]](
                raw[e * E_OBS + Int(rebind[Scalar[DT]](qa[2]))]
            )
            if q1 > Scalar[DT](1.35) and q2 < Scalar[DT](-1.35):
                rs -= dive_w
                stats[e * _S_SIZE + _S_DIVE] = (
                    rebind[Scalar[DT]](stats[e * _S_SIZE + _S_DIVE])
                    + Scalar[DT](1.0)
                )
        rsum[e] = rs
        if rebind[Scalar[DT]](
            meta[e * METADATA_SIZE + META_IDX_GOAL_HELD]
        ) > Scalar[DT](0.5):
            succ[e] = Scalar[DT](1.0)
        if rebind[Scalar[DT]](done[e]) > Scalar[DT](0.5):
            ended = True
    if ended:
        dmac[e] = Scalar[DT](1.0)
        for k in range(E_OBS):
            term[e * E_OBS + k] = raw[e * E_OBS + k]


def _post_k[N: Int, E_OBS: Int](
    raw: _V[N * E_OBS],
    term: _V[N * E_OBS],
    raw_post: _V[N * E_OBS],
    rsum: _V[N],
    spen: _V[N],
    dmac: _V[N],
    rew_out: _V[N],
    done_out: _V[N],
    env_done: _V[N],
    smooth_w: Scalar[DT],
    repeat: Int32,
):
    """The policy step's transition: the summed reward (less the smoothing
    penalty), done if it ended at any tick, a lane that ended before the last
    tick its terminal row; the done written back to the env for its reset."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var r = rebind[Scalar[DT]](rsum[e])
    if smooth_w > Scalar[DT](0.0):
        r -= rebind[Scalar[DT]](spen[e])
    rew_out[e] = r
    var d = rebind[Scalar[DT]](dmac[e]) > Scalar[DT](0.5)
    done_out[e] = Scalar[DT](1.0) if d else Scalar[DT](0.0)
    env_done[e] = Scalar[DT](1.0) if d else Scalar[DT](0.0)
    var use_term = d and repeat > 1
    for k in range(E_OBS):
        raw_post[e * E_OBS + k] = (
            term[e * E_OBS + k] if use_term else raw[e * E_OBS + k]
        )


def _augment_k[N: Int, E_OBS: Int, W: Int, T: Int](
    raw: _V[N * E_OBS],
    hist: _V[N * W + 1],
    tprev: _V[N * ACT],
    qa: _V[ACT],
    aug: _V[N * (E_OBS + W + T)],
):
    """`_augment`: the env row, the history, then (`TARGET_OBS`) the target's
    lead over the joints. `hist` is sized `N*W + 1` so W = 0 still builds."""
    comptime A = E_OBS + W + T
    var e = Int(global_idx.x)
    if e >= N:
        return
    for k in range(E_OBS):
        aug[e * A + k] = raw[e * E_OBS + k]
    for k in range(W):
        aug[e * A + E_OBS + k] = hist[e * W + k]
    comptime if T > 0:
        for j in range(ACT):
            var q = rebind[Scalar[DT]](
                raw[e * E_OBS + Int(rebind[Scalar[DT]](qa[j]))]
            )
            aug[e * A + E_OBS + W + j] = rebind[Scalar[DT]](
                tprev[e * ACT + j]
            ) - q


# ═══════════════════════════════════════════════════════════════════════════
# running statistics (`RunningMeanStd`)
# ═══════════════════════════════════════════════════════════════════════════


def _rms_update_k[N: Int, D: Int](
    x: _V[N * D],
    skip: _V[N],
    use_skip: Int32,
    mean: _V[D],
    var_: _V[D],
    count: _V[1],
):
    """`RunningMeanStd.update`, one thread per dimension (each walks the
    lanes in the host's order); rows with `skip` set are left out when
    `use_skip`. The shared count is NOT written here (`_rms_count_k`), so
    every dimension merges against the same old count."""
    var k = Int(global_idx.x)
    if k >= D:
        return
    var n = 0
    var bm: Scalar[DT] = 0.0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        n += 1
        bm += rebind[Scalar[DT]](x[i * D + k])
    if n == 0:
        return
    bm /= Scalar[DT](n)
    var bv: Scalar[DT] = 0.0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        var d = rebind[Scalar[DT]](x[i * D + k]) - bm
        bv += d * d
    bv /= Scalar[DT](n)
    var c = rebind[Scalar[DT]](count[0])
    var nf = Scalar[DT](n)
    var tot = c + nf
    var mk = rebind[Scalar[DT]](mean[k])
    var delta = bm - mk
    var m2 = rebind[Scalar[DT]](var_[k]) * c + bv * nf + delta * delta * c * nf / tot
    mean[k] = mk + delta * nf / tot
    var_[k] = m2 / tot


def _rms_count_k[N: Int](
    skip: _V[N], use_skip: Int32, count: _V[1],
):
    """The count's half of `RunningMeanStd.update`, after `_rms_update_k`."""
    if Int(global_idx.x) != 0:
        return
    var n = 0
    for i in range(N):
        if use_skip != 0 and rebind[Scalar[DT]](skip[i]) > Scalar[DT](0.5):
            continue
        n += 1
    if n > 0:
        count[0] = rebind[Scalar[DT]](count[0]) + Scalar[DT](n)


def _rms_normalize_k[N: Int, D: Int](
    x: _V[N * D],
    dst: _V[N * D],
    mean: _V[D],
    var_: _V[D],
    clip: Scalar[DT],
    zero: _V[N],
    use_zero: Int32,
):
    """`RunningMeanStd.normalize_into`, then (`use_zero`) the diverged lanes'
    rows zeroed, as `run_ppo` does to a diverged lane's terminal obs."""
    var i = Int(global_idx.x)
    if i >= N * D:
        return
    var e = i // D
    var k = i % D
    if use_zero != 0 and rebind[Scalar[DT]](zero[e]) > Scalar[DT](0.5):
        dst[i] = Scalar[DT](0.0)
        return
    var v = (rebind[Scalar[DT]](x[i]) - rebind[Scalar[DT]](mean[k])) / fsqrt(
        rebind[Scalar[DT]](var_[k]) + Scalar[DT](1e-8)
    )
    if v > clip:
        v = clip
    elif v < -clip:
        v = -clip
    dst[i] = v


# ═══════════════════════════════════════════════════════════════════════════
# rewards and episodes
# ═══════════════════════════════════════════════════════════════════════════


def _ret_k[N: Int](
    rew: _V[N], ret_acc: _V[N], raw_ret: _V[N], rets: _V[N],
    gamma: Scalar[DT],
):
    """The discounted return each lane's reward is scaled by, and its raw
    episode return."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var r = rebind[Scalar[DT]](rew[e])
    var ra = rebind[Scalar[DT]](ret_acc[e]) * gamma + r
    ret_acc[e] = ra
    rets[e] = ra
    raw_ret[e] = rebind[Scalar[DT]](raw_ret[e]) + r


def _rew_k[N: Int](
    rew: _V[N],
    done: _V[N],
    ret_var: _V[1],
    rew_n: _V[N],
    succ: _V[N],
    raw_ret: _V[N],
    ret_acc: _V[N],
    ep: _V[N * 3],
    clip: Scalar[DT],
):
    """`NormalizeReward` (r / std of the discounted return, clipped), and the
    episode record of a lane that ended (its success, its raw return) — then
    that lane's accumulators start over."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var scale = Scalar[DT](1.0) / fsqrt(
        rebind[Scalar[DT]](ret_var[0]) + Scalar[DT](1e-8)
    )
    var v = rebind[Scalar[DT]](rew[e]) * scale
    if v > clip:
        v = clip
    elif v < -clip:
        v = -clip
    rew_n[e] = v
    if rebind[Scalar[DT]](done[e]) > Scalar[DT](0.5):
        ep[e * 3 + _EP_DONE] = Scalar[DT](1.0)
        ep[e * 3 + _EP_SUCC] = succ[e]
        ep[e * 3 + _EP_RET] = raw_ret[e]
        succ[e] = Scalar[DT](0.0)
        raw_ret[e] = Scalar[DT](0.0)
        ret_acc[e] = Scalar[DT](0.0)
    else:
        ep[e * 3 + _EP_DONE] = Scalar[DT](0.0)


# ═══════════════════════════════════════════════════════════════════════════
# a lane's new episode
# ═══════════════════════════════════════════════════════════════════════════


def _reset_k[N: Int, E_OBS: Int, W: Int](
    raw: _V[N * E_OBS],
    qa: _V[ACT],
    done: _V[N],
    all_lanes: Int32,
    arm_q: _V[N * ACT],
    tprev: _V[N * ACT],
    hist: _V[N * W + 1],
    has_prev: _V[N],
    dmac: _V[N],
    diverged: _V[N],
    rsum: _V[N],
    cfg: _V[_L_SIZE],
    alpha: _V[N * ACT],
    delay: _V[N * ACT],
    y: _V[N * ACT],
    ring: _V[N * LAG_MAX_DELAY * ACT],
    vcap: _V[N * ACT],
    off: _V[N * ACT],
    seed: UInt64,
    offset: UInt64,
):
    """Every lane: the joints from the (post-reset) observation, and the next
    step's per-tick accumulators cleared. A lane that ended (or every lane,
    `all_lanes`): `ServoLag.reset_lane` (its dynamics drawn — Philox, keyed
    on (lane, offset)), `_target_reset`, `_hist_clear`, no previous action."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    for j in range(ACT):
        arm_q[e * ACT + j] = raw[e * E_OBS + Int(rebind[Scalar[DT]](qa[j]))]
    dmac[e] = Scalar[DT](0.0)
    diverged[e] = Scalar[DT](0.0)
    rsum[e] = Scalar[DT](0.0)
    if all_lanes == 0 and rebind[Scalar[DT]](done[e]) <= Scalar[DT](0.5):
        return
    for j in range(ACT):
        tprev[e * ACT + j] = arm_q[e * ACT + j]
    for k in range(W):
        hist[e * W + k] = Scalar[DT](0.0)
    has_prev[e] = Scalar[DT](0.0)
    if rebind[Scalar[DT]](cfg[_L_ON]) <= Scalar[DT](0.5):
        return
    # `ServoLag.reset_lane`: slot 0 the period, 1-3 the shared draws,
    # 4 + 4 j + (0..3) the per-joint offset / tau / delay / cap draws.
    var dt = rebind[Scalar[DT]](cfg[_L_DT])
    var dt_lo = rebind[Scalar[DT]](cfg[_L_DT_LO])
    var dt_hi = rebind[Scalar[DT]](cfg[_L_DT_HI])
    if dt_hi > Scalar[DT](0.0):
        dt = dt_lo + (dt_hi - dt_lo) * _uniform(seed, e, offset, 0)
    var u01 = _uniform(seed, e, offset, 1)
    var u02 = _uniform(seed, e, offset, 2)
    var u03 = _uniform(seed, e, offset, 3)
    var per_joint = rebind[Scalar[DT]](cfg[_L_PER_JOINT]) > Scalar[DT](0.5)
    for j in range(ACT):
        var i = e * ACT + j
        if rebind[Scalar[DT]](cfg[_L_HAS_OFF]) > Scalar[DT](0.5):
            var lo = rebind[Scalar[DT]](cfg[_L_OFF_LO + j])
            var hi = rebind[Scalar[DT]](cfg[_L_OFF_HI + j])
            off[i] = lo + (hi - lo) * _uniform(seed, e, offset, 4 + 4 * j)
        var ut = u01
        var ud = u02
        var uv = u03
        if per_joint:
            ut = _uniform(seed, e, offset, 5 + 4 * j)
            ud = _uniform(seed, e, offset, 6 + 4 * j)
            uv = _uniform(seed, e, offset, 7 + 4 * j)
        var tlo = rebind[Scalar[DT]](cfg[_L_TAU_LO])
        var thi = rebind[Scalar[DT]](cfg[_L_TAU_HI])
        if per_joint and rebind[Scalar[DT]](cfg[_L_JT_HI + j]) > Scalar[DT](0.0):
            tlo = rebind[Scalar[DT]](cfg[_L_JT_LO + j])
            thi = rebind[Scalar[DT]](cfg[_L_JT_HI + j])
        var tau = tlo + (thi - tlo) * ut
        alpha[i] = (
            Scalar[DT](1.0) - fexp(-dt / tau)
            if tau > Scalar[DT](1e-6)
            else Scalar[DT](1.0)
        )
        var dlo = Int(rebind[Scalar[DT]](cfg[_L_D_LO]))
        var dhi = Int(rebind[Scalar[DT]](cfg[_L_D_HI]))
        if per_joint and rebind[Scalar[DT]](cfg[_L_JD_LO + j]) >= Scalar[DT](0.0):
            dlo = Int(rebind[Scalar[DT]](cfg[_L_JD_LO + j]))
            dhi = Int(rebind[Scalar[DT]](cfg[_L_JD_HI + j]))
        var d = dlo + Int(ud * Scalar[DT](dhi - dlo + 1))
        if d > dhi:
            d = dhi
        delay[i] = Scalar[DT](d)
        var vmax: Scalar[DT] = 0.0
        if per_joint and rebind[Scalar[DT]](cfg[_L_JV_HI + j]) > Scalar[DT](0.0):
            var vlo = rebind[Scalar[DT]](cfg[_L_JV_LO + j])
            var vhi = rebind[Scalar[DT]](cfg[_L_JV_HI + j])
            vmax = vlo + (vhi - vlo) * uv
        elif j < ACT - 1:
            var vlo = rebind[Scalar[DT]](cfg[_L_VMAX_LO])
            var vhi = rebind[Scalar[DT]](cfg[_L_VMAX_HI])
            vmax = vlo + (vhi - vlo) * uv
        vcap[i] = vmax * dt
        var q = rebind[Scalar[DT]](arm_q[i])
        y[i] = q
        for k in range(LAG_MAX_DELAY):
            ring[(e * LAG_MAX_DELAY + k) * ACT + j] = q


def lag_cfg_words(ref lag: ServoLag) -> List[Scalar[DT]]:
    """`ServoLag`'s configuration packed as `_lag_cfg` (the host struct stays
    the parser: `--lag-*` go through `parse` / `set_*` as before)."""
    var w = List[Scalar[DT]](length=_L_SIZE, fill=Scalar[DT](0))
    w[_L_ON] = Scalar[DT](1.0) if lag.on else Scalar[DT](0.0)
    w[_L_TAU_LO] = Scalar[DT](lag.tau_lo)
    w[_L_TAU_HI] = Scalar[DT](lag.tau_hi)
    w[_L_D_LO] = Scalar[DT](lag.d_lo)
    w[_L_D_HI] = Scalar[DT](lag.d_hi)
    w[_L_DT] = Scalar[DT](lag.dt)
    w[_L_VMAX_LO] = Scalar[DT](lag.vmax_lo)
    w[_L_VMAX_HI] = Scalar[DT](lag.vmax_hi)
    w[_L_ELBOW] = Scalar[DT](lag.elbow_max)
    w[_L_PER_JOINT] = Scalar[DT](1.0) if lag.per_joint else Scalar[DT](0.0)
    w[_L_HAS_OFF] = Scalar[DT](1.0) if lag.has_off else Scalar[DT](0.0)
    w[_L_DT_LO] = Scalar[DT](lag.dt_lo)
    w[_L_DT_HI] = Scalar[DT](lag.dt_hi)
    for j in range(ACT):
        w[_L_JT_LO + j] = Scalar[DT](lag.jt_lo[j])
        w[_L_JT_HI + j] = Scalar[DT](lag.jt_hi[j])
        w[_L_JD_LO + j] = Scalar[DT](lag.jd_lo[j])
        w[_L_JD_HI + j] = Scalar[DT](lag.jd_hi[j])
        w[_L_JV_LO + j] = Scalar[DT](lag.jv_lo[j])
        w[_L_JV_HI + j] = Scalar[DT](lag.jv_hi[j])
        w[_L_OFF_LO + j] = Scalar[DT](lag.off_lo[j])
        w[_L_OFF_HI + j] = Scalar[DT](lag.off_hi[j])
    return w^


# ═══════════════════════════════════════════════════════════════════════════
# the rollout
# ═══════════════════════════════════════════════════════════════════════════


def _update_rms[N: Int, D: Int](
    ctx: DeviceContext,
    mut x: Tensor,
    mut skip: Tensor,
    mut mean: Tensor,
    mut var_: Tensor,
    mut count: Tensor,
    use_skip: Bool,
) raises:
    """`RunningMeanStd.update` on the device: the per-dimension merge, then
    the shared count."""
    var us = Int32(1) if use_skip else Int32(0)
    ctx.enqueue_function[_rms_update_k[N, D]](
        x.lt["gpu", Layout.row_major(N * D)](),
        skip.lt["gpu", Layout.row_major(N)](),
        us,
        mean.lt["gpu", Layout.row_major(D)](),
        var_.lt["gpu", Layout.row_major(D)](),
        count.lt["gpu", Layout.row_major(1)](),
        grid_dim=(D + TPB - 1) // TPB, block_dim=TPB,
    )
    ctx.enqueue_function[_rms_count_k[N]](
        skip.lt["gpu", Layout.row_major(N)](),
        us,
        count.lt["gpu", Layout.row_major(1)](),
        grid_dim=1, block_dim=1,
    )


def _normalize[N: Int, D: Int](
    ctx: DeviceContext,
    mut src: Tensor,
    mut dst: Tensor,
    mut mean: Tensor,
    mut var_: Tensor,
    mut zero: Tensor,
    clip: Float64,
    zero_rows: Bool,
) raises:
    """`RunningMeanStd.normalize_into` on the device (`zero_rows`: the rows
    flagged in `zero` written as 0)."""
    ctx.enqueue_function[_rms_normalize_k[N, D]](
        src.lt["gpu", Layout.row_major(N * D)](),
        dst.lt["gpu", Layout.row_major(N * D)](),
        mean.lt["gpu", Layout.row_major(D)](),
        var_.lt["gpu", Layout.row_major(D)](),
        Scalar[DT](clip),
        zero.lt["gpu", Layout.row_major(N)](),
        Int32(1) if zero_rows else Int32(0),
        grid_dim=(N * D + TPB - 1) // TPB, block_dim=TPB,
    )


@fieldwise_init
struct FamilyStepConfig(Copyable, Movable):
    """The per-step options of `run_ppo` (its flags), as the kernels take
    them."""

    var mode: Int  # 0 absolute, 1 delta, 2 target
    var repeat: Int
    var d_arm: Float64
    var d_grip: Float64
    var lead: Float64
    var exec_noise: Float64
    var smooth_w: Float64
    var dive_w: Float64
    var gamma: Float64
    var obs_clip: Float64
    var rew_clip: Float64
    var obs_bound: Float64
    var rew_bound: Float64


struct FamilyDeviceRollout[N_: Int, E_OBS_: Int](Movable):
    """The rollout's device state and its control step. `OBS` is the
    policy's observation (env + action history + target lead)."""

    comptime N = Self.N_
    comptime E_OBS = Self.E_OBS_
    comptime W = ACT_HIST * ACT
    comptime T = TARGET_OBS
    comptime OBS = Self.E_OBS + Self.W + Self.T

    var cfg: FamilyStepConfig
    var seed: UInt64
    var ctx: DeviceContext
    # the policy's observations and action
    var cur_n: Tensor
    var next_n: Tensor
    var aug: Tensor
    var act: Tensor
    # per-lane action state
    var hist: Tensor
    var tprev: Tensor
    var arm_q: Tensor
    var tg: Tensor
    var a_prev: Tensor
    var has_prev: Tensor
    var spen: Tensor
    # ServoLag
    var lag_cfg: Tensor
    var alpha: Tensor
    var delay: Tensor
    var y: Tensor
    var ring: Tensor
    var vcap: Tensor
    var off: Tensor
    var tick: Int
    # the step's per-tick accumulators and its transition
    var rsum: Tensor
    var dmac: Tensor
    var diverged: Tensor
    var succ: Tensor
    var term: Tensor
    var raw_post: Tensor
    var rew_raw: Tensor
    var done_out: Tensor
    var zeros: Tensor
    # reward normalisation and the episode records
    var ret_acc: Tensor
    var raw_ret: Tensor
    var rets: Tensor
    var rew_n: Tensor
    var ep: Tensor
    # running statistics
    var obs_mean: Tensor
    var obs_var: Tensor
    var obs_count: Tensor
    var ret_mean: Tensor
    var ret_var: Tensor
    var ret_count: Tensor
    # per-lane counters (log cadence)
    var stats: Tensor
    # the lane constants
    var qa: Tensor
    var a_lo: Tensor
    var a_hi: Tensor
    # the episode-record ring
    var _ring_bufs: List[HostBuffer[DT]]
    var _pending: Int
    var _sync_every: Int
    var it: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        cfg: FamilyStepConfig,
        seed: UInt64,
        ref a_qa: List[Int],
        ref a_lo: List[Float64],
        ref a_hi: List[Float64],
        ref lag: ServoLag,
        sync_every: Int = 32,
    ) raises:
        self.cfg = cfg.copy()
        self.seed = seed
        self.ctx = ctx
        comptime N = Self.N
        self.cur_n = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.next_n = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.aug = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.act = Tensor.alloc_gpu(ctx, N * ACT)
        self.hist = Tensor.alloc_gpu(ctx, N * Self.W + 1)
        self.tprev = Tensor.alloc_gpu(ctx, N * ACT)
        self.arm_q = Tensor.alloc_gpu(ctx, N * ACT)
        self.tg = Tensor.alloc_gpu(ctx, N * ACT)
        self.a_prev = Tensor.alloc_gpu(ctx, N * ACT)
        self.has_prev = Tensor.alloc_gpu(ctx, N)
        self.spen = Tensor.alloc_gpu(ctx, N)
        self.lag_cfg = Tensor.alloc(_L_SIZE)
        var w = lag_cfg_words(lag)
        for k in range(_L_SIZE):
            self.lag_cfg.data[k] = w[k]
        self.lag_cfg.upload(ctx)
        self.alpha = Tensor.alloc_gpu(ctx, N * ACT)
        fill_dev(self.alpha.dev.value(), N * ACT, Scalar[DT](1.0), ctx)
        self.delay = Tensor.alloc_gpu(ctx, N * ACT)
        self.y = Tensor.alloc_gpu(ctx, N * ACT)
        self.ring = Tensor.alloc_gpu(ctx, N * LAG_MAX_DELAY * ACT)
        self.vcap = Tensor.alloc_gpu(ctx, N * ACT)
        self.off = Tensor.alloc_gpu(ctx, N * ACT)
        self.tick = 0
        self.rsum = Tensor.alloc_gpu(ctx, N)
        self.dmac = Tensor.alloc_gpu(ctx, N)
        self.diverged = Tensor.alloc_gpu(ctx, N)
        self.succ = Tensor.alloc_gpu(ctx, N)
        self.term = Tensor.alloc_gpu(ctx, N * Self.E_OBS)
        self.raw_post = Tensor.alloc_gpu(ctx, N * Self.E_OBS)
        self.rew_raw = Tensor.alloc_gpu(ctx, N)
        self.done_out = Tensor.alloc_gpu(ctx, N)
        self.zeros = Tensor.alloc_gpu(ctx, N)
        self.ret_acc = Tensor.alloc_gpu(ctx, N)
        self.raw_ret = Tensor.alloc_gpu(ctx, N)
        self.rets = Tensor.alloc_gpu(ctx, N)
        self.rew_n = Tensor.alloc_gpu(ctx, N)
        self.ep = Tensor.alloc_gpu(ctx, N * 3)
        self.obs_mean = Tensor.alloc_gpu(ctx, Self.OBS)
        self.obs_var = Tensor.alloc_gpu(ctx, Self.OBS)
        fill_dev(self.obs_var.dev.value(), Self.OBS, Scalar[DT](1.0), ctx)
        self.obs_count = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.obs_count.dev.value(), 1, Scalar[DT](1e-4), ctx)
        self.ret_mean = Tensor.alloc_gpu(ctx, 1)
        self.ret_var = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.ret_var.dev.value(), 1, Scalar[DT](1.0), ctx)
        self.ret_count = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.ret_count.dev.value(), 1, Scalar[DT](1e-4), ctx)
        self.stats = Tensor.alloc_gpu(ctx, N * _S_SIZE)
        self.qa = Tensor.alloc(ACT)
        self.a_lo = Tensor.alloc(ACT)
        self.a_hi = Tensor.alloc(ACT)
        for j in range(ACT):
            self.qa.data[j] = Scalar[DT](a_qa[j])
            self.a_lo.data[j] = Scalar[DT](a_lo[j])
            self.a_hi.data[j] = Scalar[DT](a_hi[j])
        self.qa.upload(ctx)
        self.a_lo.upload(ctx)
        self.a_hi.upload(ctx)
        self._sync_every = max(sync_every, 1)
        self._ring_bufs = List[HostBuffer[DT]]()
        for _ in range(self._sync_every):
            self._ring_bufs.append(ctx.enqueue_create_host_buffer[DT](N * 3))
        self._pending = 0
        self.it = 0

    # ── grid helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _g(n: Int) -> Int:
        return (n + TPB - 1) // TPB

    # ── running statistics in and out (the host `RunningMeanStd`) ─────────

    def stats_from_host(
        mut self, ref mean: List[Float64], ref var_: List[Float64],
        count: Float64,
    ) raises:
        """Seed the device observation statistics (an `--init` run's, or the
        BC pre-training's)."""
        self.obs_mean.ensure(Self.OBS)
        self.obs_var.ensure(Self.OBS)
        for k in range(Self.OBS):
            self.obs_mean.data[k] = Scalar[DT](mean[k])
            self.obs_var.data[k] = Scalar[DT](var_[k])
        self.obs_mean.upload_resident(self.ctx)
        self.obs_var.upload_resident(self.ctx)
        fill_dev(self.obs_count.dev.value(), 1, Scalar[DT](count), self.ctx)
        self.ctx.synchronize()

    def stats_to_host(
        mut self, mut mean: List[Float64], mut var_: List[Float64],
        mut count: Float64,
    ) raises:
        """The device observation statistics into the host struct's lists —
        for its `save` and the greedy evaluation."""
        self.obs_mean.ensure(Self.OBS)
        self.obs_var.ensure(Self.OBS)
        self.obs_count.ensure(1)
        self.obs_mean.download(self.ctx)
        self.obs_var.download(self.ctx)
        self.obs_count.download(self.ctx)
        for k in range(Self.OBS):
            mean[k] = Float64(self.obs_mean.data[k])
            var_[k] = Float64(self.obs_var.data[k])
        count = Float64(self.obs_count.data[0])

    def read_window_stats(mut self) raises -> List[Float64]:
        """[diverged, smoothing sum, smoothing count, dive ticks, ticks],
        summed over lanes, since the last `reset_window_stats` — except the
        diverged count, which is cumulative like the host's."""
        self.stats.ensure(Self.N * _S_SIZE)
        self.stats.download(self.ctx)
        var s = List[Float64](length=_S_SIZE, fill=0.0)
        for e in range(Self.N):
            for k in range(_S_SIZE):
                s[k] += Float64(self.stats.data[e * _S_SIZE + k])
        return s^

    def reset_window_stats(mut self) raises:
        """Zero the smoothing and dive sums (the host's every-10-updates
        reset); the diverged count is kept."""
        self.stats.ensure(Self.N * _S_SIZE)
        self.stats.download(self.ctx)
        for e in range(Self.N):
            for k in range(_S_SIZE):
                if k != _S_DIVERGED:
                    self.stats.data[e * _S_SIZE + k] = Scalar[DT](0)
        self.stats.upload_resident(self.ctx)

    # ── the episode records ─────────────────────────────────────────────

    def drain(
        mut self, mut hist_succ: List[Bool], mut hist_ret: List[Float64],
        mut n_episodes: Int,
    ) raises:
        """ONE sync, then every buffered step's records in order, lanes in
        order — the host loop's `hist_succ` / `hist_ret` / `n_episodes`."""
        if self._pending == 0:
            return
        self.ctx.synchronize()
        for s in range(self._pending):
            var p = self._ring_bufs[s].unsafe_ptr()
            for e in range(Self.N):
                if p[e * 3 + _EP_DONE] > Scalar[DT](0.5):
                    hist_succ.append(p[e * 3 + _EP_SUCC] > Scalar[DT](0.5))
                    hist_ret.append(Float64(p[e * 3 + _EP_RET]))
                    n_episodes += 1
        self._pending = 0

    def ring_full(self) -> Bool:
        return self._pending >= self._sync_every

    # ── the first observation and the per-step pieces ────────────────────

    def _augment(mut self, raw_ptr: _Ptr) raises:
        comptime N = Self.N
        self.ctx.enqueue_function[
            _augment_k[N, Self.E_OBS, Self.W, Self.T]
        ](
            _V[N * Self.E_OBS](raw_ptr),
            self.hist.lt["gpu", Layout.row_major(N * Self.W + 1)](),
            self.tprev.lt["gpu", Layout.row_major(N * ACT)](),
            self.qa.lt["gpu", Layout.row_major(ACT)](),
            self.aug.lt["gpu", Layout.row_major(N * Self.OBS)](),
            grid_dim=Self._g(N), block_dim=TPB,
        )

    def _reset_lanes(
        mut self, raw_ptr: _Ptr, done_ptr: _Ptr, all_lanes: Bool
    ) raises:
        comptime N = Self.N
        self.ctx.enqueue_function[_reset_k[N, Self.E_OBS, Self.W]](
            _V[N * Self.E_OBS](raw_ptr),
            self.qa.lt["gpu", Layout.row_major(ACT)](),
            _V[N](done_ptr),
            Int32(1) if all_lanes else Int32(0),
            self.arm_q.lt["gpu", Layout.row_major(N * ACT)](),
            self.tprev.lt["gpu", Layout.row_major(N * ACT)](),
            self.hist.lt["gpu", Layout.row_major(N * Self.W + 1)](),
            self.has_prev.lt["gpu", Layout.row_major(N)](),
            self.dmac.lt["gpu", Layout.row_major(N)](),
            self.diverged.lt["gpu", Layout.row_major(N)](),
            self.rsum.lt["gpu", Layout.row_major(N)](),
            self.lag_cfg.lt["gpu", Layout.row_major(_L_SIZE)](),
            self.alpha.lt["gpu", Layout.row_major(N * ACT)](),
            self.delay.lt["gpu", Layout.row_major(N * ACT)](),
            self.y.lt["gpu", Layout.row_major(N * ACT)](),
            self.ring.lt["gpu", Layout.row_major(N * LAG_MAX_DELAY * ACT)](),
            self.vcap.lt["gpu", Layout.row_major(N * ACT)](),
            self.off.lt["gpu", Layout.row_major(N * ACT)](),
            self.seed ^ UInt64(0x5E2F0),
            UInt64(self.it),
            grid_dim=Self._g(N), block_dim=TPB,
        )

    def start(mut self, raw_ptr: _Ptr) raises:
        """The first observation (`run_ppo` before its loop): every lane's
        joints, servo model and target, the statistics updated with every
        lane, the first normalised observation."""
        self._reset_lanes(raw_ptr, mptr(self.zeros.dev.value().unsafe_ptr()), True)
        self._augment(raw_ptr)
        _update_rms[Self.N, Self.OBS](
            self.ctx, self.aug, self.diverged, self.obs_mean, self.obs_var,
            self.obs_count, False,
        )
        _normalize[Self.N, Self.OBS](
            self.ctx, self.aug, self.cur_n, self.obs_mean, self.obs_var,
            self.diverged, self.cfg.obs_clip, False,
        )

    def step[
        A: OnPolicyBatchedCore, E: BatchedEnv, USE_ENV_GRAPH: Bool = False
    ](
        mut self,
        mut trainer: A,
        mut env: E,
        meta_ptr: _Ptr,
        mut env_graph: Optional[CUDAGraph],
        mut reset_graph: Optional[CUDAGraph],
    ) raises:
        """One policy step of `run_ppo` on the device: act, the executed
        action, `repeat` ticks, the transition, normalisation, the record,
        the reset of the lanes that ended, the next observation."""
        comptime N = Self.N
        comptime EO = Self.E_OBS
        var c = self.ctx
        var cfg = self.cfg.copy()
        var stepped = cfg.mode != 0
        # 1. act on the normalised observation
        trainer.select_action_device(
            mptr(self.cur_n.dev.value().unsafe_ptr()), mptr(self.act.dev.value().unsafe_ptr())
        )
        if cfg.exec_noise > 0.0:
            c.enqueue_function[_exec_noise_k[N]](
                self.act.lt["gpu", Layout.row_major(N * ACT)](),
                Scalar[DT](cfg.exec_noise),
                self.seed ^ UInt64(0xE7EC),
                UInt64(self.it),
                grid_dim=Self._g(N * ACT), block_dim=TPB,
            )
        if cfg.smooth_w > 0.0:
            c.enqueue_function[_smooth_k[N]](
                self.act.lt["gpu", Layout.row_major(N * ACT)](),
                self.a_prev.lt["gpu", Layout.row_major(N * ACT)](),
                self.has_prev.lt["gpu", Layout.row_major(N)](),
                self.spen.lt["gpu", Layout.row_major(N)](),
                self.stats.lt["gpu", Layout.row_major(N * _S_SIZE)](),
                Scalar[DT](cfg.smooth_w),
                grid_dim=Self._g(N), block_dim=TPB,
            )
        if stepped:
            c.enqueue_function[_targets_k[N]](
                self.act.lt["gpu", Layout.row_major(N * ACT)](),
                self.arm_q.lt["gpu", Layout.row_major(N * ACT)](),
                self.tg.lt["gpu", Layout.row_major(N * ACT)](),
                self.tprev.lt["gpu", Layout.row_major(N * ACT)](),
                self.a_lo.lt["gpu", Layout.row_major(ACT)](),
                self.a_hi.lt["gpu", Layout.row_major(ACT)](),
                Int32(cfg.mode),
                Scalar[DT](cfg.d_arm),
                Scalar[DT](cfg.d_grip),
                Scalar[DT](cfg.lead),
                grid_dim=Self._g(N * ACT), block_dim=TPB,
            )
        comptime if Self.W > 0:
            c.enqueue_function[_hist_push_k[N, Self.W]](
                self.hist.lt["gpu", Layout.row_major(N * Self.W)](),
                self.act.lt["gpu", Layout.row_major(N * ACT)](),
                grid_dim=Self._g(N), block_dim=TPB,
            )
        # 2. the ticks under the held targets
        for t in range(cfg.repeat):
            c.enqueue_function[_to_env_k[N]](
                self.tg.lt["gpu", Layout.row_major(N * ACT)](),
                self.act.lt["gpu", Layout.row_major(N * ACT)](),
                _V[N * ACT](env.action_ptr()),
                self.a_lo.lt["gpu", Layout.row_major(ACT)](),
                self.a_hi.lt["gpu", Layout.row_major(ACT)](),
                Int32(1) if stepped else Int32(0),
                Int32(self.tick),
                self.lag_cfg.lt["gpu", Layout.row_major(_L_SIZE)](),
                self.alpha.lt["gpu", Layout.row_major(N * ACT)](),
                self.delay.lt["gpu", Layout.row_major(N * ACT)](),
                self.y.lt["gpu", Layout.row_major(N * ACT)](),
                self.ring.lt["gpu", Layout.row_major(N * LAG_MAX_DELAY * ACT)](),
                self.vcap.lt["gpu", Layout.row_major(N * ACT)](),
                self.off.lt["gpu", Layout.row_major(N * ACT)](),
                grid_dim=Self._g(N * ACT), block_dim=TPB,
            )
            if stepped:
                self.tick += 1

            def _env_step() capturing raises -> None:
                env.step_batch[N](
                    ctx=Optional(c),
                    rng_seed=UInt64(self.it * cfg.repeat + t + 1),
                )

            comptime if USE_ENV_GRAPH:
                maybe_capture_replay[_env_step](env_graph, c)
            else:
                _env_step()
            c.enqueue_function[_tick_k[N, EO]](
                _V[N * EO](env.obs_ptr()),
                _V[N](env.reward_ptr()),
                _V[N](env.done_ptr()),
                _V[N * METADATA_SIZE](meta_ptr),
                self.qa.lt["gpu", Layout.row_major(ACT)](),
                self.dmac.lt["gpu", Layout.row_major(N)](),
                self.diverged.lt["gpu", Layout.row_major(N)](),
                self.rsum.lt["gpu", Layout.row_major(N)](),
                self.succ.lt["gpu", Layout.row_major(N)](),
                self.term.lt["gpu", Layout.row_major(N * EO)](),
                self.stats.lt["gpu", Layout.row_major(N * _S_SIZE)](),
                Scalar[DT](cfg.dive_w),
                Scalar[DT](cfg.obs_bound),
                Scalar[DT](cfg.rew_bound),
                grid_dim=Self._g(N), block_dim=TPB,
            )
        # 3. the transition, the next observation's statistics
        c.enqueue_function[_post_k[N, EO]](
            _V[N * EO](env.obs_ptr()),
            self.term.lt["gpu", Layout.row_major(N * EO)](),
            self.raw_post.lt["gpu", Layout.row_major(N * EO)](),
            self.rsum.lt["gpu", Layout.row_major(N)](),
            self.spen.lt["gpu", Layout.row_major(N)](),
            self.dmac.lt["gpu", Layout.row_major(N)](),
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.done_out.lt["gpu", Layout.row_major(N)](),
            _V[N](env.done_ptr()),
            Scalar[DT](cfg.smooth_w),
            Int32(cfg.repeat),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        self._augment(mptr(self.raw_post.dev.value().unsafe_ptr()))
        _update_rms[Self.N, Self.OBS](
            c, self.aug, self.diverged, self.obs_mean, self.obs_var,
            self.obs_count, True,
        )
        _normalize[Self.N, Self.OBS](
            c, self.aug, self.next_n, self.obs_mean, self.obs_var,
            self.diverged, cfg.obs_clip, True,
        )
        # 4. reward normalisation and the episode records
        c.enqueue_function[_ret_k[N]](
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.ret_acc.lt["gpu", Layout.row_major(N)](),
            self.raw_ret.lt["gpu", Layout.row_major(N)](),
            self.rets.lt["gpu", Layout.row_major(N)](),
            Scalar[DT](cfg.gamma),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        _update_rms[Self.N, 1](
            c, self.rets, self.diverged, self.ret_mean, self.ret_var,
            self.ret_count, False,
        )
        c.enqueue_function[_rew_k[N]](
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.done_out.lt["gpu", Layout.row_major(N)](),
            self.ret_var.lt["gpu", Layout.row_major(1)](),
            self.rew_n.lt["gpu", Layout.row_major(N)](),
            self.succ.lt["gpu", Layout.row_major(N)](),
            self.raw_ret.lt["gpu", Layout.row_major(N)](),
            self.ret_acc.lt["gpu", Layout.row_major(N)](),
            self.ep.lt["gpu", Layout.row_major(N * 3)](),
            Scalar[DT](cfg.rew_clip),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        if self._pending >= self._sync_every:
            raise Error(
                "FamilyDeviceRollout: the episode ring is full — `drain` when"
                " `ring_full()`"
            )
        c.enqueue_copy(self._ring_bufs[self._pending], self.ep.dev.value())
        self._pending += 1
        # 5. record (normalised obs / reward; `done` is recorded, no true
        # terminal: the host loop marks none either)
        trainer.record_device(
            mptr(self.rew_n.dev.value().unsafe_ptr()),
            mptr(self.next_n.dev.value().unsafe_ptr()),
            mptr(self.done_out.dev.value().unsafe_ptr()),
            mptr(self.zeros.dev.value().unsafe_ptr()),
        )
        # 6. reset the finished lanes; the observation they restart from
        def _env_reset() capturing raises -> None:
            env.selective_reset_batch[N](
                ctx=Optional(c),
                rng_seed=self.seed * UInt64(7919) + UInt64(self.it + 1),
            )

        comptime if USE_ENV_GRAPH:
            maybe_capture_replay[_env_reset](reset_graph, c)
        else:
            _env_reset()
        self._reset_lanes(
            env.obs_ptr(), mptr(self.done_out.dev.value().unsafe_ptr()), False
        )
        self._augment(env.obs_ptr())
        _normalize[Self.N, Self.OBS](
            c, self.aug, self.cur_n, self.obs_mean, self.obs_var,
            self.diverged, cfg.obs_clip, False,
        )
        self.it += 1
