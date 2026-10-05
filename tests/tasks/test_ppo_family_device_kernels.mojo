"""The family PPO device rollout's kernels against the host code they twin.

`noeira/tasks/ppo_family_device.mojo` ports `run_ppo`'s per-step host loop
to kernels. The device rollout cannot be bit-identical to the host one (its
RNG streams differ, and it runs Float32 where the host keeps Float64), so
each kernel is held here against its HOST counterpart on identical inputs:

  targets    `_delta_targets` / `_target_targets` (lead off and on)
  servo      12 ticks of `_targets_to_env` from the host `ServoLag`'s own
             state (tau, delay, speed cap, elbow stop, offsets)
  reset      `ServoLag.reset_lane` with degenerate ranges (no draw left to
             differ), shared and per-joint, and the target / history resets
  rms        `RunningMeanStd.update` with a diverged mask, three batches,
             then `normalize_into` (the diverged rows zeroed)
  reward     the discounted-return statistics, `NormalizeReward`, the
             episode records
  tick/post  the tick loop and the transition of `run_ppo` (inline there, so
             transcribed here): NaN and out-of-bound lanes, a reward beyond
             the bound, the dive penalty, the goal bit, early ends with
             `--repeat 3`, the smoothing penalty
  augment    `_augment` with an action history and the target lead (the
             test build has neither define, so the reference is transcribed)
  smoothing  `--smooth-penalty` (transcribed): the arm words' change, no
             penalty on an episode's first step, the per-lane sums
  history    `_hist_push` (transcribed: a no-op in this build)
  exec noise `--exec-noise`: its RNG differs from the host's by design, so
             its DISTRIBUTION is checked — the unclipped part's mean and std

Tolerances are Float32 against Float64 (relative 1e-4 or absolute 1e-5).

    pixi run mojo run -I . tests/tasks/test_ppo_family_device_kernels.mojo
    pixi run -e apple mojo run -I . tests/tasks/test_ppo_family_device_kernels.mojo
"""

from layout import Layout
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.physics3d.gpu.constants import METADATA_SIZE, META_IDX_GOAL_HELD
from noeira.tasks.delta_action import LAG_MAX_DELAY, ServoLag
from noeira.tasks.ppo_family_driver import (
    N_ENVS, RunningMeanStd, _delta_targets, _target_targets, _targets_to_env,
)
from noeira.tasks.ppo_family_device import (
    ACT, _exec_noise_k, _smooth_k, _hist_push_k, _targets_k, _to_env_k, _reset_k, _tick_k, _post_k, _augment_k,
    _ret_k, _rew_k, _update_rms, _normalize, lag_cfg_words,
    _L_SIZE, _S_SIZE, _S_DIVERGED, _S_DIVE, _S_TICKS, _S_SPEN_SUM, _S_SPEN_N, _EP_DONE, _EP_SUCC,
    _EP_RET,
)


comptime N = N_ENVS
comptime E_OBS = 12
comptime G = (N + TPB - 1) // TPB
comptime GA = (N * ACT + TPB - 1) // TPB


def up(ctx: DeviceContext, ref v: List[Float64]) raises -> Tensor:
    var t = Tensor.alloc(len(v))
    for i in range(len(v)):
        t.data[i] = Scalar[DT](v[i])
    t.upload(ctx)
    return t^


def zeros(ctx: DeviceContext, n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    t.upload(ctx)
    return t^


def down(ctx: DeviceContext, mut t: Tensor) raises -> List[Float64]:
    t.download(ctx)
    var out = List[Float64](length=t.n, fill=0.0)
    for i in range(t.n):
        out[i] = Float64(t.data[i])
    return out^


def rnd(n: Int, lo: Float64, hi: Float64) -> List[Float64]:
    var v = List[Float64](length=n, fill=0.0)
    for i in range(n):
        v[i] = lo + (hi - lo) * random_float64()
    return v^


def check(
    name: String, ref got: List[Float64], ref want: List[Float64], n: Int,
    rel: Float64 = 1e-4, abs_tol: Float64 = 1e-5,
) raises:
    var worst = 0.0
    var at = -1
    for i in range(n):
        var g = got[i]
        var w = want[i]
        var both_nan = not (g == g) and not (w == w)
        if both_nan:
            continue
        var d = abs(g - w)
        var lim = abs_tol + rel * abs(w)
        if not (d <= lim):
            if not (d == d) or d - lim > worst:
                worst = d - lim if d == d else 1e30
                at = i
    if at >= 0:
        print("  FAIL", name, "at", at, ": got", got[at], "want", want[at])
    assert_true(at < 0, name + ": device and host disagree")
    print("  ok", name, "(", n, "words )")


def _lims() -> Tuple[List[Float64], List[Float64]]:
    var lo = List[Float64](length=ACT, fill=-1.9)
    var hi = List[Float64](length=ACT, fill=1.9)
    lo[ACT - 1] = -0.17
    hi[ACT - 1] = 1.74
    return (lo^, hi^)


def test_targets(ctx: DeviceContext) raises:
    print("targets")
    var lims = _lims()
    var lo = lims[0].copy()
    var hi = lims[1].copy()
    var arm_q = rnd(N * ACT, -1.7, 1.7)
    var tprev0 = rnd(N * ACT, -1.7, 1.7)
    var act = rnd(N * ACT, -1.3, 1.3)
    var act_s = List[Scalar[DT]](length=N * ACT, fill=Scalar[DT](0))
    # the host reads the action through DT, as the policy writes it
    for i in range(N * ACT):
        act_s[i] = Scalar[DT](act[i])
        act[i] = Float64(act_s[i])
    for ci in range(3):
        var mode = 1 if ci == 0 else 2
        var lead = 0.3 if ci == 2 else 0.0
        var tg_h = List[Float64](length=N * ACT, fill=0.0)
        var tprev_h = tprev0.copy()
        if mode == 1:
            _delta_targets(mptr(act_s.unsafe_ptr()), tg_h, arm_q, lo, hi, 0.1, 0.2)
        else:
            _target_targets(
                mptr(act_s.unsafe_ptr()), tg_h, tprev_h, arm_q, lo, hi, 0.1,
                0.2, lead,
            )
        var t_act = up(ctx, act)
        var t_q = up(ctx, arm_q)
        var t_tg = zeros(ctx, N * ACT)
        var t_tp = up(ctx, tprev0)
        var t_lo = up(ctx, lo)
        var t_hi = up(ctx, hi)
        ctx.enqueue_function[_targets_k[N]](
            t_act.lt["gpu", Layout.row_major(N * ACT)](),
            t_q.lt["gpu", Layout.row_major(N * ACT)](),
            t_tg.lt["gpu", Layout.row_major(N * ACT)](),
            t_tp.lt["gpu", Layout.row_major(N * ACT)](),
            t_lo.lt["gpu", Layout.row_major(ACT)](),
            t_hi.lt["gpu", Layout.row_major(ACT)](),
            Int32(mode), Scalar[DT](0.1), Scalar[DT](0.2), Scalar[DT](lead),
            grid_dim=GA, block_dim=TPB,
        )
        var tg_d = down(ctx, t_tg)
        var tp_d = down(ctx, t_tp)
        check("targets mode " + String(mode) + " lead " + String(lead), tg_d, tg_h, N * ACT)
        if mode == 2:
            check("previous targets", tp_d, tprev_h, N * ACT)


def _lag_state_to_device(
    ctx: DeviceContext, ref lag: ServoLag
) raises -> List[Tensor]:
    """alpha, delay, y, ring, vcap, off, cfg — the host lag's state."""
    var d = List[Float64](length=N * ACT, fill=0.0)
    for i in range(N * ACT):
        d[i] = Float64(lag.delay[i])
    var cfgw = lag_cfg_words(lag)
    var cfg = List[Float64](length=_L_SIZE, fill=0.0)
    for k in range(_L_SIZE):
        cfg[k] = Float64(cfgw[k])
    var out = List[Tensor]()
    out.append(up(ctx, lag.alpha))
    out.append(up(ctx, d))
    out.append(up(ctx, lag.y))
    out.append(up(ctx, lag.hist))
    out.append(up(ctx, lag.vcap))
    out.append(up(ctx, lag.off))
    out.append(up(ctx, cfg))
    return out^


def test_servo(ctx: DeviceContext) raises:
    print("servo")
    var lims = _lims()
    var lo = lims[0].copy()
    var hi = lims[1].copy()
    var lag = ServoLag.parse(N, "40,120", "0,3", 0.032)
    lag.set_limits("0.8,1.2", 1.56)
    lag.set_offset("0,0.02;0,0.02;0,0.03;0,0;0,0;0,0")
    lag.set_period("30,40")
    var arm_q = rnd(N * ACT, -1.5, 1.5)
    for e in range(N):
        lag.reset_lane(e, arm_q, e * ACT, random_float64(), random_float64(),
                       random_float64())
    var st = _lag_state_to_device(ctx, lag)
    var t_lo = up(ctx, lo)
    var t_hi = up(ctx, hi)
    var t_act = zeros(ctx, N * ACT)
    var t_env = zeros(ctx, N * ACT)
    for tick in range(12):
        var tg = rnd(N * ACT, -1.8, 1.8)
        var env_h = List[Scalar[DT]](length=N * ACT, fill=Scalar[DT](0))
        _targets_to_env(tg, mptr(env_h.unsafe_ptr()), lo, hi, lag)
        var t_tg = up(ctx, tg)
        ctx.enqueue_function[_to_env_k[N]](
            t_tg.lt["gpu", Layout.row_major(N * ACT)](),
            t_act.lt["gpu", Layout.row_major(N * ACT)](),
            t_env.lt["gpu", Layout.row_major(N * ACT)](),
            t_lo.lt["gpu", Layout.row_major(ACT)](),
            t_hi.lt["gpu", Layout.row_major(ACT)](),
            Int32(1), Int32(tick),
            st[6].lt["gpu", Layout.row_major(_L_SIZE)](),
            st[0].lt["gpu", Layout.row_major(N * ACT)](),
            st[1].lt["gpu", Layout.row_major(N * ACT)](),
            st[2].lt["gpu", Layout.row_major(N * ACT)](),
            st[3].lt["gpu", Layout.row_major(N * LAG_MAX_DELAY * ACT)](),
            st[4].lt["gpu", Layout.row_major(N * ACT)](),
            st[5].lt["gpu", Layout.row_major(N * ACT)](),
            grid_dim=GA, block_dim=TPB,
        )
        var env_d = down(ctx, t_env)
        var env_hf = List[Float64](length=N * ACT, fill=0.0)
        for i in range(N * ACT):
            env_hf[i] = Float64(env_h[i])
        check("servo tick " + String(tick) + " env action", env_d, env_hf, N * ACT, 1e-4, 1e-4)
        var y_d = down(ctx, st[2])
        check("servo tick " + String(tick) + " lagged target", y_d, lag.y, N * ACT, 1e-4, 1e-4)


def test_reset(ctx: DeviceContext) raises:
    """Degenerate ranges: every draw is the range's one value, so the
    device's Philox and the host RNG cannot differ."""
    print("reset")
    for per_joint in range(2):
        var lag = ServoLag.parse(N, "80,80", "2,2", 0.032)
        lag.set_limits("1.0,1.0", 1.5)
        lag.set_offset("0.01,0.01;0.02,0.02;0.03,0.03;0,0;0,0;0.005,0.005")
        lag.set_period("33,33")
        if per_joint == 1:
            lag.set_per_joint(
                "50,50;60,60;70,70;80,80;90,90;100,100",
                "1,1;2,2;3,3;1,1;2,2;0,0",
                "1.1,1.1;1.2,1.2;1.0,1.0;0.9,0.9;1.3,1.3;1.5,1.5",
            )
        var raw = rnd(N * E_OBS, -1.5, 1.5)
        var arm_q = List[Float64](length=N * ACT, fill=0.0)
        for e in range(N):
            for j in range(ACT):
                arm_q[e * ACT + j] = Float64(Scalar[DT](raw[e * E_OBS + j]))
            lag.reset_lane(e, arm_q, e * ACT, 0.5, 0.5, 0.5)
        var qa = List[Float64](length=ACT, fill=0.0)
        for j in range(ACT):
            qa[j] = Float64(j)
        var cfgw = lag_cfg_words(lag)
        var cfg = List[Float64](length=_L_SIZE, fill=0.0)
        for k in range(_L_SIZE):
            cfg[k] = Float64(cfgw[k])
        var t_raw = up(ctx, raw)
        var t_qa = up(ctx, qa)
        var t_done = zeros(ctx, N)
        var t_q = zeros(ctx, N * ACT)
        var t_tp = zeros(ctx, N * ACT)
        var t_hist = up(ctx, rnd(N * 12 + 1, -1.0, 1.0))
        var t_hp = up(ctx, rnd(N, 1.0, 1.0))
        var t_dm = up(ctx, rnd(N, 1.0, 1.0))
        var t_dv = up(ctx, rnd(N, 1.0, 1.0))
        var t_rs = up(ctx, rnd(N, 3.0, 3.0))
        var t_cfg = up(ctx, cfg)
        var t_al = zeros(ctx, N * ACT)
        var t_de = zeros(ctx, N * ACT)
        var t_y = zeros(ctx, N * ACT)
        var t_ring = zeros(ctx, N * LAG_MAX_DELAY * ACT)
        var t_vc = zeros(ctx, N * ACT)
        var t_off = zeros(ctx, N * ACT)
        ctx.enqueue_function[_reset_k[N, E_OBS, 12]](
            t_raw.lt["gpu", Layout.row_major(N * E_OBS)](),
            t_qa.lt["gpu", Layout.row_major(ACT)](),
            t_done.lt["gpu", Layout.row_major(N)](),
            Int32(1),
            t_q.lt["gpu", Layout.row_major(N * ACT)](),
            t_tp.lt["gpu", Layout.row_major(N * ACT)](),
            t_hist.lt["gpu", Layout.row_major(N * 12 + 1)](),
            t_hp.lt["gpu", Layout.row_major(N)](),
            t_dm.lt["gpu", Layout.row_major(N)](),
            t_dv.lt["gpu", Layout.row_major(N)](),
            t_rs.lt["gpu", Layout.row_major(N)](),
            t_cfg.lt["gpu", Layout.row_major(_L_SIZE)](),
            t_al.lt["gpu", Layout.row_major(N * ACT)](),
            t_de.lt["gpu", Layout.row_major(N * ACT)](),
            t_y.lt["gpu", Layout.row_major(N * ACT)](),
            t_ring.lt["gpu", Layout.row_major(N * LAG_MAX_DELAY * ACT)](),
            t_vc.lt["gpu", Layout.row_major(N * ACT)](),
            t_off.lt["gpu", Layout.row_major(N * ACT)](),
            UInt64(7), UInt64(0),
            grid_dim=G, block_dim=TPB,
        )
        var tag = " per-joint" if per_joint == 1 else " shared"
        var al = down(ctx, t_al)
        check("reset alpha" + tag, al, lag.alpha, N * ACT)
        var de = down(ctx, t_de)
        var dh = List[Float64](length=N * ACT, fill=0.0)
        for i in range(N * ACT):
            dh[i] = Float64(lag.delay[i])
        check("reset delay" + tag, de, dh, N * ACT)
        var vc = down(ctx, t_vc)
        check("reset speed cap" + tag, vc, lag.vcap, N * ACT)
        var off = down(ctx, t_off)
        check("reset offset" + tag, off, lag.off, N * ACT)
        var yy = down(ctx, t_y)
        check("reset lagged target" + tag, yy, lag.y, N * ACT)
        var ring = down(ctx, t_ring)
        check("reset command ring" + tag, ring, lag.hist, N * LAG_MAX_DELAY * ACT)
        var tp = down(ctx, t_tp)
        check("reset target = joints" + tag, tp, arm_q, N * ACT)
        var hist = down(ctx, t_hist)
        var hz = List[Float64](length=N * 12, fill=0.0)
        check("reset history cleared" + tag, hist, hz, N * 12)
        var hp = down(ctx, t_hp)
        var z = List[Float64](length=N, fill=0.0)
        check("reset no previous action" + tag, hp, z, N)
        var dm = down(ctx, t_dm)
        check("reset step accumulators" + tag, dm, z, N)


def test_rms(ctx: DeviceContext) raises:
    print("running statistics")
    comptime D = 9
    var host = RunningMeanStd(D)
    var t_mean = zeros(ctx, D)
    var v1 = rnd(D, 1.0, 1.0)
    var t_var = up(ctx, v1)
    var c0 = List[Float64](length=1, fill=1e-4)
    var t_cnt = up(ctx, c0)
    for b in range(3):
        var x = rnd(N * D, -3.0 + Float64(b), 2.0 + 2.0 * Float64(b))
        var xs = List[Scalar[DT]](length=N * D, fill=Scalar[DT](0))
        for i in range(N * D):
            xs[i] = Scalar[DT](x[i])
            x[i] = Float64(xs[i])
        var skip = List[Bool](length=N, fill=False)
        var sk = List[Float64](length=N, fill=0.0)
        for e in range(N):
            if random_float64() < 0.05:
                skip[e] = True
                sk[e] = 1.0
        host.update(mptr(xs.unsafe_ptr()), N, D, skip)
        var t_x = up(ctx, x)
        var t_sk = up(ctx, sk)
        _update_rms[N, D](ctx, t_x, t_sk, t_mean, t_var, t_cnt, True)
        var m = down(ctx, t_mean)
        var v = down(ctx, t_var)
        var c = down(ctx, t_cnt)
        check("rms mean batch " + String(b), m, host.mean, D)
        check("rms var batch " + String(b), v, host.var_, D)
        var hc = List[Float64](length=1, fill=host.count)
        check("rms count batch " + String(b), c, hc, 1)
        var out_h = List[Scalar[DT]](length=N * D, fill=Scalar[DT](0))
        host.normalize_into(mptr(xs.unsafe_ptr()), mptr(out_h.unsafe_ptr()), N, D, 10.0)
        var want = List[Float64](length=N * D, fill=0.0)
        for e in range(N):
            for k in range(D):
                want[e * D + k] = 0.0 if skip[e] else Float64(out_h[e * D + k])
        var t_o = zeros(ctx, N * D)
        _normalize[N, D](ctx, t_x, t_o, t_mean, t_var, t_sk, 10.0, True)
        var o = down(ctx, t_o)
        check("normalize batch " + String(b), o, want, N * D, 1e-4, 1e-4)


def test_reward(ctx: DeviceContext) raises:
    print("reward normalisation and episode records")
    var host = RunningMeanStd(1)
    var ret_acc = rnd(N, -1.0, 1.0)
    var raw_ret = rnd(N, -3.0, 3.0)
    var succ = List[Float64](length=N, fill=0.0)
    for e in range(N):
        succ[e] = 1.0 if random_float64() < 0.3 else 0.0
    var t_acc = up(ctx, ret_acc)
    var t_raw = up(ctx, raw_ret)
    var t_succ = up(ctx, succ)
    var t_rm = zeros(ctx, 1)
    var t_rv = up(ctx, rnd(1, 1.0, 1.0))
    var t_rc = up(ctx, rnd(1, 1e-4, 1e-4))
    var t_skip = zeros(ctx, N)
    for b in range(3):
        var r = rnd(N, -2.0, 4.0)
        var dn = List[Float64](length=N, fill=0.0)
        for e in range(N):
            r[e] = Float64(Scalar[DT](r[e]))
            dn[e] = 1.0 if random_float64() < 0.1 else 0.0
        # host
        var rets = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
        var rew_n = List[Float64](length=N, fill=0.0)
        var ep_d = List[Float64](length=N, fill=0.0)
        var ep_s = List[Float64](length=N, fill=0.0)
        var ep_r = List[Float64](length=N, fill=0.0)
        for e in range(N):
            ret_acc[e] = ret_acc[e] * 0.99 + r[e]
            rets[e] = Scalar[DT](ret_acc[e])
            raw_ret[e] += r[e]
        host.update(mptr(rets.unsafe_ptr()), N, 1)
        var rscale = 1.0 / sqrt(host.var_[0] + 1e-8)
        for e in range(N):
            var v = r[e] * rscale
            v = 10.0 if v > 10.0 else (-10.0 if v < -10.0 else v)
            rew_n[e] = v
            if dn[e] > 0.5:
                ep_d[e] = 1.0
                ep_s[e] = succ[e]
                ep_r[e] = raw_ret[e]
                succ[e] = 0.0
                raw_ret[e] = 0.0
                ret_acc[e] = 0.0
        # device
        var t_r = up(ctx, r)
        var t_dn = up(ctx, dn)
        var t_rets = zeros(ctx, N)
        var t_rn = zeros(ctx, N)
        var t_ep = zeros(ctx, N * 3)
        ctx.enqueue_function[_ret_k[N]](
            t_r.lt["gpu", Layout.row_major(N)](),
            t_acc.lt["gpu", Layout.row_major(N)](),
            t_raw.lt["gpu", Layout.row_major(N)](),
            t_rets.lt["gpu", Layout.row_major(N)](),
            Scalar[DT](0.99),
            grid_dim=G, block_dim=TPB,
        )
        _update_rms[N, 1](ctx, t_rets, t_skip, t_rm, t_rv, t_rc, False)
        ctx.enqueue_function[_rew_k[N]](
            t_r.lt["gpu", Layout.row_major(N)](),
            t_dn.lt["gpu", Layout.row_major(N)](),
            t_rv.lt["gpu", Layout.row_major(1)](),
            t_rn.lt["gpu", Layout.row_major(N)](),
            t_succ.lt["gpu", Layout.row_major(N)](),
            t_raw.lt["gpu", Layout.row_major(N)](),
            t_acc.lt["gpu", Layout.row_major(N)](),
            t_ep.lt["gpu", Layout.row_major(N * 3)](),
            Scalar[DT](10.0),
            grid_dim=G, block_dim=TPB,
        )
        var rn = down(ctx, t_rn)
        check("normalised reward batch " + String(b), rn, rew_n, N, 2e-4, 1e-4)
        var ep = down(ctx, t_ep)
        var g_d = List[Float64](length=N, fill=0.0)
        var g_s = List[Float64](length=N, fill=0.0)
        var g_r = List[Float64](length=N, fill=0.0)
        for e in range(N):
            g_d[e] = ep[e * 3 + _EP_DONE]
            g_s[e] = ep[e * 3 + _EP_SUCC] if g_d[e] > 0.5 else 0.0
            g_r[e] = ep[e * 3 + _EP_RET] if g_d[e] > 0.5 else 0.0
        check("episode done batch " + String(b), g_d, ep_d, N)
        check("episode success batch " + String(b), g_s, ep_s, N)
        check("episode return batch " + String(b), g_r, ep_r, N, 1e-4, 1e-4)


def test_tick_post(ctx: DeviceContext) raises:
    """Three ticks (`--repeat 3`) of `run_ppo`'s lane walk, then the
    transition — the reference transcribed from `run_ppo`."""
    print("ticks and transition")
    comptime REP = 3
    var qa = List[Float64](length=ACT, fill=0.0)
    for j in range(ACT):
        qa[j] = Float64(j)
    var dive_w = 0.5
    var smooth_w = 0.2
    var spen = rnd(N, 0.0, 0.3)
    # reference state
    var rsum = List[Float64](length=N, fill=0.0)
    var dmac = List[Bool](length=N, fill=False)
    var div = List[Bool](length=N, fill=False)
    var succ = List[Bool](length=N, fill=False)
    var term = List[Float64](length=N * E_OBS, fill=0.0)
    var n_div = List[Float64](length=N, fill=0.0)
    var n_dive = List[Float64](length=N, fill=0.0)
    var n_tick = List[Float64](length=N, fill=0.0)
    # device state
    var t_qa = up(ctx, qa)
    var t_dm = zeros(ctx, N)
    var t_dv = zeros(ctx, N)
    var t_rs = zeros(ctx, N)
    var t_su = zeros(ctx, N)
    var t_term = zeros(ctx, N * E_OBS)
    var t_stats = zeros(ctx, N * _S_SIZE)
    var raw = List[Float64]()
    for tick in range(REP):
        raw = rnd(N * E_OBS, -1.0, 1.0)
        var rew = rnd(N, -0.5, 1.0)
        var done = List[Float64](length=N, fill=0.0)
        var meta = List[Float64](length=N * METADATA_SIZE, fill=0.0)
        for e in range(N):
            var u = random_float64()
            if u < 0.02:
                raw[e * E_OBS + 7] = 0.0 / 0.0  # NaN
            elif u < 0.04:
                raw[e * E_OBS + 3] = 5.0e3  # beyond OBS_BOUND
            elif u < 0.05:
                rew[e] = 2.0e3  # beyond REW_BOUND
            if random_float64() < 0.15:
                raw[e * E_OBS + 1] = 1.5  # shoulder lift
                raw[e * E_OBS + 2] = -1.5  # elbow: a dive
            if random_float64() < 0.1:
                done[e] = 1.0
            if random_float64() < 0.2:
                meta[e * METADATA_SIZE + META_IDX_GOAL_HELD] = 1.0
        for i in range(N * E_OBS):
            raw[i] = Float64(Scalar[DT](raw[i])) if raw[i] == raw[i] else raw[i]
        # reference: the tick's lane walk
        for e in range(N):
            if dmac[e]:
                continue
            var bad = False
            var rv = rew[e]
            if not (rv == rv) or abs(rv) > 1.0e3:
                bad = True
            for k in range(E_OBS):
                var v = raw[e * E_OBS + k]
                if not (v == v) or abs(v) > 1.0e3:
                    bad = True
                    break
            if bad:
                div[e] = True
                n_div[e] += 1.0
                dmac[e] = True
            else:
                rsum[e] += rv
                n_tick[e] += 1.0
                if raw[e * E_OBS + 1] > 1.35 and raw[e * E_OBS + 2] < -1.35:
                    rsum[e] -= dive_w
                    n_dive[e] += 1.0
                if meta[e * METADATA_SIZE + META_IDX_GOAL_HELD] > 0.5:
                    succ[e] = True
                if done[e] > 0.5:
                    dmac[e] = True
            if dmac[e]:
                for k in range(E_OBS):
                    term[e * E_OBS + k] = raw[e * E_OBS + k]
        # device
        var t_raw = up(ctx, raw)
        var t_rew = up(ctx, rew)
        var t_done = up(ctx, done)
        var t_meta = up(ctx, meta)
        ctx.enqueue_function[_tick_k[N, E_OBS]](
            t_raw.lt["gpu", Layout.row_major(N * E_OBS)](),
            t_rew.lt["gpu", Layout.row_major(N)](),
            t_done.lt["gpu", Layout.row_major(N)](),
            t_meta.lt["gpu", Layout.row_major(N * METADATA_SIZE)](),
            t_qa.lt["gpu", Layout.row_major(ACT)](),
            t_dm.lt["gpu", Layout.row_major(N)](),
            t_dv.lt["gpu", Layout.row_major(N)](),
            t_rs.lt["gpu", Layout.row_major(N)](),
            t_su.lt["gpu", Layout.row_major(N)](),
            t_term.lt["gpu", Layout.row_major(N * E_OBS)](),
            t_stats.lt["gpu", Layout.row_major(N * _S_SIZE)](),
            Scalar[DT](dive_w), Scalar[DT](1.0e3), Scalar[DT](1.0e3),
            grid_dim=G, block_dim=TPB,
        )
    # the transition (reference)
    var r_out = List[Float64](length=N, fill=0.0)
    var d_out = List[Float64](length=N, fill=0.0)
    var raw_post = raw.copy()
    for e in range(N):
        r_out[e] = rsum[e] - spen[e]
        d_out[e] = 1.0 if dmac[e] else 0.0
        if dmac[e]:
            for k in range(E_OBS):
                raw_post[e * E_OBS + k] = term[e * E_OBS + k]
    var t_raw = up(ctx, raw)
    var t_sp = up(ctx, spen)
    var t_post = zeros(ctx, N * E_OBS)
    var t_ro = zeros(ctx, N)
    var t_do = zeros(ctx, N)
    var t_envd = zeros(ctx, N)
    ctx.enqueue_function[_post_k[N, E_OBS]](
        t_raw.lt["gpu", Layout.row_major(N * E_OBS)](),
        t_term.lt["gpu", Layout.row_major(N * E_OBS)](),
        t_post.lt["gpu", Layout.row_major(N * E_OBS)](),
        t_rs.lt["gpu", Layout.row_major(N)](),
        t_sp.lt["gpu", Layout.row_major(N)](),
        t_dm.lt["gpu", Layout.row_major(N)](),
        t_ro.lt["gpu", Layout.row_major(N)](),
        t_do.lt["gpu", Layout.row_major(N)](),
        t_envd.lt["gpu", Layout.row_major(N)](),
        Scalar[DT](smooth_w), Int32(REP),
        grid_dim=G, block_dim=TPB,
    )
    var dv = down(ctx, t_dv)
    var dvh = List[Float64](length=N, fill=0.0)
    var su = down(ctx, t_su)
    var suh = List[Float64](length=N, fill=0.0)
    var n_div_lanes = 0
    var n_end = 0
    for e in range(N):
        dvh[e] = 1.0 if div[e] else 0.0
        suh[e] = 1.0 if succ[e] else 0.0
        if div[e]:
            n_div_lanes += 1
        if dmac[e]:
            n_end += 1
    print("  (", n_div_lanes, "diverged lanes,", n_end, "ended )")
    assert_true(n_div_lanes > 0 and n_end > n_div_lanes, "the fixture exercises nothing")
    check("diverged", dv, dvh, N)
    check("success", su, suh, N)
    var ro = down(ctx, t_ro)
    check("step reward", ro, r_out, N, 1e-4, 1e-4)
    var do = down(ctx, t_do)
    check("step done", do, d_out, N)
    var envd = down(ctx, t_envd)
    check("done written back to the env", envd, d_out, N)
    var post = down(ctx, t_post)
    check("transition observation (terminal rows)", post, raw_post, N * E_OBS)
    var st = down(ctx, t_stats)
    var g_div = List[Float64](length=N, fill=0.0)
    var g_dive = List[Float64](length=N, fill=0.0)
    var g_tick = List[Float64](length=N, fill=0.0)
    for e in range(N):
        g_div[e] = st[e * _S_SIZE + _S_DIVERGED]
        g_dive[e] = st[e * _S_SIZE + _S_DIVE]
        g_tick[e] = st[e * _S_SIZE + _S_TICKS]
    check("diverged counter", g_div, n_div, N)
    check("dive counter", g_dive, n_dive, N)
    check("tick counter", g_tick, n_tick, N)


def test_augment(ctx: DeviceContext) raises:
    print("augment")
    comptime W = 2 * ACT
    comptime T = ACT
    comptime A = E_OBS + W + T
    var raw = rnd(N * E_OBS, -1.0, 1.0)
    var hist = rnd(N * W + 1, -1.0, 1.0)
    var tprev = rnd(N * ACT, -1.5, 1.5)
    var qa = List[Float64](length=ACT, fill=0.0)
    for j in range(ACT):
        qa[j] = Float64(ACT - 1 - j)
    var want = List[Float64](length=N * A, fill=0.0)
    for e in range(N):
        for k in range(E_OBS):
            want[e * A + k] = raw[e * E_OBS + k]
        for k in range(W):
            want[e * A + E_OBS + k] = hist[e * W + k]
        for j in range(ACT):
            want[e * A + E_OBS + W + j] = tprev[e * ACT + j] - raw[e * E_OBS + Int(qa[j])]
    var t_raw = up(ctx, raw)
    var t_hist = up(ctx, hist)
    var t_tp = up(ctx, tprev)
    var t_qa = up(ctx, qa)
    var t_aug = zeros(ctx, N * A)
    ctx.enqueue_function[_augment_k[N, E_OBS, W, T]](
        t_raw.lt["gpu", Layout.row_major(N * E_OBS)](),
        t_hist.lt["gpu", Layout.row_major(N * W + 1)](),
        t_tp.lt["gpu", Layout.row_major(N * ACT)](),
        t_qa.lt["gpu", Layout.row_major(ACT)](),
        t_aug.lt["gpu", Layout.row_major(N * A)](),
        grid_dim=G, block_dim=TPB,
    )
    var got = down(ctx, t_aug)
    check("augmented observation (history + target lead)", got, want, N * A, 1e-4, 1e-5)


def test_smooth(ctx: DeviceContext) raises:
    print("smoothing penalty")
    var w = 0.3
    var act = rnd(N * ACT, -1.4, 1.4)
    for i in range(N * ACT):
        act[i] = Float64(Scalar[DT](act[i]))
    var a_prev = rnd(N * ACT, -1.0, 1.0)
    var has_prev = List[Float64](length=N, fill=0.0)
    for e in range(N):
        has_prev[e] = 1.0 if random_float64() < 0.7 else 0.0
    var want_sp = List[Float64](length=N, fill=0.0)
    var want_prev = a_prev.copy()
    for e in range(N):
        var d2 = 0.0
        for j in range(ACT - 1):
            var v = act[e * ACT + j]
            v = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
            if has_prev[e] > 0.5:
                var dd = v - a_prev[e * ACT + j]
                d2 += dd * dd
            want_prev[e * ACT + j] = v
        want_sp[e] = w * d2 / Float64(ACT - 1)
    var t_act = up(ctx, act)
    var t_prev = up(ctx, a_prev)
    var t_hp = up(ctx, has_prev)
    var t_sp = zeros(ctx, N)
    var t_st = zeros(ctx, N * _S_SIZE)
    ctx.enqueue_function[_smooth_k[N]](
        t_act.lt["gpu", Layout.row_major(N * ACT)](),
        t_prev.lt["gpu", Layout.row_major(N * ACT)](),
        t_hp.lt["gpu", Layout.row_major(N)](),
        t_sp.lt["gpu", Layout.row_major(N)](),
        t_st.lt["gpu", Layout.row_major(N * _S_SIZE)](),
        Scalar[DT](w),
        grid_dim=G, block_dim=TPB,
    )
    var sp = down(ctx, t_sp)
    check("smoothing penalty", sp, want_sp, N, 1e-4, 1e-6)
    var pv = down(ctx, t_prev)
    check("previous executed action", pv, want_prev, N * ACT)
    var hp = down(ctx, t_hp)
    var ones = List[Float64](length=N, fill=1.0)
    check("has a previous action", hp, ones, N)
    var st = down(ctx, t_st)
    var g_sum = List[Float64](length=N, fill=0.0)
    var g_n = List[Float64](length=N, fill=0.0)
    for e in range(N):
        g_sum[e] = st[e * _S_SIZE + _S_SPEN_SUM]
        g_n[e] = st[e * _S_SIZE + _S_SPEN_N]
    check("smoothing sums", g_sum, want_sp, N, 1e-4, 1e-6)
    check("smoothing counts", g_n, ones, N)


def test_hist_push(ctx: DeviceContext) raises:
    print("action history")
    comptime W = 3 * ACT
    var hist = rnd(N * W, -1.0, 1.0)
    var act = rnd(N * ACT, -1.4, 1.4)
    var want = hist.copy()
    for e in range(N):
        for k in range(W - 1, ACT - 1, -1):
            want[e * W + k] = hist[e * W + k - ACT]
        for j in range(ACT):
            var v = act[e * ACT + j]
            want[e * W + j] = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
    var t_h = up(ctx, hist)
    var t_a = up(ctx, act)
    ctx.enqueue_function[_hist_push_k[N, W]](
        t_h.lt["gpu", Layout.row_major(N * W)](),
        t_a.lt["gpu", Layout.row_major(N * ACT)](),
        grid_dim=G, block_dim=TPB,
    )
    var got = down(ctx, t_h)
    check("history after a push", got, want, N * W)


def test_exec_noise(ctx: DeviceContext) raises:
    """Zero actions + sigma 0.2: no word reaches the clip, so the result IS
    the noise — mean ~0, std ~0.2 over 6144 draws; and two offsets draw two
    different streams."""
    print("exec noise")
    var sigma = 0.2
    var t_a = zeros(ctx, N * ACT)
    ctx.enqueue_function[_exec_noise_k[N]](
        t_a.lt["gpu", Layout.row_major(N * ACT)](),
        Scalar[DT](sigma), UInt64(99), UInt64(0),
        grid_dim=GA, block_dim=TPB,
    )
    var a = down(ctx, t_a)
    var m = 0.0
    for i in range(N * ACT):
        m += a[i]
    m /= Float64(N * ACT)
    var v = 0.0
    for i in range(N * ACT):
        v += (a[i] - m) * (a[i] - m)
    var sd = sqrt(v / Float64(N * ACT))
    print("  noise mean", m, "std", sd)
    assert_true(abs(m) < 0.01, "exec noise mean is off")
    assert_true(abs(sd - sigma) < 0.01, "exec noise std is off")
    var t_b = zeros(ctx, N * ACT)
    ctx.enqueue_function[_exec_noise_k[N]](
        t_b.lt["gpu", Layout.row_major(N * ACT)](),
        Scalar[DT](sigma), UInt64(99), UInt64(1),
        grid_dim=GA, block_dim=TPB,
    )
    var b = down(ctx, t_b)
    var same = 0
    for i in range(N * ACT):
        if a[i] == b[i]:
            same += 1
    assert_true(same < 10, "two steps drew the same exec noise")
    print("  ok exec noise distribution, and a fresh draw per step")


def main() raises:
    seed(1234)
    var ctx = DeviceContext()
    print("--- family PPO device rollout: kernels vs host, N =", N, "---")
    test_targets(ctx)
    test_servo(ctx)
    test_reset(ctx)
    test_rms(ctx)
    test_reward(ctx)
    test_tick_post(ctx)
    test_augment(ctx)
    test_smooth(ctx)
    test_hist_push(ctx)
    test_exec_noise(ctx)
    print("ALL PASSED")
