"""Probe a trained FB-MEBE G1 checkpoint — `noeira-docs/FB_MEBE_G1_PLAN.md`
§12.3's open diagnostic: why do the velocity tasks stay upright but score
~13/250 when stand scores 177?

The checkpoint holds the nets and the normaliser, not the ring, so a fresh
ring is collected first: `--collect` batched steps of the agent's own
exploration (MEBE pool refreshed as in training, NO updates, so the nets
and the normaliser stay the checkpoint's). Then, for five reward pools:

    ring          32 768 uniform ring rows, centred rewards (= the training eval)
    upright       the upright rows (g_z ≤ −0.93) of 65 536 uniform ring rows
    upright_wide  the same rows, reward r^¼ (every σ × 2) for the z only
    calm          the upright rows with |ωz| < 0.3 (§13.2)
    calm_sharp    the same rows, reward r⁴: only the rows that match count

it prints, per task:

    POOL  how many pool rows the reward sees (r > 0.5, r > 0.1, mean r) and
          the reward-weighted mean of (vx, vy, ωz, h) — what the z is asked for
    EVAL  return / fall / the mean (vx, vy, ωz, h) actually held over the
          250 eval steps, |vx − vx*|, |vy − vy*|, |ωz − ωz*| (velocity tasks),
          the return the SAME rollout earns on vx0 (standing still), and the
          mean vx and |vx − vx*| over the last 125 steps (the held speed,
          once the start from the reset stand is over)

and, with `--dump`, writes the `ring` eval's qpos of the first lane of every
task, every step, as text (`t lane q0 … q35`) for an offline render.
With `--dump-pool`, it writes the 65 536 sampled ring rows' (vx, vy, ωz,
h, g_z) as text, for offline joint histograms of what the ring holds.

`--prior FILE` attaches a seeded arm's prior rows as the driver does
(same `PriorPrice`), so the collection's MEBE pool draws them at the
batch's share. `--dump-zpool FILE` then writes the TRAINING EVAL's reward
pool (32 768 rows, each a prior row w.p. `PRIOR_FRAC`, drawn as `_eval`
draws them) as a `TrajectoryStore`, one episode, columns `obs` (the raw
527-wide row) and `b` (its `B(s)`, 64). A laptop demo can then build
`z = E_pool[r · B]` for any command with no ring and no GPU.

    pixi run -e nvidia mojo build -I . examples/fb/fb_mebe_g1_probe_gpu.mojo -o fb_mebe_g1_probe
    ./fb_mebe_g1_probe --ckpt <run>/checkpoints/step_19988480.ckpt --reg 1 --dump qpos.txt

`--reg` must match the run (it loads the `.reg` sidecar); `--explore`,
`--phi`, `--wz-max`, `--upright-gz`, `--vy-max` and `--h-min` too (they set the MEBE pool the collection refreshes).
"""

from std.math import sqrt
from std.random import seed
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from noeira.core.tiles import tile
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.deep_agents.fb import z_from_reward
from noeira.deep_agents.fb.bfm_towers import BFMFTower, BFMActorTower
from noeira.deep_agents.fb.kernels import _blocks
from noeira.deep_agents.fb.density import InverseDensityConfig, SplitMix64
from noeira.deep_agents.fb.online_mebe import (
    FBMebeOnlineAgent,
    MebeConfig,
    RegRewardConfig,
)
from noeira.deep_agents.fb.g1_mebe import (
    G1BNetProj,
    G1M_CMD_P,
    G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H, G1M_COL_GX, G1M_COL_GY,
    G1M_COL_GZ, G1M_COL_QD,
    g1_cmd_suite,
    g1_cmd_reward,
    g1_cmd_reward_obs,
    g1_cmd_step_kernel,
    g1_fall_kernel,
    g1_reset_noise_kernel,
)
from noeira.deep_agents.fb.prior import PriorPrice, load_prior_rows
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.envs.robots import UnitreeG1Batched
from noeira.envs.robots.unitree_g1_xml import UNITREE_G1_OBS_DIM


comptime N_ENVS: Int = 1024
comptime OBS: Int = UNITREE_G1_OBS_DIM   # 527
comptime ACT: Int = 29
comptime D: Int = 64
comptime H: Int = 1024
comptime L: Int = 3
comptime LA: Int = 2
comptime HB: Int = 512
comptime BATCH: Int = 1024
comptime CAP: Int = 524_288   # ≥ the default --collect's 409 600 rows (still
                              # OOMs beside two G1 training runs on a 5090)
comptime ZBUF: Int = 8192
comptime PRIOR_ROWS: Int = get_defined_int["PRIOR_ROWS", 256]()
comptime EnvT = UnitreeG1Batched[N_ENVS]
comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = G1BNetProj[OBS, D, HB]
comptime ANet = BFMActorTower[OBS, D, H, LA, ACT]
comptime QNet = BFMFTower[OBS, ACT, D, H, L, 1]
comptime Agent = FBMebeOnlineAgent[
    FNet, BNet, ANet, QNet, OBS, ACT, D, BATCH, CAP, N_ENVS, ZBUF, 0, DT,
    PRIOR_ROWS,
]

comptime CTRL_DT: Float64 = 0.02
comptime FALL_H: Float64 = 0.45
comptime FALL_GZ: Float64 = -0.5
comptime FALL_DELAY: Int = 25
comptime EVAL_LEN: Int = 250
comptime EVAL_POOL: Int = 32_768
comptime EVAL_FALL_H: Float64 = 0.5
comptime UPRIGHT_GZ: Float64 = -0.93
comptime CALM_WZ: Float64 = 0.3


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _full_reset(
    mut env: EnvT, ctx: DeviceContext, seed_v: UInt64, noise: Float64
) raises:
    env._done.enqueue_fill(Scalar[DT](1.0))
    env.selective_reset_batch[N_ENVS](Optional(ctx), seed_v)
    _after_reset(env, ctx, seed_v, noise)


def _after_reset(
    mut env: EnvT, ctx: DeviceContext, seed_v: UInt64, noise: Float64
) raises:
    if noise > 0.0:
        ctx.enqueue_function[g1_reset_noise_kernel[EnvT.NQ, N_ENVS]](
            mptr(env.d.qpos.dev.value().unsafe_ptr()),
            mptr(env._reset_mask.unsafe_ptr()),
            Scalar[DT](noise), seed_v,
            grid_dim=_blocks(N_ENVS), block_dim=TPB,
        )
    env._run_fields_fk(ctx)
    env._run_fields_vel(ctx)
    env._extract_obs_only(ctx)


def _fmt(x: Float64) -> String:
    """3 decimals, for a table the eye can scan."""
    var s = Float64(Int(abs(x) * 1000.0 + 0.5)) / 1000.0
    var t = String(s)
    return ("-" if x < 0.0 and s > 0.0 else "") + t


def main() raises:
    var ckpt = _flag(String("--ckpt"), String(""))
    if ckpt.byte_length() == 0:
        raise Error("--ckpt <path> is required")
    var explore = _flag(String("--explore"), String("mebe"))
    var beta = atof(_flag(String("--beta"), String(2.0)))
    var reg = atof(_flag(String("--reg"), String(0.0)))
    var n_collect = atol(_flag(String("--collect"), String(400)))
    var reset_noise = atof(_flag(String("--reset-noise"), String(0.1)))
    var seed_v = atol(_flag(String("--seed"), String(1)))
    var dump = _flag(String("--dump"), String(""))
    var dump_pool = _flag(String("--dump-pool"), String(""))
    var prior_path = _flag(String("--prior"), String(""))
    var dump_zpool = _flag(String("--dump-zpool"), String(""))
    var w_yaw = atof(_flag(String("--w-yaw"), String(0.0)))
    var w_angxy = atof(_flag(String("--w-angxy"), String(0.0)))
    var phi = _flag(String("--phi"), String("vxvywzh"))
    var wz_max = atof(_flag(String("--wz-max"), String(0.0)))
    var upright_gz = atof(_flag(String("--upright-gz"), String(-0.93)))
    var vy_max = atof(_flag(String("--vy-max"), String(1.3)))
    var h_min = atof(_flag(String("--h-min"), String(0.0)))
    var mebe = explore == "mebe"
    var use_reg = reg > 0.0
    if phi != "vxvywzh" and phi != "vxvyh" and phi != "vxh" and phi != "vx":
        raise Error("--phi must be 'vxvywzh', 'vxvyh', 'vxh' or 'vx'")

    var phi_cols: List[Int] = [G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H]
    if phi == "vxvyh":
        phi_cols = [G1M_COL_VX, G1M_COL_VY, G1M_COL_H]
    elif phi == "vxh":
        phi_cols = [G1M_COL_VX, G1M_COL_H]
    elif phi == "vx":
        phi_cols = [G1M_COL_VX]
    var cfg = MebeConfig(phi_cols^)
    cfg.add_filter(G1M_COL_GZ, -1.01, upright_gz)
    cfg.add_filter(G1M_COL_VX, -1.8, 1.8)
    cfg.add_filter(G1M_COL_VY, -vy_max, vy_max)
    if h_min > 0.0:
        cfg.add_filter(G1M_COL_H, h_min, 2.0)
    if wz_max > 0.0:
        cfg.add_filter(G1M_COL_WZ, -wz_max, wz_max)
    cfg.density = InverseDensityConfig(beta=beta, eps_rel=0.1, bw_scale=1.0)
    cfg.n_cand = 8192
    cfg.n_ref = 2048
    cfg.refresh_every = 25
    var regc = RegRewardConfig(
        qd_off=G1M_COL_QD, qd_n=ACT, dt=CTRL_DT, w_acc=2.5e-7, w_ar=0.1, w_env=1.0,
    )
    var suite = g1_cmd_suite()
    var n_task = len(suite)
    var reps = N_ENVS // n_task
    var k_vx0 = -1
    for k in range(n_task):
        if suite[k].name == "vx0.0":
            k_vx0 = k
    if k_vx0 < 0:
        raise Error("the suite has no 'vx0.0' task")

    seed(seed_v)
    print("=" * 70)
    print("FB-MEBE G1 probe —", ckpt)
    print("  explore", explore, " β", beta, " reg", reg, " collect", n_collect,
          "batched steps (", n_collect * N_ENVS, "rows )  seed", seed_v)
    print("  φ", phi, " |ωz| box", wz_max, "(0 = off)")
    print("=" * 70)

    with DeviceContext() as ctx:
        var agent = Agent.make(
            ctx, cfg^, regc,
            mebe=mebe, use_reg=use_reg, reg_coeff=reg, reg_raw=True,
            lr=3e-4, lr_b=1e-5, lr_q=3e-4, ortho_weight=100.0,
            max_grad_norm=0.0, act_l2_weight=0.0, act_l2_margin=0.8,
            learning_starts=BATCH, action_scale=1.0, expl_std=0.05,
            z_hold=100, explore_frac=0.8 if mebe else 0.0, keep_frac=0.0,
            window_size=100, seed=UInt64(seed_v), normalize_obs=True,
            episode_len=0,
        )
        agent.load_state(ckpt)
        if prior_path.byte_length() > 0:
            # the driver's attach (`fb_mebe_g1_gpu.mojo`), same price
            var price = PriorPrice.none()
            if use_reg and (w_yaw > 0.0 or w_angxy > 0.0):
                price = PriorPrice(G1M_COL_WZ - 2, w_yaw, w_angxy)
            var pr = load_prior_rows(prior_path, OBS, ACT, regc, price)
            var nxt = pr.upload(ctx)
            var nv = pr.n_valid
            agent.attach_prior(pr.take_obs(), pr.take_act(), nxt^, pr.take_rew(), nv)
            print("  [prior]", nv, "transitions,", pr.n_term, "episodes attached")
        var env = EnvT(ctx)
        var eval_env = EnvT(ctx)

        # ── 1. a fresh ring from the agent's own exploration, no updates ──
        var prev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
        var ao = ctx.enqueue_create_buffer[DT](N_ENVS * 2 * ACT)
        var alp = ctx.enqueue_create_buffer[DT](N_ENVS * (ACT + 1))
        var r_env = ctx.enqueue_create_buffer[DT](N_ENVS)
        var done_rec = ctx.enqueue_create_buffer[DT](N_ENVS)
        var fall_cnt = ctx.enqueue_create_buffer[DT](N_ENVS)
        var n_falls = ctx.enqueue_create_buffer[DT](N_ENVS)
        r_env.enqueue_fill(Scalar[DT](0.0))
        fall_cnt.enqueue_fill(Scalar[DT](0.0))
        n_falls.enqueue_fill(Scalar[DT](0.0))
        env.reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v))
        _full_reset(env, ctx, UInt64(seed_v) * 1000003, reset_noise)
        eval_env.reset_batch[N_ENVS](Optional(ctx), UInt64(1))
        var t0 = perf_counter_ns()
        for s in range(n_collect):
            ctx.enqueue_copy(prev_obs, env._obs)
            agent.select_action_batched[N_ENVS](
                tile[N_ENVS, OBS](env._obs),
                tile[N_ENVS, ACT](env._action),
                tile[N_ENVS, 2 * ACT](ao),
                tile[N_ENVS, ACT + 1](alp),
                s * N_ENVS,
            )
            env.step_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            ctx.enqueue_function[g1_fall_kernel[OBS, N_ENVS]](
                mptr(env._obs.unsafe_ptr()), mptr(env._done.unsafe_ptr()),
                mptr(fall_cnt.unsafe_ptr()), mptr(done_rec.unsafe_ptr()),
                mptr(n_falls.unsafe_ptr()),
                Scalar[DT](FALL_H), Scalar[DT](FALL_GZ), Int32(FALL_DELAY),
                grid_dim=_blocks(N_ENVS), block_dim=TPB,
            )
            agent.record_batch_gpu[N_ENVS](
                ctx, prev_obs, env._action, r_env, env._obs, done_rec
            )
            env.selective_reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            _after_reset(env, ctx, UInt64(seed_v) * 1000003 + UInt64(s + 1), reset_noise)
        ctx.synchronize()
        var h_falls = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        ctx.enqueue_copy(h_falls, n_falls)
        ctx.synchronize()
        var nf = 0.0
        for l in range(N_ENVS):
            nf += Float64(h_falls[l])
        print("[collect]", agent.replay_size(), "rows in",
              Float64(perf_counter_ns() - t0) / 1e9, "s   falls/1k lane-steps",
              nf / Float64(n_collect * N_ENVS) * 1000.0)

        # ── 2. the pools: 65 536 uniform ring rows, embedded once ──────────
        comptime NROW = 2 * EVAL_POOL
        var rng = SplitMix64(UInt64(seed_v) * 7919 + 1)
        var rows = List[Int]()
        for _ in range(NROW):
            rows.append(rng.below(agent.replay_size()))
        var raw = List[Scalar[DT]]()
        var bs = List[Scalar[DT]]()
        agent.embed_rows(rows, raw, bs)
        var up_idx = List[Int]()
        var calm_idx = List[Int]()
        for i in range(NROW):
            if Float64(raw[i * OBS + G1M_COL_GZ]) <= UPRIGHT_GZ:
                up_idx.append(i)
                if abs(Float64(raw[i * OBS + G1M_COL_WZ])) < CALM_WZ:
                    calm_idx.append(i)
        print("[pool] upright rows", len(up_idx), "/", NROW,
              "  calm (upright, |ωz| <", CALM_WZ, ")", len(calm_idx))
        if dump_pool.byte_length() > 0:
            var txt = String("# vx vy wz h gz\n")
            for i in range(NROW):
                var o = i * OBS
                txt += (String(Float64(raw[o + G1M_COL_VX])) + " "
                        + String(Float64(raw[o + G1M_COL_VY])) + " "
                        + String(Float64(raw[o + G1M_COL_WZ])) + " "
                        + String(Float64(raw[o + G1M_COL_H])) + " "
                        + String(Float64(raw[o + G1M_COL_GZ])) + "\n")
            with open(dump_pool, "w") as fh:
                fh.write(txt)
            print("[dump] pool rows ->", dump_pool)
        if dump_zpool.byte_length() > 0:
            # the training eval's pool, drawn as `_eval` draws it
            var zrng = SplitMix64(UInt64(seed_v) * 7919)
            var zrows = List[Int]()
            var zfrom = List[Bool]()
            var n_prior = agent.prior_rows()
            for _ in range(EVAL_POOL):
                if n_prior > 0 and zrng.uniform() < Agent.PRIOR_FRAC:
                    zrows.append(zrng.below(n_prior))
                    zfrom.append(True)
                else:
                    zrows.append(zrng.below(agent.replay_size()))
                    zfrom.append(False)
            var zraw = List[Scalar[DT]]()
            var zbs = List[Scalar[DT]]()
            if n_prior > 0:
                agent.embed_rows_mixed(zrows, zfrom, zraw, zbs)
            else:
                agent.embed_rows(zrows, zraw, zbs)
            var n_p = 0
            for i in range(EVAL_POOL):
                if zfrom[i]:
                    n_p += 1
            var fo = List[Float32](length=EVAL_POOL * OBS, fill=0)
            var fb = List[Float32](length=EVAL_POOL * D, fill=0)
            for i in range(EVAL_POOL * OBS):
                fo[i] = Float32(zraw[i])
            for i in range(EVAL_POOL * D):
                fb[i] = Float32(zbs[i])
            var zcols = List[ColumnSpec]()
            zcols.append(ColumnSpec(String("obs"), DType.float32, OBS))
            zcols.append(ColumnSpec(String("b"), DType.float32, D))
            var zw = TrajectoryStoreWriter(
                dump_zpool, zcols^, env_id=String("builtin:unitree_g1"), seed=seed_v,
            )
            zw.append[DType.float32](String("obs"), fo.unsafe_ptr().as_unsafe_any_origin(), EVAL_POOL)
            zw.append[DType.float32](String("b"), fb.unsafe_ptr().as_unsafe_any_origin(), EVAL_POOL)
            zw.end_episode()
            zw.close()
            print("[dump] z pool:", EVAL_POOL, "rows (", n_p, "prior ) ->", dump_zpool)

        # ── eval buffers ───────────────────────────────────────────────────
        var e_task = ctx.enqueue_create_buffer[DType.int32](N_ENVS)
        var e_params = ctx.enqueue_create_buffer[DT](n_task * G1M_CMD_P)
        var e_z = ctx.enqueue_create_buffer[DT](N_ENVS * D)
        var e_ret = ctx.enqueue_create_buffer[DT](N_ENVS)
        var e_fell = ctx.enqueue_create_buffer[DT](N_ENVS)
        var e_ar = ctx.enqueue_create_buffer[DT](N_ENVS)
        var e_prev_a = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
        var h_task = ctx.enqueue_create_host_buffer[DType.int32](N_ENVS)
        var h_params = ctx.enqueue_create_host_buffer[DT](n_task * G1M_CMD_P)
        for l in range(N_ENVS):
            h_task[l] = Int32(l % n_task) if l < reps * n_task else Int32(-1)
        for k in range(n_task):
            var p = suite[k].params()
            for j in range(G1M_CMD_P):
                h_params[k * G1M_CMD_P + j] = Scalar[DT](p[j])
        ctx.enqueue_copy(e_task, h_task)
        ctx.enqueue_copy(e_params, h_params)
        var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var h_qpos = ctx.enqueue_create_host_buffer[DT](len(eval_env.d.qpos.dev.value()))

        var pool_names: List[String] = [
            "ring", "upright", "upright_wide", "calm", "calm_sharp"
        ]
        for v in range(len(pool_names)):
            # the pool's rows (indices into `rows`) and the reward power
            var sel = List[Int]()
            if v == 0:
                for i in range(EVAL_POOL):
                    sel.append(i)
            elif v <= 2:
                for i in range(min(EVAL_POOL, len(up_idx))):
                    sel.append(up_idx[i])
            else:
                for i in range(min(EVAL_POOL, len(calm_idx))):
                    sel.append(calm_idx[i])
            var n = len(sel)
            if n == 0:
                print("# pool", pool_names[v], "is empty: skipped")
                continue
            var pw = 0.25 if v == 2 else (4.0 if v == 4 else 1.0)
            var sub_b = List[Scalar[DT]](length=n * D, fill=Scalar[DT](0))
            for i in range(n):
                for j in range(D):
                    sub_b[i * D + j] = bs[sel[i] * D + j]
            print()
            print("#" * 70)
            print("# pool", pool_names[v], " rows", n, " reward power", pw)
            print("#" * 70)
            print("POOL task  r>0.5  r>0.1  mean_r   E_r[vx]  E_r[vy]  E_r[wz]  E_r[h]")
            var h_z = ctx.enqueue_create_host_buffer[DT](N_ENVS * D)
            for i in range(N_ENVS * D):
                h_z[i] = Scalar[DT](0.0)
            for k in range(n_task):
                var rew = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
                var mu = 0.0
                var n5 = 0
                var n1 = 0
                var sw = 0.0
                var wvx = 0.0
                var wvy = 0.0
                var wwz = 0.0
                var wh = 0.0
                for i in range(n):
                    var off = sel[i] * OBS
                    var r0 = g1_cmd_reward_obs(suite[k], raw, off)
                    if r0 > 0.5:
                        n5 += 1
                    if r0 > 0.1:
                        n1 += 1
                    sw += r0
                    wvx += r0 * Float64(raw[off + G1M_COL_VX])
                    wvy += r0 * Float64(raw[off + G1M_COL_VY])
                    wwz += r0 * Float64(raw[off + G1M_COL_WZ])
                    wh += r0 * Float64(raw[off + G1M_COL_H])
                    var r = r0 ** pw if r0 > 0.0 else 0.0
                    rew[i] = Scalar[DT](r)
                    mu += r
                mu /= Float64(n)
                for i in range(n):
                    rew[i] = Scalar[DT](Float64(rew[i]) - mu)
                var sw_ = sw if sw > 0.0 else 1.0
                print("POOL", suite[k].name, n5, n1, _fmt(sw / Float64(n)),
                      _fmt(wvx / sw_), _fmt(wvy / sw_), _fmt(wwz / sw_), _fmt(wh / sw_))
                var zk = z_from_reward[D](sub_b, rew, n)
                for l in range(N_ENVS):
                    if Int(h_task[l]) == k:
                        for j in range(D):
                            h_z[l * D + j] = zk[j]
            ctx.enqueue_copy(e_z, h_z)

            # ── 3. the eval, with host-side tracking statistics ────────────
            _full_reset(eval_env, ctx, UInt64(0xE7A1), reset_noise)
            e_ret.enqueue_fill(Scalar[DT](0.0))
            e_fell.enqueue_fill(Scalar[DT](0.0))
            e_ar.enqueue_fill(Scalar[DT](0.0))
            e_prev_a.enqueue_fill(Scalar[DT](0.0))
            # per task: Σ vx, vy, wz, h, |e_vx|, |e_vy|, |e_wz|, r_vx0,
            # and over the last half: Σ vx, |e_vx|
            comptime NS = 10
            var acc = List[Float64](length=n_task * NS, fill=0.0)
            var vx0p = suite[k_vx0].params()
            var dump_txt = String()
            for t in range(EVAL_LEN):
                agent.policy_with_z(
                    mptr(eval_env._obs.unsafe_ptr()), mptr(e_z.unsafe_ptr()),
                    mptr(eval_env._action.unsafe_ptr()),
                )
                eval_env.step_batch[N_ENVS](Optional(ctx), UInt64(t))
                ctx.enqueue_function[g1_cmd_step_kernel[OBS, N_ENVS]](
                    mptr(eval_env._obs.unsafe_ptr()), mptr(e_task.unsafe_ptr()),
                    mptr(e_params.unsafe_ptr()), mptr(e_ret.unsafe_ptr()),
                    mptr(e_fell.unsafe_ptr()), mptr(eval_env._action.unsafe_ptr()),
                    mptr(e_prev_a.unsafe_ptr()), mptr(e_ar.unsafe_ptr()),
                    Scalar[DT](EVAL_FALL_H), Int32(ACT),
                    grid_dim=_blocks(N_ENVS), block_dim=TPB,
                )
                ctx.enqueue_copy(h_obs, eval_env._obs)
                if v == 0 and dump.byte_length() > 0:
                    ctx.enqueue_copy(h_qpos, eval_env.d.qpos.dev.value())
                ctx.synchronize()
                for l in range(reps * n_task):
                    var k = l % n_task
                    var o = l * OBS
                    var vx = Float64(h_obs[o + G1M_COL_VX])
                    var vy = Float64(h_obs[o + G1M_COL_VY])
                    var wz = Float64(h_obs[o + G1M_COL_WZ])
                    var hh = Float64(h_obs[o + G1M_COL_H])
                    var b = k * NS
                    acc[b + 0] += vx
                    acc[b + 1] += vy
                    acc[b + 2] += wz
                    acc[b + 3] += hh
                    acc[b + 4] += abs(vx - suite[k].vx)
                    acc[b + 5] += abs(vy - suite[k].vy)
                    acc[b + 6] += abs(wz - suite[k].wz)
                    acc[b + 7] += g1_cmd_reward(
                        vx0p[0], vx0p[1], vx0p[2], vx0p[3], vx0p[4], vx0p[5],
                        vx0p[6], vx0p[7], vx, vy, wz,
                        Float64(h_obs[o + G1M_COL_GX]),
                        Float64(h_obs[o + G1M_COL_GY]), hh,
                    )
                    if t >= EVAL_LEN // 2:
                        acc[b + 8] += vx
                        acc[b + 9] += abs(vx - suite[k].vx)
                if v == 0 and dump.byte_length() > 0:
                    for k in range(n_task):
                        dump_txt += String(t) + " " + String(k)
                        for j in range(EnvT.NQ):
                            dump_txt += " " + String(Float64(h_qpos[k * EnvT.NQ + j]))
                        dump_txt += "\n"
            var h_ret = ctx.enqueue_create_host_buffer[DT](N_ENVS)
            var h_fell = ctx.enqueue_create_host_buffer[DT](N_ENVS)
            ctx.enqueue_copy(h_ret, e_ret)
            ctx.enqueue_copy(h_fell, e_fell)
            ctx.synchronize()
            print("EVAL task  ret  fall   vx  vy  wz  h   |e_vx| |e_vy| |e_wz|   ret_on_vx0   late_vx late_|e_vx|")
            var fam_sum = List[Float64](length=3, fill=0.0)
            var fam_n = List[Int](length=3, fill=0)
            for k in range(n_task):
                var r = 0.0
                var f = 0.0
                for l in range(reps * n_task):
                    if l % n_task == k:
                        r += Float64(h_ret[l])
                        f += Float64(h_fell[l])
                r /= Float64(reps)
                f /= Float64(reps)
                fam_sum[suite[k].family] += r
                fam_n[suite[k].family] += 1
                var nn = Float64(reps * EVAL_LEN)
                var nl = Float64(reps * (EVAL_LEN - EVAL_LEN // 2))
                var b = k * NS
                print("EVAL", suite[k].name, _fmt(r), _fmt(f),
                      _fmt(acc[b] / nn), _fmt(acc[b + 1] / nn), _fmt(acc[b + 2] / nn),
                      _fmt(acc[b + 3] / nn), _fmt(acc[b + 4] / nn),
                      _fmt(acc[b + 5] / nn), _fmt(acc[b + 6] / nn),
                      _fmt(acc[b + 7] / Float64(reps)),
                      _fmt(acc[b + 8] / nl), _fmt(acc[b + 9] / nl))
            print("[eval", pool_names[v], "] velocity", fam_sum[0] / Float64(fam_n[0]),
                  " posture", fam_sum[1] / Float64(fam_n[1]),
                  " composite", fam_sum[2] / Float64(fam_n[2]))
            if v == 0 and dump.byte_length() > 0:
                var names = String("# tasks:")
                for k in range(n_task):
                    names += " " + suite[k].name
                with open(dump, "w") as fh:
                    fh.write(names + "\n# t lane qpos[" + String(EnvT.NQ) + "]\n")
                    fh.write(dump_txt)
                print("[dump] qpos of lane k (task k) ->", dump)
        _ = agent.replay_size()
