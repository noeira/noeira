"""FB-MEBE on the Unitree G1 — rung G of `noeira-docs/FB_MEBE_G1_PLAN.md` §4.

A data-free base BFM: no LAFAN, no RSI, no tracking pin, no expert windows,
no discriminator. What rung W (§10-11) says carries over is the base arm:

    B on ϑ_g (10-D: gravity, height, heading-frame lin/ang velocity)
    sphere z for the lanes and the relabel (keep 0)
    truncation boundaries flagged per lane (time limit AND fall reset)

and MEBE (β 2, the inverse-density z = B(s) pool on φ = (vx, vy, ωz, h))
and Q_reg (λ ≤ 1 there; recalibrated here) are the add-ons:

    arm      flags
    G-FB     --explore sphere --reg 0
    G-Reg    --explore sphere --reg λ
    G-MEBE   --explore mebe --beta 2 --reg λ
    G-MEBE0  --explore mebe --beta 0 --reg λ

and the §12.6 levers against the spin MEBE bought on φ ∋ ωz:

    --phi vxvyh            φ = (vx, vy, h): density no longer rewards ωz
    --wz-max 1.0           a |ωz| box in the validity filter (0 = off)
    --upright-gz -0.995    the filter's g_z bound (default −0.93 ≈ 21° of
                           tilt; −0.995 ≈ 6°, what σ_g 0.1 rewards)
    --w-yaw 0.5 --w-angxy 0.05
                           −w_yaw·ωz² − w_angxy·(ωx² + ωy²) in Q_reg's
                           simulator half (needs --reg > 0)

    pixi run -e nvidia mojo build -I . examples/fb/fb_mebe_g1_gpu.mojo -o fb_mebe_g1
    ./fb_mebe_g1 --explore sphere --reg 0 --tag gfb_s1 --seed 1
    ./fb_mebe_g1 --smoke                  # 3 segments of 40 batched steps + an eval

## What differs from the BFM-Zero G1 driver (`examples/g1/bfm_zero_train_gpu.mojo`)

  * The agent row is the env's 527-D observation; there is NO history tail
    and the actor sees the privileged block. The tail's derived columns read
    a single batch-wide age, which per-lane resets would break; a data-free
    base model is a simulation result first (sim-to-real is §5's problem).
  * Resets are PER LANE: the env's time limit (500 steps) and a fall (pelvis
    < 0.45 m or tilt > 60° for 0.5 s) both end the lane's episode, both are
    BOUNDARIES in the ring (never terminals), and the lane restarts from the
    stand keyframe with ±`--reset-noise` rad on every joint.
  * The regularity reward is the walker's (joint-acceleration finite
    difference + action rate, from the ring) plus the simulator's half
    (`g1_reg_env_kernel`: undesired contacts, feet orientation, soft joint
    limits, feet slide), fed as the env reward with `w_env 1`.

## The eval (§3's base-command suite, in the loop)

At each of the last `--eval-rungs` segment ends (and every
`--eval-every` segments if set): 32 768 ring rows are the reward pool,
`z_k = E_pool[r_k(s) · B(s)]` per task (rewards centred, as the walker
eval), then a SEPARATE 1024-lane env runs all 22 tasks at once (46 lanes
each, fixed reset noise, so rungs are paired) for 250 steps from the stand
keyframe. Printed as `SUITE <env step> <task> <family> <return> <fall>
<action-rate>` lines for the harness; return is in [0, 250]. The eval
leaves the training lanes untouched (`policy_with_z`).

⚠ Build, do not `mojo run` with defines. ⚠ NVIDIA: the G1 fields path does
not run on Metal at 1024 lanes in useful time.
"""

from std.math import sqrt, log
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
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
    G1RegWeights,
    G1Cmd,
    G1M_CMD_P,
    G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H, G1M_COL_GZ, G1M_COL_QD,
    g1_cmd_suite,
    g1_cmd_reward_obs,
    g1_cmd_step_kernel,
    g1_fall_kernel,
    g1_reset_noise_kernel,
    g1_reg_env_kernel,
    g1_angvel_price_kernel,
)
from noeira.envs.robots import UnitreeG1Batched
from noeira.envs.robots.unitree_g1_xml import UNITREE_G1_OBS_DIM


comptime N_ENVS: Int = 1024
comptime OBS: Int = UNITREE_G1_OBS_DIM   # 527
comptime ACT: Int = 29
comptime D: Int = 64        # the paper's 50, rounded to a GEMM-friendly width
comptime H: Int = 1024
comptime L: Int = 3         # BFMEmbed needs L >= 3
comptime LA: Int = 2
comptime HB: Int = 512
comptime BATCH: Int = 1024
comptime CAP: Int = 1_000_000
comptime ZBUF: Int = 8192
comptime EnvT = UnitreeG1Batched[N_ENVS]
comptime FNet = BFMFTower[OBS, ACT, D, H, L, D]
comptime BNet = G1BNetProj[OBS, D, HB]
comptime ANet = BFMActorTower[OBS, D, H, LA, ACT]
comptime QNet = BFMFTower[OBS, ACT, D, H, L, 1]
comptime Agent = FBMebeOnlineAgent[
    FNet, BNet, ANet, QNet, OBS, ACT, D, BATCH, CAP, N_ENVS, ZBUF
]

comptime CTRL_DT: Float64 = 0.02
comptime FALL_H: Float64 = 0.45
comptime FALL_GZ: Float64 = -0.5     # cos 60°
comptime FALL_DELAY: Int = 25        # 0.5 s
comptime EVAL_LEN: Int = 250
comptime EVAL_POOL: Int = 32_768
comptime EVAL_FALL_H: Float64 = 0.5

comptime LoggerT = CompositeLogger[CsvLogger, RemoteLogger]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _full_reset(
    mut env: EnvT, ctx: DeviceContext, seed_v: UInt64, noise: Float64
) raises:
    """Every lane to the stand keyframe + joint noise (the per-lane path,
    with every lane flagged)."""
    env._done.enqueue_fill(Scalar[DT](1.0))
    env.selective_reset_batch[N_ENVS](Optional(ctx), seed_v)
    _after_reset(env, ctx, seed_v, noise)


def _after_reset(
    mut env: EnvT, ctx: DeviceContext, seed_v: UInt64, noise: Float64
) raises:
    """Noise on the lanes `selective_reset_batch` just reset, then FK,
    velocities and the obs (the 463 privileged dims need the velocity
    fields, which the reset alone does not refresh)."""
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


def main() raises:
    var smoke = _has(String("--smoke"))
    var total = atol(_flag(String("--steps"), String(20_000_000)))
    var seg_steps = atol(_flag(String("--segment"), String(1_000_000)))
    var ups = atol(_flag(String("--ups"), String(16)))
    var warmup = atol(_flag(String("--warmup"), String(102_400)))
    var z_hold = atol(_flag(String("--z-hold"), String(100)))
    var explore = _flag(String("--explore"), String("sphere"))
    var beta = atof(_flag(String("--beta"), String(2.0)))
    var eps_rel = atof(_flag(String("--eps-rel"), String(0.1)))
    var refresh = atol(_flag(String("--refresh"), String(25)))
    var reg = atof(_flag(String("--reg"), String(0.0)))
    var w_acc = atof(_flag(String("--w-acc"), String(2.5e-7)))
    var w_ar = atof(_flag(String("--w-ar"), String(0.1)))
    var w_env = atof(_flag(String("--w-env"), String(1.0)))
    var expl_std = atof(_flag(String("--expl-std"), String(0.05)))
    var act_l2 = atof(_flag(String("--act-l2"), String(0.0)))
    var reset_noise = atof(_flag(String("--reset-noise"), String(0.1)))
    var eval_rungs = atol(_flag(String("--eval-rungs"), String(5)))
    var eval_every = atol(_flag(String("--eval-every"), String(0)))
    var seed_v = atol(_flag(String("--seed"), String(1)))
    var tag = _flag(String("--tag"), String(""))
    var phi = _flag(String("--phi"), String("vxvywzh"))
    var wz_max = atof(_flag(String("--wz-max"), String(0.0)))
    var upright_gz = atof(_flag(String("--upright-gz"), String(-0.93)))
    var w_yaw = atof(_flag(String("--w-yaw"), String(0.0)))
    var w_angxy = atof(_flag(String("--w-angxy"), String(0.0)))
    if smoke:
        total = 3 * 40 * N_ENVS
        seg_steps = 40 * N_ENVS
        warmup = 10 * N_ENVS
        eval_rungs = 1
    if explore != "mebe" and explore != "sphere":
        raise Error("--explore must be 'mebe' or 'sphere'")
    var mebe = explore == "mebe"
    if phi != "vxvywzh" and phi != "vxvyh":
        raise Error("--phi must be 'vxvywzh' or 'vxvyh'")
    if (w_yaw > 0.0 or w_angxy > 0.0) and reg <= 0.0:
        raise Error("--w-yaw / --w-angxy price Q_reg: they need --reg > 0")
    if warmup < BATCH:
        raise Error("--warmup must be >= BATCH")
    var n_batched = total // N_ENVS
    var seg_b = seg_steps // N_ENVS
    if seg_b < 1:
        raise Error("--segment must be >= N_ENVS")
    # whole segments only: a 20 M run at 1 M segments is 19 531 batched steps,
    # and the remainder made a 21st segment of 11 k steps whose checkpoint
    # and eval duplicated the 20th (run gfb_s1)
    if n_batched >= seg_b:
        n_batched = (n_batched // seg_b) * seg_b
    var n_seg = (n_batched + seg_b - 1) // seg_b

    var run = RunContext(
        project=String("fb"),
        driver=String("examples/fb/fb_mebe_g1_gpu.mojo"),
        slug=String("fb-mebe-g1") + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:unitree_g1"),
    )
    run.set_tag(tag)
    var csv_path = run.metrics_path()

    # φ = (vx, vy, ωz, h); valid = upright and in the release's speed box
    var phi_cols: List[Int] = [G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H]
    if phi == "vxvyh":
        phi_cols = [G1M_COL_VX, G1M_COL_VY, G1M_COL_H]
    var cfg = MebeConfig(phi_cols^)
    cfg.add_filter(G1M_COL_GZ, -1.01, upright_gz)
    cfg.add_filter(G1M_COL_VX, -1.8, 1.8)
    cfg.add_filter(G1M_COL_VY, -1.3, 1.3)
    if wz_max > 0.0:
        cfg.add_filter(G1M_COL_WZ, -wz_max, wz_max)
    cfg.density = InverseDensityConfig(beta=beta, eps_rel=eps_rel, bw_scale=1.0)
    cfg.n_cand = 8192
    cfg.n_ref = 2048
    cfg.refresh_every = refresh
    var regc = RegRewardConfig(
        qd_off=G1M_COL_QD, qd_n=ACT, dt=CTRL_DT,
        w_acc=w_acc, w_ar=w_ar, w_env=w_env,
    )
    var rw = G1RegWeights.default()
    var use_reg = reg > 0.0
    var explore_frac = 0.8 if mebe else 0.0
    var suite = g1_cmd_suite()
    var n_task = len(suite)
    var reps = N_ENVS // n_task

    seed(seed_v)
    print("=" * 70)
    print("FB-MEBE — Unitree G1, batched GPU, no dataset (rung G)")
    print("=" * 70)
    print("  nets                =", "F/Q tower H", H, "L", L, "| actor H", H, "L", LA, "| B ϑ(10) ->", HB, "->", D)
    print("  explore             =", explore, " β", beta, " eps_rel", eps_rel, " refresh", refresh, " explore_frac", explore_frac)
    print("  upright g_z bound    =", upright_gz)
    print("  φ / |ωz| box        =", phi, "/", wz_max, "(0 = off)   ang-vel price yaw", w_yaw, " xy", w_angxy)
    print("  Q_reg λ (raw)       =", reg, " w_acc", w_acc, " w_ar", w_ar, " w_env", w_env,
          " (und", rw.undesired, " fo", rw.feet_orient, " lim", rw.limits, " slide", rw.slide, ")")
    print("  N_ENVS / BATCH / D  =", N_ENVS, "/", BATCH, "/", D)
    print("  env steps           =", n_batched * N_ENVS, " in", n_seg, "segments of", seg_b * N_ENVS)
    print("  ups / warmup / hold =", ups, "/", warmup, "/", z_hold)
    print("  expl_std / act_l2   =", expl_std, "/", act_l2, "  reset noise ±", reset_noise, "rad")
    print("  fall reset          = h <", FALL_H, "or g_z >", FALL_GZ, "for", FALL_DELAY, "steps (boundary)")
    print("  eval                =", n_task, "tasks x", reps, "lanes x", EVAL_LEN, "steps, last", eval_rungs, "segments",
          " every", eval_every)
    print("  seed / tag          =", seed_v, "/ '", tag, "'   run:", run.dir)
    print("=" * 70)

    with DeviceContext() as ctx:
        var env_vars = load_dotenv()
        var logger = CompositeLogger(
            CsvLogger(csv_path, buffer_size=64),
            RemoteLogger(
                server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
                run_name=run.name(), run_id=run.id, buffer_size=64,
                api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
            ),
        )
        logger.set_config("algorithm", "FB-MEBE-online")
        logger.set_config("env", "unitree_g1 (data-free)")
        logger.set_config("explore", explore)
        logger.set_config("beta", String(beta))
        logger.set_config("reg_coeff", String(reg))
        logger.set_config("w_env", String(w_env))
        logger.set_config("d", String(D))
        logger.set_config("ups", String(ups))
        logger.set_config("z_hold", String(z_hold))
        logger.set_config("expl_std", String(expl_std))
        logger.set_config("reset_noise", String(reset_noise))
        logger.set_config("seed", String(seed_v))
        logger.set_config("tag", tag)
        logger.set_config("phi", phi)
        logger.set_config("wz_max", String(wz_max))
        logger.set_config("upright_gz", String(upright_gz))
        logger.set_config("w_yaw", String(w_yaw))
        logger.set_config("w_angxy", String(w_angxy))
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = Agent.make(
            ctx, cfg^, regc,
            mebe=mebe, use_reg=use_reg, reg_coeff=reg, reg_raw=True,
            lr=3e-4, lr_b=1e-5, lr_q=3e-4, ortho_weight=100.0,
            max_grad_norm=0.0, act_l2_weight=act_l2, act_l2_margin=0.8,
            learning_starts=warmup, action_scale=1.0, expl_std=expl_std,
            z_hold=z_hold, explore_frac=explore_frac, keep_frac=0.0,
            window_size=100, seed=UInt64(seed_v), normalize_obs=True,
            episode_len=0,
        )
        var env = EnvT(ctx)
        var eval_env = EnvT(ctx)
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        # ── rollout buffers ───────────────────────────────────────────
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
        var h_falls = ctx.enqueue_create_host_buffer[DT](N_ENVS)

        # ── eval buffers ──────────────────────────────────────────────
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
        var h_eobs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)

        def _eval(at: Int) capturing raises:
            var t0 = perf_counter_ns()
            # 1. z per task, relabelled over the ring
            var rng = SplitMix64(UInt64(seed_v) * 7919 + UInt64(at))
            var rows = List[Int]()
            for _ in range(EVAL_POOL):
                rows.append(rng.below(agent.replay_size()))
            var raw = List[Scalar[DT]]()
            var bs = List[Scalar[DT]]()
            agent.embed_rows(rows, raw, bs)
            var h_z = ctx.enqueue_create_host_buffer[DT](N_ENVS * D)
            for i in range(N_ENVS * D):
                h_z[i] = Scalar[DT](0.0)
            for k in range(n_task):
                var rew = List[Scalar[DT]](length=EVAL_POOL, fill=Scalar[DT](0))
                var mu = 0.0
                for i in range(EVAL_POOL):
                    var r = g1_cmd_reward_obs(suite[k], raw, i * OBS)
                    rew[i] = Scalar[DT](r)
                    mu += r
                mu /= Float64(EVAL_POOL)
                for i in range(EVAL_POOL):
                    rew[i] = Scalar[DT](Float64(rew[i]) - mu)
                var zk = z_from_reward[D](bs, rew, EVAL_POOL)
                for l in range(N_ENVS):
                    if Int(h_task[l]) == k:
                        for j in range(D):
                            h_z[l * D + j] = zk[j]
            ctx.enqueue_copy(e_z, h_z)
            # 2. 250 steps from the stand keyframe, fixed noise (paired rungs)
            _full_reset(eval_env, ctx, UInt64(0xE7A1), reset_noise)
            e_ret.enqueue_fill(Scalar[DT](0.0))
            e_fell.enqueue_fill(Scalar[DT](0.0))
            e_ar.enqueue_fill(Scalar[DT](0.0))
            e_prev_a.enqueue_fill(Scalar[DT](0.0))
            # §12.6: the spin, read back per step — Σ|ωz|, Σ|vx − vx*|
            var sp = List[Float64](length=2 * n_task, fill=0.0)
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
                ctx.enqueue_copy(h_eobs, eval_env._obs)
                ctx.synchronize()
                for l in range(reps * n_task):
                    var k = l % n_task
                    sp[2 * k] += abs(Float64(h_eobs[l * OBS + G1M_COL_WZ]))
                    sp[2 * k + 1] += abs(Float64(h_eobs[l * OBS + G1M_COL_VX]) - suite[k].vx)
            var h_ret = ctx.enqueue_create_host_buffer[DT](N_ENVS)
            var h_fell = ctx.enqueue_create_host_buffer[DT](N_ENVS)
            var h_ar = ctx.enqueue_create_host_buffer[DT](N_ENVS)
            ctx.enqueue_copy(h_ret, e_ret)
            ctx.enqueue_copy(h_fell, e_fell)
            ctx.enqueue_copy(h_ar, e_ar)
            ctx.synchronize()
            # 3. per task
            var fam_sum = List[Float64](length=3, fill=0.0)
            var fam_n = List[Int](length=3, fill=0)
            var fall_all = 0.0
            var wz_all = 0.0
            for k in range(n_task):
                var r = 0.0
                var f = 0.0
                var a = 0.0
                for l in range(reps * n_task):
                    if l % n_task == k:
                        r += Float64(h_ret[l])
                        f += Float64(h_fell[l])
                        a += Float64(h_ar[l]) / Float64(EVAL_LEN)
                r /= Float64(reps)
                f /= Float64(reps)
                a /= Float64(reps)
                fall_all += f
                fam_sum[suite[k].family] += r
                fam_n[suite[k].family] += 1
                var nn = Float64(reps * EVAL_LEN)
                wz_all += sp[2 * k] / nn
                print("SUITE", at, suite[k].name, suite[k].family, r, f, a,
                      sp[2 * k] / nn, sp[2 * k + 1] / nn)
            print(
                "  [eval @", at, "] velocity", fam_sum[0] / Float64(fam_n[0]),
                " posture", fam_sum[1] / Float64(fam_n[1]),
                " composite", fam_sum[2] / Float64(fam_n[2]),
                " fall", fall_all / Float64(n_task),
                " |wz|", wz_all / Float64(n_task),
                "  (", Float64(perf_counter_ns() - t0) / 1e9, "s )",
            )
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("suite/velocity")); vals.append(fam_sum[0] / Float64(fam_n[0]))
            names.append(String("suite/posture")); vals.append(fam_sum[1] / Float64(fam_n[1]))
            names.append(String("suite/composite")); vals.append(fam_sum[2] / Float64(fam_n[2]))
            names.append(String("suite/fall")); vals.append(fall_all / Float64(n_task))
            logger.log_scalars(names, vals, at)

        def _coverage(at: Int, window_lane_steps: Int) capturing raises:
            """The plan's early kill checks, on the ring: upright share,
            upright |vx| >= 0.5 share, entropy of the upright (vx, vy)
            histogram (0.25 m/s cells on [-2, 2]²), and the fall-reset rate."""
            var cols: List[Int] = [G1M_COL_VX, G1M_COL_VY, G1M_COL_GZ, G1M_COL_WZ]
            comptime NS = 8192
            var x = agent.sample_ring_cols(cols, NS, UInt64(at) + 17)
            var up = 0
            var fast = 0
            var calm = 0
            var hist = List[Int](length=256, fill=0)
            for i in range(NS):
                var vx = x[i * 4]
                var vy = x[i * 4 + 1]
                if x[i * 4 + 2] <= -0.93:
                    up += 1
                    if abs(x[i * 4 + 3]) < 0.3:
                        calm += 1
                    if abs(vx) >= 0.5:
                        fast += 1
                    var cx = Int((vx + 2.0) / 0.25)
                    var cy = Int((vy + 2.0) / 0.25)
                    if cx >= 0 and cx < 16 and cy >= 0 and cy < 16:
                        hist[cy * 16 + cx] += 1
            var ent = 0.0
            var tot = 0
            for c in range(256):
                tot += hist[c]
            for c in range(256):
                if hist[c] > 0:
                    var p = Float64(hist[c]) / Float64(tot)
                    ent -= p * log(p)
            ctx.enqueue_copy(h_falls, n_falls)
            n_falls.enqueue_fill(Scalar[DT](0.0))
            ctx.synchronize()
            var nf = 0.0
            for l in range(N_ENVS):
                nf += Float64(h_falls[l])
            var fall_rate = nf / Float64(window_lane_steps) * 1000.0
            print(
                "   [cov] step", at, " upright", Float64(up) / Float64(NS),
                " upright|vx|>=0.5", Float64(fast) / Float64(NS),
                " upright|wz|<0.3", Float64(calm) / Float64(NS),
                " H(vx,vy)", ent, " falls/1k lane-steps", fall_rate,
            )
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("cov/upright")); vals.append(Float64(up) / Float64(NS))
            names.append(String("cov/upright_fast")); vals.append(Float64(fast) / Float64(NS))
            names.append(String("cov/upright_calm")); vals.append(Float64(calm) / Float64(NS))
            names.append(String("cov/entropy_vxvy")); vals.append(ent)
            names.append(String("cov/falls_per_1k")); vals.append(fall_rate)
            logger.log_scalars(names, vals, at)

        def _captured_updates() capturing raises -> None:
            for _ in range(ups):
                agent.train_device_kernels()

        env.reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v))
        _full_reset(env, ctx, UInt64(seed_v) * 1000003, reset_noise)
        eval_env.reset_batch[N_ENVS](Optional(ctx), UInt64(1))

        var train_graph: Optional[CUDAGraph] = None
        var print_every = 5 if smoke else 250
        var last_print = 0
        var t_start = perf_counter_ns()
        for s in range(n_batched):
            var env_steps = s * N_ENVS
            ctx.enqueue_copy(prev_obs, env._obs)
            agent.select_action_batched[N_ENVS](
                LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](env._obs),
                LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin](env._action),
                LayoutTensor[DT, Layout.row_major(N_ENVS, 2 * ACT), MutAnyOrigin](ao),
                LayoutTensor[DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin](alp),
                env_steps,
            )
            env.step_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            if use_reg and w_env != 0.0:
                ctx.enqueue_function[
                    g1_reg_env_kernel[N_ENVS, EnvT.NQ, EnvT.NBODY, EnvT.MC]
                ](
                    env.d.qpos.lt["gpu", type_of(env.d).L_QPOS](),
                    env.d.xquat.lt["gpu", type_of(env.d).L_B4](),
                    env.d.xvel.lt["gpu", type_of(env.d).L_B3](),
                    env.d.contacts.lt["gpu", type_of(env.d).L_CONTACTS](),
                    env.d.meta.lt["gpu", type_of(env.d).L_META](),
                    mptr(r_env.unsafe_ptr()),
                    Scalar[DT](rw.undesired), Scalar[DT](rw.feet_orient),
                    Scalar[DT](rw.limits), Scalar[DT](rw.slide),
                    grid_dim=_blocks(N_ENVS), block_dim=TPB,
                )
            if use_reg and (w_yaw > 0.0 or w_angxy > 0.0):
                ctx.enqueue_function[g1_angvel_price_kernel[OBS, N_ENVS]](
                    mptr(env._obs.unsafe_ptr()), mptr(r_env.unsafe_ptr()),
                    Scalar[DT](w_yaw), Scalar[DT](w_angxy),
                    grid_dim=_blocks(N_ENVS), block_dim=TPB,
                )
            ctx.enqueue_function[g1_fall_kernel[OBS, N_ENVS]](
                mptr(env._obs.unsafe_ptr()), mptr(env._done.unsafe_ptr()),
                mptr(fall_cnt.unsafe_ptr()), mptr(done_rec.unsafe_ptr()),
                mptr(n_falls.unsafe_ptr()),
                Scalar[DT](FALL_H), Scalar[DT](FALL_GZ), Int32(FALL_DELAY),
                grid_dim=_blocks(N_ENVS), block_dim=TPB,
            )
            # the transition is recorded BEFORE the reset: its successor row
            # will hold the post-reset obs, and `done_rec` flags it a boundary
            agent.record_batch_gpu[N_ENVS](
                ctx, prev_obs, env._action, r_env, env._obs, done_rec
            )
            env.selective_reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            _after_reset(env, ctx, UInt64(seed_v) * 1000003 + UInt64(s + 1), reset_noise)

            if env_steps >= warmup:
                maybe_capture_replay[_captured_updates](train_graph, ctx)
                for _ in range(ups):
                    agent.note_train_update()

            if (s + 1) % print_every == 0:
                ctx.synchronize()
                var el = Float64(perf_counter_ns() - t_start) / 1e9
                var at = (s + 1) * N_ENVS
                print(
                    "[g1] step", at, "/", n_batched * N_ENVS,
                    "  ", Float64(at) / el, "env st/s  train steps",
                    agent.total_train_steps(),
                )
                if env_steps >= warmup:
                    agent.flush_metrics_through_logger[LoggerT](logger_ptr, at)
                _coverage(at, (s + 1 - last_print) * N_ENVS)
                last_print = s + 1

            if (s + 1) % seg_b == 0 or s + 1 == n_batched:
                var at = (s + 1) * N_ENVS
                var seg_i = (s + 1 + seg_b - 1) // seg_b
                var path = run.checkpoint_path(String("step_") + String(at))
                agent.save_state(path)
                var el = Float64(perf_counter_ns() - t_start) / 1e9
                print(
                    "  [segment", seg_i, "/", n_seg, "]  env steps", at,
                    "  train steps", agent.total_train_steps(),
                    "  replay", agent.replay_size(), "  ", Float64(at) / el,
                    "env st/s ->", path,
                )
                var late = seg_i > n_seg - eval_rungs
                var periodic = eval_every > 0 and seg_i % eval_every == 0
                if env_steps >= warmup and (late or periodic):
                    _eval(at)

        var pf = run.checkpoint_path(String("final"))
        agent.save_state(pf)
        announce_checkpoint(pf, artifacts, run.dir)
        finish_run(run, logger, artifacts)
        _ = logger
        print("=" * 70)
        print("done. final checkpoint ->", pf, "   metrics ->", csv_path)
        print("=" * 70)
