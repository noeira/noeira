"""FB-MEBE on the dm_control walker — rung W of `noeira-docs/FB_MEBE_G1_PLAN.md` §4.

The online walker arc measured a null (`BFM_ZERO_SHOT_RL.md` §18.7-18.9):
pure online FB with no prior (`a3c`) recovers `stand` and NO locomotion
(5-rung means 1.59 / 0.93 / 0.99 × random); only the CPR style term, fed by
expert states, broke it (walk 1.93, run 1.51). FB-MEBE claims to break the
same null with NO data — inverse-density exploration plus a hand-written
regularity critic. This driver runs the paper's four arms on the cheapest
body where we already know the null:

    arm   B input   exploration                       Q_reg   flags
    W1    ϑ_w       sphere (lanes + relabel)          –       --explore sphere --reg 0
    W2    ϑ_w       sphere                            λ       --explore sphere --reg 20
    W3    ϑ_w       MEBE β = 2 (80 % pool / 20 % ⊙)   λ       --explore mebe --beta 2 --reg 20
    W4    ϑ_w       MEBE β = 0                        λ       --explore mebe --beta 0 --reg 20

W0 (the a3c re-run, B on the full 24-D obs) is `fb_online_walker_gpu.mojo
--store ""` unchanged; this driver with `-D B_PROJ=0` is the same B under
the new mixture, not W0.

B's input is COMPTIME — build, do not `mojo run`, a define (a stale
define-less compile is silently reused by `run`):

    pixi run -e nvidia mojo build -I . -D B_PROJ=1 examples/fb/fb_mebe_walker_gpu.mojo -o fb_mebe_walker
    ./fb_mebe_walker --explore mebe --beta 2 --reg 20 --tag w3_s1 --seed 1

    pixi run -e nvidia mojo build -I . -D B_PROJ=1 examples/fb/fb_eval_walker_online.mojo -o fb_eval_walker_proj
    ./fb_eval_walker_proj runs/<run>/checkpoints/step_<k> --episodes 64

Everything not under test is the a3c configuration (§18.7.3): 256 envs,
D 128, F / actor 1024 wide, batch 1024, 8 updates per iteration, ortho 100,
lr_b 1e-5, grad clip 1, the action hinge 100 @ 0.8, 5 M env steps, 25 600
warmup steps, expl_std 0.2. The z mixture follows the release (lanes held
100 steps; relabel keeps 0 % of the stored z, 80 / 20 pool / sphere).

The regularity reward is the paper's appendix F minus the feet term (the
walker has no feet contact sensor here): `−2.5e-7·‖q̈‖² − 0.1·‖a − a_prev‖²`,
with `q̈` the leg joints' finite difference at the control step (25 ms).
`λ` is added RAW (the release's `reg_coeff 20`); read
`actor_grad_value → actor_grad_total` at the first flush to see what it
weighs against `F·z`, and sweep it before reading any arm.

⚠ `episode_len 1000`: walker lanes truncate together at the time limit, and
the batched driver passes `terminated` only — without the schedule the ring
would treat each reset as a transition (`FBMebeOnlineAgent.make`).
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns

from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.deep_agents.fb.online_mebe import (
    FBMebeOnlineAgent,
    MebeConfig,
    RegRewardConfig,
)
from noeira.deep_agents.fb.density import InverseDensityConfig
from noeira.deep_agents.fb.walker_nets import (
    WalkerFNet, WalkerANet, WalkerQNet, WalkerBNetFull, WalkerBNetProj,
    WALKER_OBS, WALKER_ACT, WALKER_D, WALKER_HID,
    WALKER_COL_HEIGHT, WALKER_COL_ROOTX_VEL, WALKER_COL_JOINT_VEL, WALKER_N_JOINT,
)
from noeira.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig


comptime B_PROJ: Int = get_defined_int["B_PROJ", 1]()

comptime WalkerCfg = DMWalkerConfig[1.0]   # `walk` for the coverage readout only
comptime OBS: Int = WALKER_OBS
comptime NACT: Int = WALKER_ACT
comptime N_ENVS: Int = 256
comptime EnvT = Phyics3dBatchedEnv[
    DMWalkerModel, WalkerCfg, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime D: Int = WALKER_D
comptime BATCH: Int = 1024
comptime CAP: Int = 1_000_000
comptime ZBUF: Int = 10_000
comptime AgentOf[BN: Module] = FBMebeOnlineAgent[
    WalkerFNet, BN, WalkerANet, WalkerQNet, OBS, NACT, D, BATCH, CAP, N_ENVS, ZBUF
]

comptime TOTAL_ENV_STEPS: Int = 5_000_000
comptime SEGMENT_STEPS: Int = 250_000
comptime UPDATES_PER_ITER: Int = 8
comptime WARMUP_STEPS: Int = 25_600
comptime Z_HOLD: Int = 100
comptime EXPL_STD: Float64 = 0.2
comptime ACT_L2: Float64 = 100.0
comptime ACT_MARGIN: Float64 = 0.8
comptime ORTHO_WEIGHT: Float64 = 100.0
comptime LR_B: Float64 = 1e-5
comptime MAX_GRAD_NORM: Float64 = 1.0
comptime EPISODE_LEN: Int = 1000
comptime CTRL_DT: Float64 = 0.025
comptime DIAG_EVERY: Int = N_ENVS * 100
comptime PRINT_EVERY: Int = N_ENVS * 500
comptime USE_TRAIN_CUDA_GRAPH: Bool = True
comptime SEED: Int = 20261012

comptime LoggerT = CompositeLogger[CsvLogger, RemoteLogger]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    comptime if B_PROJ == 1:
        _main[WalkerBNetProj]()
    else:
        _main[WalkerBNetFull]()


def _main[BN: Module]() raises:
    var total = atol(_flag(String("--steps"), String(TOTAL_ENV_STEPS)))
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_ITER)))
    var warmup = atol(_flag(String("--warmup"), String(WARMUP_STEPS)))
    var z_hold = atol(_flag(String("--z-hold"), String(Z_HOLD)))
    var explore = _flag(String("--explore"), String("mebe"))
    var beta = atof(_flag(String("--beta"), String(2.0)))
    var eps_rel = atof(_flag(String("--eps-rel"), String(0.1)))
    var refresh = atol(_flag(String("--refresh"), String(25)))
    var reg = atof(_flag(String("--reg"), String(20.0)))
    var w_acc = atof(_flag(String("--w-acc"), String(2.5e-7)))
    var w_ar = atof(_flag(String("--w-ar"), String(0.1)))
    var act_l2 = atof(_flag(String("--act-l2"), String(ACT_L2)))
    var seed_v = atol(_flag(String("--seed"), String(SEED)))
    var tag = _flag(String("--tag"), String(""))
    if explore != "mebe" and explore != "sphere":
        raise Error("--explore must be 'mebe' or 'sphere'")
    var mebe = explore == "mebe"
    var use_reg = reg > 0.0
    if warmup < BATCH:
        raise Error("--warmup must be >= BATCH (" + String(BATCH) + ")")

    var run = RunContext(
        project=String("fb"),
        driver=String("examples/fb/fb_mebe_walker_gpu.mojo"),
        slug=String("fb-mebe-walker") + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:dm_control/walker-walk"),
    )
    run.set_tag(tag)
    var csv_path = run.metrics_path()
    var seg = SEGMENT_STEPS if SEGMENT_STEPS < total else total
    var n_segments = (total + seg - 1) // seg

    # φ = (ẋ_root, torso height); no fall filter — the walker's reset is a
    # random orientation and `stand` IS getting up, so lying states are not
    # degenerate here. The box only drops numerical spikes.
    var cfg = MebeConfig([WALKER_COL_ROOTX_VEL, WALKER_COL_HEIGHT])
    cfg.add_filter(WALKER_COL_ROOTX_VEL, -15.0, 15.0)
    cfg.density = InverseDensityConfig(beta=beta, eps_rel=eps_rel, bw_scale=1.0)
    cfg.n_cand = 8192
    cfg.n_ref = 2048
    cfg.refresh_every = refresh
    var regc = RegRewardConfig(
        qd_off=WALKER_COL_JOINT_VEL, qd_n=WALKER_N_JOINT, dt=CTRL_DT,
        w_acc=w_acc, w_ar=w_ar, w_env=0.0,
    )
    # sphere arms: no pool, and the base's non-uniform share at 0 — lanes and
    # relabel draw the sphere only (the paper's undirected FB / FB-Reg)
    var explore_frac = 0.8 if mebe else 0.0

    seed(seed_v)
    print("=" * 70)
    print("FB-MEBE — dm_control walker, batched GPU, no dataset (rung W)")
    print("=" * 70)
    print("  B input             =", "ϑ_w (projection, 6)" if B_PROJ == 1 else "full obs (24)")
    print("  explore             =", explore, " β", beta, " eps_rel", eps_rel,
          " refresh", refresh, " explore_frac", explore_frac)
    print("  Q_reg λ (raw)       =", reg, " w_acc", w_acc, " w_ar", w_ar)
    print("  N_ENVS / BATCH / D  =", N_ENVS, "/", BATCH, "/", D)
    print("  env steps           =", total, " in", n_segments, "segments of", seg)
    print("  ups / warmup / hold =", ups, "/", warmup, "/", z_hold)
    print("  act_l2 @ margin     =", act_l2, "@", ACT_MARGIN, "  seed", seed_v)
    print("  tag                 = '", tag, "'   run:", run.dir)
    print("=" * 70)

    with DeviceContext() as ctx:
        var env_vars = load_dotenv()
        var logger = CompositeLogger(
            CsvLogger(csv_path, buffer_size=64),
            RemoteLogger(
                server_url=env_vars.get("NOEIRA_CLOUD_URL", ""),
                run_name=run.name(),
                run_id=run.id,
                buffer_size=64,
                api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
            ),
        )
        logger.set_config("algorithm", "FB-MEBE-online")
        logger.set_config("env", "dm_control/walker-walk (coverage readout)")
        logger.set_config("b_input", "proj6" if B_PROJ == 1 else "full24")
        logger.set_config("explore", explore)
        logger.set_config("beta", String(beta))
        logger.set_config("eps_rel", String(eps_rel))
        logger.set_config("refresh_every", String(refresh))
        logger.set_config("reg_coeff", String(reg))
        logger.set_config("w_acc", String(w_acc))
        logger.set_config("w_ar", String(w_ar))
        logger.set_config("act_l2", String(act_l2))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("d", String(D))
        logger.set_config("batch", String(BATCH))
        logger.set_config("hidden", String(WALKER_HID))
        logger.set_config("updates_per_iter", String(ups))
        logger.set_config("z_hold", String(z_hold))
        logger.set_config("seed", String(seed_v))
        logger.set_config("tag", tag)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = AgentOf[BN].make(
            ctx, cfg^, regc,
            mebe=mebe, use_reg=use_reg, reg_coeff=reg, reg_raw=True,
            lr=3e-4, lr_b=LR_B, lr_q=3e-4, ortho_weight=ORTHO_WEIGHT,
            max_grad_norm=MAX_GRAD_NORM, act_l2_weight=act_l2,
            act_l2_margin=ACT_MARGIN, learning_starts=warmup,
            action_scale=1.0, expl_std=EXPL_STD, z_hold=z_hold,
            explore_frac=explore_frac, keep_frac=0.0, window_size=100,
            seed=UInt64(seed_v), episode_len=EPISODE_LEN,
        )
        var env = EnvT(ctx)
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var t_start = perf_counter_ns()
        for s in range(n_segments):
            var done_steps = s * seg
            var this_seg = seg if done_steps + seg <= total else total - done_steps
            _ = run_offpolicy_train_batched[
                AgentOf[BN], EnvT, N_ENVS=N_ENVS,
                USE_TRAIN_CUDA_GRAPH=USE_TRAIN_CUDA_GRAPH,
                USE_ENV_CUDA_GRAPH=False,
                L=LoggerT,
            ](
                Optional(ctx), agent, env, this_seg,
                rng_seed=UInt64(seed_v + s),
                updates_per_step=ups,
                print_every=PRINT_EVERY,
                verbose=True,
                logger=logger_ptr,
                diag_every=DIAG_EVERY,
                episode_sync_every=32,
                base_step=done_steps,
                progress_label="fb-mebe",
            )
            var at = done_steps + this_seg
            var path = run.checkpoint_path(String("step_") + String(at))
            agent.save_state(path)
            var el = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "  [segment", s + 1, "/", n_segments, "]  env steps", at,
                "  train steps", agent.total_train_steps(),
                "  replay", agent.replay_size(), "  ", Float64(at) / el, "env st/s",
                " ->", path,
            )
        var pf = run.checkpoint_path(String("final"))
        agent.save_state(pf)
        announce_checkpoint(pf, artifacts, run.dir)
        finish_run(run, logger, artifacts)
        _ = logger
        print("=" * 70)
        print("done. final checkpoint ->", pf, "   metrics ->", csv_path)
        print("eval (same B_PROJ):  mojo build -D B_PROJ=", B_PROJ,
              " examples/fb/fb_eval_walker_online.mojo")
        print("=" * 70)
