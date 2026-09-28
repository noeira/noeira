"""FB-CPR ONLINE on dm_control walker — the reference's own setting (§18.9.1).

`fb_online_walker_gpu.mojo` (A3: FB online, no D) + `FBCPRHead` (A4). The
policy rolls out in `N_ENVS` lanes; D(s, z) tells the ring's own rows under
their rolled-out z from windows of the SAC-tail expert store under their
window encoding; `Q_D` is a critic of the POLICY'S OWN outcomes under
`r_D`; the actor maximises `F·z + reg·|F·z|·Q_D`. Offline (§18.9.1) the
style critic was a constant because the dataset's rows do not depend on
the action; online they do.

    pixi run -e nvidia mojo run -I . examples/fb/fb_online_cpr_walker_gpu.mojo --tag cpr_online

Expert set = the per-task top `--expert-frac` (0.2) of the store's
episodes by `ep_return`, as `fb_train_cpr_gpu.mojo`; the observation table
is the env's 24-D vector via `obs_at`; windows of `SEQ = 8` never cross an
episode end (valid-start table from the store's episode index).

Flags: `--steps --ups --warmup --z-hold --store --expert-frac --reg --gp
--lr-d --lr-q --bc --act-l2 --act-margin --ortho --lr-b --expl-std
--p-goal --p-expert --seed --tag`. Defaults: `bc 0` (CPR replaces BC —
the reference's setting; the FB batch is the ring, which has no expert
action to clone), hinge 100 @ 0.8 (A3 run 3's stabiliser), `reg 0.01`,
`gp 10`. `--seed` sets the host RNG, the agent and the driver's per-
segment env seed together — the ONE knob a replicate changes.

⚠⚠ Score with `fb_rescore.sh`, never one checkpoint: seed 20260908 ran
1.74 / 1.87 / 1.28 over five late rungs and its FINAL rung alone read
2.53 / 3.12 / 2.06 (§18.9.3–4). Bars at the same five-rung protocol:
`a3c` online without D 1.59 / 0.93 / 0.99, `a35` 0.97 / 0.77 / 0.96,
offline 24-D pair 1.38 / 2.06 / 1.51 (three rungs). The FB file is
unchanged in format, so `fb_eval_walker_online.mojo` scores it directly.

Verdict (§18.9.6, two seeds x `reg 0.01` / `reg 0`, ten rungs a side):
this arm is 1.58 / 1.93 / 1.51 and `--reg 0` is 1.66 / 1.14 / 0.89 —
**the actor's style gradient is the ingredient**, on locomotion only
(walk p 0.0093, run p 0.0106, stand null). `--reg 0` keeps D, Q_D and the
z mixture, and against `a3c` it is null on walk and worse on run: the
mixture alone buys nothing. The FB batch never holds an expert
transition (`EXPERT_ROWS = 0`) — the store enters as D's positives and as
`z` encodings, expert STATES only — and on that footing the arm matches
the offline pair, which trains on the SAC transitions themselves.

⚠ `Qpi > Q` is NOT a signal here, contrary to what this header said
before the first run: online the batch action IS the policy's own action
plus exploration noise, so the two agree by construction. It was a real
null offline, where the batch action is the store's.
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.sys import argv
from std.time import perf_counter_ns

from noeira.core.dotenv import load_dotenv
from noeira.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU, Tanh
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.online_cpr import FBCPROnlineAgent
from noeira.nn.core.tensor import Tensor
from noeira.data.store import TrajectoryStore
from noeira.data.resident import ResidentColumn, IDX_DT
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig


comptime WalkerCfg = DMWalkerConfig[1.0]
comptime OBS: Int = DMWalkerModel.OBS_DIM      # 24 — dm_control's vector
comptime NACT: Int = DMWalkerModel.ACTION_DIM  # 6
comptime N_ENVS: Int = 256
comptime EnvT = Phyics3dBatchedEnv[
    DMWalkerModel, WalkerCfg, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime D: Int = 128
comptime BATCH: Int = 1024
comptime SEQ: Int = 8
comptime HID: Int = 1024
comptime D_HID: Int = 1024
comptime CAP: Int = 1_000_000
comptime ZBUF: Int = 10_000
comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D
comptime D_IN = OBS + D
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
comptime DNet = Sequential[
    Linear[D_IN, D_HID], LayerNorm[D_HID], Tanh[D_HID],
    Linear[D_HID, D_HID], ReLU[D_HID],
    Linear[D_HID, D_HID], ReLU[D_HID],
    Linear[D_HID, 1],
]
comptime QNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, 1]]
comptime Agent = FBCPROnlineAgent[
    FNet, BNet, ANet, DNet, QNet, OBS, NACT, D, BATCH, CAP, N_ENVS, SEQ, ZBUF
]
comptime NQ: Int = 9
comptime NV: Int = 9
comptime ScorerEnv = Phyics3dEnv[
    DMWalkerModel, WalkerCfg, DType.float64, False
]
comptime TOTAL_ENV_STEPS: Int = 5_000_000
comptime SEGMENT_STEPS: Int = 250_000
comptime UPDATES_PER_ITER: Int = 8
comptime WARMUP_STEPS: Int = 25_600
comptime Z_HOLD: Int = 150
comptime EXPL_STD: Float64 = 0.2
comptime BC_WEIGHT: Float64 = 0.0
comptime STORE_PATH: StaticString = "fb_walker_all_sac.h5"
comptime EXPERT_FRAC: Float64 = 0.2
comptime ACT_L2: Float64 = 100.0
comptime ACT_MARGIN: Float64 = 0.8
comptime ORTHO_WEIGHT: Float64 = 100.0
comptime LR_B: Float64 = 1e-5
comptime REG_COEFF: Float64 = 0.01
comptime GP_COEF: Float64 = 10.0
comptime LR_D: Float64 = 1e-5
comptime LR_Q: Float64 = 1e-4
comptime P_GOAL: Float64 = 0.2
comptime P_EXPERT: Float64 = 0.6
comptime MAX_GRAD_NORM: Float64 = 1.0
comptime DIAG_EVERY: Int = N_ENVS * 100
comptime PRINT_EVERY: Int = N_ENVS * 500
comptime USE_TRAIN_CUDA_GRAPH: Bool = True
# ⚠ ON APPLE, BUILD WITH `-Xlinker -ld_classic`. The fully-expanded graph
# type mangles to a symbol longer than the new macOS linker accepts:
#   ld: Assertion failed: (name.size() <= maxLength)
# Healthy source, toolchain limit — `act_so101_train_gpu.mojo` carries the
# same note. `mojo run` JITs and never invokes ld; NVIDIA needs no flag.
#
# ⚠ NO PATH CONSTANTS HERE ANY MORE. Every path this driver writes comes
# from `RunContext` in `main`, so two runs cannot collide — see
# `core/run.mojo` and docs/PROJECT_LAYER_PLAN.md P0d.
comptime SEED: Int = 20260908
comptime LoggerT = CompositeLogger[CsvLogger, RemoteLogger]


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _quantile_threshold(mut v: List[Float64], top_frac: Float64) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var k = Int(Float64(len(v)) * (1.0 - top_frac))
    if k >= len(v):
        k = len(v) - 1
    if k < 0:
        k = 0
    return v[k]


def main() raises:
    var total = atol(_flag(String("--steps"), String(TOTAL_ENV_STEPS)))
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_ITER)))
    var warmup = atol(_flag(String("--warmup"), String(WARMUP_STEPS)))
    var z_hold = atol(_flag(String("--z-hold"), String(Z_HOLD)))
    var store_path = _flag(String("--store"), String(STORE_PATH))
    var expert_frac = atof(_flag(String("--expert-frac"), String(EXPERT_FRAC)))
    var bc_w = atof(_flag(String("--bc"), String(BC_WEIGHT)))
    var act_l2 = atof(_flag(String("--act-l2"), String(ACT_L2)))
    var act_margin = atof(_flag(String("--act-margin"), String(ACT_MARGIN)))
    var ortho_w = atof(_flag(String("--ortho"), String(ORTHO_WEIGHT)))
    var lr_b = atof(_flag(String("--lr-b"), String(LR_B)))
    var expl = atof(_flag(String("--expl-std"), String(EXPL_STD)))
    var reg = atof(_flag(String("--reg"), String(REG_COEFF)))
    var gp = atof(_flag(String("--gp"), String(GP_COEF)))
    var lr_d = atof(_flag(String("--lr-d"), String(LR_D)))
    var lr_q = atof(_flag(String("--lr-q"), String(LR_Q)))
    var p_goal = atof(_flag(String("--p-goal"), String(P_GOAL)))
    var p_expert = atof(_flag(String("--p-expert"), String(P_EXPERT)))
    var tag = _flag(String("--tag"), String(""))
    var seed_v = atol(_flag(String("--seed"), String(SEED)))
    # ⚠⚠ ONE OF **FIVE** COPIES OF THIS BLOCK IN THE FB FAMILY, all replaced by
    # `RunContext` together. Deriving three paths from a `--tag` a human has to
    # remember to vary is one forgotten flag away from a run silently
    # overwriting the previous one; `checkpoints/` holds 26 `fb_walker_*` files
    # because of it. The tag survives as the SLUG so sweep arms stay legible in
    # a directory listing — uniqueness now comes from the id, not the human.
    var run = RunContext(
        project=String("fb"),
        driver=String("examples/fb/fb_online_cpr_walker_gpu.mojo"),
        slug=String("fb-cpr-online")
             + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:dm_control/walker-walk"),
        dataset=store_path,
        seed=seed_v,
    )
    run.set_tag(tag)
    var csv_path = run.metrics_path()
    print("run:", run.dir)
    if warmup < BATCH:
        raise Error("--warmup must be >= BATCH (" + String(BATCH) + ")")
    if store_path.byte_length() == 0:
        raise Error("--store is required: CPR needs an expert set for D")
    var seg = SEGMENT_STEPS if SEGMENT_STEPS < total else total
    var n_segments = (total + seg - 1) // seg
    seed(seed_v)
    print("=" * 70)
    print("FB-CPR ONLINE — dm_control walker, batched GPU")
    print("=" * 70)
    print("  OBS / NACT / D      =", OBS, "/", NACT, "/", D)
    print("  N_ENVS / BATCH / SEQ=", N_ENVS, "/", BATCH, "/", SEQ)
    print("  env steps           =", total, " in", n_segments, "segments of", seg)
    print("  updates / iteration =", ups, " (", ups * BATCH // N_ENVS, "samples per env step )")
    print("  warmup env steps    =", warmup)
    print("  z_hold / ZBUF       =", z_hold, "/", ZBUF)
    print("  expl_std / bc / act_l2@margin / ortho / lr_b =", expl, "/", bc_w, "/", act_l2, "@", act_margin, "/", ortho_w, "/", lr_b)
    print("  reg / gp / lr_d / lr_q =", reg, "/", gp, "/", lr_d, "/", lr_q)
    print("  mixture goal / expert / uniform =", p_goal, "/", p_expert, "/", 1.0 - p_goal - p_expert)
    print("  expert store        =", store_path, " top", expert_frac, "per task")
    print("  CUDA graph (train)  =", USE_TRAIN_CUDA_GRAPH)
    print("  seed                =", seed_v)
    print("  tag                 = '", tag, "'")
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
        logger.set_config("algorithm", "FB-CPR-online")
        logger.set_config("seed", String(seed_v))
        logger.set_config("env", "dm_control/walker-walk (coverage readout)")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("d", String(D))
        logger.set_config("batch", String(BATCH))
        logger.set_config("seq", String(SEQ))
        logger.set_config("hidden", String(HID))
        logger.set_config("replay_cap", String(CAP))
        logger.set_config("updates_per_iter", String(ups))
        logger.set_config("warmup", String(warmup))
        logger.set_config("z_hold", String(z_hold))
        logger.set_config("expl_std", String(expl))
        logger.set_config("bc_weight", String(bc_w))
        logger.set_config("act_l2", String(act_l2))
        logger.set_config("act_margin", String(act_margin))
        logger.set_config("ortho_weight", String(ortho_w))
        logger.set_config("lr_b", String(lr_b if lr_b >= 0.0 else 3e-4))
        logger.set_config("max_grad_norm", String(MAX_GRAD_NORM))
        logger.set_config("reg_coeff", String(reg))
        logger.set_config("gp_coef", String(gp))
        logger.set_config("lr_d", String(lr_d))
        logger.set_config("lr_q", String(lr_q))
        logger.set_config("p_goal", String(p_goal))
        logger.set_config("p_expert", String(p_expert))
        logger.set_config("expert_frac", String(expert_frac))
        logger.set_config("tag", tag)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()
        var agent = Agent.make(
            ctx,
            lr=3e-4, lr_b=lr_b, lr_d=lr_d, lr_q=lr_q,
            ortho_weight=ortho_w, max_grad_norm=MAX_GRAD_NORM,
            bc_weight=bc_w, act_l2_weight=act_l2, act_l2_margin=act_margin,
            reg_coeff=reg, gp_coef=gp,
            learning_starts=warmup, action_scale=1.0, expl_std=expl,
            z_hold=z_hold, p_goal=p_goal, p_expert=p_expert,
            window_size=100, initial_episode_fill=0.0, seed=UInt64(seed_v),
        )
        var env = EnvT(ctx)

        # ── the expert set: per-task top `expert_frac` of the store ──────
        print("[expert] loading", store_path, "...")
        var store = TrajectoryStore(store_path)
        var n_rows = store.n_rows()
        var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
        var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
        var ep_return = ResidentColumn[DType.float32].load(store, String("ep_return"))
        var task_col = ResidentColumn[DType.int32].load(store, String("task"))
        var scorer = ScorerEnv()
        _ = scorer.reset()
        var q = List[Float64](length=NQ, fill=0.0)
        var v = List[Float64](length=NV, fill=0.0)
        var eobs = Tensor.alloc(n_rows * OBS)
        var t0 = perf_counter_ns()
        for r in range(n_rows):
            for k in range(NQ):
                q[k] = Float64(qpos.host[r * NQ + k])
            for k in range(NV):
                v[k] = Float64(qvel.host[r * NV + k])
            var o = scorer.obs_at(q, v)
            for k in range(OBS):
                eobs.data[r * OBS + k] = Scalar[DT](Float64(o.data[k]))
        print("[expert] obs table:", n_rows, "rows in",
              Float64(perf_counter_ns() - t0) / 1e9, "s")
        var moving = 0
        for k in range(OBS):
            var mn = Float64(1e30)
            var mx = Float64(-1e30)
            for r in range(n_rows):
                var x = Float64(eobs.data[r * OBS + k])
                if x < mn:
                    mn = x
                if x > mx:
                    mx = x
            if mx - mn > 1e-6:
                moving += 1
        print("[expert] obs dims that vary across rows:", moving, "/", OBS)
        if moving < OBS - 2:
            raise Error("expert obs table: too few varying dims — obs_at is not producing the env's observation")
        var n_eps = store.episodes.n_episodes()
        var n_tasks = 0
        for e in range(n_eps):
            var off = Int(store.episodes.ep_offset[e])
            var t_id = Int(task_col.host[off])
            if t_id + 1 > n_tasks:
                n_tasks = t_id + 1
        var thresholds = List[Float64](length=n_tasks, fill=0.0)
        for t_id in range(n_tasks):
            var rets = List[Float64]()
            for e in range(n_eps):
                var off = Int(store.episodes.ep_offset[e])
                if Int(task_col.host[off]) == t_id:
                    rets.append(Float64(ep_return.host[off]))
            if len(rets) > 0:
                thresholds[t_id] = _quantile_threshold(rets, expert_frac)
        var starts_host = List[Scalar[IDX_DT]]()
        var expert_eps = 0
        for e in range(n_eps):
            var off = Int(store.episodes.ep_offset[e])
            var ln = Int(store.episodes.ep_len[e])
            var t_id = Int(task_col.host[off])
            if Float64(ep_return.host[off]) < thresholds[t_id]:
                continue
            expert_eps += 1
            var last_start = off + ln - SEQ - 1
            for s0 in range(off, last_start + 1):
                starts_host.append(Scalar[IDX_DT](s0))
        var n_starts = len(starts_host)
        print("[expert] per-task top", expert_frac, "->", expert_eps, "/", n_eps,
              "episodes,", n_starts, "valid window starts")
        for t_id in range(n_tasks):
            print("[expert]   task", t_id, "return threshold", thresholds[t_id])
        var sh = ctx.enqueue_create_host_buffer[IDX_DT](n_starts)
        for i in range(n_starts):
            sh[i] = starts_host[i]
        var sd = ctx.enqueue_create_buffer[IDX_DT](n_starts)
        ctx.enqueue_copy(sd, sh)
        eobs.upload(ctx)
        ctx.synchronize()
        agent.attach_expert_windows(eobs^, sd^, n_starts)
        logger.set_config("expert_store", store_path)
        logger.set_config("expert_episodes", String(expert_eps))
        logger.set_config("expert_window_starts", String(n_starts))

        # ⚠ AFTER the config, before step 0 — `register_run` seeds the
        # dashboard config from the run and POSTs `/runs`. Registering lazily on
        # the first metric batch (which `flush` still does for drivers that never
        # call this) means a run that dies before step 0 never appears at all.
        register_run(run, logger)
        # The uplink: `final` is uploaded (step_* only on request); None without
        # a monitor in .env, and every call below is then a no-op.
        var artifacts = sink_for_run(run.id, run.dir)

        var t_start = perf_counter_ns()
        for s in range(n_segments):
            var done_steps = s * seg
            var this_seg = seg if done_steps + seg <= total else total - done_steps
            _ = run_offpolicy_train_batched[
                Agent, EnvT, N_ENVS=N_ENVS,
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
                progress_label="fb-cpr-online",
            )
            var at = done_steps + this_seg
            var path = run.checkpoint_path(String("step_") + String(at))
            agent.save_state(path)
            var el = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "  [segment", s + 1, "/", n_segments, "]  env steps", at,
                "  train steps", agent.total_train_steps(),
                "  mean_ret(walk)", agent.mean_return(),
                "  replay", agent.replay_size(), "  ", Float64(at) / el, "env st/s",
                " ->", path, "(+ .cpr)",
            )
        var pf = run.checkpoint_path(String("final"))
        agent.save_state(pf)
        announce_checkpoint(pf, artifacts, run.dir)
        finish_run(run, logger, artifacts)
        _ = logger
        print("=" * 70)
        print("done. final checkpoint ->", pf, "   metrics ->", csv_path)
        print("next:  pixi run mojo run -I . examples/fb/fb_eval_walker_online.mojo", pf)
        print("=" * 70)
