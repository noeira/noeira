"""A3 — ONLINE off-policy FB on dm_control walker, batched GPU env, NO dataset.

`docs/BFM_ZERO_SHOT_RL.md` §18.3 step A3. The question this run answers is
one sentence long: does `pi_z` trained with NO dataset — generating its own
coverage in `N_ENVS` parallel walkers — recover more of the SAC experts than
the ~19 / 5 / 5 % the offline runs are stuck at (§13)? §16.2 says the
offline number is the coverage limit; this is the experiment that isolates
that claim, because it changes nothing else: same `FBTrainer`, same nets as
`fb_train_gpu.mojo`, same eval protocol.

    pixi run -e nvidia mojo run -I . examples/fb/fb_online_walker_gpu.mojo \
        [--steps N] [--ups K] [--warmup N] [--z-hold N] [--bc X] [--ortho X] \
        [--lr-b X] [--expl-std X] [--act-l2 X] [--act-margin X] [--tag NAME] \
        [--store PATH|""] [--expert-max N]

Then score it — the online arm has its OWN eval, because the batched env's
observation is dm_control's 24-D vector and not the store's `[qpos | qvel]`:

    pixi run mojo run -I . examples/fb/fb_eval_walker_online.mojo <ckpt>

## A3.5 — the expert store in the replay (default ON)

`--store fb_walker_all_sac.h5` (default) attaches the SAC ladder: 512 of
every 1024 batch rows come from it (s, a, s', s+), BC clones its actions on
those rows only, and the other 512 rows are the online ring. `--store ""`
is the pure-online arm (run 3: stand 1.51, walk 0.97, run 0.94). The
question: does online interaction ADD anything on top of the data — walk /
run above the offline 1.82 / 1.44 — or merely match it? §18.7.2.

## What the numbers in the log mean

  * `mean_ret` is the env's OWN reward (walker `walk`) under the exploring,
    z-conditioned policy. FB never trains on it. It is a COVERAGE signal —
    "does the population ever walk" — not a zero-shot score.
  * `fb/*` at each diag flush are the collapse detectors §14 insists on:
    `|B|` must read sqrt(d) = 11.314 at every flush (hard invariant under
    `LayerNormNoAffine`), `ortho_Q` must not grow without bound, `|F|` and
    `grad_norm_f1` must settle, `mean|a|` must stay well below 1.
  * `replay_size` shows the ring filling; nothing is sampled below one batch.

## Knobs, and where their defaults come from

  N_ENVS 256 / `--ups` 8 / BATCH 1024: BFM-Zero runs 1024 envs and 16
  updates per env step (Table 1). At tau = 0.01, 16 updates move the Polyak
  target 15 % per iteration; 8 moves it 7.7 %. The walker SAC ladder measured
  64 updates/iter as a target that chased itself (§13's SAC note), so this
  starts at 8 and exposes it.
  `--z-hold` 150 and the 10 k ZBuffer are BFM-Zero's rollout rule.
  `--bc` 0.0, `--act-l2` 100 at `--act-margin` 0.8: two runs bracket it.
  Run 1 (no penalty) went bang-bang — mean|a| 0.82 → 0.88, 82–90 %
  saturated, walk 0.71x / run 0.48x random. Run 2 (plain L2 at 1.0) went
  NULL — replay mean|a| 0.19, eval 0.10, walker lying still, all three
  tasks ≤ random — and a probe showed the penalty was not the cause: the
  adaptive scale (BC's) had been extended to it and left the value term at
  ~1/200 of its size. See `FBTrainer.act_l2_margin`. Now the value gradient
  is RAW, the hinge caps the corner, and the flush line prints `gA value
  -> total`: the value-term RMS of the actor gradient and the RMS after the
  penalty. Expect mean|a| near the margin; read `saturated` at eval.
  `--ortho` 100 and `--lr-b` 1e-5: the A2 winner (§18.6.1), which is the
  reference's own PAIR — each was null alone, together stand 1.57 / walk
  1.92 / run 1.63x random offline with every rung SIGNAL. That offline arm
  (`ortho100_lrb1e5_u`) is the number this run has to beat with no dataset.

⚠ Segmented, with STEP-STAMPED checkpoints — the driver overwrites its
`checkpoint_path` on every save, and §13 records a good early checkpoint
being destroyed that way. One agent, `N_SEGMENTS` calls into the driver.

⚠ `USE_ENV_CUDA_GRAPH=False`: the fields path's blocked Newton kernel does not
replay (same as the SAC walker script). The TRAIN step is captured.
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
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.online import FBOnlineAgent
from noeira.nn.core.tensor import Tensor
from noeira.data.store import TrajectoryStore
from noeira.data.resident import ResidentColumn, IDX_DT
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.deep_agents.training.driver_offpolicy import (
    run_offpolicy_train_batched,
)
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig


# ── the env ──────────────────────────────────────────────────────────────
# `walk` for the coverage readout only; FB never reads the reward.
comptime WalkerCfg = DMWalkerConfig[1.0]
comptime OBS: Int = DMWalkerModel.OBS_DIM      # 24 — dm_control's vector
comptime NACT: Int = DMWalkerModel.ACTION_DIM  # 6
comptime N_ENVS: Int = 256
comptime EnvT = Phyics3dBatchedEnv[
    DMWalkerModel, WalkerCfg, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

# ── the model — MUST match fb_eval_walker_online.mojo ────────────────────
comptime D: Int = 128
comptime BATCH: Int = 1024
comptime HID: Int = 1024
comptime CAP: Int = 1_000_000
comptime ZBUF: Int = 10_000
comptime F_IN = OBS + NACT + D
comptime A_IN = OBS + D
comptime FNet = Sequential[Linear[F_IN, HID], ReLU[HID], Linear[HID, D]]
comptime BNet = Sequential[
    Linear[OBS, 256], ReLU[256], Linear[256, D], LayerNormNoAffine[D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, NACT], Tanh[NACT]
]
# A3.5 (§18.7.2): half of every batch from the SAC ladder store when one is
# attached (`--store`), the other half from the online ring. With
# `--store ""` the same binary is the pure-online arm of run 3.
comptime EXPERT_ROWS: Int = 512
comptime Agent = FBOnlineAgent[
    FNet, BNet, ANet, OBS, NACT, D, BATCH, CAP, N_ENVS, ZBUF, EXPERT_ROWS
]
comptime NQ: Int = 9
comptime NV: Int = 9
comptime ScorerEnv = Phyics3dEnv[
    DMWalkerModel, WalkerCfg, DType.float64, False
]

# ── the run ──────────────────────────────────────────────────────────────
comptime TOTAL_ENV_STEPS: Int = 5_000_000
comptime SEGMENT_STEPS: Int = 250_000
comptime UPDATES_PER_ITER: Int = 8
comptime WARMUP_STEPS: Int = 25_600      # 100 iterations of random actions
comptime Z_HOLD: Int = 150
comptime EXPL_STD: Float64 = 0.2
# -1 = auto: 1.0 when an expert store is attached (BC on its rows only, the
# offline arm's value), 0 otherwise (BC toward the ring's own actions is
# circular — §18.7). `--bc` overrides.
comptime BC_WEIGHT: Float64 = -1.0
comptime STORE_PATH: StaticString = "fb_walker_all_sac.h5"
comptime ACT_L2: Float64 = 100.0
comptime ACT_MARGIN: Float64 = 0.8
comptime ORTHO_WEIGHT: Float64 = 100.0
comptime LR_B: Float64 = 1e-5
comptime MAX_GRAD_NORM: Float64 = 1.0
comptime DIAG_EVERY: Int = N_ENVS * 100  # 100 iterations
comptime PRINT_EVERY: Int = N_ENVS * 500
comptime USE_TRAIN_CUDA_GRAPH: Bool = True
# ⚠ NO PATH CONSTANTS HERE ANY MORE. Every path this driver writes comes
# from `RunContext` in `main`, so two runs cannot collide — see
# `core/run.mojo` and docs/PROJECT_LAYER_PLAN.md P0d.
comptime SEED: Int = 20260907

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
    var total = atol(_flag(String("--steps"), String(TOTAL_ENV_STEPS)))
    var ups = atol(_flag(String("--ups"), String(UPDATES_PER_ITER)))
    var warmup = atol(_flag(String("--warmup"), String(WARMUP_STEPS)))
    var z_hold = atol(_flag(String("--z-hold"), String(Z_HOLD)))
    var store_path = _flag(String("--store"), String(STORE_PATH))
    var expert_max = atol(_flag(String("--expert-max"), String(0)))
    var use_expert = store_path.byte_length() > 0
    var bc_w = atof(_flag(String("--bc"), String(BC_WEIGHT)))
    if bc_w < 0.0:
        bc_w = 1.0 if use_expert else 0.0
    var act_l2 = atof(_flag(String("--act-l2"), String(ACT_L2)))
    var act_margin = atof(_flag(String("--act-margin"), String(ACT_MARGIN)))
    var ortho_w = atof(_flag(String("--ortho"), String(ORTHO_WEIGHT)))
    var lr_b = atof(_flag(String("--lr-b"), String(LR_B)))
    var expl = atof(_flag(String("--expl-std"), String(EXPL_STD)))
    var tag = _flag(String("--tag"), String(""))
    # ⚠⚠ ONE OF **FIVE** COPIES OF THIS BLOCK IN THE FB FAMILY, all replaced by
    # `RunContext` together. Deriving three paths from a `--tag` a human has to
    # remember to vary is one forgotten flag away from a run silently
    # overwriting the previous one; `checkpoints/` holds 26 `fb_walker_*` files
    # because of it. The tag survives as the SLUG so sweep arms stay legible in
    # a directory listing — uniqueness now comes from the id, not the human.
    var run = RunContext(
        project=String("fb"),
        driver=String("examples/fb/fb_online_walker_gpu.mojo"),
        slug=String("fb-online-walker")
             + ("-" + tag if tag.byte_length() > 0 else ""),
        env=String("builtin:dm_control/walker-walk"),
    )
    run.set_tag(tag)
    var csv_path = run.metrics_path()
    print("run:", run.dir)
    if warmup < BATCH:
        raise Error("--warmup must be >= BATCH (" + String(BATCH) + ")")
    var seg = SEGMENT_STEPS if SEGMENT_STEPS < total else total
    var n_segments = (total + seg - 1) // seg

    seed(SEED)
    print("=" * 70)
    print("FB ONLINE — dm_control walker, batched GPU, no dataset")
    print("=" * 70)
    print("  OBS / NACT / D      =", OBS, "/", NACT, "/", D)
    print("  N_ENVS / BATCH      =", N_ENVS, "/", BATCH)
    print("  env steps           =", total, " in", n_segments, "segments of", seg)
    print("  updates / iteration =", ups, " (", ups * BATCH // N_ENVS, "samples per env step )")
    print("  warmup env steps    =", warmup)
    print("  z_hold / ZBUF       =", z_hold, "/", ZBUF)
    print("  expl_std / bc / act_l2@margin / ortho / lr_b =", expl, "/", bc_w, "/", act_l2, "@", act_margin, "/", ortho_w, "/", lr_b)
    print("  expert store        =", store_path if use_expert else "(none — pure online)",
          " rows/batch", EXPERT_ROWS if use_expert else 0)
    print("  CUDA graph (train)  =", USE_TRAIN_CUDA_GRAPH)
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
        logger.set_config("algorithm", "FB-online")
        logger.set_config("env", "dm_control/walker-walk (coverage readout)")
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("d", String(D))
        logger.set_config("batch", String(BATCH))
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
        logger.set_config("tag", tag)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = Agent.make(
            ctx,
            lr=3e-4,
            lr_b=lr_b,
            ortho_weight=ortho_w,
            max_grad_norm=MAX_GRAD_NORM,
            bc_weight=bc_w,
            act_l2_weight=act_l2,
            act_l2_margin=act_margin,
            learning_starts=warmup,
            action_scale=1.0,
            expl_std=expl,
            z_hold=z_hold,
            window_size=100,
            initial_episode_fill=0.0,
            seed=UInt64(SEED),
        )
        var env = EnvT(ctx)

        if use_expert:
            # ── the expert store, in the ENV'S observation layout ─────────
            # The store holds qpos/qvel; the batched env emits dm_control's
            # 24-D vector. `obs_at` is the one producer of that vector on the
            # CPU path (the eval uses it for the same reason) — feeding
            # `[qpos | qvel]` here would train fine and evaluate to noise.
            print("[expert] loading", store_path, "...")
            var store = TrajectoryStore(store_path)
            var n_all = store.n_rows()
            var n_rows = n_all if expert_max <= 0 or expert_max > n_all else expert_max
            var qpos = ResidentColumn[DType.float32].load(store, String("qpos"))
            var qvel = ResidentColumn[DType.float32].load(store, String("qvel"))
            var action = ResidentColumn[DType.float32].load(store, String("action"))
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
                if r % 250_000 == 0 and r > 0:
                    print("[expert]   ", r, "/", n_rows, "rows through obs_at")
            print("[expert] obs table:", n_rows, "rows in",
                  Float64(perf_counter_ns() - t0) / 1e9, "s")
            # ⚠ A constant table trains fine and evaluates to noise. Count the
            # dimensions that actually vary across rows; walker's 24-D vector
            # should move on all of them.
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
                raise Error(
                    "expert obs table: only " + String(moving) + " of "
                    + String(OBS) + " dims vary — obs_at is not producing the"
                    " env's observation"
                )
            var eact = Tensor.alloc(n_rows * NACT)
            for i in range(n_rows * NACT):
                eact.data[i] = Scalar[DT](Float64(action.host[i]))
            # Episode-safe next-row table, from the store's OWN index — the
            # same rule fb_train_gpu.mojo applies, for the same reason.
            var nh = ctx.enqueue_create_host_buffer[IDX_DT](n_rows)
            for r in range(n_rows):
                nh[r] = Scalar[IDX_DT](r + 1 if r + 1 < n_rows else r)
            var n_eps = store.episodes.n_episodes()
            var marked = 0
            for e in range(n_eps):
                var off = Int(store.episodes.ep_offset[e])
                var ln = Int(store.episodes.ep_len[e])
                if ln <= 0:
                    continue
                var last = off + ln - 1
                if last < n_rows:
                    nh[last] = Scalar[IDX_DT](last)
                    marked += 1
            print("[expert]", n_eps, "episodes,", marked, "boundaries inside the",
                  n_rows, "rows used")
            var nd = ctx.enqueue_create_buffer[IDX_DT](n_rows)
            ctx.enqueue_copy(nd, nh)
            eobs.upload(ctx)
            eact.upload(ctx)
            ctx.synchronize()
            agent.attach_expert(eobs^, eact^, nd^, n_rows)
            logger.set_config("expert_store", store_path)
            logger.set_config("expert_rows_used", String(n_rows))
            logger.set_config("expert_rows_per_batch", String(EXPERT_ROWS))
            print("[expert] attached:", n_rows, "rows,", EXPERT_ROWS, "of every",
                  BATCH, "batch rows; BC on those rows at", bc_w)

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
                rng_seed=UInt64(SEED + s),
                updates_per_step=ups,
                print_every=PRINT_EVERY,
                verbose=True,
                logger=logger_ptr,
                diag_every=DIAG_EVERY,
                episode_sync_every=32,
                base_step=done_steps,
                progress_label="fb-online",
            )
            var at = done_steps + this_seg
            var path = run.checkpoint_path(String("step_") + String(at))
            agent.save_state(path)
            var el = Float64(perf_counter_ns() - t_start) / 1e9
            print(
                "  [segment", s + 1, "/", n_segments, "]  env steps", at,
                "  train steps", agent.total_train_steps(),
                "  mean_ret(walk)", agent.mean_return(),
                "  replay", agent.size, "  ", Float64(at) / el, "env st/s",
                " ->", path,
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
