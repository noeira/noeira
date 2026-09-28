"""SAC on dm_control `walker` (GPU, multi-env) — and the POLICY LADDER the FB
dataset is collected from.

Counterpart of `examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`, with
one structural difference that is the whole point of this script.

## Why this is not just "SAC on walker"

`docs/BFM_ZERO_SHOT_RL.md` §13 measured the limit of the M2 run: a
random-policy walker dataset contains no trajectory that ever stands or walks,
so `F`'s argmax over it is arbitrary and `pi_z` loses to random jitter. §6
component 1 ranks the fix — dumping the states visited across SAC training —
above every other collection lever, because the falls from EARLY training are
the coverage and the gait from LATE training is the signal, and one run
produces both.

So this script does not save one checkpoint. It trains in `N_SEGMENTS`
segments on ONE agent (nets + replay + optimizers persist across `train`
calls) and writes a STEP-STAMPED checkpoint after each, into the run's own
directory (`RunContext`, project `dm-control`; `metrics.csv` and `run.kv` sit
beside it, and each rung is uploaded through the run's artifact sink):

    runs/<id>/checkpoints/sac_dm_walker_stand.ckpt.00025000
    runs/<id>/checkpoints/sac_dm_walker_stand.ckpt.00050000
    ...

`examples/fb/collect_walker_sac.mojo` then rolls out every rung (point its
prefix at that run's `checkpoints/`) and writes one dataset spanning random →
expert, tagging each row with the rung it came from.

⚠ The ladder exists because the driver OVERWRITES `checkpoint_path` on every
save (`run_offpolicy_train_batched`, and `train`'s docstring now says so). A
single 600 k-step call with `checkpoint_every=25_000` would leave exactly one
file — the expert — and the early-training diversity would be gone. That is
not hypothetical: the same overwrite destroyed the good early states of an FB
run during M2 and cost a re-run.

## Why the ladder and not the replay buffer itself

The literal replay buffer stores OBSERVATIONS, and dm_control rewards read
`Data` (FK products, `xmat`, `subtree_linvel`), not the observation vector.
A dataset of observations can only ever be scored under the reward that was
labelled at collection time, which is the one thing zero-shot inference must
not assume — see the header of `deep_agents/fb/collect.mojo`. Re-rolling the
ladder writes `qpos`/`qvel` instead, so `reward_at` can relabel it under a
reward invented afterwards. It is also re-runnable at any dataset size without
retraining, and free of FIFO eviction.

## Tasks

    stand  MOVE_SPEED = 0.0    walk  MOVE_SPEED = 1.0    run  MOVE_SPEED = 8.0

Edit `TASK`, rebuild, run — only the selected branch is instantiated. All three
share the model, so the three runs produce three ladders over one body, which
is what makes a single FB dataset scorable under all three rewards.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/sac_dm_walker_training_gpu.mojo
    pixi run -e apple  mojo run -I . examples/dm_control/sac_dm_walker_training_gpu.mojo
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.deep_agents.primitives.stochastic_actor import StochasticActor
from noeira.deep_agents.sac import SACAgent
from noeira.deep_agents.training.blocks import UniformSampleGpuStep
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.dm_control.walker import DMWalkerModel, DMWalkerConfig


# ── pick ONE ─────────────────────────────────────────────────────────────
comptime TASK: StaticString = "walk"  # "stand" | "walk" | "run"

comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)
comptime WalkerCfg = DMWalkerConfig[MOVE_SPEED]

comptime OBS_DIM = DMWalkerModel.OBS_DIM
comptime ACT_DIM = DMWalkerModel.ACTION_DIM
comptime HIDDEN = 256

comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000
comptime N_ENVS = 32

# ══ LADDER GEOMETRY — `collect_walker_sac.mojo` MUST DECLARE THE SAME ═════
# The collector reconstructs rung filenames arithmetically as
# `(k+1) * SEGMENT_STEPS`, so a change here that is not mirrored there makes
# every rung "MISSING". The collector prints each path it failed to open, so
# the mismatch is one run away from obvious — but check both files first.
#
# ⚠ `SEGMENT_STEPS` is TOTAL env-steps across all envs, not per env. The
# driver's progress counter divides by `N_ENVS` to get iterations, so a
# segment shorter than `EPISODE_LEN * N_ENVS` cannot complete even one episode
# per env and the `mean_ret` printed per rung is stale from earlier segments —
# measured on the smoke run, where 2000 steps over 8 envs finished 0 episodes.
# Tying it to `EPISODE_LEN * N_ENVS` makes each rung exactly one episode deep,
# so its reported return is real.
comptime EPISODE_LEN = 1000  # dm_control's own MAX_STEPS
comptime SEGMENT_STEPS = EPISODE_LEN * N_ENVS  # 32 000
comptime N_SEGMENTS = 20  # → 640 k env-steps total
comptime NUM_STEPS = SEGMENT_STEPS * N_SEGMENTS

comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = SEGMENT_STEPS
comptime DIAG_EVERY = 1000

# dm_control rewards are in [0, 1] per step over 1000 steps, so an episode
# return is bounded by 1000 regardless of task — unlike HalfCheetah's unbounded
# forward-velocity reward. The bands below read against that ceiling.
comptime MAX_RETURN = 1000.0

comptime BatchedEnvT = Phyics3dBatchedEnv[
    DMWalkerModel, WalkerCfg, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def _stamped(prefix: String, step: Int) raises -> String:
    """Zero-padded to 8 digits so the ladder sorts lexicographically.

    `collect_walker_sac.mojo` reconstructs these names arithmetically rather
    than globbing, so the padding here and the padding there must agree.
    """
    var s = String(step)
    var pad = String("")
    for _ in range(8 - s.byte_length()):
        pad += "0"
    return prefix + ".ckpt." + pad + s


def main() raises:
    # ⚠ In a function body — a top-level `comptime assert` does not parse.
    # Without it a typo'd TASK falls through the ternary above to 8.0 (run)
    # and silently trains the wrong task under the right filename.
    comptime assert (
        TASK == "stand" or TASK == "walk" or TASK == "run"
    ), "TASK must be 'stand', 'walk' or 'run'"

    seed(42)
    print("=" * 70)
    print("SAC — dm_control walker", TASK, "(GPU, multi-env) + policy ladder")
    print("=" * 70)
    print("  TASK / MOVE_SPEED  =", TASK, "/", MOVE_SPEED)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  N_ENVS             =", N_ENVS)
    print("  SEGMENT_STEPS      =", SEGMENT_STEPS)
    print("  N_SEGMENTS         =", N_SEGMENTS)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  WARMUP_STEPS       =", WARMUP_STEPS)
    print("=" * 70)

    var run = RunContext(
        project=String("dm-control"),
        driver=String("examples/dm_control/sac_dm_walker_training_gpu.mojo"),
        slug=String("sac-dm-walker-") + String(TASK),
        env=String("builtin:dm_control/walker-") + String(TASK),
        seed=42,
    )
    # The ladder keeps its step-stamped names, now inside the run directory.
    var prefix = run.dir + "/checkpoints/sac_dm_walker_" + String(TASK)
    print("  Run                =", run.dir)

    with DeviceContext() as ctx:
        # ─── Logger ──────────────────────────────────────────────────────
        var logger = run_logger(run, buffer_size=64)
        logger.set_config("algorithm", "SAC")
        logger.set_config("env", String("dm_control/walker-") + String(TASK))
        logger.set_config("target", "gpu")
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(REPLAY_CAPACITY))
        logger.set_config("ladder_rungs", String(N_SEGMENTS))
        logger.set_config("segment_steps", String(SEGMENT_STEPS))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent + batched GPU env ─────────────────────────────────────
        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=0.2,
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=WARMUP_STEPS,
            window_size=100,
            initial_episode_fill=0.0,
        )
        var env = BatchedEnvT(ctx)

        print("Starting GPU training —", N_SEGMENTS, "ladder segments ...")
        print("-" * 70)
        var t_start = perf_counter_ns()

        # ─── Segmented loop — ONE agent, N checkpoints ───────────────────
        # `base_step` keeps the logger's x-axis cumulative; without it every
        # segment would restart the dashboard at 0 and the run would look like
        # 24 unrelated experiments.
        #
        # USE_ENV_CUDA_GRAPH=False: the fields path solves PYRAMIDAL contacts
        # with the one-env-per-block blocked Newton kernel on NVIDIA, and
        # replaying that capture illegal-addresses. Eager env stepping is
        # correct; only the per-step launch-collapse speedup is lost.
        for seg in range(N_SEGMENTS):
            var done_steps = seg * SEGMENT_STEPS
            _ = agent.train[
                BatchedEnvT,
                N_ENVS=N_ENVS,
                USE_TRAIN_CUDA_GRAPH=True,
                USE_ENV_CUDA_GRAPH=False,
                L=RunLogger,
            ](
                env,
                SEGMENT_STEPS,
                rng_seed=UInt64(42 + seg),
                updates_per_step=N_ENVS,
                print_every=PRINT_EVERY,
                verbose=True,
                logger=logger_ptr,
                diag_every=DIAG_EVERY,
                episode_sync_every=32,
                base_step=done_steps,
            )
            var at = done_steps + SEGMENT_STEPS
            var path = _stamped(prefix, at)
            agent.save(path)
            announce_checkpoint(path, artifacts, run.dir)
            print(
                "  [rung", seg + 1, "/", N_SEGMENTS, "]  step", at,
                "  mean_ret", agent.mean_return(), " ->", path,
            )

        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        finish_run(
            run, logger, artifacts,
            String("mean_return_100=") + String(agent.mean_return()),
        )
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete —", TASK)
        print("  total env_steps           =", NUM_STEPS)
        print("  elapsed                   =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed        =", agent.ep_count())
        print("  ladder rungs written      =", N_SEGMENTS)
        print("  ladder prefix             =", prefix + ".ckpt.*")
        print("  run record                =", run.kv_path())
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        var frac = final_avg / MAX_RETURN
        if frac > 0.8:
            print("EXCELLENT — near the dm_control ceiling (>0.8 x 1000).")
        elif frac > 0.5:
            print("STRONG — solved the task (>0.5 x 1000).")
        elif frac > 0.2:
            print("PROGRESS — partial competence (>0.2 x 1000).")
        else:
            print("WEAK — check the run before collecting from this ladder.")
        print("")
        print("⚠ The ladder is only useful if the LAST rungs are competent AND")
        print("  the first are not. A run that plateaus at rung 2 gives FB no")
        print("  gradient of behaviour to cover — that is the M2 failure again.")
        print("")
        print("Next:")
        print("  pixi run mojo run -I . examples/fb/collect_walker_sac.mojo")
        print("=" * 70)
