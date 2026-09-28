"""SAC training on InvertedDoublePendulum (CPU) via the new `SACAgent` facade.

InvertedDoublePendulum counterpart of
`examples/half_cheetah/sac_half_cheetah_training.mojo`. Uses the new
`deep_agents/` surface:

  * `SACAgent[...]` — facade over `SACTrainer` + the single-env off-policy
    driver.
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor, at every chunk boundary
    AND at the driver's `print_every` cadence.
  * Single-file checkpointing — `run.checkpoint_path("last")`, overwritten
    each chunk and uploaded through the run's artifact sink.

After training, the final checkpoint is reloaded into the same agent and a
greedy probe confirms the action reproduces to `|diff| < 1e-5`.

InvertedDoublePendulum (Phyics3dEnv, MuJoCo-style):
  * 9D observation (cart_x, sin/cos of both pole angles, clipped velocities)
  * 1D continuous action (cart slider force)
  * Reward = alive bonus − distance/velocity penalties; episode ends when the
    tip drops (`TERMINATE_ON_UNHEALTHY=True`). Max return ≈ 9300.

Run:
    pixi run mojo run -I . examples/inverted_double_pendulum/sac_inverted_double_pendulum_training.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.inverted_double_pendulum import InvertedDoublePendulum


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = InvertedDoublePendulum[DT, TERMINATE_ON_UNHEALTHY=True]
comptime OBS_DIM = EnvT.OBS_DIM  # 9
comptime ACT_DIM = EnvT.ACTION_DIM  # 1
comptime HIDDEN = 128
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000

# Training duration. CPU single-env; drop NUM_STEPS to 20_000 for a smoke run.
comptime NUM_STEPS = 150_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 25_000

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac):
# the canonical fused-`LinearReLU` `SACActorNet` / `SACCriticNet`. Using the
# preset here keeps the CPU checkpoint layout identical to the GPU trainer's,
# so a checkpoint trained on either target loads in the other (and in the
# eval script).


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — InvertedDoublePendulum CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN             =", HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  PRINT_EVERY        =", PRINT_EVERY)
    print("  DIAG_EVERY         =", DIAG_EVERY)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)

    # ─── Run + logger ───────────────────────────────────────────────────────
    var run = RunContext(
        project=String("mujoco"),
        driver=String("examples/inverted_double_pendulum/sac_inverted_double_pendulum_training.mojo"),
        slug=String("sac-inverted-double-pendulum"),
        env=String("builtin:mujoco/inverted_double_pendulum"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run                =", run.dir)
    print("=" * 70)
    var logger = run_logger(run, buffer_size=200)
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "InvertedDoublePendulum")
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)

    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    # ─── Agent + env ─────────────────────────────────────────────────────
    # `SAC[target, OBS, ACT, BATCH, CAP, HIDDEN]` builds the SACAgent with
    # the fused default nets + SAC's tuned defaults (lr=3e-4, gamma=0.99,
    # tau=0.005, init_alpha=0.2, target_entropy=-ACT, …). Override only the
    # example-specific knobs; the rest come from the preset.
    var agent = SAC[
        "cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN
    ](
        window_size=100,
        initial_episode_fill=0.0,
    )
    var env = EnvT()

    # ─── Single train() call — auto-flush + auto-checkpoint ──────────────
    var t_start = perf_counter_ns()
    _ = agent.train_single[
        EnvT,
        L=RunLogger,
    ](
        env,
        NUM_STEPS,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=checkpoint_path,
        checkpoint_every=CHECKPOINT_EVERY,
        artifacts=artifacts,
        run_dir=run.dir,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    var total = NUM_STEPS
    var sent = logger.b.total_logged()
    finish_run(
        run, logger, artifacts,
        String("mean_return_100=") + String(agent.mean_return()),
    )
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps        =", total)
    print("  elapsed                =", elapsed_s, "s")
    print("  mean ep return (last 100) =", agent.mean_return())
    print("  episodes completed     =", agent.ep_count())
    print("  remote points sent     =", sent)
    print("  run record             =", run.kv_path())
    print("=" * 70)

    var final_avg = Float64(agent.mean_return())
    if final_avg > 9000.0:
        print("EXCELLENT — balancing reliably (mean > 9000).")
    elif final_avg > 5000.0:
        print("STRONG — mostly upright (mean > 5000).")
    elif final_avg > 1000.0:
        print("PROGRESS — learning to balance (mean > 1000).")
    elif final_avg > 100.0:
        print("LEARNING — some control (mean > 100).")
    else:
        print("EARLY — still falling fast (mean < 100).")
    print("=" * 70)

    # ─── Save/load round-trip smoke test ─────────────────────────────────
    var probe_obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    for d in range(OBS_DIM):
        probe_obs[d] = Scalar[DT](0.1 * Float64(d - OBS_DIM // 2))
    var act_before = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_before)

    agent.load(checkpoint_path)
    var act_after = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe_obs, act_after)

    print("Save/load round-trip on probe obs:")
    var ok = True
    for j in range(ACT_DIM):
        var diff = Float64(act_after[j] - act_before[j])
        if diff < 0:
            diff = -diff
        print(
            "  dim",
            j,
            " before =",
            act_before[j],
            " after =",
            act_after[j],
            " |diff| =",
            diff,
        )
        if diff > 1e-5:
            ok = False
    if ok:
        print("Round-trip OK (max |diff| < 1e-5 on every action dim).")
    else:
        print("Round-trip MISMATCH — investigate save/load semantics.")
    print("=" * 70)
