"""SAC training on Swimmer (CPU) via the new `SACAgent` facade.

Swimmer counterpart of `examples/half_cheetah/sac_half_cheetah_training.mojo`.
Uses the new `deep_agents/` surface:

  * `SAC[...]` — preset facade over `SACTrainer` + the single-env off-policy
    driver (canonical fused-`LinearReLU` `SACActorNet` / `SACCriticNet` +
    SAC's tuned defaults).
  * A `RunContext` (project `mujoco`): the checkpoint, `metrics.csv` and
    `run.kv` all live in `runs/<id>/`, and the monitor row carries the same id.
  * `run_logger(run)` — `metrics.csv` + the monitor, at every chunk boundary
    AND at the driver's `print_every` cadence.
  * Single-file checkpointing — `run.checkpoint_path("last")`, overwritten
    each chunk and uploaded through the run's artifact sink: actor + twin
    critics + their Adam states + `alpha_opt` ScalarAdam.

After training, the final checkpoint is reloaded into the same agent and a
greedy probe confirms the action vector reproduces dimension-by-dimension to
`|diff| < 1e-5`.

Swimmer (Phyics3dEnv, MuJoCo-style):
  * 8D observation (qpos[2:5] + qvel[0:5])
  * 2D continuous action (2 rotational motor torques)
  * Reward ≈ x_velocity - 0.0001·||action||²
  * No early termination (`TERMINATE_ON_UNHEALTHY=False`).

Run:
    pixi run mojo run -I . examples/swimmer/sac_swimmer_training.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.swimmer import Swimmer


# =============================================================================
# Architecture
# =============================================================================

comptime EnvT = Swimmer[DT, TERMINATE_ON_UNHEALTHY=False]
comptime OBS_DIM = EnvT.OBS_DIM  # 8
comptime ACT_DIM = EnvT.ACTION_DIM  # 2
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000

# Training duration. CPU single-env; drop NUM_STEPS to 20_000 for a smoke run.
comptime NUM_STEPS = 200_000
comptime PRINT_EVERY = 5_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000

# Actor + twin critics come from the `SAC[...]` preset (deep_agents.sac):
# the canonical fused-`LinearReLU` `SACActorNet` / `SACCriticNet`. Using the
# preset here keeps the CPU checkpoint layout identical to the GPU trainer's,
# so a checkpoint trained on either target loads in the other.


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (deep_agents) — Swimmer CPU + checkpoints + logger")
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
        driver=String("examples/swimmer/sac_swimmer_training.mojo"),
        slug=String("sac-swimmer"),
        env=String("builtin:mujoco/swimmer"),
    )
    var checkpoint_path = run.checkpoint_path(String("last"))
    print("  Run                =", run.dir)
    print("=" * 70)
    var logger = run_logger(run, buffer_size=200)
    logger.set_config("algorithm", "SAC")
    logger.set_config("env", "Swimmer")
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
    if final_avg > 100.0:
        print("EXCELLENT — swimming fast (mean > 100).")
    elif final_avg > 40.0:
        print("STRONG — learned to swim (mean > 40).")
    elif final_avg > 20.0:
        print("PROGRESS — early locomotion (mean > 20).")
    elif final_avg > 0.0:
        print("LEARNING — positive return (mean > 0).")
    else:
        print("EARLY — still exploring (mean < 0).")
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
