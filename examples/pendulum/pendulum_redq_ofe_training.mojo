"""REDQ-OFE training on Pendulum V1 via the REDQOFEAgent facade.

User-facing entry point for the deep_agents REDQ-OFE port. Shows
the one-line `REDQOFE6` preset constructor (Design F) which bundles
the OFE feature extractor (6-block DenseNet-style branches), the
REDQ N=2/M=2 critic ensemble, and SAC-shape (UTD=1/POLICY_DELAY=1)
update cadence.

Architecture (matching `OFENet-main/gins/` HalfCheetah/Hopper preset):
  - State branch: 6 × DenseBlock[Linear+LayerNorm+SiLU+skip-concat],
                  per_unit=8 (smaller than the paper's 40 — fast
                  compile/run on a 3-D Pendulum obs)
  - Action branch: same shape, IN = φ(s) + action
  - Predictor: Linear[φ(s,a) → predicted next-obs]
  - Actor: 2-layer MLP on φ(s) → (mean, log_std)
  - Critic: 2 × 2-layer MLP on φ(s, a) → Q

Training:
  - 5k env steps
  - aux MSE loss runs once per train_step (legacy REDQ-OFE pattern)
  - critic + actor + α step on the same UTD=1 cadence as SAC

Run:
    pixi run mojo run -I . examples/pendulum/pendulum_redq_ofe_training.mojo

Reference numbers from the smoke test (seed=42, 5k steps,
HIDDEN=64, PER_UNIT=8):
    step 4000 → mean_ret(10) ≈ -752
    step 5000 → mean_ret(10) ≈ -420
"""

from std.random import seed

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.deep_agents.redq_ofe import REDQOFE6
from noeira.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 128
comptime REPLAY_CAPACITY = 20_000
comptime HIDDEN = 64
comptime PER_UNIT = 8
comptime TOTAL_TIMESTEPS = 5_000
comptime WARMUP = 500


def main() raises:
    seed(42)
    print("=" * 70)
    print("nn REDQ-OFE (REDQOFEAgent facade) — Pendulum V1 (CPU)")
    print("=" * 70)

    var agent = REDQOFE6[
        "cpu", OBS, ACT, BATCH, REPLAY_CAPACITY, HIDDEN, PER_UNIT,
    ](
        actor_lr=3e-4,
        critic_lr=1e-3,
        ofe_lr=3e-4,
        alpha_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        init_alpha=0.2,
        target_entropy=-1.0,
        learning_starts=WARMUP,
        window_size=10,
        initial_episode_fill=-1250.0,
    )
    var env = PendulumEnv[DT]()

    # ─── Run: one directory per run, and the monitor row shares its id ────
    var run = RunContext(
        project=String("classic-control"),
        driver=String("examples/pendulum/pendulum_redq_ofe_training.mojo"),
        slug=String("redq-ofe-pendulum"),
        env=String("builtin:classic-control/pendulum"),
    )
    var logger = run_logger(run)
    logger.set_config("algorithm", "REDQ-OFE")
    logger.set_config("per_unit", String(PER_UNIT))
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("batch", String(BATCH))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()
    print("Run:", run.dir)

    var ep_returns = agent.train_single[L=RunLogger](
        env,
        total_timesteps=TOTAL_TIMESTEPS,
        print_every=1_000,
        verbose=True,
        logger=logger_ptr,
    )

    print("=" * 70)
    var final_mean = agent.mean_return()
    print("Final mean ep return (last 10):", final_mean)
    print("Episodes completed:            ", agent.ep_count())
    print("ep_returns list length:        ", len(ep_returns))
    print("Total inner train steps:       ", agent.total_train_steps())
    if final_mean > -200.0:
        print("EXCELLENT — solved swing-up (>-200).")
    elif final_mean > -500.0:
        print("SUCCESS — substantially learned (>-500).")
    elif final_mean > -1000.0:
        print("PROGRESS — learning (>-1000).")
    else:
        print("EARLY — still exploring (<-1000).")
    print("=" * 70)

    # The checkpoint lands in the run, and is uploaded if a monitor is set.
    var ckpt = run.checkpoint_path(String("last"))
    agent.save(ckpt)
    announce_checkpoint(ckpt, artifacts, run.dir)
    print("Saved checkpoint to:", ckpt)

    var eval_mean = agent.eval(env, num_episodes=5)
    print("Greedy eval mean return (5 eps):", eval_mean)
    finish_run(
        run, logger, artifacts,
        String("eval_return=") + String(eval_mean)
        + " mean_return_10=" + String(final_mean),
    )
    _ = logger  # lifetime extender for logger_ptr
    print("Run record:", run.kv_path())
