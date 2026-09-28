"""DreamerV3 training on CartPole (GPU) via the `DreamerV3Agent` facade.

GPU successor of `cartpole_dreamerv3_training.mojo` and counterpart of
`examples/humanoid/sac_humanoid_training_gpu.mojo`. The world model + actor-
critic train on-device (`train_target="gpu"`); the env steps on CPU and obs are
marshalled H2D inside `select_action` (the env=cpu / train=gpu cross-target
pattern — DreamerV3 is single-env with a belief carry, so it uses `train_single`
rather than the batched off-policy driver).

This exercises the DISCRETE GPU actor-critic path (categorical actor): the
imagination rollout samples a unimix categorical on host (policy logits
downloaded) and scatters the discrete policy gradient back — mirroring the CPU
path, so the WM forward stays CPU↔GPU bit-matched. The continue head learns
`latent(fall)→0` (terminal-obs stored), and the T_IMAG=15 + lr=1.5e-4 recipe
gives enough horizon to credit the slow cart drift → solves (mean_ret(10)=500).

Run:
    pixi run -e apple  mojo run -I . examples/cartpole/cartpole_dreamerv3_training_gpu.mojo  # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/cartpole/cartpole_dreamerv3_training_gpu.mojo  # NVIDIA GPU
"""

from max.gpu.host import DeviceContext
from std.random import seed
from std.time import perf_counter_ns

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.deep_agents.dreamerv3.agent import DreamerV3Agent
from noeira.envs.cartpole import CartPoleEnv


# =============================================================================
# Architecture (matches the CPU solving recipe; CPU↔GPU bit-matched WM)
# =============================================================================
comptime EnvT = CartPoleEnv[DT]
comptime OBS = 4
comptime ACT = 2
comptime DETER = 128
comptime H = 32
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "gpu",
    OBS,
    ACT,
    DETER,
    H,
    STOCH,
    CLASSES,
    BLOCKS,
    TOKEN,
    DEC_U,
    HU,
    VU,
    PU,
    BINS,
    B,
    T,
    T_IMAG,
    CAP,
    True,  # DISCRETE=True, train_target="gpu"
    OUT_INIT=Kaiming,  # full reward/critic output init (positive-reward optimism)
]

comptime NUM_STEPS = 150_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime EVAL_EVERY = 2500
comptime EVAL_EPISODES = 10
comptime EP_LEN = 500
comptime CHECKPOINT_EVERY = 25_000


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 (facade) — CartPole GPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS / ACT          =", OBS, "/", ACT)
    print("  DETER/STOCH/CLASSES=", DETER, "/", STOCH, "/", CLASSES)
    print("  T / T_IMAG         =", T, "/", T_IMAG)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Run + logger ────────────────────────────────────────────────
        var run = RunContext(
            project=String("classic-control"),
            driver=String("examples/cartpole/cartpole_dreamerv3_training_gpu.mojo"),
            slug=String("dreamerv3-cartpole-gpu"),
            env=String("builtin:classic-control/cartpole"),
        )
        var checkpoint_path = run.checkpoint_path(String("last"))
        print("  Run:", run.dir)
        var logger = run_logger(run, buffer_size=200)
        logger.set_config("algorithm", "DreamerV3")
        logger.set_config("env", "CartPole")
        logger.set_config("target", "gpu")
        logger.set_config("t_imag", String(T_IMAG))
        register_run(run, logger)
        var artifacts = sink_for_run(run.id, run.dir)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        # ─── Agent (GPU) + env (CPU; obs marshalled H2D in select_action) ──
        var agent = Ag.make(
            ctx=ctx,
            lr=Scalar[DT](1.5e-4),
            learning_starts=LEARN_START,
            warmup_steps=500,
            )
        var env = EnvT()

        # ─── Single train() call — auto-eval + auto-log + auto-checkpoint ──
        print("Starting GPU training...")
        print("-" * 70)
        var t_start = perf_counter_ns()
        # USE_TRAIN_CUDA_GRAPH=True: replay the WM+AC device-kernel sequence from
        # a captured CUDA graph on non-diag steps (Stage 3 — ~2-3x train-step
        # throughput, launch-bound). Bit-identical to eager (capture-parity gate
        # `tests/nn/test_dreamerv3_capture_parity.mojo`, ΔWM=ΔAC=0.0) so
        # convergence is unchanged; a no-op on non-NVIDIA (runs eagerly).
        var final_ret = agent.train_single[
            EnvT, L=RunLogger, USE_TRAIN_CUDA_GRAPH=True
        ](
            env,
            NUM_STEPS,
            learn_start=LEARN_START,
            train_every=TRAIN_EVERY,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            ep_len=EP_LEN,
            print_every=EVAL_EVERY,
            verbose=True,
            logger=logger_ptr,
            checkpoint_path=checkpoint_path,
            artifacts=artifacts,
            run_dir=run.dir,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        var sent = logger.b.total_logged()
        finish_run(
            run, logger, artifacts,
            String("eval_return=") + String(final_ret),
        )
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("-" * 70)
        print("=" * 70)
        print("Training complete")
        print("  total env_steps        =", NUM_STEPS)
        print("  elapsed                =", elapsed_s, "s")
        print("  FINAL mean_ret(", EVAL_EPISODES, ")  =", final_ret)
        print("  remote points sent     =", sent)
        if Float64(final_ret) >= 475.0:
            print("SOLVED — mean_ret >= 475.")
        elif Float64(final_ret) >= 200.0:
            print("STRONG — sustained balancing (>= 200).")
        elif Float64(final_ret) >= 50.0:
            print("LEARNING — climbing (>= 50).")
        else:
            print("EARLY — still exploring (< 50).")
        print("=" * 70)
