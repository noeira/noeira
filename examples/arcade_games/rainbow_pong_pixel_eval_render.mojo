"""Rainbow DQN CNN Pong — Live Render Eval (pixel obs, deep_agents).

Loads a checkpoint saved by `rainbow_pong_pixel_training_gpu.mojo` and plays
Pong in an SDL3 window using deterministic greedy actions (NoisyLinear noise
off), so you can watch the trained CNN agent live. The agent sees the same
4×84×84 stacked grayscale frames it trained on; the window renders the
underlying game state.

The CNN q-net identity below MUST match the training script (conv geometry,
HIDDEN, NUM_ATOMS, V_MIN, V_MAX). The checkpoint stores only the q-net +
optimizer + epsilon, so the replay buffer here is deliberately tiny.

Run with (GPU env):
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_pixel_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_pixel_eval_render.mojo
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_pong_pixel_eval_render.mojo --ckpt <run_id>

Reads checkpoints/rainbow_pong_pixel.ckpt (or `--ckpt`). The window closes on quit
(ESC / window close) or after EVAL_EPISODES games.
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT

from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.primitives.noisy_linear import NoisyLinear
from noeira.nn.primitives.dueling_head_c51 import DuelingHeadC51

from noeira.deep_agents.c51.trainer import C51Trainer
from noeira.deep_agents.training.blocks import NStepSampleStep
from noeira.deep_agents.data.any_per_replay import AnyPerReplay
from noeira.envs.arcade_games.pong import PongPixelEnv


# =============================================================================
# Config — must match rainbow_pong_pixel_training_gpu.mojo's q-net identity
# =============================================================================

comptime OBS_DIM = PongPixelEnv[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = PongPixelEnv[DType.float64].NUM_ACTIONS  # 3
comptime FRAMES = 4

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 1
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)
comptime HIT_REWARD = 0.0

comptime CKPT_PATH = "checkpoints/rainbow_pong_pixel.ckpt"

# Eval-only knobs.
comptime EVAL_EPISODES = 5
comptime FRAME_DELAY_MS = 16  # ~60 FPS
comptime MAX_STEPS = 20_000

# Replay is unused at eval time — keep it tiny (checkpoint excludes it).
comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

# Same Nature backbone + noisy dueling distributional heads as training.
comptime RainbowCNNNet = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84], ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20], ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9], ReLU[64 * 7 * 7],
    Flatten[64 * 7 * 7],
    LinearReLU[64 * 7 * 7, HIDDEN],
    NoisyLinear[HIDDEN, (1 + NUM_ACTIONS) * NUM_ATOMS],
    DuelingHeadC51[NUM_ACTIONS, NUM_ATOMS],
]

comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["gpu", OBS_DIM, 1, EVAL_CAP], BATCH_SIZE
]
comptime RainbowTrainer = C51Trainer[
    "gpu", SAMPLE, RainbowCNNNet, NUM_ATOMS, NUM_ACTIONS, True
]


def _flag(name: String, dflt: String) raises -> String:
    """Value of `--name X`, or `dflt` when the flag is absent."""
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def main() raises:
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), String(CKPT_PATH)), String("last"))
    print("=" * 70)
    print("Rainbow Pong — Live Render Eval (pixel obs)")
    print("=" * 70)
    print("  Checkpoint:", ckpt)
    print("  Episodes:", EVAL_EPISODES)
    print()

    with DeviceContext() as ctx:
        var trainer = RainbowTrainer.make(
            ctx=ctx,
            lr=Scalar[DT](6.25e-5),
            gamma=Scalar[DT](0.99),
            tau=Scalar[DT](0.005),
            epsilon=Scalar[DT](0.0),
            learning_starts=0,
            target_update_freq=500,
            max_grad_norm=Scalar[DT](10.0),
            per_alpha=Scalar[DT](0.5),
            per_beta=Scalar[DT](0.4),
            per_epsilon=Scalar[DT](1e-6),
            nstep=N_STEP,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        trainer.load_state(ckpt)
        print("Checkpoint loaded. Starting live play...")

        # Deterministic greedy: zero out NoisyLinear exploration noise.
        trainer.set_noise_scale(Scalar[DT](0.0))

        var env = PongPixelEnv[DT, HIT_REWARD]()
        _ = env.init_renderer()

        var ep: Int = 0
        var obs = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        while env.is_renderer_open() and ep < EVAL_EPISODES:
            env.render_frame()

            var action = trainer.select_greedy_action(obs)
            var result = env.step_obs(action)
            obs = result[0].copy()
            ep_return += result[1]
            ep_steps += 1

            var done = result[2]
            if done or ep_steps >= MAX_STEPS:
                ep += 1
                print(
                    "Episode",
                    ep,
                    "return:",
                    ep_return,
                    "steps:",
                    ep_steps,
                )
                obs = env.reset_obs_list()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

            env.renderer_delay(FRAME_DELAY_MS)

            if env.check_renderer_quit():
                break

        env.close_renderer()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
