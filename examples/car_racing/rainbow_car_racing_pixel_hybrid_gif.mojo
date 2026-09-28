"""Rainbow CNN CarRacing — Record a GIF of a HYBRID-trained checkpoint.

Loads a checkpoint from `rainbow_car_racing_pixel_hybrid_training.mojo` and drives
CarRacing in the real top-down SDL3 COLOR scene while the agent acts on its
4x84x84 grayscale view, recording the color scene to a GIF.

IMPORTANT: a hybrid checkpoint was trained on the CPU `CarRacingMB` env (faithful
Gymnasium track), so it MUST be eval'd on that same env — NOT the GPU
`CarRacingPixel` env, whose embedded track differs (the agent would be
out-of-distribution there). This runs the CPU env directly: `render_frame()`
draws the color scene; `step_action_pixel()` provides the pixel obs the CNN saw
in training.

The CNN q-net identity below MUST match the hybrid training script.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_gif.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_gif.mojo
    pixi run -e apple  mojo run -I . examples/car_racing/rainbow_car_racing_pixel_hybrid_gif.mojo --ckpt <run_id>

Reads checkpoints/rainbow_car_racing_pixel_hybrid.ckpt, or the run/file given by `--ckpt`.
Writes gifs/rainbow_car_racing.gif.
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
from noeira.envs.car_racing import CarRacingMB, CarRacingPixel


# =============================================================================
# Config — must match rainbow_car_racing_pixel_hybrid_training.mojo
# =============================================================================

comptime OBS_DIM = CarRacingPixel[DType.float64].OBS_DIM  # 28224
comptime NUM_ACTIONS = CarRacingPixel[DType.float64].NUM_ACTIONS  # 5
comptime FRAMES = 4

comptime NUM_ATOMS = 101
comptime HIDDEN = 512
comptime N_STEP = 3
comptime V_MIN = Scalar[DT](-100.0)
comptime V_MAX = Scalar[DT](200.0)

comptime CKPT_PATH = "checkpoints/rainbow_car_racing_pixel_hybrid.ckpt"
comptime GIF_PATH = "gifs/rainbow_car_racing.gif"

comptime GIF_EPISODES = 1
comptime GIF_FPS = 25
comptime GIF_FRAME_SKIP = 2  # record every 2nd frame (50 sim FPS → 25 GIF FPS)
comptime MAX_STEPS = 1_000  # Gymnasium CarRacing-v3 max_episode_steps

comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

comptime RainbowCNNNet = Sequential[
    Conv2D[FRAMES, 32, 8, 4, 0, 84, 84],
    ReLU[32 * 20 * 20],
    Conv2D[32, 64, 4, 2, 0, 20, 20],
    ReLU[64 * 9 * 9],
    Conv2D[64, 64, 3, 1, 0, 9, 9],
    ReLU[64 * 7 * 7],
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
    print("Rainbow CarRacing — HYBRID checkpoint GIF export (color scene)")
    print("=" * 70)
    print("  Checkpoint:", ckpt, "  Episodes:", GIF_EPISODES)
    print("  Output:", GIF_PATH)
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
        print("Checkpoint loaded. Recording...")
        trainer.set_noise_scale(Scalar[DT](0.0))  # deterministic greedy

        var env = CarRacingMB[DT](max_steps=MAX_STEPS)
        _ = env.init_renderer()
        env.start_recording(String(GIF_PATH), fps=GIF_FPS, skip=GIF_FRAME_SKIP)

        var ep = 0
        var obs = env.reset_pixel()
        var ep_return = Scalar[DT](0.0)
        var ep_steps = 0

        while env.is_renderer_open() and ep < GIF_EPISODES:
            env.render_frame()  # real color scene (captured into the GIF)

            var action = trainer.select_greedy_action(obs)
            var result = env.step_action_pixel(action)
            obs = result[0].copy()
            ep_return += Scalar[DT](result[1])
            ep_steps += 1

            if result[2] or ep_steps >= MAX_STEPS:
                ep += 1
                print(
                    "Race",
                    ep,
                    " return:",
                    ep_return,
                    " steps:",
                    ep_steps,
                    " tiles:",
                    env.tiles_visited,
                    "/",
                    env.track_length(),
                )
                obs = env.reset_pixel()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

            if env.check_renderer_quit():
                break

        env.stop_recording()
        env.close_renderer()
        print("=" * 70)
        print("Saved:", GIF_PATH)
        print("=" * 70)
