"""Rainbow DQN CNN Atari Pong — Live Render Eval (pixel obs, deep_agents).

Loads a checkpoint saved by `rainbow_atari_pong_pixel_training_gpu.mojo` and
plays the real Atari 2600 Pong ROM (6502/TIA/RIOT emulation) in an SDL3 window
using deterministic greedy actions (NoisyLinear noise off), so you can watch
the trained CNN agent live. The agent sees the same 4×84×84 stacked grayscale
frames it trained on; the window renders the emulator's native 160×210 display.

This is the Atari counterpart to `rainbow_pong_pixel_eval_render.mojo` (which
evals on the *native* GPU Pong physics engine). The Atari emulator is CPU-only,
so the env steps on the CPU while the CNN q-net selects actions on the GPU.
The frame the agent's last `step_obs` rendered (`raw_frame_b`, the most recent
of the two max-pooled sub-frames) is blitted straight into the `AtariRenderer`
pixel buffer — no extra emulation, no display desync.

The q-net identity comes from the same `RainbowCNNConfig` preset the training
script used (Nature-CNN backbone + noisy dueling distributional heads, HIDDEN
512, 51 atoms), so the checkpoint's q-net + optimizer load cleanly. Only the
Pong-specific value support (V_MIN/V_MAX ±2) is overridden, matching training.
Replay capacity is irrelevant to the q-net params, so it's kept tiny here (the
checkpoint excludes the replay buffer anyway).

Window controls: P pauses, ESC/Q or window-close quits. The window closes on
quit or after EVAL_EPISODES games.

Requires the Pong ROM at `roms/pong.bin` (symlink to ale_py/roms/).

Run with (GPU env):
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_eval_render.mojo
    pixi run -e apple  mojo run -I . examples/arcade_games/rainbow_atari_pong_pixel_eval_render.mojo --ckpt <run_id>
"""

from std.memory import memcpy
from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT

from noeira.deep_agents.c51.config import RainbowCNNConfig, trainer_from_config

from noeira.envs.atari import AtariEnv
from noeira.envs.atari.games.registry import AtariGame
from noeira.envs.atari.flags import OBS_WIDTH, OBS_HEIGHT
from noeira.envs.atari.frame_render import FRAME_BUF_SIZE
from noeira.envs.atari.renderer import AtariRenderer


# =============================================================================
# Config — must match rainbow_atari_pong_pixel_training_gpu.mojo's q-net identity
# =============================================================================

comptime FRAMES = 4
comptime OBS_DIM = FRAMES * OBS_WIDTH * OBS_HEIGHT  # 4 * 84 * 84 = 28224
comptime NUM_ACTIONS = 6  # Pong minimal set (NOOP/FIRE/RIGHT/LEFT/R+F/L+F)

comptime NUM_ATOMS = 51
comptime HIDDEN = 512
comptime N_STEP = 3
comptime OBS_STORE_DT = DType.uint8

# Distributional support — must match training (brackets the discounted return,
# NOT the raw ±21 episode score).
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

comptime CKPT_PATH = "checkpoints/rainbow_atari_pong_pixel.ckpt"

# Eval-only knobs.
comptime EVAL_EPISODES = 5
comptime MAX_STEPS = 20_000

# Replay is unused at eval time and excluded from the checkpoint — keep it tiny.
# Capacity does not enter the q-net / optimizer params, so any value loads.
comptime EVAL_CAP = 256
comptime BATCH_SIZE = 32

# Pixel obs env (OBS_MODE=1) using the nn dtype, so step_obs emits List[DT].
comptime AtariPongPixel = AtariEnv[1, DT]

# Same preset the training script used → identical q-net, clean checkpoint load.
comptime EvalConfig = RainbowCNNConfig[
    "gpu", NUM_ACTIONS, BATCH_SIZE, EVAL_CAP,
    FRAMES, NUM_ATOMS, HIDDEN, N_STEP, OBS_STORE_DT,
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
    print("Rainbow Atari Pong — Live Render Eval (pixel obs)")
    print("=" * 70)
    print("  Checkpoint:", ckpt)
    print("  Episodes:", EVAL_EPISODES)
    print()

    with DeviceContext() as ctx:
        var trainer = trainer_from_config[EvalConfig](
            ctx=ctx,
            v_min=V_MIN,
            v_max=V_MAX,
        )

        trainer.load_state(ckpt)
        print("Checkpoint loaded. Starting live play...")

        # Deterministic greedy: zero out NoisyLinear exploration noise.
        trainer.set_noise_scale(Scalar[DT](0.0))

        # Pixel env (auto-loads roms/pong.bin) + SDL renderer for the display.
        # Each step advances 4 emulator frames (frame skip = 4), so cap the
        # display at 15 steps/s → 15 × 4 = 60 emulator FPS = real Atari speed.
        var env = AtariPongPixel(AtariGame.PONG)
        var renderer = AtariRenderer(fps=15)
        if not renderer.init_display():
            print("Failed to initialize display")
            return

        var ep: Int = 0
        var obs = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        while not renderer.should_quit and ep < EVAL_EPISODES:
            if not renderer.handle_events():
                break

            if not renderer.paused:
                var action = trainer.select_greedy_action(obs)
                var result = env.step_obs(action)
                obs = result[0].copy()
                ep_return += result[1]
                ep_steps += 1

                # Blit the frame this step rendered into the display buffer.
                memcpy(
                    dest=renderer.get_pixel_buffer(),
                    src=env.raw_frame_b.value(),
                    count=FRAME_BUF_SIZE,
                )

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

            # display_buffer_with_hud presents + caps the framerate (60 FPS).
            renderer.display_buffer_with_hud(
                Int(env.env.state.score),
                Int(env.env.state.lives),
                Int(env.env.state.frame_number),
            )

        renderer.close()
        env.close()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
