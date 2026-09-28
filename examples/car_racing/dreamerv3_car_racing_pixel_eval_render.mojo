"""DreamerV3 pixel-CarRacing — live COLOR eval for a trained checkpoint.

Loads a checkpoint from `dreamerv3_car_racing_pixel_training.mojo` and drives
CarRacing in the real top-down SDL3 COLOR scene while the agent acts on its
4×96×96 grayscale view (the same obs it trained on). Continuous control: the
agent outputs normalized [-1,1] [steer, gas, brake]; the env remaps gas/brake
to [0,1] internally (Gymnasium), so we pass the action straight through.

DreamerV3 is a closed-loop policy: each step the encoder+RSSM update a posterior
belief from the real frame, then the actor acts on it. So we `reset_belief()` at
the start of every episode and let `select_action` advance the belief. We act by
SAMPLING the policy (`explore=True`), matching how the DreamerV3 reference evals
— the deterministic mode degenerates early in training (constant hard steer →
spin); sampling reflects true on-policy behavior.

The agent identity below MUST match the training script (arch + DETER). For eval
RECON_SIGMOID is irrelevant — eval never decodes pixels (no recon, no
imagination), so the checkpoint loads regardless of which recon it trained with.

Run with:
    pixi run -e apple  mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_eval_render.mojo
    pixi run -e apple  mojo run -I . examples/car_racing/dreamerv3_car_racing_pixel_eval_render.mojo --ckpt <run_id>

Reads dreamerv3_carracing_pixel_gpu.ckpt, or the run/file given by `--ckpt`.
"""

from std.memory import alloc
from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.deep_agents.dreamerv3.agent import DreamerV3Agent
from noeira.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from noeira.envs.car_racing.car_racing_mb import CarRacingMB

# ── arch (MUST match dreamerv3_car_racing_pixel_training.mojo) ──
comptime C = 4
comptime IMG = 96
comptime BASE = 48
comptime OBS = C * IMG * IMG  # 36864
comptime ACT = 3
comptime DETER = 2048  # MUST match the training config (checkpoint compatibility)
comptime H = 256
comptime STOCH = 32
comptime CLASSES = 32
comptime BLOCKS = 8
comptime TOKEN = 1024
comptime DEC_U = 1024
comptime HU = 256
comptime VU = 256
comptime PU = 256
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 256  # eval only → tiny replay (params load from ckpt)

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, False, ENC, DEC,  # DISCRETE=False (continuous)
    # RECON_SIGMOID left default — eval never decodes, so it has no effect here.
]
comptime Env = CarRacingMB[DT, True, IMG]  # PIXEL_OBS=True, PIX_RES=96

comptime CHECKPOINT_PATH = "dreamerv3_carracing_pixel_gpu.ckpt"
comptime EVAL_EPISODES = 5
comptime FRAME_DELAY_MS = 20  # ~50 FPS
comptime MAX_STEPS = 1_000  # Gymnasium CarRacing-v3 max_episode_steps (env frames)
# MUST match the training FRAME_REPEAT — the agent decides once per ACTION_REPEAT
# env frames (holding the action), exactly as it was trained.
comptime ACTION_REPEAT = 4


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
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), String(CHECKPOINT_PATH)), String("last"))
    print("=" * 70)
    print("DreamerV3 CarRacing — checkpoint eval (color scene, CPU env)")
    print("=" * 70)
    print("  Checkpoint:", ckpt, "  Episodes:", EVAL_EPISODES)
    print()

    with DeviceContext() as ctx:
        var agent = Ag.make(ctx=ctx, action_scale=Scalar[DT](1.0))
        agent.load(ckpt)
        print("Checkpoint loaded. Starting live play...")

        var env = Env(max_steps=MAX_STEPS)
        _ = env.init_renderer()

        var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

        var ep = 0
        agent.reset_belief()
        var obs = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps = 0  # env frames
        var quit = False

        while env.is_renderer_open() and ep < EVAL_EPISODES and not quit:
            # one greedy decision, held for ACTION_REPEAT env frames (matches
            # training); render every sub-frame so the scene stays smooth.
            for i in range(OBS):
                ob[i] = obs[i]
            # SAMPLE the policy (explore=True), NOT the deterministic mode — the
            # DreamerV3 reference evaluates by sampling the actor. Early on the
            # mode degenerates (constant hard steer → spin); sampling reflects
            # the true on-policy behavior and tracks training episode_reward.
            agent.select_action(ob, ac, explore=True)  # normalized [-1,1], advances belief
            var act_list = List[Scalar[DT]](capacity=ACT)
            for a in range(ACT):
                act_list.append(ac[a])
            var ep_done = False
            for _r in range(ACTION_REPEAT):
                env.render_frame()  # real color scene
                var res = env.step_continuous_vec[DT](act_list)
                obs = res[0].copy()
                ep_return += res[1]
                ep_steps += 1
                env.renderer_delay(FRAME_DELAY_MS)
                if env.check_renderer_quit():
                    quit = True
                    break
                if res[2] or ep_steps >= MAX_STEPS:
                    ep_done = True
                    break

            if ep_done:
                ep += 1
                print(
                    "Race", ep, " return:", ep_return, " steps:", ep_steps,
                    " tiles:", env.tiles_visited, "/", env.track_length(),
                )
                agent.reset_belief()
                obs = env.reset_obs_list()
                ep_return = Scalar[DT](0.0)
                ep_steps = 0

        env.close_renderer()
        ob.free()
        ac.free()
        print("=" * 70)
        print("Eval complete.")
        print("=" * 70)
        _ = env^
        _ = agent^
