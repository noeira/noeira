"""DreamerV3 Atari Pong — live render eval for a trained checkpoint.

Loads a checkpoint from `dreamerv3_atari_pong_training.mojo` and plays the real
Atari 2600 Pong ROM (6502/TIA/RIOT emulation) in an SDL3 window. The agent sees
the same single 96×96 grayscale frame it trained on (OBS_MODE=3 — the RSSM
carries temporal state, no stacking); the window renders the emulator's native
160×210 display (the `raw_frame_b` the step rendered is blitted straight into
the AtariRenderer — no extra emulation).

DreamerV3 is a closed-loop policy: each step the encoder+RSSM update a posterior
belief from the real frame, then the actor acts on it. So we `reset_belief()` at
the start of every episode and let `select_action` advance the belief. We act by
SAMPLING the actor (`explore=True`), matching how the DreamerV3 reference evals —
the deterministic mode degenerates early in training; sampling reflects true
on-policy behavior. The env's internal frame-skip=4 means one decision = 4 ROM
frames (real Atari speed at ~15 decisions/s).

The agent identity below MUST match the training script — same TIER, pool conv
geometry, and raw-token width — or the checkpoint will not load. Requires the
Pong ROM at `roms/pong.bin` (`pixi run setup-roms`).

Window controls: P pauses, ESC/Q or window-close quits.

Run:
    pixi run -e apple  mojo run -I . examples/atari/dreamerv3_atari_pong_eval_render.mojo
    pixi run -e nvidia mojo run -I . examples/atari/dreamerv3_atari_pong_eval_render.mojo
    pixi run -e apple  mojo run -I . examples/atari/dreamerv3_atari_pong_eval_render.mojo --ckpt <run_id>

Reads dreamerv3_atari_pong_gpu.ckpt, or the run/file given by `--ckpt`.
"""

from std.memory import alloc, memcpy
from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.primitives.ops.swish_op import SwishOp
from noeira.deep_agents.dreamerv3.agent import DreamerV3Agent
from noeira.deep_agents.dreamerv3.nets_cnn import (
    DreamerEncoderCNNPool,
    DreamerDecoderCNNPool,
)
from noeira.envs.atari import AtariEnv
from noeira.envs.atari.games.registry import AtariGame
from noeira.envs.atari.frame_render import FRAME_BUF_SIZE
from noeira.envs.atari.renderer import AtariRenderer

# ── arch (MUST match dreamerv3_atari_pong_training.mojo, incl. TIER) ──
comptime C = 1  # single grayscale frame (reference parity — no stacking)
comptime IMG = 96
comptime TIER = "50m"  # "200m" | "50m" — checkpoints are tier-specific
comptime BASE = 64 if TIER == "200m" else 32
comptime OBS = C * IMG * IMG  # 9216
comptime ACT = 6
comptime DETER = 8192 if TIER == "200m" else 4096
comptime H = 1024 if TIER == "200m" else 512
comptime STOCH = 32
comptime CLASSES = 64 if TIER == "200m" else 32
comptime BLOCKS = 8
comptime TOKEN = 4 * BASE * (IMG // 16) * (IMG // 16)  # raw conv tokens
comptime UNITS = H
comptime DEC_U = H
comptime HU = H
comptime VU = H
comptime PU = H
comptime BINS = 255
comptime B = 16
comptime T = 16  # eval only → small replay window (params load from ckpt)
comptime T_IMAG = 15
comptime CAP = 256  # eval only → tiny replay

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNNPool[C, IMG, IMG, BASE, SwishOp]
comptime DEC = DreamerDecoderCNNPool[
    FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
]

comptime Ag = DreamerV3Agent[
    "gpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True, ENC, DEC,  # DISCRETE=True
    RECON_SIGMOID=True,  # must match training (decode = sigmoid)
    # OUT_INIT/NET_INIT left default: eval-only — ckpt load overwrites params.
]
comptime Env = AtariEnv[3, DT]  # OBS_MODE=3 (gray-96 single frame)

comptime CHECKPOINT_PATH = "dreamerv3_atari_pong_gpu.ckpt"
comptime EVAL_EPISODES = 5
comptime MAX_STEPS = 4000  # agent decisions per episode cap (Pong ends on 21 pts)


def _argmax(p: Pointer[Scalar[DT], MutAnyOrigin], n: Int) -> Int:
    var best = 0
    var bv = p[0]
    for i in range(1, n):
        if p[i] > bv:
            bv = p[i]
            best = i
    return best


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
    print("DreamerV3 Atari Pong — checkpoint eval (live SDL, CPU emulator)")
    print("=" * 70)
    print("  Checkpoint:", ckpt, "  Episodes:", EVAL_EPISODES)
    print("  TIER:", TIER, " DETER:", DETER, " TOKEN:", TOKEN)
    print()

    with DeviceContext() as ctx:
        var agent = Ag.make(ctx=ctx)
        agent.load(ckpt)
        print("Checkpoint loaded. Starting live play...")

        # Pixel env (auto-loads roms/pong.bin) + SDL renderer. Cap 15 decisions/s
        # → 15 × 4 = 60 emulator FPS = real Atari speed.
        var env = Env(AtariGame.PONG)
        var renderer = AtariRenderer(fps=15)
        if not renderer.init_display():
            print("Failed to initialize display")
            return

        var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

        var ep: Int = 0
        agent.reset_belief()
        var obs = env.reset_obs_list()
        var ep_return = Scalar[DT](0.0)
        var ep_steps: Int = 0

        while not renderer.should_quit and ep < EVAL_EPISODES:
            if not renderer.handle_events():
                break

            if not renderer.paused:
                for i in range(OBS):
                    ob[i] = obs[i]
                agent.select_action(ob, ac, explore=True)  # sample; advances belief
                var res = env.step_obs(_argmax(ac, ACT))
                obs = res[0].copy()
                ep_return += res[1]
                ep_steps += 1

                # Blit the frame this step rendered into the display buffer.
                memcpy(
                    dest=renderer.get_pixel_buffer(),
                    src=env.raw_frame_b.value(),
                    count=FRAME_BUF_SIZE,
                )

                if res[2] or ep_steps >= MAX_STEPS:
                    ep += 1
                    print(
                        "Game", ep, " return:", ep_return, " steps:", ep_steps,
                    )
                    agent.reset_belief()
                    obs = env.reset_obs_list()
                    ep_return = Scalar[DT](0.0)
                    ep_steps = 0

            renderer.display_buffer_with_hud(
                Int(env.env.state.score),
                Int(env.env.state.lives),
                Int(env.env.state.frame_number),
            )

        renderer.close()
        ob.free()
        ac.free()
        env.close()
        print("=" * 70)
        print("Eval complete.  (Pong: -21 = shutout loss, +21 = shutout win)")
        print("=" * 70)
        _ = env^
        _ = agent^
