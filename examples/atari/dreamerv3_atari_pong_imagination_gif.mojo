"""DreamerV3 Atari Pong — imagination accuracy GIF (GPU decode).

Discrete-pixel twin of `examples/car_racing/dreamerv3_car_racing_imagination_gif.mojo`.
Loads a checkpoint from the Pong training run, collects one episode (sampling the
actor), then asks the world model to "dream": seed the posterior belief from CTX
real frames, roll the RSSM PRIOR forward open-loop on the recorded actions (no
further observations), and DECODE each imagined latent back to pixels. The decode
runs on-device (reusing the live training GPU graphs, B=1); only the small
per-step decoded frame is brought back to host (`openloop_decode_gpu`).

Output = an animated GIF, one frame per horizon step, three panels:

    [ REAL | RECON | IMAGINED ]

  * REAL     — ground-truth frame from the env.
  * RECON    — teacher-forced posterior decode (the decode UPPER BOUND — if this
               is blurry the decoder/representation is the limit, not dynamics).
  * IMAGINED — open-loop prior decode (what imagination actually trains on). If it
               tracks REAL the world model is faithful; if it drifts while RECON
               stays sharp, the dynamics are the bottleneck.

Obs is a 4×96×96 grayscale stack (C=4); the panels show the NEWEST frame
(channel C-1). GIF encoding is pure Mojo (`save_frame_sequence_gif`) — no Python.

Run (NVIDIA, after training has written a checkpoint; needs roms/pong.bin):
    pixi run -e nvidia mojo run -I . \\
        examples/atari/dreamerv3_atari_pong_imagination_gif.mojo [--ckpt <run_id>]
"""

from std.memory import alloc
from std.random import seed
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
from noeira.render.image_writer import save_frame_sequence_gif

# ── arch (MUST match the training run that WROTE the checkpoint) ──
# C is the frame-stack depth and selects the env obs mode:
#   C=1 ↔ AtariEnv OBS_MODE=3 (gray-96 single frame — atari100k-aligned cfg)
#   C=4 ↔ AtariEnv OBS_MODE=4 (gray-96 4-frame stack — older debug runs)
# A checkpoint from one C cannot load into the other (conv/OBS shapes differ).
comptime C = 1
comptime IMG = 96
comptime TIER = "50m"  # MUST match the checkpoint's training TIER ("200m" | "50m")
comptime BASE = 64 if TIER == "200m" else 32
comptime OBS = C * IMG * IMG  # 9216
comptime ACT = 6
comptime DETER = 8192 if TIER == "200m" else 4096
comptime H = 1024 if TIER == "200m" else 512
comptime STOCH = 32
comptime CLASSES = 64 if TIER == "200m" else 32
comptime BLOCKS = 8
comptime TOKEN = 4 * BASE * (IMG // 16) * (IMG // 16)  # 9216 (pool geometry)
comptime UNITS = H  # decoder bspace-stem MLP width (= hidden, per tier)
comptime DEC_U = H
comptime HU = H
comptime VU = H
comptime PU = H
comptime BINS = 255
comptime B = 16
comptime T = 16
comptime T_IMAG = 15
comptime CAP = 256  # we never train here → tiny replay (params load from ckpt)

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNNPool[C, IMG, IMG, BASE, SwishOp]
comptime DEC = DreamerDecoderCNNPool[
    FEATIN, DETER, C, IMG, IMG, BASE, UNITS, SwishOp
]

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
    True,  # DISCRETE=True
    ENC,
    DEC,
    RECON_SIGMOID=True,  # must match training (decode = sigmoid)
]
comptime OBS_MODE = 3 if C == 1 else 4  # keyed to C (see comment above)
comptime Env = AtariEnv[OBS_MODE, DT]

comptime CHECKPOINT_PATH = "dreamerv3_atari_pong_gpu.ckpt"
comptime GIF_PATH = "dreamerv3_atari_pong_imagination.gif"

comptime CTX = 5  # real context frames to seed the belief
comptime HOR = 45  # imagination horizon (max GIF frames)
comptime NEWCH = (C - 1) * IMG * IMG  # newest-frame offset within the stacked OBS

# triptych layout
comptime SEP = 2
comptime WC = 3 * IMG + 2 * SEP
comptime HC = IMG


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
    print("DreamerV3 Atari Pong — imagination GIF (GPU decode)")
    print("  CTX", CTX, " HOR", HOR, " OBS", OBS, "(", C, "x", IMG, "x", IMG, ")")
    print("=" * 70)
    seed(42)

    with DeviceContext() as ctx:
        var agent = Ag.make(ctx=ctx)
        print("loading checkpoint", ckpt, "...")
        agent.load(ckpt)

        # ── collect one episode (closed-loop, sampling the actor) ──
        var env = Env(AtariGame.PONG)
        var robs = alloc[Scalar[DT]](
            (CTX + HOR + 1) * OBS
        ).as_unsafe_any_origin()
        var ract = alloc[Scalar[DT]]((CTX + HOR) * ACT).as_unsafe_any_origin()
        var ob = alloc[Scalar[DT]](OBS).as_unsafe_any_origin()
        var ac = alloc[Scalar[DT]](ACT).as_unsafe_any_origin()

        agent.reset_belief()
        var obs = env.reset_obs_list()
        for i in range(OBS):
            robs[i] = obs[i]
        var steps = CTX + HOR
        var collected = 0
        print("collecting episode (up to", steps, "steps)...")
        for t in range(steps):
            for i in range(OBS):
                ob[i] = obs[i]
            # sample the actor (one-hot into `ac`); record the one-hot the WM
            # dynamics conditions on, and argmax for the discrete env step.
            agent.select_action(ob, ac, explore=True)
            for a in range(ACT):
                ract[t * ACT + a] = ac[a]
            var res = env.step_obs(_argmax(ac, ACT))
            obs = res[0].copy()
            for i in range(OBS):
                robs[(t + 1) * OBS + i] = obs[i]
            collected = t + 1
            if res[2]:  # env terminated early
                print("  episode terminated early at step", collected)
                break

        var eff_hor = collected - CTX
        if eff_hor < 1:
            raise Error(
                "episode too short to trace (collected "
                + String(collected)
                + " < CTX+1)"
            )
        print("  collected", collected, "steps → eff_hor", eff_hor)

        # ── GPU open-loop decode (imagined) + teacher-forced (recon) ──
        var ol = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        var tf = alloc[Scalar[DT]](HOR * OBS).as_unsafe_any_origin()
        print("decoding imagination on GPU...")
        agent.trainer.openloop_decode_gpu(robs, ract, CTX, eff_hor, ol, tf)

        # ── compose [REAL | RECON | IMAGINED] grayscale triptych per step ──
        var comp = alloc[Scalar[DType.float32]](eff_hor * HC * WC)
        var sepval = Float32(0.12)
        for h in range(eff_hor):
            var fbase = h * HC * WC
            for p in range(HC * WC):
                comp[fbase + p] = sepval
            for y in range(IMG):
                var row = fbase + y * WC
                var rb = (CTX + h) * OBS + NEWCH + y * IMG
                var ob_ = h * OBS + NEWCH + y * IMG
                for x in range(IMG):
                    comp[row + x] = Float32(robs[rb + x])
                    comp[row + IMG + SEP + x] = Float32(tf[ob_ + x])
                    comp[row + 2 * IMG + 2 * SEP + x] = Float32(ol[ob_ + x])

        save_frame_sequence_gif(
            GIF_PATH,
            comp,
            eff_hor,
            HC,
            WC,
            channels=1,
            fps=10,
            loop=True,
            vmin=0.0,
            vmax=1.0,
        )

        ob.free()
        ac.free()
        robs.free()
        ract.free()
        ol.free()
        tf.free()
        comp.free()
        print("=" * 70)
        print("DONE — open ", GIF_PATH)
        print("  panels: [ REAL | RECON | IMAGINED ]")
        print("=" * 70)
        _ = env^
        _ = agent^
