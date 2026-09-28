"""DreamerV3 pixel-CarRacing — imagination accuracy GIF (GPU decode).

Loads a checkpoint from the GPU pixel training run, collects one greedy episode,
then asks the world model to "dream": seed the posterior belief from CTX real
frames, roll the RSSM PRIOR forward open-loop on the recorded actions (no further
observations), and DECODE each imagined latent back to pixels. The decode runs
on-device (reusing the live training GPU graphs, B=1); only the small per-step
decoded frame is brought back to host (`openloop_decode_gpu`).

The output is an animated GIF, one frame per horizon step, three panels:

    [ REAL | RECON | IMAGINED ]

  * REAL      — ground-truth frame from the env (what actually happened).
  * RECON     — teacher-forced: the model re-observes each real frame (posterior)
                and decodes it. The decode UPPER BOUND — if this is blurry the
                decoder/representation is the limit, not the dynamics.
  * IMAGINED  — open-loop: prior dynamics only, decoded. This is what imagination
                actually trains on. If IMAGINED tracks REAL the world model is
                faithful; if it drifts/blurs while RECON stays sharp, the
                dynamics are the bottleneck (the classic model-exploitation gap).

Only the newest frame of the 4-frame grayscale stack (channel C-1) is shown.
GIF encoding is pure Mojo (`save_frame_sequence_gif`) — no Python, no SDL.

Run (NVIDIA, after the training run has written a checkpoint):
    pixi run -e nvidia mojo run -I . \\
        examples/car_racing/dreamerv3_car_racing_imagination_gif.mojo [--ckpt <run_id>]
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
    DreamerEncoderCNN,
    DreamerDecoderCNN,
)
from noeira.envs.car_racing.car_racing_mb import CarRacingMB
from noeira.render.image_writer import save_frame_sequence_gif

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
comptime CAP = 256  # we never train here → tiny replay (params load from ckpt)

comptime FEATIN = STOCH * CLASSES + DETER
comptime ENC = DreamerEncoderCNN[C, IMG, IMG, BASE, TOKEN, SwishOp]
comptime DEC = DreamerDecoderCNN[FEATIN, C, IMG, IMG, BASE, SwishOp]

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
    False,
    ENC,
    DEC,  # DISCRETE=False (continuous)
    RECON_SIGMOID=True,  # RECON_SIGMOID — must match the training config (decode = sigmoid)
]
comptime Env = CarRacingMB[DT, True, IMG]  # PIXEL_OBS=True, PIX_RES=96

comptime CHECKPOINT_PATH = "dreamerv3_carracing_pixel_gpu.ckpt"
comptime GIF_PATH = "dreamerv3_carracing_imagination.gif"

comptime CTX = 5  # real context frames to seed the belief
comptime HOR = 45  # imagination horizon (max GIF frames)
comptime NEWCH = (C - 1) * IMG * IMG  # newest-frame channel offset within OBS

# triptych layout
comptime SEP = 2
comptime WC = 3 * IMG + 2 * SEP
comptime HC = IMG


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
    print("DreamerV3 pixel-CarRacing — imagination GIF (GPU decode)")
    print(
        "  CTX", CTX, " HOR", HOR, " OBS", OBS, "(", C, "x", IMG, "x", IMG, ")"
    )
    print("=" * 70)
    seed(42)

    with DeviceContext() as ctx:
        var agent = Ag.make(ctx=ctx, action_scale=Scalar[DT](1.0))
        print("loading checkpoint", ckpt, "...")
        agent.load(ckpt)

        # ── collect one greedy episode (closed-loop) ──
        var env = Env()
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
        print("collecting greedy episode (up to", steps, "steps)...")
        for t in range(steps):
            for i in range(OBS):
                ob[i] = obs[i]
            agent.select_greedy_action(ob, ac)
            var act_list = List[Scalar[DT]](capacity=ACT)
            for a in range(ACT):
                ract[t * ACT + a] = ac[a]
                act_list.append(ac[a])
            var res = env.step_continuous_vec[DT](act_list)
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
        print("  panels: [ REAL | RECON | IMAGINED ]  (newest stack frame)")
        print("=" * 70)
        _ = env^
        _ = agent^
