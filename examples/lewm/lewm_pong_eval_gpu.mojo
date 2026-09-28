"""LeWM (nn) — evaluate a trained Pong world model (GPU, §10.9).

Loads a checkpoint produced by `lewm_pong_train_gpu.mojo`, samples a
context window from the offline buffer, and runs the teacher-forced
action-awareness eval: scores the EXPERT (recorded) actions vs random vs
CEM by latent-prediction MSE. A healthy, action-aware world model scores
  expert < random_min   (it can tell good actions from bad)
  cem    <= random_min  (the planner finds good actions in latent space)
A COLLAPSED model (var_min→0, gram_off→1) scores expert ≈ random — it
can't distinguish actions, so it's useless for planning. This is the
quantitative read on the collapse the training probes flagged.

Run (after training):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_eval_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_eval_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.memory import alloc
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.eval import lewm_action_awareness_eval
from noeira.experimental.lewm.pixel_convert import u8_to_fp32_norm
from noeira.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


# ── §10.7 Pong-ViT recipe (must match the training run) ────────────────
comptime IN_CH = 4
comptime IMG = 84
comptime PATCH = 14
comptime HIDDEN = 128
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 1
comptime EMB = 128
comptime ENC_PROJ_H = 64
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 3
comptime SMOOTHED = 16
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 64
comptime DEPTH = 6
comptime PRED_PROJ_H = 256
comptime SIG_PROJ = 256       # must match training
comptime SIG_KNOTS = 5
comptime B = 16

comptime IMG_DIM = IN_CH * IMG * IMG     # 28224 == PONG_FRAME_BYTES
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime BUFFER_PATH: String = "/tmp/lewm_pong_buffer.bin"
comptime CKPT_PATH: String = "/tmp/lewm_pong_world_model.txt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


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
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), CKPT_PATH), String("last"))
    print("=" * 70)
    print("LeWM nn — Pong world-model eval (GPU, §10.9 action-awareness)")
    print("=" * 70)
    var ctx = DeviceContext()

    var buf = PongOfflineBuffer.load(BUFFER_PATH)
    print("buffer n_frames =", buf.n_frames)

    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading checkpoint", ckpt, "...")
    tr.load_params(ckpt)

    # Sample one context window → host fp32 pixels + host one-hot actions.
    var pix_u8: Pointer[Scalar[DType.uint8], MutAnyOrigin] = alloc[
        Scalar[DType.uint8]
    ](B * PIX)
    var act_host = _a(B * ACTIN)
    buf.sample_batch_uint8(B, T, pix_u8, act_host)
    var pix_host = _a(B * PIX)
    u8_to_fp32_norm["cpu", B * PIX](pix_u8, pix_host)

    print("eval (expert vs random vs CEM) ...")
    var r = lewm_action_awareness_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
    ](
        tr, pix_host, act_host,
        num_random=32, cem_iters=5, cem_samples=64, cem_topk=8,
        ctx=ctx,
    )

    print()
    print("   expert/random_min =", r[0] / r[2],
          "  (< 1.0 ⇒ action-aware; ≈ 1.0 ⇒ collapsed/useless)")

    pix_u8.free(); act_host.free(); pix_host.free()
    _ = tr^
    _ = buf^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
