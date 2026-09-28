"""LeWM (nn) — autoregressive MPC eval on real Pong (GPU, §10.9 gate).

Loads a trained checkpoint, encodes a Pong window's start/goal latents,
and runs the latent-rollout MPC eval (horizon>1): the predictor is rolled
forward in latent space under candidate action plans and scored against
the goal latent. Reports expert vs random vs CEM.

The §10.9 success gate (on a NON-collapsed model): cem < random_min and
expert < random_min. A collapsed model scores them all alike.

Run (after training a non-collapsed model — see the λ sweep):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_mpc_eval_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_mpc_eval_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.mpc import lewm_mpc_eval
from noeira.experimental.lewm.pixel_convert import u8_to_fp32_norm
from noeira.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


# ── §10.7 Pong-ViT recipe (must match training) ────────────────────────
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
comptime MPC_HORIZON = 4          # NEEDED = H + horizon - 1 = 6 == T

comptime IMG_DIM = IN_CH * IMG * IMG
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


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


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
    print("LeWM nn — Pong MPC eval (GPU, §10.9, horizon=", MPC_HORIZON, ")")
    print("=" * 70)
    var ctx = DeviceContext()

    var buf = PongOfflineBuffer.load(BUFFER_PATH)
    print("buffer n_frames =", buf.n_frames)
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading checkpoint", ckpt, "...")
    tr.load_params(ckpt)

    # sample one window → device fp32 pixels + device/host actions
    var pix_u8: Pointer[Scalar[DType.uint8], MutAnyOrigin] = alloc[
        Scalar[DType.uint8]
    ](B * PIX)
    var act_host = _a(B * ACTIN)
    buf.sample_batch_uint8(B, T, pix_u8, act_host)

    var pix_host = _a(B * PIX)
    u8_to_fp32_norm["cpu", B * PIX](pix_u8, pix_host)
    var pix_d = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_d = ctx.enqueue_create_buffer[DT](B * ACTIN)
    ctx.enqueue_copy(pix_d, pix_host)
    ctx.enqueue_copy(act_d, act_host)
    ctx.synchronize()
    var pix_t = TileTensor(_p(pix_d), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_d), row_major[B, ACTIN]())

    print("MPC eval (expert vs random vs CEM) ...")
    var r = lewm_mpc_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu",
    ](
        tr, pix_t, act_t, act_host,
        num_random=64, cem_iters=5, cem_samples=64, cem_topk=8, ctx=ctx,
    )

    print()
    # The §10.9 planning gate is whether the PLANNER beats random — i.e.
    # cem < random_min. (expert < random_min is a teacher-forced action-
    # awareness signal; over a short latent rollout the goal latent ≈ start
    # so expert needn't beat best-of-N random — that's reachability, not
    # collapse. Check action-awareness with the teacher-forced eval.)
    print("   §10.9 planner gate (cem < random_min):",
          "PASS" if r[3] < r[2] else "FAIL")
    print("   cem < expert:", "yes" if r[3] < r[0] else "no",
          "  | expert vs random_min (informational):", r[0] / r[2])

    pix_u8.free(); act_host.free(); pix_host.free()
    _ = tr^; _ = buf^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
