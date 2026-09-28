"""LeWM (nn) — evaluate a trained PushT world model (GPU).

Loads a checkpoint from `lewm_pusht_train_gpu.mojo` and runs the
teacher-forced action-awareness eval (legacy H6): score the REAL (expert)
actions vs BATCH-shuffled actions by latent-prediction MSE. PushT actions
are continuous, so this shuffle test — not the categorical CEM/shooter —
is the right action-awareness probe. An action-aware (non-collapsed) model
scores expert < shuffled; a collapsed one scores them alike.

Run (after training):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_eval_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_eval_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.experimental.lewm.eval import lewm_shuffled_eval
from noeira.envs.pusht import PushTOfflineSampler


# ── must match training ────────────────────────────────────────────────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14
comptime HIDDEN = 96
comptime ENC_HEADS = 4
comptime ENC_LAYERS = 2
comptime EMB = 96
comptime ENC_PROJ_H = 256
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 10
comptime SMOOTHED = 32
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 4
comptime PRED_FF = 256
comptime DEPTH = 6
comptime PRED_PROJ_H = 256
comptime SIG_PROJ = 1024
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime CKPT_PATH: String = "/tmp/lewm_pusht_world_model.txt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
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
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), CKPT_PATH), String("last"))
    print("=" * 70)
    print("LeWM nn — PushT world-model eval (GPU, H6 action-awareness)")
    print("=" * 70)
    var ctx = DeviceContext()

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading checkpoint", ckpt, "...")
    tr.load_params(ckpt)

    # one real window (device fp32 pixels + device actions)
    src.next_batch()
    var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())

    # expert actions need to be on HOST for the shuffle. D2H the device acts.
    var act_host = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    var act_dev = DeviceBuffer[DT](ctx, src.act_ptr(), B * ACTIN, owning=False)
    ctx.enqueue_copy(act_host, act_dev)
    ctx.synchronize()

    print("shuffled action-awareness eval ...")
    var r = lewm_shuffled_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
    ](
        tr, pix_t,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](act_host.unsafe_ptr()),
        ctx=ctx,
    )

    print()
    print("   action-aware (expert < shuffled_min):",
          "YES" if r[0] < r[2] else "no (collapsed/insufficient?)")

    _ = src^; _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
