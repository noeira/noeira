"""LeWM (nn) — eval the PAPER-WIDTH PushT world model (GPU, H6).

Loads the `lewm_pusht_train_gpu_paper.mojo` checkpoint and runs the
shuffled action-awareness eval (expert vs BATCH-shuffled actions). Config
MUST match the paper-width trainer. See that file for the predictor
attention-expansion fidelity note.

Run (after the paper-width train run):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_eval_gpu_paper.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_eval_gpu_paper.mojo --ckpt <run_id>
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


# ── must match lewm_pusht_train_gpu_paper.mojo ────────────────────────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14
comptime HIDDEN = 192
comptime ENC_HEADS = 3
comptime ENC_LAYERS = 12
comptime EMB = 192
comptime ENC_PROJ_H = 2048
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 10
comptime SMOOTHED = 32
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 16
comptime PRED_DIM_HEAD = 64     # expanded attention, must match training
comptime PRED_FF = 2048
comptime DEPTH = 6
comptime PRED_PROJ_H = 2048
comptime SIG_PROJ = 2048
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime CKPT_PATH: String = "lewm_pusht_paper.ckpt"

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD,
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
    print("LeWM nn — PushT PAPER-WIDTH eval (GPU, H6 action-awareness)")
    print("=" * 70)
    var ctx = DeviceContext()

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading checkpoint", ckpt, "...")
    tr.load_params(ckpt)

    src.next_batch()
    var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
    var act_host = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    var act_dev = DeviceBuffer[DT](ctx, src.act_ptr(), B * ACTIN, owning=False)
    ctx.enqueue_copy(act_host, act_dev)
    ctx.synchronize()

    print("shuffled action-awareness eval ...")
    var r = lewm_shuffled_eval[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
        PRED_DIM_HEAD,
    ](
        tr, pix_t,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](act_host.unsafe_ptr()),
        ctx=ctx,
    )
    print()
    print("   action-aware (expert < shuffled_min):",
          "YES" if r[0] < r[2] else "no")

    _ = src^; _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
