"""LeWM (nn) — PushT JEPA training, PAPER-WIDTH with the CLS-TOKEN encoder.

Identical to `lewm_pusht_train_gpu_paper.mojo` EXCEPT the encoder is the
CLS-token variant (`LeWMEncoderCLS`, passed as the trailing `ENC` param of
LeWMTrainer) instead of mean-pooling. Motivation: the closed-loop probe
showed the mean-pooled latent under-encodes the small agent/pusher (washed
out across 256 patches), which capped closed-loop control; a [CLS] token can
attend selectively to control-relevant patches (the action-conditioned
prediction objective rewards encoding the pusher, since actions move it).

This is the WM for a SECOND closed-loop attempt. Writes a SEPARATE checkpoint
so the mean-pooled WM is preserved for comparison. The run (project `lewm`) writes its checkpoint to
`runs/<id>/checkpoints/last.ckpt`, beside `metrics.csv` (loss, var_min,
gram_off) and `run.kv`.

Validation gates after this run (cheap → expensive):
  1. decoder probe on this WM — does the agent dot now reconstruct?
     (re-point lewm_pusht_decode_gpu.mojo at this checkpoint + CLS encoder)
  2. offline continuous-CEM eval — still cem ≪ random?
  3. closed-loop — does the block now track to goal?

Run (NVIDIA; long, like the paper-width run):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_gpu_paper_cls.mojo
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.encoder import LeWMEncoderCLS
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.envs.pusht import PushTOfflineSampler


# ── paper-width recipe (matches lewm_pusht_train_gpu_paper.mojo) ──────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14            # → 256 patches (+1 CLS = 257 tokens)
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
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
comptime PRED_DIM_HEAD = 64
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

comptime STEPS: Int = 32000
comptime LOG_EVERY: Int = 200
comptime LAM: Scalar[DT] = 0.09
comptime LR: Scalar[DT] = 1e-3
# Global grad-norm clip (standard ViT/transformer training). The mean-pooled
# WM trained stably without it, but the single-token CLS readout concentrates
# the encoder gradient and blew up at ~step 1800 (loss→thousands, emb var→100s);
# clipping caps the per-step update so an outlier batch can't explode it.
comptime MAX_GRAD_NORM: Scalar[DT] = 1.0

comptime EncCLS = LeWMEncoderCLS[
    IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
    ENC_PROJ_H, ENC_FF_MULT,
]
comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD, EncCLS,
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT JEPA training (GPU, PAPER-WIDTH, CLS-token)")
    print("=" * 70)
    print("encoder: CLS-token, hidden=", HIDDEN, " heads=", ENC_HEADS,
          " layers=", ENC_LAYERS, " tokens=", N_PATCHES + 1, " | emb=", EMB)
    print()

    var ctx = DeviceContext()
    var run = RunContext(
        project=String("lewm"),
        driver=String("examples/lewm/lewm_pusht_train_gpu_paper_cls.mojo"),
        slug=String("lewm-pusht-paper-cls"),
        env=String("builtin:pusht"),
        dataset=String("lewm_pusht"),
        device=String(ctx.name()),
    )
    print("run:", run.dir)
    var ckpt_path = run.checkpoint_path(String("last"))
    var logger = run_logger(run)
    logger.set_config("algorithm", "LeWM")
    logger.set_config("env", "PushT")
    logger.set_config("steps", String(STEPS))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    print("opening PushT expert dataset ...")
    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(
        lam=LAM, lr=LR, max_grad_norm=MAX_GRAD_NORM, ctx=ctx
    )

    print("training", STEPS, "steps (CLS encoder) ...")
    var last_loss = Float64(0.0)
    tr.reset_loss_accum()
    for s in range(STEPS):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = tr.train_step(pix_t, act_t)
        if (s + 1) % LOG_EVERY == 0:
            var wl = tr.read_loss_accum()
            tr.reset_loss_accum()
            var probes = tr.collapse_probes()
            last_loss = Float64(wl)
            var ln = List[String]()
            var lv = List[Float64]()
            ln.append(String("loss"))
            lv.append(Float64(wl))
            ln.append(String("var_min"))
            lv.append(Float64(probes[0]))
            ln.append(String("gram_off"))
            lv.append(Float64(probes[1]))
            logger.log_scalars(ln, lv, s + 1)
            print("   step", s + 1, "/", STEPS,
                  " loss=", wl, " var_min=", probes[0],
                  " gram_off=", probes[1])

    print()
    print("saving →", ckpt_path)
    tr.save_params(ckpt_path)
    announce_checkpoint(ckpt_path, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("final_loss=") + String(last_loss),
    )
    _ = src^
    _ = tr^
    print("=" * 70)
    print("DONE — next: decoder-probe this WM for agent encoding")
    print("=" * 70)
