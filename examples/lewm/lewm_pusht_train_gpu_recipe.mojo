"""LeWM (nn) — PushT JEPA training, REFERENCE RECIPE (CLS encoder, GPU).

The recipe retrain (reference audit items 1-3, docs/LEWM_REFERENCE_AUDIT.md):

  1. NO stop-gradient on the target (graph change in loss_graph.mojo —
     gradients flow through `tgt`, the paper's headline design).
  2. AdamW-style training: decoupled weight decay 1e-3, peak lr 5e-5 with
     linear warmup + cosine annealing, global grad clip 1.0 (the reference
     trainer config; our previous runs used flat Adam 1e-3 — 20× too hot).
  3. z-scored actions (dataset per-dim mean/std, the reference's
     get_column_normalizer). The WM now conditions on ~N(0,1) actions;
     planning must sample in z-space (Σ₀=I — the paper's CEM init) and
     de-normalize before execution: raw = z·std + mean, env = agent+raw·100.

Plus audit item 5 (architecture parity):
  4. SIGReg projections RESAMPLED every step (sigreg_resample=True — the
     reference draws fresh torch.randn projections per forward; a fixed A
     lets training game the sketch).
  5. ENC_FF_MULT=4 — ViT-Tiny's standard mlp_ratio (intermediate 768; our
     previous runs used 2 = half-width FFN).
  (Deferred from item 5: predictor final LayerNorm — its effect is largely
  absorbed by PredProj's immediate BatchNorm, and an optional graph node
  needs conditional-type-alias support Mojo lacks; predictor dropout 0.1 —
  regularization that matters at 10-epoch scale, invasive to thread.)

Writes a NEW checkpoint (z-action semantics — incompatible with the raw
CLS/mean-pool checkpoints). Saves periodically so partial runs are usable.
The run (project `lewm`) writes its checkpoint to
`runs/<id>/checkpoints/last.ckpt`, beside `metrics.csv` (loss, var_min,
gram_off) and `run.kv`.
Eval after: lewm_pusht_paper_protocol_gpu_recipe.mojo.

Run (NVIDIA; long):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_gpu_recipe.mojo
"""

from std.math import cos, pi
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


# ── paper-width dims (matches lewm_pusht_train_gpu_paper_cls.mojo) ────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
comptime HIDDEN = 192
comptime ENC_HEADS = 3
comptime ENC_LAYERS = 12
comptime EMB = 192
comptime ENC_PROJ_H = 2048
comptime ENC_FF_MULT = 4    # ViT-Tiny mlp_ratio 4 (reference; was 2)
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

# ── reference recipe knobs ──────────────────────────────────────────────
comptime STEPS: Int = 100000        # ~3× the 32k run; raise if time allows
comptime WARMUP_STEPS: Int = 2000
comptime PEAK_LR: Float64 = 5e-5    # reference: AdamW lr 5e-5
comptime WEIGHT_DECAY: Scalar[DT] = 1e-3
comptime MAX_GRAD_NORM: Scalar[DT] = 1.0
comptime LAM: Scalar[DT] = 0.09
comptime LOG_EVERY: Int = 500
comptime CKPT_EVERY: Int = 10000    # periodic saves → partial runs usable

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


def _lr_at(step: Int) -> Float64:
    """Linear warmup → cosine annealing to 0 (reference scheduler shape)."""
    if step < WARMUP_STEPS:
        return PEAK_LR * Float64(step + 1) / Float64(WARMUP_STEPS)
    var t = Float64(step - WARMUP_STEPS) / Float64(STEPS - WARMUP_STEPS)
    return PEAK_LR * 0.5 * (1.0 + cos(pi * t))


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT JEPA training, REFERENCE RECIPE (CLS, GPU)")
    print("=" * 70)
    print("no-stop-grad target | AdamW wd=", WEIGHT_DECAY,
          " peak lr=", PEAK_LR, " warmup", WARMUP_STEPS,
          "+cosine | clip", MAX_GRAD_NORM, "| z-scored actions")
    print("SIGReg resampled/step | ENC_FF_MULT=4 (ViT mlp_ratio)")
    print()

    var ctx = DeviceContext()
    var run = RunContext(
        project=String("lewm"),
        driver=String("examples/lewm/lewm_pusht_train_gpu_recipe.mojo"),
        slug=String("lewm-pusht-recipe"),
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
    print("opening PushT expert dataset (z-scored actions) ...")
    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=T, normalize_actions=True
    )
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(
        lam=LAM, lr=Scalar[DT](_lr_at(0)),
        max_grad_norm=MAX_GRAD_NORM, weight_decay=WEIGHT_DECAY,
        sigreg_resample=True, ctx=ctx,
    )

    print("training", STEPS, "steps ...")
    var last_loss = Float64(0.0)
    tr.reset_loss_accum()
    for s in range(STEPS):
        tr.opt.lr = Scalar[DT](_lr_at(s))
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
                  " lr=", _lr_at(s),
                  " loss=", wl, " var_min=", probes[0],
                  " gram_off=", probes[1])
        if (s + 1) % CKPT_EVERY == 0:
            tr.save_params(ckpt_path)
            announce_checkpoint(ckpt_path, artifacts, run.dir)
            print("   checkpoint saved →", ckpt_path, "(step", s + 1, ")")

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
    print("DONE — eval: lewm_pusht_paper_protocol_gpu_recipe.mojo")
    print("=" * 70)
