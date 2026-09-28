"""LeWM (nn) — train the JEPA world model on real PushT pixels (GPU).

Reproduces the legacy `lewm_pusht_pixel_train_gpu.mojo` on nn. PushT is the
paper's main benchmark: 224×224 RGB demonstrations from the HuggingFace
`quentinll/lewm-pusht` expert dataset (auto-downloaded + cached on first run,
~13 GB compressed → ~15-25 GB h5).

Pipeline (all real, HWC→CHW handled in the bridge):
  PushTOfflineSampler (HDF5 expert demos, HWC uint8, INPUT_LAYOUT_HWC=True)
    → WindowSource (sample window → H2D → u8_hwc_to_chw_norm permute+÷255)
    → LeWMTrainer.train_step (encoder → AR predictor → MSE + SIGReg)

Config — the legacy "scaled baseline" `LeWMPushTViTConfig[batch=16, t=6, h=3,
depth=6]` (hidden=emb=96, enc_heads=4, enc_layers=2, pred_heads=4, pred_ff=256,
sig_num_proj=1024). This is EXACTLY reproducible on nn: its predictor
head_dim = EMB/HEADS = 96/4 = 24 matches the legacy pred_dim_head=24.

  NOTE — the paper-WIDTH config (hidden=emb=192, pred_heads=16, pred_dim_head=64,
  pred_ff=2048, enc_layers=12) wants an INDEPENDENT pred_dim_head=64, decoupled
  from emb/heads. nn's ARPredictor (locked decision #5) ties head_dim=EMB/HEADS
  (=192/16=12 ≠ 64), so the paper-width predictor isn't bit-faithful here. For a
  paper-width run, set PRED_HEADS=3 (→ head_dim 192/3=64) to match the paper's
  per-head width instead.

  SIGReg P=1024 ≫ EMB=96 (heavily over-determined), so λ=0.09 (the paper value)
  should hold without the collapse that bit Pong at P=64 — see the Pong λ sweep.

The run (project `lewm`) writes its checkpoint to
`runs/<id>/checkpoints/last.ckpt`, beside `metrics.csv` (loss, var_min,
gram_off) and `run.kv`.

Run (NVIDIA; first run downloads the dataset):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_gpu.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.envs.pusht import PushTOfflineSampler


# ── scaled-baseline PushT-ViT recipe ───────────────────────────────────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14           # 224 // 14 == 16 → 256 patches
comptime HIDDEN = 96
comptime ENC_HEADS = 4        # head_dim = 96/4 = 24
comptime ENC_LAYERS = 2
comptime EMB = 96
comptime ENC_PROJ_H = 256
comptime ENC_FF_MULT = 2
comptime T = 6
comptime ACT = 10             # FRAMESKIP(5) × ACTION_DIM(2)
comptime SMOOTHED = 32
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 4       # head_dim = 96/4 = 24 == legacy pred_dim_head
comptime PRED_FF = 256
comptime DEPTH = 6
comptime PRED_PROJ_H = 256
comptime SIG_PROJ = 1024      # ≫ EMB=96, over-determined (paper value)
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime IMG_DIM = IN_CH * IMG * IMG     # 150528
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime STEPS: Int = 8000
comptime LOG_EVERY: Int = 200
comptime LAM: Scalar[DT] = 0.09
comptime LR: Scalar[DT] = 1e-3

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
# HWC source → WindowSource needs C/FRAME for the permute branch.
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]


def main() raises:
    print("=" * 70)
    print("LeWM nn — PushT JEPA world model training (GPU, real pixels)")
    print("=" * 70)
    print("recipe: 224x224x3, patch=14, EMB=", EMB, " DEPTH=", DEPTH,
          " B=", B, " T=", T, " ACT=", ACT, " SIG_PROJ=", SIG_PROJ)
    print()

    var ctx = DeviceContext()

    var run = RunContext(
        project=String("lewm"),
        driver=String("examples/lewm/lewm_pusht_train_gpu.mojo"),
        slug=String("lewm-pusht"),
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
    print("opening PushT expert dataset (downloads on first run) ...")
    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(lam=LAM, lr=LR, ctx=ctx)

    print("training", STEPS, "steps ...")
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
    print("saving world-model checkpoint →", ckpt_path)
    tr.save_params(ckpt_path)
    announce_checkpoint(ckpt_path, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("final_loss=") + String(last_loss),
    )

    _ = src^
    _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
