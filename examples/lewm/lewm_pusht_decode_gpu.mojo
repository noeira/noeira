"""LeWM (nn) — train the reconstruction DECODER probe on PushT (GPU).

Diagnostic only (paper §"Decoder (Visualization Only)"): the LeWM JEPA has
no decoder; this trains a lightweight transformer decoder on the **frozen**
paper-width world model to visualize what the pooled `emb` representation
retains. The decoder = learnable per-patch query tokens that cross-attend to
the (replicated) global `emb` through `DEC_LAYERS` residual-MLP layers, then
linearly project to 16×16×3 pixel patches that are un-patchified to a
224×224 RGB image (decoder patch size 16, independent of the encoder's 14).

Pipeline per step (B·T = 96 frames):
  WindowSource → pixels/actions → frozen WM forward → emb (B·T, EMB)
  patchify(pixels) → target patches → decoder.train_step(emb, target)
Every VIZ_EVERY steps a side-by-side original-vs-reconstruction PPM grid is
written to /tmp (open in Preview). The encoder is never updated.

Loads the paper-width checkpoint `lewm_pusht_paper.ckpt`.

Run (after the paper-width train run, NVIDIA — 224² is too heavy for Apple):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_decode_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_decode_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import RunContext, register_run, resolve_checkpoint
from noeira.core.run_session import finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.experimental.lewm.decoder import patchify, unpatchify
from noeira.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from noeira.envs.pusht import PushTOfflineSampler
from noeira.render.image_writer import save_reconstruction_grid


# ── frozen WM config — must match lewm_pusht_train_gpu_paper.mojo ──────
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
comptime WM_CKPT: String = "lewm_pusht_paper.ckpt"

# ── decoder probe config ────────────────────────────────────────────────
comptime PATCH_D = 16                       # paper decoder patch size
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)   # 196 query tokens
comptime PATCH_PX = IN_CH * PATCH_D * PATCH_D        # 768
comptime DEC_HID = 192
comptime DEC_FF = 4 * DEC_HID
comptime DEC_LAYERS = 4
comptime DEC_BATCH = B * T                  # 96 frames per batch

comptime STEPS = 20_000
comptime PRINT_EVERY = 500
comptime VIZ_EVERY = 4_000
comptime N_VIZ = 8                          # frames per reconstruction grid

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD,
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]
comptime Decoder = LeWMDecoderTrainer[
    EMB, DEC_HID, N_Q, PATCH_PX, DEC_FF, DEC_LAYERS, DEC_BATCH, "gpu"
]


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
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), WM_CKPT), String("last"))
    # ⚠ THE DECODER PROBE IS ITSELF A TRAINING RUN — it fits weights and keeps
    # them — so it opens a run like any other: the decoder lands in
    # `runs/<id>/checkpoints/`, beside its own metrics.csv and run.kv.
    var run = RunContext(
        project=String("lewm"),
        driver=String("examples/lewm/lewm_pusht_decode_gpu.mojo"),
        slug=String("lewm-pusht-decoder"),
        env=String("builtin:pusht"),
        dataset=ckpt,
    )
    var logger = run_logger(run)
    logger.set_config("algorithm", "LeWM decoder probe")
    logger.set_config("wm_checkpoint", ckpt)
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var dec_ckpt = run.checkpoint_path(String("decoder"))
    print("run:", run.dir)
    print("=" * 70)
    print("LeWM nn — PushT reconstruction DECODER probe (frozen WM, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading frozen WM checkpoint", ckpt, "...")
    wm.load_params(ckpt)

    var dec = Decoder.make(lr=Scalar[DT](1e-3), ctx=ctx)
    print("  decoder:", DEC_LAYERS, "layers, hid", DEC_HID, ", queries", N_Q,
          ", patch", PATCH_D, "→", PATCH_PX, "px,  batch", DEC_BATCH)

    # device target patches + host viz staging
    var tgt_dev = ctx.enqueue_create_buffer[DT](DEC_BATCH * N_Q * PATCH_PX)
    var recon_host = ctx.enqueue_create_host_buffer[DT](
        DEC_BATCH * N_Q * PATCH_PX
    )
    var recon_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    var orig_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    ctx.synchronize()

    print("training decoder", STEPS, "steps (encoder frozen) ...")
    dec.reset_loss_accum()
    for step in range(1, STEPS + 1):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        # frozen encoder forward → emb node (B, T·EMB) == (B·T, EMB) flat
        _ = wm.eval_loss(pix_t, act_t)
        ref emb_tensor = wm.graph.node_output["emb"]()
        var emb_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
            emb_tensor.dev.value().unsafe_ptr()
        )
        var emb_t = TileTensor(emb_ptr, row_major[DEC_BATCH, EMB]())
        # target patches from the same frames (CHW, normalized [0,1])
        patchify["gpu", DEC_BATCH, IN_CH, IMG, PATCH_D](
            ctx, src.pix_ptr(), _p(tgt_dev)
        )
        var tgt_t = TileTensor(
            _p(tgt_dev), row_major[DEC_BATCH, N_Q * PATCH_PX]()
        )
        _ = dec.train_step(emb_t, tgt_t)

        if step % PRINT_EVERY == 0:
            var ml = dec.read_loss_accum()
            dec.reset_loss_accum()
            print("   step", step, "/", STEPS, " recon_mse=", ml)

        if step % VIZ_EVERY == 0 or step == STEPS:
            # reconstruct the current batch's emb → patches → image; dump
            # the first N_VIZ frames as original|reconstruction pairs.
            dec.recon_into(
                emb_t,
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    recon_host.unsafe_ptr()
                ),
            )
            unpatchify["cpu", N_VIZ, IN_CH, IMG, PATCH_D](
                None,
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    recon_host.unsafe_ptr()
                ),
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    recon_img.unsafe_ptr()
                ),
            )
            # D2H the first N_VIZ original frames (CHW [0,1]). Mirror the
            # working read idiom: Pointer dst + non-owning device src.
            var orig_dev = DeviceBuffer[DT](
                ctx,
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](src.pix_ptr()),
                N_VIZ * IMG_DIM, owning=False,
            )
            ctx.enqueue_copy(
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    orig_img.unsafe_ptr()
                ),
                orig_dev,
            )
            ctx.synchronize()
            var path = "/tmp/lewm_pusht_recon_" + String(step) + ".ppm"
            save_reconstruction_grid(
                path,
                orig_img.unsafe_ptr(),
                recon_img.unsafe_ptr(),
                n=N_VIZ, height=IMG, width=IMG, channels=IN_CH,
                vmin=0.0, vmax=1.0,
            )

    dec.save_params(dec_ckpt)
    announce_checkpoint(dec_ckpt, artifacts, run.dir)
    print("decoder weights →", dec_ckpt)
    finish_run(run, logger, artifacts)
    _ = src^; _ = wm^; _ = dec^
    print("=" * 70)
    print("DONE — reconstruction grids in /tmp/lewm_pusht_recon_*.ppm")
    print("=" * 70)
