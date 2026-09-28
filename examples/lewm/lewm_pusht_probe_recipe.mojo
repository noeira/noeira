"""LeWM recipe-WM light probes — Apple-runnable checkpoint health checks.

Two cheap diagnostics on a (possibly mid-training) RECIPE checkpoint,
sized so they run on the Apple laptop while the NVIDIA box keeps
training. Copy the checkpoint over first:

  scp <nvidia-box>:/tmp/lewm_pusht_recipe_world_model.txt lewm_pusht_recipe.ckpt
  # (or from lewm_pusht_recipe.ckpt if the box runs the renamed driver)

1. SHUFFLED ACTION-AWARENESS EVAL (~a minute): prediction loss under the
   expert actions vs batch-shuffled actions. Healthy WM: expert clearly
   below shuffled (it learned action-conditioned dynamics). The single
   best cheap predictor of planning quality.

2. DECODER RECONSTRUCTION PROBE (the "reconstruction images"): trains a
   small pixel decoder on the FROZEN WM latents and writes
   original-vs-reconstruction grids to /tmp/lewm_pusht_recipe_recon_*.ppm
   (open with Preview). What to look for: block CLEARLY reconstructed at
   the right pose, and — the CLS retrain's whole point — the agent/pusher
   dot visible. Grids are written progressively (every VIZ_EVERY steps),
   so Ctrl-C anytime after the first grid keeps the images. DEC_STEPS=0
   skips this part.

First run on a fresh Mac downloads the lewm-pusht dataset (~13 GB HTTP →
~47 GB on disk — check free space first; resumable, with a progress bar
and the output kept out of the OS page cache).

Run (Apple):
  pixi run -e apple mojo run -I . examples/lewm/lewm_pusht_probe_recipe_apple.mojo
  pixi run -e apple mojo run -I . examples/lewm/lewm_pusht_probe_recipe.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.encoder import LeWMEncoderCLS
from noeira.experimental.lewm.eval import lewm_shuffled_eval
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.experimental.lewm.decoder import patchify, unpatchify
from noeira.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from noeira.envs.pusht import PushTOfflineSampler
from noeira.render.image_writer import save_reconstruction_grid


# ── must match lewm_pusht_train_gpu_recipe.mojo ───────────────────────
comptime IN_CH = 3
comptime IMG = 224
comptime PATCH = 14
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
comptime HIDDEN = 192
comptime ENC_HEADS = 3
comptime ENC_LAYERS = 12
comptime EMB = 192
comptime ENC_PROJ_H = 2048
comptime ENC_FF_MULT = 4  # recipe: ViT-Tiny mlp_ratio 4
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
comptime CKPT_PATH: String = "lewm_pusht_recipe.ckpt"

# ── probe knobs (Apple-sized) ─────────────────────────────────────────
comptime DEC_STEPS = 20_000  # decoder probe budget (0 = eval only);
# first recon grid after VIZ_EVERY steps
comptime DEC_PRINT = 100
comptime VIZ_EVERY = 1_000
comptime N_VIZ = 8
comptime BN_WARMUP_STEPS = 200  # legacy flat ckpts only (v3 carries stats)

# decoder arch (same as the decode_gpu probes)
comptime PATCH_D = 16
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)
comptime PATCH_PX = IN_CH * PATCH_D * PATCH_D
comptime DEC_HID = 192
comptime DEC_FF = 4 * DEC_HID
comptime DEC_LAYERS = 4
comptime DEC_BATCH = B * T

comptime EncCLS = LeWMEncoderCLS[
    IN_CH,
    IMG,
    PATCH,
    N_PATCHES,
    HIDDEN,
    ENC_HEADS,
    ENC_LAYERS,
    EMB,
    ENC_PROJ_H,
    ENC_FF_MULT,
]
comptime Trainer = LeWMTrainer[
    IN_CH,
    IMG,
    PATCH,
    HIDDEN,
    ENC_HEADS,
    ENC_LAYERS,
    EMB,
    ENC_PROJ_H,
    ENC_FF_MULT,
    T,
    ACT,
    SMOOTHED,
    AE_MLP,
    H,
    N_PREDS,
    PRED_HEADS,
    PRED_FF,
    DEPTH,
    PRED_PROJ_H,
    SIG_PROJ,
    SIG_KNOTS,
    B,
    "gpu",
    PRED_DIM_HEAD,
    EncCLS,
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
    var ckpt = resolve_checkpoint(_flag(String("--ckpt"), CKPT_PATH), String("last"))
    print("=" * 70)
    print("LeWM recipe-WM light probes (Apple): shuffled eval + recon grids")
    print("=" * 70)
    var ctx = DeviceContext()

    # z-scored actions — the recipe WM's input convention.
    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=T, normalize_actions=True
    )
    var src = Source.make(sampler^, ctx=ctx)
    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](5e-5), ctx=ctx)
    print("loading recipe WM", ckpt, "...")
    wm.load_params(ckpt)
    if wm.last_load_had_state:
        print("v3 checkpoint carried BN running stats — no warmup needed")
    else:
        print(
            "legacy ckpt: warming BatchNorm running stats (",
            BN_WARMUP_STEPS,
            "training-mode forwards) ...",
        )
        for _ in range(BN_WARMUP_STEPS):
            src.next_batch()
            var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
            var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
            _ = wm.eval_loss(pix_t, act_t)

    # ── 1. shuffled action-awareness eval ───────────────────────────────
    src.next_batch()
    var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
    var act_host = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    var act_dev = DeviceBuffer[DT](ctx, src.act_ptr(), B * ACTIN, owning=False)
    ctx.enqueue_copy(act_host, act_dev)
    ctx.synchronize()
    print()
    print("── probe 1: shuffled action-awareness ──")
    var r = lewm_shuffled_eval[
        IN_CH,
        IMG,
        PATCH,
        HIDDEN,
        ENC_HEADS,
        ENC_LAYERS,
        EMB,
        ENC_PROJ_H,
        ENC_FF_MULT,
        T,
        ACT,
        SMOOTHED,
        AE_MLP,
        H,
        N_PREDS,
        PRED_HEADS,
        PRED_FF,
        DEPTH,
        PRED_PROJ_H,
        SIG_PROJ,
        SIG_KNOTS,
        B,
        "gpu",
        PRED_DIM_HEAD,
        EncCLS,
    ](
        wm,
        pix_t,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](act_host.unsafe_ptr()),
        ctx=ctx,
    )
    print(
        "   action-aware (expert < shuffled_min):",
        "YES" if r[0] < r[2] else "no",
    )

    # ── 2. decoder reconstruction probe ─────────────────────────────────
    comptime if DEC_STEPS > 0:
        print()
        print(
            "── probe 2: decoder reconstruction (",
            DEC_STEPS,
            "steps; grids every",
            VIZ_EVERY,
            ") ──",
        )
        var dec = Decoder.make(lr=Scalar[DT](1e-3), ctx=ctx)
        var tgt_dev = ctx.enqueue_create_buffer[DT](DEC_BATCH * N_Q * PATCH_PX)
        var recon_host = ctx.enqueue_create_host_buffer[DT](
            DEC_BATCH * N_Q * PATCH_PX
        )
        var recon_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
        var orig_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
        ctx.synchronize()

        dec.reset_loss_accum()
        for step in range(1, DEC_STEPS + 1):
            src.next_batch()
            var dpix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
            var dact_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
            _ = wm.eval_loss(dpix_t, dact_t)
            ref emb_tensor = wm.graph.node_output["emb"]()
            var emb_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                emb_tensor.dev.value().unsafe_ptr()
            )
            var emb_t = TileTensor(emb_ptr, row_major[DEC_BATCH, EMB]())
            patchify["gpu", DEC_BATCH, IN_CH, IMG, PATCH_D](
                ctx, src.pix_ptr(), _p(tgt_dev)
            )
            var tgt_t = TileTensor(
                _p(tgt_dev), row_major[DEC_BATCH, N_Q * PATCH_PX]()
            )
            _ = dec.train_step(emb_t, tgt_t)

            if step % DEC_PRINT == 0:
                var ml = dec.read_loss_accum()
                dec.reset_loss_accum()
                print("   step", step, "/", DEC_STEPS, " recon_mse=", ml)

            if step % VIZ_EVERY == 0 or step == DEC_STEPS:
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
                var orig_dev = DeviceBuffer[DT](
                    ctx,
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        src.pix_ptr()
                    ),
                    N_VIZ * IMG_DIM,
                    owning=False,
                )
                ctx.enqueue_copy(
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        orig_img.unsafe_ptr()
                    ),
                    orig_dev,
                )
                ctx.synchronize()
                var path = (
                    "/tmp/lewm_pusht_recipe_recon_" + String(step) + ".ppm"
                )
                save_reconstruction_grid(
                    path,
                    orig_img.unsafe_ptr(),
                    recon_img.unsafe_ptr(),
                    n=N_VIZ,
                    height=IMG,
                    width=IMG,
                    channels=IN_CH,
                    vmin=0.0,
                    vmax=1.0,
                )
                print("   grid →", path)
        _ = dec^

    _ = src^
    _ = wm^
    print("=" * 70)
    print("DONE — recon grids: /tmp/lewm_pusht_recipe_recon_*.ppm (Preview)")
    print("  healthy: block pose sharp AND the agent/pusher dot visible")
    print("=" * 70)
