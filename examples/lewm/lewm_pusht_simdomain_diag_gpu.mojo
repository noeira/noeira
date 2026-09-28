"""LeWM (nn) — PushT sim-domain transfer diagnostic (GPU).

THE GATE for closed-loop control. The world model was trained on the
HuggingFace gym-pusht renders; closed-loop control would drive the mojo
`PushTEnv` (its own `render.mojo`, built to mirror gym-pusht's palette).
This probes whether the frozen encoder transfers across renderers — cheaply,
by reusing the trained decoder: render real sim frames at 224², encode →
decode → reconstruct, and compare to the originals.

  PushTEnv (stepped) → render 224² CHW → frozen WM encode → emb
    → trained decoder → recon → side-by-side grid + sim_recon_mse

Read: if the reconstructions show the goal-T + block pose (like the HF-domain
grid did) and sim_recon_mse is in the HF ballpark (~0.0018), the encoder
transfers → closed-loop is worth building. If the reconstructions are garbage
/ mse is far higher, the encoder is out-of-distribution on the mojo renderer
→ closed-loop needs the WM retrained on sim frames (or the renderer matched).

Loads `lewm_pusht_paper.ckpt` + `/tmp/lewm_pusht_decoder.txt`.
Run (NVIDIA; 224² WM):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_simdomain_diag_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_simdomain_diag_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.random import random_float64, seed as rng_seed
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.decoder import patchify, unpatchify
from noeira.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from noeira.experimental.lewm.pusht_sim_bridge import sim_frame_chw_norm
from noeira.render.image_writer import save_reconstruction_grid
from noeira.envs.pusht import PushTEnv, PushTAction


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

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime WM_CKPT: String = "lewm_pusht_paper.ckpt"

# ── decoder config — must match lewm_pusht_decode_gpu.mojo ─────────────
comptime PATCH_D = 16
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)   # 196
comptime PATCH_PX = IN_CH * PATCH_D * PATCH_D        # 768
comptime DEC_HID = 192
comptime DEC_FF = 4 * DEC_HID
comptime DEC_LAYERS = 4
comptime DEC_BATCH = B * T                           # 96
comptime DEC_CKPT: String = "/tmp/lewm_pusht_decoder.txt"

comptime N_VIZ = 8
comptime N_WARMUP = 6     # random steps before collecting the window

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD,
]
comptime Decoder = LeWMDecoderTrainer[
    EMB, DEC_HID, N_Q, PATCH_PX, DEC_FF, DEC_LAYERS, DEC_BATCH, "gpu"
]


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _rand_target() -> PushTAction[DT]:
    var x = Scalar[DT](random_float64() * 512.0)
    var y = Scalar[DT](random_float64() * 512.0)
    return PushTAction[DT](x, y)


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
    print("=" * 70)
    print("LeWM nn — PushT sim-domain transfer diagnostic (GPU)")
    print("=" * 70)
    rng_seed(0)
    var ctx = DeviceContext()

    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading frozen WM", ckpt, "...")
    wm.load_params(ckpt)
    var dec = Decoder.make(lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading decoder", DEC_CKPT, "...")
    dec.load_params(DEC_CKPT)

    # ── collect B·T sim frames (B envs × T steps), rendered at 224² CHW ──
    var pix_host = ctx.enqueue_create_host_buffer[DT](DEC_BATCH * IMG_DIM)
    ctx.synchronize()
    print("stepping", B, "PushT envs ×", T, "frames, rendering 224² ...")
    for b in range(B):
        var env = PushTEnv[DT](seed=UInt64(b + 1))
        _ = env.reset()
        for _ in range(N_WARMUP):
            _ = env.step(_rand_target())
        for t in range(T):
            _ = env.step(_rand_target())
            var bp = env.block_pose()
            var ap = env.agent_pos()
            var off = (b * T + t) * IMG_DIM
            sim_frame_chw_norm[IMG](
                bp[0], bp[1], bp[2], ap[0], ap[1],
                rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                    pix_host.unsafe_ptr()
                ) + off,
            )

    var pix_dev = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_dev = ctx.enqueue_create_buffer[DT](B * ACTIN)
    act_dev.enqueue_fill(0.0)   # emb depends only on pixels
    ctx.enqueue_copy(pix_dev, pix_host)
    ctx.synchronize()

    # ── frozen encode → emb (B, T·EMB) == (B·T, EMB) ────────────────────
    var pix_t = TileTensor(_p(pix_dev), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_dev), row_major[B, ACTIN]())
    _ = wm.eval_loss(pix_t, act_t)
    ref emb_tensor = wm.graph.node_output["emb"]()
    var emb_ptr = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        emb_tensor.dev.value().unsafe_ptr()
    )
    var emb_t = TileTensor(emb_ptr, row_major[DEC_BATCH, EMB]())

    # ── decode → recon (patch space) ────────────────────────────────────
    var recon_host = ctx.enqueue_create_host_buffer[DT](
        DEC_BATCH * N_Q * PATCH_PX
    )
    ctx.synchronize()
    dec.recon_into(
        emb_t,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](recon_host.unsafe_ptr()),
    )

    # ── quantitative sim-domain recon MSE (patch space, host) ───────────
    var tgt_host = ctx.enqueue_create_host_buffer[DT](
        DEC_BATCH * N_Q * PATCH_PX
    )
    ctx.synchronize()
    patchify["cpu", DEC_BATCH, IN_CH, IMG, PATCH_D](
        None,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](pix_host.unsafe_ptr()),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](tgt_host.unsafe_ptr()),
    )
    var sse: Float64 = 0.0
    comptime NEL = DEC_BATCH * N_Q * PATCH_PX
    for i in range(NEL):
        var d = Float64(recon_host.unsafe_ptr()[i] - tgt_host.unsafe_ptr()[i])
        sse += d * d
    var sim_mse = sse / Float64(NEL)
    print("   sim_recon_mse =", sim_mse, " (HF-domain ref ≈ 0.0018)")

    # ── viz: original (sim) vs reconstruction, first N_VIZ frames ───────
    var recon_img = ctx.enqueue_create_host_buffer[DT](N_VIZ * IMG_DIM)
    ctx.synchronize()
    unpatchify["cpu", N_VIZ, IN_CH, IMG, PATCH_D](
        None,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](recon_host.unsafe_ptr()),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](recon_img.unsafe_ptr()),
    )
    save_reconstruction_grid(
        String("/tmp/lewm_pusht_simdomain_recon.ppm"),
        pix_host.unsafe_ptr(),
        recon_img.unsafe_ptr(),
        n=N_VIZ, height=IMG, width=IMG, channels=IN_CH, vmin=0.0, vmax=1.0,
    )

    _ = wm^; _ = dec^
    print("=" * 70)
    print("DONE — grid at /tmp/lewm_pusht_simdomain_recon.ppm")
    print("  (recognizable goal-T + block pose ⇒ encoder transfers ⇒ build"
          " closed-loop; garbage ⇒ retrain WM on sim frames)")
    print("=" * 70)
