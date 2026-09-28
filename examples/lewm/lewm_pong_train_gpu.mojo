"""LeWM (nn) — train the JEPA world model on Pong pixels (GPU).

End-to-end driver for the nn LeWM port. Streams length-T windows through the
generic `WindowSource` into `LeWMTrainer` at the §10.7 Pong-ViT recipe, and
prints the loss + representation-collapse probes. The data source is swappable
at compile time via `USE_ONLINE`:

  OFFLINE (USE_ONLINE = False) — replay a pre-collected on-disk buffer:
    PongOfflineBuffer.load (collect with lewm_pong_collect_buffer.mojo)
      → WindowSource (sample window → H2D → uint8→fp32 ÷255) → train_step

  ONLINE  (USE_ONLINE = True)  — generate windows live from the simulator,
  no dataset, no recording:
    OnlinePongSampler (step PongPixelEnv pool under a scripted policy)
      → WindowSource (H2D → uint8→fp32 ÷255) → train_step
    (CPU env-step × GPU train hybrid; env stepping is the throughput floor.)

Both paths feed byte-identical (B, T·IMG_DIM) fp32 CHW pixels + (B, T·ACT)
one-hot actions, so the training loop (`_train_loop`) is shared verbatim.

Recipe (LeWMPongViTConfig[batch=16, t=6, depth=6, hidden=128, emb=128]):
  84×84×4 frames, patch=14 → 36 patches, H=3 context, EMB=128, DEPTH=6.

The run (project `lewm`) writes its checkpoint to
`runs/<id>/checkpoints/last.ckpt`, beside `metrics.csv` (loss, var_min,
gram_off) and `run.kv`.

Run:
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pong_train_gpu.mojo

Watch for: loss falling smoothly, var_min rising > 0.1, gram_off < 0.5
(legacy §10.7 healthy-representation thresholds). On Apple it runs but is
slow at this scale — NVIDIA is the intended target.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.io.artifact_sink import sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.core.offline_buffer import OfflineBuffer
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)
from noeira.envs.arcade_games.pong.online_sampler import (
    OnlinePongSampler,
    ScriptedPongPolicy,
)


# ── §10.7 Pong-ViT recipe ──────────────────────────────────────────────
comptime IN_CH = 4
comptime IMG = 84
comptime PATCH = 14          # 84 // 14 == 6 → 36 patches
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
comptime SIG_PROJ = 256       # > D=128: over-determines the latent so SIGReg
                              # can't be gamed by collapsing orthogonal dims
                              # (P=64 was too coarse → real-Pong collapse)
comptime SIG_KNOTS = 5
comptime B = 16

comptime IMG_DIM = IN_CH * IMG * IMG       # 28224 == PONG_FRAME_BYTES
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

# ── run config ──────────────────────────────────────────────────────────
# Data source: False = replay offline buffer, True = live simulator windows.
comptime USE_ONLINE: Bool = False
comptime ONLINE_EPS: Float64 = 0.3    # scripted-policy exploration (online)

comptime BUFFER_PATH: String = "/tmp/lewm_pong_buffer.bin"
comptime STEPS: Int = 2000
comptime LOG_EVERY: Int = 50
comptime LAM: Scalar[DT] = 1.0    # healthy at SIG_PROJ=256 (λ-sweep: var_min
                                  # 0.136>0.1, gram_off 0.423<0.5); 0.09/0.3
                                  # under-regularize, 3.0 over-regularizes
comptime LR: Scalar[DT] = 1e-3

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]
comptime OnlineBuf = OnlinePongSampler[ScriptedPongPolicy, B, T]


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _train_loop[
    BUF: OfflineBuffer
](
    var src: WindowSource[IMG_DIM, ACT, T, B, "gpu", BUF],
    mut tr: Trainer,
    mut logger: RunLogger,
) raises -> Float64:
    """Shared GPU training loop — identical for offline and online sources.
    Returns the last logged window's loss."""
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
    _ = src^
    return last_loss


def main() raises:
    print("=" * 70)
    print("LeWM nn — Pong JEPA world model training (GPU)")
    print("=" * 70)
    print("recipe: 84x84x4, patch=14, EMB=", EMB, " DEPTH=", DEPTH,
          " B=", B, " T=", T)
    print("data:", "ONLINE (live sim)" if USE_ONLINE else "OFFLINE (buffer)")
    print()

    var ctx = DeviceContext()
    var tr = Trainer.make(lam=LAM, lr=LR, ctx=ctx)
    var run = RunContext(
        project=String("lewm"),
        driver=String("examples/lewm/lewm_pong_train_gpu.mojo"),
        slug=String("lewm-pong"),
        env=String("builtin:arcade/pong"),
        dataset=String("") if USE_ONLINE else String(BUFFER_PATH),
        device=String(ctx.name()),
    )
    print("run:", run.dir)
    var ckpt_path = run.checkpoint_path(String("last"))
    var logger = run_logger(run)
    logger.set_config("algorithm", "LeWM")
    logger.set_config("env", "Pong")
    logger.set_config("source", "online" if USE_ONLINE else "offline")
    logger.set_config("steps", String(STEPS))
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var last_loss = Float64(0.0)

    print("training", STEPS, "steps ...")
    comptime if USE_ONLINE:
        print("   source: OnlinePongSampler eps=", ONLINE_EPS)
        var src = WindowSource[IMG_DIM, ACT, T, B, "gpu", OnlineBuf].make(
            OnlineBuf.make(ScriptedPongPolicy(eps=ONLINE_EPS)), ctx=ctx
        )
        last_loss = _train_loop[OnlineBuf](src^, tr, logger)
    else:
        print("   source: offline buffer", BUFFER_PATH)
        var buf = PongOfflineBuffer.load(BUFFER_PATH)
        print("   n_frames =", buf.n_frames)
        var src = WindowSource[IMG_DIM, ACT, T, B, "gpu"].make(buf^, ctx=ctx)
        last_loss = _train_loop[PongOfflineBuffer](src^, tr, logger)

    print()
    print("saving world-model checkpoint →", ckpt_path)
    tr.save_params(ckpt_path)
    announce_checkpoint(ckpt_path, artifacts, run.dir)
    finish_run(
        run, logger, artifacts,
        String("final_loss=") + String(last_loss),
    )

    _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
