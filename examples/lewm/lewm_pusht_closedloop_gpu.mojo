"""LeWM (nn) — CLOSED-LOOP MPC control on PushT (mean-pool WM, GPU).

Mean-pool baseline of the Gate C retry — same three eval-path fixes as the
CLS variant (docs/LEWM_REFERENCE_AUDIT.md):

  1. ACTION SCALE 100 (ground truth: swm PushT `relative=True,
     action_scale=100`; the centroid calibration 142/148 was ~1.45× large).
  2. BATCHNORM EVAL MODE at planning, after warming running stats with
     BN_WARMUP_STEPS training-mode forwards over dataset windows.
  3. PAPER CEM BUDGET 300×30 top-30.

Loads `lewm_pusht_paper.ckpt`. Heavy one-shot (per cycle:
2 encode forwards + CEM_ITERS·CEM_SAMPLES latent rollouts).
Run (NVIDIA):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_closedloop_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_closedloop_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.closedloop import run_lewm_closedloop
from noeira.experimental.lewm.pong_data import WindowSource
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

comptime MPC_HORIZON = 4          # NEEDED = H+horizon-1 = 6 = T (in-window max)
comptime CKPT_PATH: String = "lewm_pusht_paper.ckpt"

# control / planning budget — PAPER values (App D)
comptime N_CYCLES = 25
comptime CEM_ITERS = 30
comptime CEM_SAMPLES = 300
comptime CEM_TOPK = 30
comptime INIT_STD = 0.2           # ≈ stored-action RMS (≡ paper's Σ₀=I z-scored)
comptime SCALE_X = 100.0          # GROUND TRUTH: swm PushT relative=True, action_scale=100
comptime SCALE_Y = 100.0
comptime BN_WARMUP_STEPS = 200    # EMA momentum 0.1 → time constant 10 batches

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
    print("LeWM nn — PushT CLOSED-LOOP MPC (mean-pool WM, GPU) — retry")
    print("=" * 70)
    var ctx = DeviceContext()

    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading frozen WM", ckpt, "...")
    wm.load_params(ckpt)

    print("warming BatchNorm running stats (", BN_WARMUP_STEPS,
          "training-mode forwards over dataset windows) ...")
    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    for _ in range(BN_WARMUP_STEPS):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = wm.eval_loss(pix_t, act_t)
    _ = src^

    print("controlling", B, "PushT envs,", N_CYCLES, "cycles, horizon",
          MPC_HORIZON, "(CEM", CEM_SAMPLES, "×", CEM_ITERS, ", eval-mode BN,",
          "scale", SCALE_X, ") ...")
    var r = run_lewm_closedloop[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", PRED_DIM_HEAD, 2, 96,
    ](
        wm,
        n_cycles=N_CYCLES,
        scale_x=SCALE_X, scale_y=SCALE_Y,
        cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
        init_std=INIT_STD,
        goal_agent_x=256.0, goal_agent_y=256.0,
        goal_match_agent=True,   # goal = block@goal + CURRENT agent → block-pose-only objective
        seed0=1,
        viz_path=String("/tmp/lewm_pusht_closedloop.ppm"),
        ctx=ctx,
        verbose=True,
    )
    print()
    print("   SUCCESS RATE =", r[0], "  mean final coverage =", r[1])
    print("   (success = coverage > 0.95; trajectory: /tmp/lewm_pusht_closedloop.ppm)")

    _ = wm^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
