"""LeWM PushT closed-loop MPC — E1: frozen vs AdaJEPA test-time adaptation.

The go/no-go experiment of docs/ADAJEPA_LEWM_TTA_PLAN.md §6: run the paper
mean-pool WM through the closed-loop CEM controller twice with identical
seeds — once frozen (the existing baseline), once with AdaJEPA test-time
adaptation (one masked gradient step on the pretraining JEPA loss per MPC
cycle, predictor side only, planner re-synced every cycle) — and compare
success rate + mean coverage. The paper predicts adapt ≥ frozen even
in-distribution when the WM is data-limited (their Fig. 2: +20 % on PushT),
with the gap widening over replanning steps.

Config = lewm_pusht_train_gpu_paper.mojo / lewm_pusht_closedloop_gpu.mojo
(paper-width ViT-Tiny encoder, expanded predictor attention, 224², B=16
envs). TTA follows the paper's rule "same self-supervised signal + same
learning rate as training": our WM trains at flat LR 1e-3, raw actions,
no grad clip — so the adapt step uses exactly that.

Per adapt run the wm's params + BN state are snapshot/restored inside
run_lewm_closedloop; each seed set still re-makes the trainer + reloads
the checkpoint because Adam moments survive the restore (constant-mask
invariant, plan §4).

Requires (NVIDIA box):
  - the trained paper WM at CKPT_PATH (32k-step run of
    lewm_pusht_train_gpu_paper.mojo, ~6-10 h). New saves are v3 binary
    checkpoints carrying the BN running stats, so no warmup is needed;
    a legacy flat-text ckpt still loads but then also needs the cached
    lewm-pusht HDF5 dataset for the BN warmup.

Run:
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_closedloop_tta_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_closedloop_tta_gpu.mojo --ckpt <run_id>
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

comptime CKPT_PATH: String = "lewm_pusht_paper.ckpt"

# ── E1 protocol (plan §6) ─────────────────────────────────────────────
# 40 cycles (paper runs 25 then extends to 30 to show adapt keeps improving
# where frozen saturates); paper CEM budget; adapt at every cycle once the
# T=6-cycle window buffer is full (cycles 0-5 run frozen in both arms).
comptime N_CYCLES = 40
comptime CEM_ITERS = 30
comptime CEM_SAMPLES = 300
comptime CEM_TOPK = 30
comptime INIT_STD = 0.2
comptime SCALE_X = 100.0
comptime SCALE_Y = 100.0
comptime BN_WARMUP_STEPS = 200
comptime TTA_STEPS = 1             # paper default: one GD step per replan
comptime TTA_LR: Scalar[DT] = 1e-3  # = our training LR (paper's rule)
comptime LAM: Scalar[DT] = 0.09
comptime SEED_SETS = 1             # crank to 3 for the full E1 protocol
comptime SEED_STRIDE = 100         # seed0 = 1, 101, 201, ... per set

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu", PRED_DIM_HEAD,
]
comptime Source = WindowSource[
    IMG_DIM, ACT, T, B, "gpu", PushTOfflineSampler, IN_CH, IMG
]

comptime MPC_HORIZON = 4           # NEEDED = H+horizon-1 = 6 = T


def _make_warmed_wm(ctx: DeviceContext, ckpt: String) raises -> Trainer:
    """Fresh trainer (fresh Adam — TTA precondition) + ckpt (+ BN warmup
    only for legacy flat checkpoints — v3 carries the BN running stats)."""
    # wd=0, no grad clip: the adapt step matches training exactly.
    var wm = Trainer.make(lam=LAM, lr=TTA_LR, ctx=ctx)
    print("   loading frozen WM", ckpt, "...")
    wm.load_params(ckpt)
    # A v3 ckpt may carry the training run's Adam moments; the TTA mask
    # invariant needs zero moments (plan §4).
    wm.reset_opt_moments()
    if wm.last_load_had_state:
        print("   v3 checkpoint carried BN running stats — skipping warmup")
    else:
        print("   legacy ckpt: warming BatchNorm running stats (",
              BN_WARMUP_STEPS, "training-mode forwards) ...")
        var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
        var src = Source.make(sampler^, ctx=ctx)
        for _ in range(BN_WARMUP_STEPS):
            src.next_batch()
            var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
            var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
            _ = wm.eval_loss(pix_t, act_t)
        _ = src^
    return wm^


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
    print("LeWM PushT closed-loop — E1: FROZEN vs ADAPT (AdaJEPA TTA)")
    print("=" * 70)
    print("paper-width WM, B=", B, "envs,", N_CYCLES, "cycles, CEM",
          CEM_SAMPLES, "×", CEM_ITERS, ", tta_steps=", TTA_STEPS,
          ", tta_lr=", TTA_LR, ",", SEED_SETS, "seed set(s)")
    var ctx = DeviceContext()

    var frozen_succ = List[Float64]()
    var frozen_cov = List[Float64]()
    var adapt_succ = List[Float64]()
    var adapt_cov = List[Float64]()

    for s in range(SEED_SETS):
        var seed0 = 1 + s * SEED_STRIDE
        print()
        print("── seed set", s, "(seed0 =", seed0, ") ──")
        # One warmed wm per set serves both arms: the frozen run never
        # steps the optimizer, so the adapt run still sees a fresh Adam.
        var wm = _make_warmed_wm(ctx, ckpt)

        print("   [frozen] ...")
        var rf = run_lewm_closedloop[
            IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
            ENC_PROJ_H, ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS,
            PRED_HEADS, PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
            B, MPC_HORIZON, "gpu", PRED_DIM_HEAD, 2, 96,
        ](
            wm,
            n_cycles=N_CYCLES,
            scale_x=SCALE_X, scale_y=SCALE_Y,
            cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
            init_std=INIT_STD,
            goal_match_agent=True,
            seed0=seed0,
            viz_path=String("/tmp/lewm_tta_e1_frozen_s") + String(s) + ".ppm",
            ctx=ctx,
            verbose=True,
        )
        print("   [frozen] success=", rf[0], " mean_cov=", rf[1])

        print("   [adapt] ...")
        var ra = run_lewm_closedloop[
            IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
            ENC_PROJ_H, ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS,
            PRED_HEADS, PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
            B, MPC_HORIZON, "gpu", PRED_DIM_HEAD, 2, 96,
        ](
            wm,
            n_cycles=N_CYCLES,
            scale_x=SCALE_X, scale_y=SCALE_Y,
            cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
            init_std=INIT_STD,
            goal_match_agent=True,
            seed0=seed0,
            viz_path=String("/tmp/lewm_tta_e1_adapt_s") + String(s) + ".ppm",
            ctx=ctx,
            verbose=True,
            tta_enabled=True,
            tta_steps=TTA_STEPS,
            # tta_keep default = predictor side (predfull+encfrozen, plan §4)
        )
        print("   [adapt]  success=", ra[0], " mean_cov=", ra[1])

        frozen_succ.append(rf[0]); frozen_cov.append(rf[1])
        adapt_succ.append(ra[0]); adapt_cov.append(ra[1])
        _ = wm^

    print()
    print("=" * 70)
    print("E1 RESULTS (", SEED_SETS, "seed set(s) ×", B, "envs,", N_CYCLES,
          "cycles )")
    var fs: Float64 = 0.0
    var fc: Float64 = 0.0
    var as_: Float64 = 0.0
    var ac: Float64 = 0.0
    for s in range(SEED_SETS):
        print("   set", s, ": frozen succ=", frozen_succ[s], " cov=",
              frozen_cov[s], " | adapt succ=", adapt_succ[s], " cov=",
              adapt_cov[s])
        fs += frozen_succ[s]; fc += frozen_cov[s]
        as_ += adapt_succ[s]; ac += adapt_cov[s]
    var n = Float64(SEED_SETS)
    print()
    print("   FROZEN  mean success=", fs / n, "  mean coverage=", fc / n)
    print("   ADAPT   mean success=", as_ / n, "  mean coverage=", ac / n)
    print("   (per-cycle pre-adapt window losses in the [adapt] logs above")
    print("    are the paper's Fig. 11-14 diagnostic — adapt should sit")
    print("    below frozen's implied trajectory once cycles > 6)")
    print("=" * 70)
    print("DONE")
    print("=" * 70)
