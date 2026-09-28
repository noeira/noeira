"""LeWM PushT paper protocol — E1: frozen vs AdaJEPA test-time adaptation.

THE go/no-go experiment (docs/ADAJEPA_LEWM_TTA_PLAN.md §6, corrected): the
paper-protocol benchmark (dataset start, real +25-step-frame goal, swm
20px/20° success, budget 50 env steps) is BOTH the LeWM paper's PushT eval
(~90%) and the benchmark AdaJEPA itself evaluates PushT on ("goals sampled
25 steps away") — unlike the coverage closed loop, whose synthetic
composite goal has "stay still" as its cost minimizer (LEWM_REFERENCE_AUDIT
2026-06-10).

Per round: sample BATCH (start, goal) pairs from the expert dataset, then
run the SAME pairs twice — frozen, then with AdaJEPA TTA (one masked
gradient step on the pretraining JEPA loss per replan, predictor-side,
planner re-synced + goal latent re-encoded every adapt). MPC_HORIZON=1 =
AdaJEPA's plan-execute-adapt-replan shape (one chunk per replan; 10
replans at budget 50 — the T=6 window buffer fills after 6, so ~4 adapted
replans; EVAL_BUDGET=100 ≈ AdaJEPA's 20 MPC steps gives 14).

TTA at the recipe's training hyperparams (paper rule "same as training"):
lr 5e-5 (the recipe's peak), wd 0 + zeroed moments (mask invariant — the
recipe ckpt carries AdamW moments), clip 1.0.

Requires (NVIDIA box): the trained RECIPE WM at lewm_pusht_recipe.ckpt
(lewm_pusht_train_gpu_recipe.mojo) + the cached lewm-pusht HDF5 (episode
start/goal states come from the dataset regardless of ckpt format).

Run:
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_paper_protocol_tta_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_paper_protocol_tta_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.memory import alloc
from std.random import seed as rng_seed, random_float64
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.nn.datasets.lewm_pusht import LewmPushTExpert
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.encoder import LeWMEncoderCLS
from noeira.experimental.lewm.paper_protocol import run_lewm_paper_protocol
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.envs.pusht import PushTOfflineSampler


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
comptime ENC_FF_MULT = 4    # recipe: ViT-Tiny mlp_ratio 4
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

# ── E1 protocol knobs (run 2: lookahead + AdaJEPA receding horizon) ───
# Run 1 (horizon 1 = greedy plan-1-execute-1, budget 50) showed both arms
# diverging (~+130 px/episode) — no lookahead toward a goal 25 steps out.
# AdaJEPA actually PLANS 25 steps and EXECUTES 5: here plan 4 chunks
# (20 steps — NEEDED = H+4-1 = 6 = T is the action-embedder cap), execute
# 1, replan. Budget 100 ≈ AdaJEPA's 20 MPC steps → 20 replans, ~14 adapted.
comptime MPC_HORIZON = 4          # plan 4 chunks ahead (in-window max)
comptime EXECUTE_BLOCKS = 1       # execute 1 chunk per replan (AdaJEPA)
comptime ROUNDS = 3               # 3×16 = 48 episode pairs
comptime EVAL_BUDGET = 100        # ≈ AdaJEPA's PushT budget
comptime GOAL_FRAME = 5           # window frame +5 = 25 dense steps ahead
comptime CEM_ITERS = 30
comptime CEM_SAMPLES = 300
comptime CEM_TOPK = 30
comptime INIT_STD = 1.0           # z-space Σ₀ = I — the paper's CEM init
comptime SCALE_X = 100.0
comptime SCALE_Y = 100.0
comptime BN_WARMUP_STEPS = 200    # legacy flat ckpts only (v3 carries stats)
comptime TTA_STEPS = 1
comptime TTA_LR: Scalar[DT] = 5e-5  # = the recipe's training peak LR
comptime LAM: Scalar[DT] = 0.09

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
    print("LeWM PushT PAPER PROTOCOL — E1: FROZEN vs ADAPT (AdaJEPA TTA)")
    print("=" * 70)
    print("recipe WM (CLS, z-actions), plan", MPC_HORIZON, "chunks / execute",
          EXECUTE_BLOCKS, ", budget", EVAL_BUDGET, ",", ROUNDS, "rounds ×",
          B, "episodes, CEM", CEM_SAMPLES, "×", CEM_ITERS, ", tta_steps=",
          TTA_STEPS, ", tta_lr=", TTA_LR)
    var ctx = DeviceContext()

    # wd=0 + clip 1.0: training-faithful adapt step that keeps the mask
    # invariant (decoupled wd would move masked params).
    var wm = Trainer.make(
        lam=LAM, lr=TTA_LR, max_grad_norm=Scalar[DT](1.0), ctx=ctx
    )
    print("loading frozen recipe WM", ckpt, "...")
    wm.load_params(ckpt)
    # The recipe ckpt carries AdamW moments (exact-resume); TTA needs a
    # fresh optimizer (plan §4).
    wm.reset_opt_moments()

    # z-score stats come from the dataset; BN warmup only for legacy ckpts.
    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=T, normalize_actions=True
    )
    var am_x = sampler.action_mean(0)
    var am_y = sampler.action_mean(1)
    var as_x = sampler.action_std(0)
    var as_y = sampler.action_std(1)
    if wm.last_load_had_state:
        print("v3 checkpoint carried BN running stats — skipping warmup")
        _ = sampler^
    else:
        print("legacy ckpt: warming BatchNorm running stats (",
              BN_WARMUP_STEPS, "training-mode forwards) ...")
        var src = Source.make(sampler^, ctx=ctx)
        for _ in range(BN_WARMUP_STEPS):
            src.next_batch()
            var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
            var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
            _ = wm.eval_loss(pix_t, act_t)
        _ = src^

    # dataset windows → (start, goal) state pairs
    var dataset = LewmPushTExpert(frameskip=FRAMESKIP, num_steps=T)
    if dataset.state_dim != 7:
        raise Error("expected 7-dim swm PushT state column, got "
                    + String(dataset.state_dim))
    var sdim = dataset.state_dim
    var window = dataset.make_window()
    rng_seed(7)
    var starts = alloc[Scalar[DT]](B * 5)
    var goals = alloc[Scalar[DT]](B * 5)
    var n_clips = len(dataset)

    var frozen_sr = List[Float64]()
    var frozen_pd = List[Float64]()
    var adapt_sr = List[Float64]()
    var adapt_pd = List[Float64]()

    for round in range(ROUNDS):
        for b in range(B):
            var r = random_float64() * Float64(n_clips)
            var idx = Int(r)
            if idx >= n_clips:
                idx = n_clips - 1
            dataset.sample_window(idx, window)
            for j in range(5):
                starts[b * 5 + j] = Scalar[DT](
                    Float64(window.state[0 * sdim + j])
                )
                goals[b * 5 + j] = Scalar[DT](
                    Float64(window.state[GOAL_FRAME * sdim + j])
                )
        print("─" * 70)
        print("round", round + 1, "/", ROUNDS)

        print("   [frozen] ...")
        # CEM samples from the global RNG: re-seed identically before each
        # arm so frozen and adapt see the SAME CEM noise (paired episodes
        # AND paired planner noise; run 1 arms differed pre-adapt).
        rng_seed(1000 + round)
        var rf = run_lewm_paper_protocol[
            IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
            ENC_PROJ_H, ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS,
            PRED_HEADS, PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
            B, MPC_HORIZON, "gpu", PRED_DIM_HEAD, 2, 96, EncCLS,
        ](
            wm, starts.as_unsafe_any_origin(), goals.as_unsafe_any_origin(),
            eval_budget=EVAL_BUDGET,
            scale_x=SCALE_X, scale_y=SCALE_Y,
            act_mean_x=am_x, act_mean_y=am_y,
            act_std_x=as_x, act_std_y=as_y,
            cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
            init_std=INIT_STD,
            seed0=1 + round * B,
            viz_path=String("/tmp/lewm_protocol_e1_frozen_r")
                + String(round) + String(".ppm"),
            ctx=ctx,
            verbose=True,
            execute_blocks=EXECUTE_BLOCKS,
        )
        print("   [frozen] success=", rf[0], " pos_diff=", rf[1])

        print("   [adapt] ...")
        # The adapt arm restores params+state at exit, but Adam moments
        # survive — zero them so every round's mask invariant holds.
        wm.reset_opt_moments()
        rng_seed(1000 + round)  # pair the CEM noise with the frozen arm
        var ra = run_lewm_paper_protocol[
            IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
            ENC_PROJ_H, ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS,
            PRED_HEADS, PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
            B, MPC_HORIZON, "gpu", PRED_DIM_HEAD, 2, 96, EncCLS,
        ](
            wm, starts.as_unsafe_any_origin(), goals.as_unsafe_any_origin(),
            eval_budget=EVAL_BUDGET,
            scale_x=SCALE_X, scale_y=SCALE_Y,
            act_mean_x=am_x, act_mean_y=am_y,
            act_std_x=as_x, act_std_y=as_y,
            cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
            init_std=INIT_STD,
            seed0=1 + round * B,
            viz_path=String("/tmp/lewm_protocol_e1_adapt_r")
                + String(round) + String(".ppm"),
            ctx=ctx,
            verbose=True,
            execute_blocks=EXECUTE_BLOCKS,
            tta_enabled=True,
            tta_steps=TTA_STEPS,
            # tta_keep default = predictor side (predfull+encfrozen, §4)
        )
        print("   [adapt]  success=", ra[0], " pos_diff=", ra[1])

        frozen_sr.append(rf[0]); frozen_pd.append(rf[1])
        adapt_sr.append(ra[0]); adapt_pd.append(ra[1])

    print()
    print("=" * 70)
    print("E1 RESULTS (paper protocol,", ROUNDS * B, "episode pairs )")
    var fs: Float64 = 0.0
    var fp: Float64 = 0.0
    var as_: Float64 = 0.0
    var ap: Float64 = 0.0
    for r in range(ROUNDS):
        print("   round", r, ": frozen succ=", frozen_sr[r], " pos=",
              frozen_pd[r], " | adapt succ=", adapt_sr[r], " pos=",
              adapt_pd[r])
        fs += frozen_sr[r]; fp += frozen_pd[r]
        as_ += adapt_sr[r]; ap += adapt_pd[r]
    var n = Float64(ROUNDS)
    print()
    print("   FROZEN  success=", fs / n, "  mean pos_diff=", fp / n, "px")
    print("   ADAPT   success=", as_ / n, "  mean pos_diff=", ap / n, "px")
    print("   (LeWM paper frozen reference: ~0.90; AdaJEPA predicts")
    print("    adapt ≥ frozen, largest when the WM is data-limited)")
    print("=" * 70)

    starts.free(); goals.free()
    _ = window^; _ = dataset^; _ = wm^
    print("DONE")
