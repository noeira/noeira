"""LeWM (nn) — PAPER-PROTOCOL planning eval for the RECIPE WM (GPU).

Identical protocol to `lewm_pusht_paper_protocol_gpu_cls.mojo` (dataset
start, real +25-step goal, budget 50, swm 20px/20° success — THE number
comparable to the paper's ~90%) but for the recipe-retrained WM
(no-stop-grad, AdamW 5e-5, z-scored actions):

  * loads lewm_pusht_recipe.ckpt,
  * the BN warm-up and planning run with z-scored actions (the WM's input
    convention), so CEM samples Σ₀ = I (INIT_STD = 1.0 — exactly the
    paper's init), and execution de-normalizes raw = z·std + mean before
    the ·100 delta mapping (stats from the dataset, printed at load).

Run (NVIDIA, after lewm_pusht_train_gpu_recipe.mojo):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_paper_protocol_gpu_recipe.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_paper_protocol_gpu_recipe.mojo --ckpt <run_id>
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
comptime ENC_FF_MULT = 4    # must match lewm_pusht_train_gpu_recipe.mojo
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
comptime CKPT_PATH: String = "lewm_pusht_recipe.ckpt"

# protocol knobs (paper App D / F.1)
comptime ROUNDS = 3               # 3×16 = 48 episodes ≈ paper's 50
comptime EVAL_BUDGET = 50         # env steps per episode
comptime GOAL_FRAME = 5           # window frame +5 = 25 dense steps ahead
comptime CEM_ITERS = 30
comptime CEM_SAMPLES = 300
comptime CEM_TOPK = 30
comptime INIT_STD = 1.0           # z-space Σ₀ = I — the paper's CEM init
comptime SCALE_X = 100.0          # swm PushT: relative=True, action_scale=100
comptime SCALE_Y = 100.0
comptime BN_WARMUP_STEPS = 200

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
    print("LeWM nn — PushT PAPER-PROTOCOL eval, RECIPE WM (GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var wm = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](5e-5), ctx=ctx)
    print("loading frozen recipe WM", ckpt, "...")
    wm.load_params(ckpt)

    # BN running-stats warm-up with the WM's z-scored action convention.
    # Read the de-normalization stats BEFORE the sampler moves into Source.
    print("warming BatchNorm running stats (", BN_WARMUP_STEPS,
          "training-mode forwards, z-scored actions) ...")
    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=T, normalize_actions=True
    )
    var am_x = sampler.action_mean(0)
    var am_y = sampler.action_mean(1)
    var as_x = sampler.action_std(0)
    var as_y = sampler.action_std(1)
    var src = Source.make(sampler^, ctx=ctx)
    for _ in range(BN_WARMUP_STEPS):
        src.next_batch()
        var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
        var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
        _ = wm.eval_loss(pix_t, act_t)
    _ = src^

    # dataset windows → (start, goal) state pairs (7-dim swm state:
    # [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy])
    var dataset = LewmPushTExpert(frameskip=FRAMESKIP, num_steps=T)
    if dataset.state_dim != 7:
        raise Error("expected 7-dim swm PushT state column, got "
                    + String(dataset.state_dim))
    var sdim = dataset.state_dim
    var window = dataset.make_window()
    rng_seed(7)

    dataset.sample_window(0, window)
    var p0 = Float64(window.proprio[0])
    var p1 = Float64(window.proprio[1])
    var s0 = Float64(window.state[0])
    var s1 = Float64(window.state[1])
    print("  state[0:2]=", s0, s1, " proprio[0:2]=", p0, p1)
    var d0 = s0 - p0
    var d1 = s1 - p1
    if d0 * d0 + d1 * d1 > 1.0:
        raise Error("state[0:2] != proprio agent position")

    var starts = alloc[Scalar[DT]](B * 5)
    var goals = alloc[Scalar[DT]](B * 5)
    var n_clips = len(dataset)
    print("dataset:", n_clips, "clips;", ROUNDS, "rounds ×", B,
          "episodes, budget", EVAL_BUDGET, "steps, goal +25 steps")

    var total_sr: Float64 = 0.0
    var total_pd: Float64 = 0.0
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
        var r = run_lewm_paper_protocol[
            IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB,
            ENC_PROJ_H, ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS,
            PRED_HEADS, PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS,
            B, MPC_HORIZON, "gpu", PRED_DIM_HEAD, 2, 96, EncCLS,
        ](
            wm, starts, goals,
            eval_budget=EVAL_BUDGET,
            scale_x=SCALE_X, scale_y=SCALE_Y,
            act_mean_x=am_x, act_mean_y=am_y,
            act_std_x=as_x, act_std_y=as_y,
            cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES, cem_topk=CEM_TOPK,
            init_std=INIT_STD,
            seed0=1 + round * B,
            viz_path=String("/tmp/lewm_pusht_protocol_recipe_r")
                + String(round) + String(".ppm"),
            ctx=ctx,
            verbose=True,
        )
        print("   round success_rate=", r[0], " mean_pos_diff=", r[1])
        total_sr += r[0]
        total_pd += r[1]

    print("=" * 70)
    print("PAPER-PROTOCOL RESULT (", ROUNDS * B, "episodes ):")
    print("   SUCCESS RATE =", total_sr / Float64(ROUNDS),
          "  (paper LeWM: ~0.90)")
    print("   mean final pos_diff =", total_pd / Float64(ROUNDS), "px",
          " (success needs < 20 px joint + angle < 20°)")
    print("=" * 70)

    starts.free(); goals.free()
    _ = window^; _ = dataset^; _ = wm^
    print("DONE")
