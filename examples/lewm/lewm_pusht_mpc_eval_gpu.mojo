"""LeWM (nn) — solve PushT with the Cross-Entropy Method (GPU).

Plans in LATENT space through the frozen paper-width PushT world model with
the paper's planner: continuous Gaussian CEM (`ContinuousCEMOptimizer`).
Encodes a window's start (frame 0) + goal (frame T-1) latents, then has CEM
optimize a continuous action sequence to drive the predicted latent toward
the goal latent (MSE), versus a Gaussian random-shooter baseline and the
dataset's expert actions.

Paper CEM config: 300 candidates, 30 iterations, top-30 elites, initial
variance 1 (here data-scaled to the expert action magnitude), horizon 5.
Our trained WM has T=6 / H=3, so the in-window max rollout is MPC_HORIZON=4
(NEEDED = H+horizon-1 = 6 = T, using all recorded expert action blocks).

Gate (planner works): cem < random_min — CEM finds action sequences that
reach the goal latent better than random, on a non-collapsed model. expert
is the dataset-action reference. (NOTE: this is latent-space planning toward
a goal latent, the continuous analogue of the Pong §10.9 MPC eval — not yet
closed-loop control on the PushT simulator.)

Loads the paper-width checkpoint `lewm_pusht_paper.ckpt`.
Run (NVIDIA; 224² + ~9k host-synced rollouts — a heavy one-shot eval):
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_mpc_eval_gpu.mojo
  pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_mpc_eval_gpu.mojo --ckpt <run_id>
"""

from std.sys import argv
from noeira.core.run import resolve_checkpoint
from std.math import sqrt
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer
from noeira.experimental.lewm.pong_data import WindowSource
from noeira.experimental.lewm.mpc_continuous import lewm_mpc_eval_continuous
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
comptime PRED_DIM_HEAD = 64     # expanded attention, must match training
comptime PRED_FF = 2048
comptime DEPTH = 6
comptime PRED_PROJ_H = 2048
comptime SIG_PROJ = 2048
comptime SIG_KNOTS = 17
comptime B = 16
comptime FRAMESKIP = 5

comptime MPC_HORIZON = 4         # NEEDED = H+horizon-1 = 6 = T (in-window max)
comptime CEM_ITERS = 30          # paper
comptime CEM_SAMPLES = 300       # paper
comptime CEM_TOPK = 30           # paper
comptime NUM_RANDOM = 300

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT
comptime CKPT_PATH: String = "lewm_pusht_paper.ckpt"

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
    print("LeWM nn — PushT CEM planning eval (latent MPC, GPU)")
    print("=" * 70)
    var ctx = DeviceContext()

    var sampler = PushTOfflineSampler(frameskip=FRAMESKIP, num_steps=T)
    var src = Source.make(sampler^, ctx=ctx)
    var tr = Trainer.make(lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3), ctx=ctx)
    print("loading checkpoint", ckpt, "...")
    tr.load_params(ckpt)

    src.next_batch()
    var pix_t = TileTensor(src.pix_ptr(), row_major[B, PIX]())
    var act_t = TileTensor(src.act_ptr(), row_major[B, ACTIN]())
    var act_host = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    var act_dev = DeviceBuffer[DT](ctx, src.act_ptr(), B * ACTIN, owning=False)
    ctx.enqueue_copy(act_host, act_dev)
    ctx.synchronize()

    # data-scaled initial std: RMS of the expert action blocks (the CEM mean
    # starts at 0 and refits, so this just sets the exploration scale).
    var ss: Float64 = 0.0
    for i in range(B * ACTIN):
        var v = Float64(act_host.unsafe_ptr()[i])
        ss += v * v
    var init_std = sqrt(ss / Float64(B * ACTIN))
    if init_std < 0.1:
        init_std = 0.1
    print("   init_std (expert RMS) =", init_std)

    print("CEM planning (", CEM_SAMPLES, "samples ×", CEM_ITERS, "iters,",
          "horizon", MPC_HORIZON, ") ...")
    var r = lewm_mpc_eval_continuous[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, MPC_HORIZON,
        "gpu", PRED_DIM_HEAD,
    ](
        tr, pix_t, act_t,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](act_host.unsafe_ptr()),
        num_random=NUM_RANDOM, cem_iters=CEM_ITERS, cem_samples=CEM_SAMPLES,
        cem_topk=CEM_TOPK, init_std=init_std, ctx=ctx,
    )
    print()
    print("   planner works (cem < random_min):",
          "YES" if r[3] < r[2] else "no")
    print("   cem beats expert (cem < expert):",
          "yes" if r[3] < r[0] else "no")

    _ = src^; _ = tr^
    print("=" * 70)
    print("DONE")
    print("=" * 70)
