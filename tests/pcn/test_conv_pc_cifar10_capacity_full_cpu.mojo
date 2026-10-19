"""FULL-BUDGET capacity-matched conv-PCN — the one lever that transferred.

Everything in P11-P23 ran at 10k/2ep, 1.6% of the reference's budget. Of all
the levers tried, exactly ONE produced a transferable gain: capacity (P18,
+0.046). The algorithmic fixes (Qi et al.'s spiking schedule and forward
updates, ePC, beta-nudging) were neutral, harmful, or capacity-dependent.

So this is the honest end of the line: the best-understood configuration at a
real budget.

    capacity-matched net (3,859,210 params -- parameter-identical to their
    VGG5), sPC, T=6, beta=1, the P10 recipe at full budget:
    50k + h-flip, 15 epochs, ADAM_LR=2e-4, lr_x=0.025, WD=2e-4, adaptive clip.

Reference points:
    P10 net, same recipe, full budget          0.6565  (~58 min)
    capacity-matched, same recipe, 10k/2ep     0.4905
    their VGG5 with ePC, 50k/25ep              ~0.90 (paper)

Runtime ~2.5-3 h on CPU at ~75 sample-presentations/s. It prints test accuracy
EVERY epoch, so a run killed by laptop sleep still yields the whole curve up to
that point -- only the ability to continue is lost, not the results. If it
dies, read the log and relaunch; there is no checkpoint.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_capacity_full_cpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from std.math import sqrt
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.nn.core.ptr import mptr
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.cifar10 import CIFAR10
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)
from noeira.experimental.pcn.pc_trainer import PCTrainResult
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime EPOCHS = 15                   # full budget
comptime N_TRAIN_SAMPLES = 50000       # full budget
comptime LR_X: Float32 = 0.025
comptime ADAM_LR: Float64 = 0.0002
comptime CLIP_FACTOR: Float64 = 5.0
comptime WD: Float64 = 0.0002
comptime N_TEST_SAMPLES = 2000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32


@always_inline
def _flip_into(
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
):
    for c in range(3):
        for h in range(32):
            var row = c * 1024 + h * 32
            for w in range(32):
                dst[row + w] = src[row + (31 - w)]


comptime NET = PCSequential[
    ConvPCBlock[3, 128, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[128, 256, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[256, 512, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[512, 512, 3, 2, 1, 4, 4, PCReLU],
    PCBlock[2048, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 128, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[128, 256, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[256, 512, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[512, 512, 3, 2, 1, 4, 4, PCReLU],
    PCBlock[2048, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 72)
    print("Conv-PCN CIFAR-10 — FULL-BUDGET capacity-matched (CPU)")
    print("=" * 72)
    print("  net       : capacity-matched to VGG5, N =", NET.N, "levels")
    print("  budget    :", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep  (SCREEN)")
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var ds = CIFAR10()

    # ── Buffers: allocated ONCE, shared by every arm ─────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var snap_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(snap_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)

    # One draw, snapshotted — every arm restores from this.
    NET.pc_init_params[PCXavier, dtype](params)
    for i in range(NET.PARAM_SIZE):
        snap_buf[i] = params_buf[i]

    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var err_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var errors = tile[BATCH, NET.LATENT_DIM](err_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    # ── Arms: (T_infer, spike_sigma) ─────────────────────────────────────────
    var arm_T = [6]
    var arm_LR: List[Float64] = [0.025]
    var arm_EPC = [False]
    var arm_name = [String("sPC T=6 β=1 — full budget")]
    var results: List[Float64] = []
    var walls: List[Float64] = []

    for a in range(len(arm_T)):
        var T_infer = arm_T[a]
        var lr_a = Scalar[dtype](arm_LR[a])
        var use_epc = arm_EPC[a]

        # Reset EVERYTHING that carries state between arms.
        for i in range(NET.PARAM_SIZE):
            params_buf[i] = snap_buf[i]
        memset(grads_buf, 0, NET.PARAM_SIZE)
        memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
        memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
        memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
        memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
        memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
        memset(err_buf_raw, 0, BATCH * NET.LATENT_DIM)

        var step_num: Int = 0
        var best_acc: Float64 = 0.0
        var ema_norm: Float64 = 0.0
        var t0 = perf_counter_ns()

        print("\n  ── arm", arm_name[a], "──")
        print("     epoch | sup_loss  | test_acc | best   | clips | wall_t (s)")
        print("     ------+-----------+----------+--------+-------+-----------")

        for epoch in range(EPOCHS):
            var ep_loss: Float64 = 0.0
            var n_clips: Int = 0
            for batch_idx in range(N_TRAIN_BATCHES):
                for i in range(BATCH):
                    var sidx = batch_idx * BATCH + i
                    var do_flip = ((sidx * 2654435761 + epoch * 40503) >> 13) & 1
                    if do_flip == 1:
                        _flip_into(mptr(ds.train_images).unsafe_offset(sidx * IN), x_buf + i * IN)
                    else:
                        for j in range(IN):
                            x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
                    for c in range(NET.OUT_DIM):
                        y_buf[i * NET.OUT_DIM + c] = 0
                    y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1

                var r: PCTrainResult
                if use_epc:
                    r = TRAINER.compute_grads_only_epc[BATCH](
                        params, grads, errors, latents, mu_eps_buf,
                        a_below_buf, z_below_buf, dx_buf, x_batch, y_batch,
                        T_infer=T_infer, lr_e=lr_a,
                    )
                else:
                    r = TRAINER.compute_grads_only[BATCH](
                        params, grads, latents, mu_eps_buf, a_below_buf,
                        z_below_buf, dx_buf, x_batch, y_batch,
                        T_infer=T_infer, lr_x=lr_a,
                    )
                ep_loss += r.output_loss_final

                var gn: Float64 = 0.0
                for i in range(NET.PARAM_SIZE):
                    var gv = Float64(grads_buf[i])
                    gn += gv * gv
                gn = sqrt(gn)
                if ema_norm > 0.0 and gn > CLIP_FACTOR * ema_norm:
                    var sc = Scalar[dtype]((CLIP_FACTOR * ema_norm) / gn)
                    for i in range(NET.PARAM_SIZE):
                        grads_buf[i] = grads_buf[i] * sc
                    n_clips += 1
                    gn = CLIP_FACTOR * ema_norm
                if ema_norm == 0.0:
                    ema_norm = gn
                else:
                    ema_norm = 0.99 * ema_norm + 0.01 * gn

                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](params, grads, opt_state, opt_global, step_num)
                var keep = Scalar[dtype](1.0 - WD)
                for i in range(NET.PARAM_SIZE):
                    params_buf[i] = params_buf[i] * keep

            var correct: Int = 0
            for tb in range(N_TEST_BATCHES):
                for i in range(BATCH):
                    var sidx = tb * BATCH + i
                    for j in range(IN):
                        x_buf[i * IN + j] = ds.test_images[sidx * IN + j]
                NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
                for i in range(BATCH):
                    var best_c: Int = 0
                    var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
                    for c in range(1, NET.OUT_DIM):
                        var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                        if v > best_v:
                            best_v = v; best_c = c
                    if best_c == Int(ds.test_labels[tb * BATCH + i]):
                        correct += 1
            var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
            if acc > best_acc:
                best_acc = acc
            var el = Float64(perf_counter_ns() - t0) / 1e9
            print("       ", epoch, "  ", fit(String(ep_loss / Float64(N_TRAIN_BATCHES)), 9),
                  "  ", fit(String(acc), 8), "  ", fit(String(best_acc), 6),
                  "  ", n_clips, "  ", el)

        results.append(best_acc)
        walls.append(Float64(perf_counter_ns() - t0) / 1e9)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FULL-BUDGET SUMMARY —", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep")
    print("=" * 72)
    print("  arm                                | best_acc | Δ vs B   | wall (s)")
    print("  ---------------------------+----------+----------+---------")
    for a in range(len(results)):
        print("  ", fit(arm_name[a], 34), " ", fit(String(results[a]), 8),
              " ", fit(String(results[a] - results[0]), 8), " ",
              fit(String(walls[a]), 7))
    print("\n  P10 net, same recipe, full budget : 0.6565")
    print("  this net, same recipe, 10k/2ep     : 0.4905")
    print("  their VGG5 + ePC, 50k/25ep         : ~0.90 (paper)")
    print("  ADAM_LR inherited from P10, NOT retuned for this net.")
