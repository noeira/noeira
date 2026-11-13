"""Backprop control for conv-PCN on CIFAR-10 — the SAME net, recipe and harness.

Every conv-PCN accuracy in `docs/PCN_CONV_DESIGN.md` (0.465 -> ... -> 0.734)
was read against a PAPER's number on a different architecture (VGG-7 with
max-pooling, 25 epochs). None was read against backprop on OUR net, so "the
remaining gap is ~0.156" never said how much of that gap is predictive coding
and how much is the stride-2 all-conv net, the MSE readout, 15 epochs and an
un-retuned learning rate.

This file is `test_conv_pc_cifar10_capacity_full_cos_wd_cpu.mojo` (P26, the
line's best: 0.734) with ONE axis changed: the gradient. `--arm bp` (default)
uses `PCTrainer.compute_grads_only_bp`, the exact gradient of the same
0.5*||y - y_hat||^2 readout loss (gated in `test_bp_grad_parity.mojo`);
`--arm spc` re-runs P26 itself in this binary for an interleaved comparison.
Same parameter-identical-to-VGG5 net, same init draw, same 50k + h-flip,
15 epochs, Adam 2e-4, cosine + 2-epoch warmup, scaled weight decay, adaptive
clipping, same 2000-image test protocol.

Read it as:
    BP ~= 0.734   PC is at parity on this net; the gap to the paper is the
                  net and the recipe, not the learning rule.
    BP >> 0.734   the PC training itself is leaving accuracy on the table.

Beyond the P26 protocol it also prints the last-epoch accuracy on the full
10k test set and on 2000 training images, so the comparison does not rest on
a best-of-15 pick over 2000 images.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_capacity_full_bp_cpu.mojo
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_capacity_full_bp_cpu.mojo --arm spc
"""

from std.memory import alloc, unsafe_memset
from std.sys import argv
from std.time import perf_counter_ns
from std.math import sqrt
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.nn.core.ptr import mptr
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.experimental.pcn.pc_scheduler import CosineWarmupSchedule
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
comptime N_FULL_TEST_BATCHES = 10000 // BATCH
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
comptime SCHED = CosineWarmupSchedule[WARMUP_EPOCHS=2, MIN_SCALE=0.1]


def main() raises:
    print("=" * 72)
    print("Conv CIFAR-10 — BACKPROP CONTROL on the P26 net and recipe (CPU)")
    print("=" * 72)
    print("  net       : capacity-matched to VGG5, N =", NET.N, "levels")
    print("  budget    :", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep")
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var ds = CIFAR10()

    # ── Buffers: allocated ONCE, shared by every arm ─────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var snap_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(snap_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
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
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    # ── Arms: (T_infer, spike_sigma) ─────────────────────────────────────────
    var use_bp = True
    var args = argv()
    if len(args) > 2 and String(args[1]) == "--arm":
        if String(args[2]) == "spc":
            use_bp = False
        elif String(args[2]) != "bp":
            raise Error("--arm takes bp or spc")
    var arm_T = [6]
    var arm_LR: List[Float64] = [0.025]
    var arm_name = [
        String("BACKPROP, same net + recipe") if use_bp
        else String("sPC T=6 β=1 (P26 re-run)")
    ]
    var results: List[Float64] = []
    var walls: List[Float64] = []

    for a in range(len(arm_T)):
        var T_infer = arm_T[a]
        var lr_a = Scalar[dtype](arm_LR[a])

        # Reset EVERYTHING that carries state between arms.
        for i in range(NET.PARAM_SIZE):
            params_buf[i] = snap_buf[i]
        unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
        unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
        unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
        unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
        unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
        unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

        var step_num: Int = 0
        var best_acc: Float64 = 0.0
        var ema_norm: Float64 = 0.0
        var t0 = perf_counter_ns()

        print("\n  ── arm", arm_name[a], "──")
        print("     epoch | sup_loss  | test_acc | best   | clips | wall_t (s)")
        print("     ------+-----------+----------+--------+-------+-----------")

        for epoch in range(EPOCHS):
            var lr_scale = SCHED.lr_scale_at(epoch, EPOCHS)
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
                if use_bp:
                    r = TRAINER.compute_grads_only_bp[BATCH](
                        params, grads, latents, mu_eps_buf, a_below_buf,
                        z_below_buf, x_batch, y_batch,
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
                OPT.step[NET.PARAM_SIZE, dtype](
                    params, grads, opt_state, opt_global, step_num, lr_scale
                )
                var keep = Scalar[dtype](1.0 - WD * lr_scale)
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

        # Last-epoch weights on the FULL test set and on seen training images.
        var full_correct: Int = 0
        for tb in range(N_FULL_TEST_BATCHES):
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
                    full_correct += 1
        var train_correct: Int = 0
        for tb in range(N_TEST_BATCHES):
            for i in range(BATCH):
                var sidx = tb * BATCH + i
                for j in range(IN):
                    x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
            NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
            for i in range(BATCH):
                var best_c: Int = 0
                var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
                for c in range(1, NET.OUT_DIM):
                    var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                    if v > best_v:
                        best_v = v; best_c = c
                if best_c == Int(ds.train_labels[tb * BATCH + i]):
                    train_correct += 1
        print("     last epoch, full test set (", N_FULL_TEST_BATCHES * BATCH, "):",
              Float64(full_correct) / Float64(N_FULL_TEST_BATCHES * BATCH))
        print("     last epoch, train images  (", N_TEST_BATCHES * BATCH, "):",
              Float64(train_correct) / Float64(N_TEST_BATCHES * BATCH))

        results.append(best_acc)
        walls.append(Float64(perf_counter_ns() - t0) / 1e9)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BACKPROP CONTROL SUMMARY —", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep")
    print("=" * 72)
    print("  arm                                | best_acc | Δ vs B   | wall (s)")
    print("  ---------------------------+----------+----------+---------")
    for a in range(len(results)):
        print("  ", fit(arm_name[a], 34), " ", fit(String(results[a]), 8),
              " ", fit(String(results[a] - results[0]), 8), " ",
              fit(String(walls[a]), 7))
    print("\n  P10 net, no schedule, full budget  : 0.6565")
    print("  this net, no schedule, full budget : 0.7035 (plateau from ep 3)")
    print("  this net, cosine, UNSCALED WD      : 0.7155 (tail declining)")
    print("  this net, cosine, scaled WD (P26)  : 0.734  (sPC T=6, recorded)")
    print("  their VGG5 + ePC, 50k/25ep         : ~0.90 (paper)")
    print("  ADAM_LR inherited from P10, NOT retuned. Only the gradient differs.")
