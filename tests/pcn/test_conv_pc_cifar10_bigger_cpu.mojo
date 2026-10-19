"""Conv-PCN CIFAR-10 — MORE CAPACITY/DEPTH (CPU). Background run.

Builds on the stable bigger-budget recipe (best 0.6165) by adding capacity where
it is CHEAP — at low spatial resolution (8×8, 4×4) — leaving the expensive
high-res early convs unchanged. Wider conv3 (64→128), a new stride-1 conv at
4×4 (extra depth, tiny FLOPs), and a doubled readout.

  ConvPCBlock[3,   32, 3, 2, 1, 32, 32, PCIdentity]  # → 32×16×16  (unchanged, expensive)
  ConvPCBlock[32,  64, 3, 2, 1, 16, 16, PCReLU]      # → 64×8×8    (unchanged, expensive)
  ConvPCBlock[64, 128, 3, 2, 1, 8,  8,  PCReLU]      # → 128×4×4   (widened)
  ConvPCBlock[128,128, 3, 1, 1, 4,  4,  PCReLU]      # → 128×4×4   (NEW depth, cheap)
  PCBlock[2048, 512, PCReLU]                          # 2048 → 512  (wider head)
  PCBlock[512, 10, PCIdentity]

Stable recipe: 50k + h-flip, 15 ep, T_INFER=20, ADAM_LR=2e-4, lr_x=0.025,
decoupled WD=2e-4, adaptive grad-clip. Baseline to beat: 0.6165.

Run (long — background it):
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_bigger_cpu.mojo
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
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime EPOCHS = 15
comptime T_INFER = 20
comptime LR_X: Float32 = 0.025
comptime ADAM_LR: Float64 = 0.0002
comptime CLIP_FACTOR: Float64 = 5.0
comptime WD: Float64 = 0.0002
comptime N_TRAIN_SAMPLES = 50000
comptime N_TEST_SAMPLES = 2000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32
comptime PASS_ACC = 0.62


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
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[64, 128, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[128, 128, 3, 1, 1, 4, 4, PCReLU],
    PCBlock[2048, 512, PCReLU],
    PCBlock[512, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[64, 128, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[128, 128, 3, 1, 1, 4, 4, PCReLU],
    PCBlock[2048, 512, PCReLU],
    PCBlock[512, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("Conv-PCN CIFAR-10 BIGGER (CPU): 6 levels, 50k /", EPOCHS,
          "ep / T", T_INFER, "\n")
    var ds = CIFAR10()
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
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
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    var step_num: Int = 0
    var best_acc: Float64 = 0.0
    var ema_norm: Float64 = 0.0
    var t0 = perf_counter_ns()
    print("\n  epoch | sup_loss  | test_acc | best | clips | gnorm | wall_t (s)")
    print("  ------+-----------+----------+------+-------+-------+-----------")
    for epoch in range(EPOCHS):
        var ep_loss: Float64 = 0.0
        var n_clips: Int = 0
        var last_gn: Float64 = 0.0
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

            var r = TRAINER.compute_grads_only[BATCH](
                params, grads, latents, mu_eps_buf, a_below_buf,
                z_below_buf, dx_buf, x_batch, y_batch,
                T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
            )
            ep_loss += r.output_loss_final

            var gn: Float64 = 0.0
            for i in range(NET.PARAM_SIZE):
                var gv = Float64(grads_buf[i])
                gn += gv * gv
            gn = sqrt(gn)
            last_gn = gn
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
        print("    ", epoch, "  ", ep_loss / Float64(N_TRAIN_BATCHES), "  ",
              acc, "  ", best_acc, "  ", n_clips, "  ",
              fit(String(last_gn), 6), "  ", el)

    print("\n  best test accuracy =", best_acc)
    print("  (bigger-budget baseline 0.6165 @ 32/64/64)")
    if best_acc >= PASS_ACC:
        print("✅ PASS — more capacity lifts conv-PCN CIFAR (best", best_acc, ")")
    else:
        print("⚠️  best", best_acc, "< target", PASS_ACC, "(still informative)")
